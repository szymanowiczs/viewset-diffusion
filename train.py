import json
import glob
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt

from pytorch_lightning.lite import LightningLite

from data_manager import get_data_manager

from model.reconstructor import Reconstructor
from model.diffusion import ViewsetDiffusion

from utils import set_seed

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups

from ema_pytorch import EMA

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class Lite(LightningLite):
    def run(self, cfg):
        vis_dir = os.getcwd()

        dict_cfg = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

        wandb_run = wandb.init(project=cfg.wandb.project, reinit=True,
                    config=dict_cfg,group="decoder_debug")

        set_seed(cfg.general.random_seed)

        reconstructor = Reconstructor(cfg)

        objective = 'pred_x0'
        diffuser = ViewsetDiffusion(cfg,
                                    reconstructor,
                                    objective=objective,
                                    image_size=cfg.data.input_size[0],
                                    timesteps=cfg.model.diffuser.steps,
                                    sampling_timesteps=cfg.eval.sampling_timesteps,
                                    loss_type=cfg.optimization.loss,
                                    min_snr_loss_weight=cfg.optimization.clamp_min_snr,
                                    beta_schedule=cfg.model.diffuser.beta_schedule)

        # optimizer settings
        lr = cfg.optimization.lr  # for MLP
        n_iter = cfg.optimization.n_iter

        if cfg.optimization.continue_from_checkpoint != 'none':
            print('Loading a pretrained model from ', 
                  cfg.optimization.continue_from_checkpoint)
            checkpoint = self.load(os.path.join(cfg.optimization.continue_from_checkpoint,
                                                "model.pth")) 
            pretrained_dict = {}
            non_loaded_keys = []
            model_dict = diffuser.state_dict()
            for k, v in checkpoint["diffuser"].items():
                if "reconstructor" in k and ("reconstructor.init_conv" not in k and \
                                             "reconstructor.out" not in k):
                    pretrained_dict[k.split("_module.module.")[1]] = v
                else:
                    non_loaded_keys.append(k)
            model_dict.update(pretrained_dict)
            diffuser.load_state_dict(model_dict)
            iteration_start = checkpoint["iteration"]
            print('Loaded layers: ', pretrained_dict.keys())
            print('Not loaded layers: ', non_loaded_keys)
            if cfg.optimization.freeze_pretrained:
                print('freezing the loaded parameters')
                for name, param in diffuser.named_parameters():
                    if name in pretrained_dict.keys():
                        param.requires_grad = False
        else:
            iteration_start = 0

            print('No model found in {}, training from scratch'.format(os.path.join(vis_dir, "model.pth")))
        print('Model loaded.')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, diffuser.parameters()),
                            lr=lr,
                            betas=(cfg.optimization.betas[0],
                                    cfg.optimization.betas[1]))
        print('Setting up ddp')
        diffuser, optimizer = self.setup(diffuser, optimizer)
        
        print('Setting up ema')
        if cfg.optimization.ema.use and self.is_global_zero:
            ema = EMA(diffuser, 
                      beta=cfg.optimization.ema.decay,
                      update_every=cfg.optimization.ema.update_every)
            ema = self.to_device(ema)
        print('EMA set up.')

        diffuser.train()
        optimizer.zero_grad()
        dataset = get_data_manager(cfg)
        print('Loaded data manager')

        # a second time setting random seed ensures the same order of dataloading
        set_seed(cfg.general.random_seed)

        # batch size
        # number of images will be batch_size x number of input conditioning images
        batch_size = cfg.optimization.batch_size 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=True, num_workers=4,
                                persistent_workers=True)

        if cfg.data.dataset_type == "srn":
            val_single = True
        else:
            val_single = False
        val_dataset = get_data_manager(cfg, split='val',
                convert_to_single_conditioning=val_single,
                convert_to_double_conditioning=False,
                for_training=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True, pin_memory=True, 
                                    num_workers=4, persistent_workers=True)

        dataloader, val_dataloader = self.setup_dataloaders(dataloader, val_dataloader)
        val_dataloader_iterator = iter(val_dataloader)
        iteration = iteration_start - 1
        print('Starting training')
        for num_epoch in range((n_iter - iteration_start) // len(dataloader) + 1):
            dataloader.sampler.set_epoch(num_epoch)
            for data in dataloader:
                iteration += 1

                losses = diffuser(data["x_in"], data, iteration = iteration)

                total_loss = sum([l  for l in losses if not torch.isnan(l)])
                self.backward(total_loss)

                optimizer.step()
                optimizer.zero_grad()

                if cfg.optimization.ema.use and self.is_global_zero:
                    ema.update()

                if (iteration + 1) % 10 == 0:
                    try:
                        val_data = next(val_dataloader_iterator)
                    except StopIteration:
                        val_dataloader_iterator = iter(val_dataloader)
                        val_data = next(val_dataloader_iterator)

                    for k, v in val_data.items():
                        val_data[k] = self.to_device(v)
                    with torch.no_grad():
                        losses = diffuser(val_data["x_in"], val_data)
                        val_loss = sum([l for l in losses if not torch.isnan(l)])

                if self.is_global_zero and (iteration + 1) % 10 == 0:
                    if len(losses) == 2:
                        wandb.log({"loss_unseen": np.log(losses[0].item() + 1e-8)}, step=iteration)
                        wandb.log({"loss_seen": np.log(losses[1].item() +1e-8)}, step=iteration)
                    
                    wandb.log({"total_loss": np.log(total_loss.item())}, step=iteration)

                    wandb.log({"val_loss": np.log(val_loss.item())}, step=iteration)

                if iteration % (cfg.optimization.n_iter // 10) == 0 and self.is_global_zero:
                    try:
                        val_data = next(val_dataloader_iterator)
                    except StopIteration:
                        val_dataloader_iterator = iter(val_dataloader)
                        val_data = next(val_dataloader_iterator)
                    for k, v in val_data.items():
                        val_data[k] = self.to_device(v)
                    # forward function of diffuser normalizes the input to [-1, 1]
                    # in visualisation we need to do it manually
                    noisy_input = torch.randint(0, cfg.model.diffuser.steps, (val_data["x_in"].shape[0],), ).long()
                    noisy_input = self.to_device(noisy_input)
                    log_visualisations(diffuser, val_data, noisy_input, iteration, cfg, split="val")

                    noisy_input = torch.randint(0, cfg.model.diffuser.steps, (data["x_in"].shape[0],), ).long()
                    noisy_input = self.to_device(noisy_input)
                    log_visualisations(diffuser, data, noisy_input, iteration, cfg, split="training")

                if (iteration+1) % cfg.optimization.save_every == 0:
                    diffuser.eval()
                    if cfg.optimization.ema.use and self.is_global_zero:
                        print('Saving ema model')
                        torch.save({"diffuser": ema.ema_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "iteration": iteration},
                                os.path.join(vis_dir, "model.pth"))
                    else:
                        self.print('Saving non-ema model')
                        self.save({"diffuser": diffuser.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "iteration": iteration},
                                os.path.join(vis_dir, "model.pth"))
                    self.print("Saved model at iteration {}".format(iteration))
                    diffuser.train()                

                # without noise we train for fewer iterations so need to earlier
                if cfg.optimization.hard_mining_proportion == 0.0:
                    interval = 5000
                    cutoff_iteration = 30000
                else:
                    interval = 8000
                    cutoff_iteration = 120000

                if (iteration+1) % interval == 0 and iteration >= cutoff_iteration:
                    diffuser.eval()
                    if cfg.optimization.ema.use and self.is_global_zero:
                        print('Saving ema model')
                        torch.save({"diffuser": ema.ema_model.state_dict(),
                                    "iteration": iteration},
                                os.path.join(vis_dir, 
                                             "model_{}.pth".format(iteration + 1)))
                    else:
                        self.print('Saving non-ema model')
                        self.save({"diffuser": diffuser.state_dict(),
                                "iteration": iteration},
                                os.path.join(vis_dir, 
                                             "model_{}.pth".format(iteration + 1)))
                    diffuser.train()

        wandb_run.finish()

def log_visualisations(diffuser, data, noisy_input, iteration, cfg,
                       split):
    diffuser.module.module.vis_iteration(data["x_in"] * 2 - 1, 
                        noisy_input,
                        data, 
                        iteration = iteration,
                        split=split)

    vis_batch_size = 4
    n_rows = int(np.sqrt(vis_batch_size))

    diffuser.eval()
    batches = num_to_groups(vis_batch_size, min(cfg.optimization.batch_size, data["x_in"].shape[0]))
    i_start = 0
    resizing = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)

    # training with triplanes does not support visualisation during training
    if cfg.model.unet.volume_repr == "triplanes":
        vis_rot_cf_guidances = []
    else:
        vis_rot_cf_guidances = [-1.0, 0.0]

    for cf_guidance_weight in vis_rot_cf_guidances:
        all_images_list = []
        gt_cond_list = []
        for batch_size in batches:
            all_images_list.append(diffuser.module.module.ddim_sample(
                        (batch_size, data["x_in"].shape[1], 3, cfg.data.input_size[0], cfg.data.input_size[0]),
                        {k: v[i_start:i_start + batch_size] for k, v in data.items()},
                                classifier_free_guidance_w=cf_guidance_weight,
                                render_spinning_volume=True)
                                )
            gt_cond_list.append(data["x_cond"][i_start:i_start + batch_size] * 0.5 + 0.5)
        all_images = torch.cat(all_images_list, dim = 0)
        all_gt = torch.cat(gt_cond_list, dim = 0)
        if cf_guidance_weight == -1.0:
            diffused_views_start = 0
        else:
            diffused_views_start = data["x_cond"].shape[1]
            for cond_view_idx in range(data["x_cond"].shape[1]):
                cond_view_result = all_images[:, cond_view_idx].permute(0, 2, 3, 1)
                cond_view_result = cond_view_result.reshape(
                    n_rows,n_rows, *cond_view_result.shape[1:]
                    )
                rows = [torch.hstack([im for im in sample_row]) for sample_row in cond_view_result]
                cond_view_result = torch.vstack(rows).permute(2, 0, 1)
                wandb.log({"cond_view_{}_{}".format(cond_view_idx, split): 
                            wandb.Image(cond_view_result)},
                            step=iteration)
                
                cond_view_gt = all_gt[:, cond_view_idx].permute(0, 2, 3, 1)
                cond_view_gt = cond_view_gt.reshape(
                    n_rows,n_rows, *cond_view_gt.shape[1:]
                    )
                rows = [torch.hstack([im for im in sample_row]) for sample_row in cond_view_gt]
                cond_view_gt = torch.vstack(rows).permute(2, 0, 1)
                wandb.log({"cond_gt_{}_{}".format(cond_view_idx, split):
                            wandb.Image(cond_view_gt)},
                            step=iteration)

        for diffused_view_idx in range(data["x_in"].shape[1]):
            diffused_view_result = all_images[:, diffused_views_start + diffused_view_idx].permute(0, 2, 3, 1)
            diffused_view_result = diffused_view_result.reshape(
                    n_rows,n_rows, *diffused_view_result.shape[1:]
                    )
            rows = [torch.hstack([im for im in sample_row]) for sample_row in diffused_view_result]
            diffused_view_result = torch.vstack(rows).permute(2, 0, 1)
            print(diffused_view_result.shape)
            wandb.log({"diffused_view_{}_{}_{}".format(diffused_view_idx, cf_guidance_weight, split): 
                        wandb.Image(diffused_view_result)},
                        step=iteration)
        
        grids = []
        for rot_idx in range(diffused_views_start+data["x_in"].shape[1], all_images.shape[1]):
            samples_this_angle = resizing(all_images[:, rot_idx, ...]).permute(0, 2, 3, 1)
            samples_this_angle = samples_this_angle.reshape(n_rows, n_rows, *samples_this_angle.shape[1:])
            rows = [torch.hstack([im for im in sample_row]) for sample_row in samples_this_angle]
            grid = torch.vstack(rows)
            grids.append((np.clip(grid.permute(2, 0, 1).detach().cpu().numpy(), 0, 1)*255).astype(np.uint8))
        
        wandb.log({"rot_{}_{}".format(cf_guidance_weight, split):
                   wandb.Video(np.asarray(grids), fps=4, format="gif")},
                   step=iteration)

    diffuser.train()

@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):
    Lite(strategy="ddp", devices=cfg.general.devices, accelerator="gpu").run(cfg)
if __name__=="__main__":
    main()