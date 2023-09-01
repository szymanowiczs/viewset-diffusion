import os

from omegaconf import OmegaConf

import torch
import torchvision.utils as tv_uils
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

from . import set_seed
from . import Reconstructor
from . import ViewsetDiffusion
from . import get_data_manager

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups

class Generator():
    """ 
    Universal generator for qualitative evaluation
    takes in parameters
      - indexes of images in the dataset to use for camera pose / conditioning
      - number of clean conditioning images
      - number of noisy images in viewset
    loads appropriate model and dataset
    load:
      - volume model
      - diffusion model
      - dataset"""
    def __init__(self, model_path, device, seed=0,
                 deterministic=False):
        # for reproducibility
        set_seed(seed=seed)
        # experiment path should be a folder that contains /hydra folder
        # with the config.yaml file and a model.pth file
        experiment_path = os.path.dirname(model_path)
        cfg = OmegaConf.load(os.path.join(experiment_path, ".hydra", "config.yaml"))
        self.experiment_path = experiment_path
        self.cfg = cfg

        self.device = device

        # keep both the reconstructor and the diffuser as
        # attributes of the generator - reconstructor can
        # be used for deterministic reconstruction testing
        self.reconstructor = Reconstructor(cfg)
        self.diffuser = ViewsetDiffusion(cfg,
                                    self.reconstructor,
                                    objective='pred_x0',
                                    image_size=cfg.data.input_size[0],
                                    timesteps=cfg.model.diffuser.steps,
                                    sampling_timesteps=cfg.eval.sampling_timesteps,
                                    loss_type=cfg.optimization.loss,
                                    min_snr_loss_weight=cfg.optimization.clamp_min_snr,
                                    beta_schedule=cfg.model.diffuser.beta_schedule).to(self.device)
        
        if deterministic:
            self.diffuser.num_timesteps = 1
            self.diffuser.sampling_timesteps = 1

        checkpoint = torch.load(model_path,
                                map_location=self.device) 
        self.diffuser.load_state_dict({k.split("_module.module.")[1]: v for k, v in checkpoint["diffuser"].items()})
        self.diffuser.eval()

        self.dataset = None
        self.convert_to_double_conditioning=None
        self.convert_to_single_conditioning=None

    def prepare_dataset(self, split):
        """
        Updates dataset with the appropriate conversion (single vs double conditioning)
        """
        # load dataset - use validation by default
        self.dataset = get_data_manager(self.cfg, split=split,
                convert_to_double_conditioning=self.convert_to_double_conditioning,
                convert_to_single_conditioning=self.convert_to_single_conditioning)

    def update_dataset(self, N_clean, split, cf_guidance,
                       with_index_selection=False):
        if N_clean == 1:
            if not self.convert_to_single_conditioning:
                self.convert_to_single_conditioning = True
                self.convert_to_double_conditioning = False
        elif N_clean == 2:
            if not self.convert_to_double_conditioning:
                self.convert_to_single_conditioning = False
                self.convert_to_double_conditioning = True
        else:
            if not self.convert_to_single_conditioning:
                self.convert_to_single_conditioning = True
                self.convert_to_double_conditioning = False
            assert N_clean == 0, "N_clean must be 0, 1, or 2"
            assert cf_guidance == -1.0, "CFG must be -1.0 for unconditional generation"

        self.prepare_dataset(split = split)

        if with_index_selection and self.cfg.data.dataset_type == "co3d":
            self.dataset.select_testing_idxs()

    def reshape_and_save_samples(self, samples, data, N_clean, 
                                 N_noisy, split, cf_guidance, seed=0):
        out_dir_name = os.path.join(self.experiment_path, '{}_{}_{}_{}_{}'.format(
            split, N_clean, N_noisy, cf_guidance, seed))
        if not os.path.exists(out_dir_name):
            os.makedirs(out_dir_name)
        n_rows = int(np.sqrt(samples.shape[0]))

        resizing = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)

        if cf_guidance != -1.0:
            for cond_view_idx in range(N_clean):
                cond_view_result = resizing(samples[:, cond_view_idx])
                tv_uils.save_image(cond_view_result, 
                    os.path.join(out_dir_name, "cond_view_{}.png".format(cond_view_idx)),
                    nrow=n_rows)

                cond_view_gt = resizing(data["x_cond"][:, cond_view_idx]) * 0.5 + 0.5
                tv_uils.save_image(cond_view_gt, 
                    os.path.join(out_dir_name, "cond_gt_{}.png".format(cond_view_idx)),
                    nrow=n_rows)
                
        for diffused_view_idx in range(N_noisy-1):
            diffused_view_result = resizing(samples[:, N_clean + diffused_view_idx])
            tv_uils.save_image(diffused_view_result, 
                    os.path.join(out_dir_name, "diffused_view_{}.png".format(diffused_view_idx)),
                    nrow=n_rows)
        
        diffused_target = resizing(samples[:, N_clean + N_noisy - 1])
        tv_uils.save_image(diffused_target, 
                os.path.join(out_dir_name, "diffused_view_target.png"),
                nrow=n_rows)
        
        gt_target = resizing(data["validation_imgs"][:, -1])
        tv_uils.save_image(gt_target, 
                os.path.join(out_dir_name, "gt_target.png"),
                nrow=n_rows)

        grids = []
        n_rows = int(np.sqrt(samples.shape[0]))
        for rot_idx in range(N_clean+N_noisy, samples.shape[1]):
            samples_this_angle = resizing(samples[:, rot_idx, ...]).permute(0, 2, 3, 1)
            samples_this_angle = samples_this_angle.reshape(n_rows, n_rows, *samples_this_angle.shape[1:])
            rows = [torch.hstack([im for im in sample_row]) for sample_row in samples_this_angle]
            grid = torch.vstack(rows)
            grids.append(Image.fromarray((
                np.clip(grid.detach().cpu().numpy(), 0, 1)*255
                ).astype(np.uint8)))
            tv_uils.save_image(grid.permute(2, 0, 1), 
                os.path.join(out_dir_name, "rot_{}.png".format(str(rot_idx).zfill(3))),
                nrow=n_rows)
        
        gif = grids[0]
        gif.save(fp=os.path.join(out_dir_name, 'volume.gif'),
                 format='GIF',
                 append_images=grids[1:],
                 save_all=True,
                 duration=100,
                 loop=0)

    @torch.no_grad()
    def generate_samples(self, dataset_idxs, N_clean, N_noisy, split='val',
                         cf_guidance = 0.0, use_testing_protocol = False):
        """
        Args:
            dataset_idxs: list of indexes of images in the dataset to use for 
                camera pose / conditioning. 
            N_clean: number of clean conditioning images
            N_noisy: number of noisy images in viewset
        """
        # comment the line below to avoid repeated dataset loading in quantitative eval
        self.update_dataset(N_clean, split, cf_guidance)

        # assert N_noisy > 0, "Wrong function for deterministic reconstruction"
        # group samples into batches
        batches = num_to_groups(len(dataset_idxs), 
            # a heuristic for how many images will fit on the GPU
            (self.cfg.optimization.batch_size * 4) // (N_clean + N_noisy))

        batch_idx_start = 0

        output_all_samples = None
        gt_data = {"validation_imgs": [], 
                   "x_cond": [],
                   "test_imgs": []}
        for batch_size in batches:
            batch_idxs = dataset_idxs[batch_idx_start:batch_idx_start + batch_size]
            # conditioning and target images
            batch_data = {k: [] for k in ["training_imgs", 
                                          "validation_imgs",
                                          "x_in", 
                                          "x_cond",
                                          "pose_embed",
                                          "val_pose_embeds",
                                          "target_Rs",
                                          "target_Ts",
                                          "background"]}
            if self.cfg.data.dataset_type == "co3d":
                batch_data["principal_points"] = []
                batch_data["focal_lengths"] = []
            if use_testing_protocol:
                for k in ["test_Rs", "test_Ts", "test_imgs"]:
                    batch_data[k] = []
            for ex_idx in batch_idxs:
                if use_testing_protocol:
                    ex_with_virtual_views = self.dataset.get_item_for_testing(ex_idx, N_noisy)
                else:
                    ex_with_virtual_views = self.dataset.get_item_with_virtual_views(ex_idx, N_noisy - 1)
                for k, v in ex_with_virtual_views.items():
                    batch_data[k].append(v.unsqueeze(0).to(self.device))
            for k, v in batch_data.items():
                batch_data[k] = torch.cat(v, dim=0)
            # generate samples
            for k in gt_data.keys():
                if k in batch_data.keys() or use_testing_protocol:
                    gt_data[k].append(batch_data[k])
            # The first N_clean images are conditioning images
            # The next N_noisy - 1 images are noisy virtual views
            # The last image is the target image
            # DDIM sampling will also return a set of N_vis renders, at the end of the viewset
            samples = self.diffuser.ddim_sample((len(batch_idxs), N_noisy, 3, self.cfg.data.input_size[0], 
                                                 self.cfg.data.input_size[1]),
                                                 batch_data,
                                                 render_spinning_volume=True,
                                                 classifier_free_guidance_w=cf_guidance)
            if output_all_samples is None:
                output_all_samples = samples
            else:
                output_all_samples = torch.cat([output_all_samples, samples], dim=0)

            batch_idx_start += batch_size
        for k in gt_data.keys():
            if k in batch_data.keys() or use_testing_protocol:
                gt_data[k] = torch.cat(gt_data[k], dim=0)

        return output_all_samples, gt_data
