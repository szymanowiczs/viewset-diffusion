from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, reduce, extract
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import identity

from . import get_Ts_and_Rs_loop

import torch
import numpy as np

from collections import namedtuple
from functools import partial
import tqdm

import wandb

# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L97
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class ViewsetDiffusion(GaussianDiffusion):
    def __init__(
        self,
        cfg,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__(model,
                        image_size = image_size,
                        timesteps = timesteps,
                        sampling_timesteps = sampling_timesteps,
                        loss_type = loss_type,
                        objective = objective,
                        beta_schedule = beta_schedule,
                        schedule_fn_kwargs = schedule_fn_kwargs,
                        ddim_sampling_eta = ddim_sampling_eta,
                        auto_normalize = auto_normalize,
                        min_snr_loss_weight = min_snr_loss_weight, # https://arxiv.org/abs/2303.09556
                        min_snr_gamma = min_snr_gamma)
        self.cfg = cfg

    def view_dropout(self, Cond):
        """
        Gets the indices of images that should be used for reconstruction 
        and the indices of images that should be rendered.
        """
        clean_idx = torch.arange(Cond+1)[:1]
        noisy_idxs = torch.arange(Cond+1)[1:]

        # drop out views
        if self.cfg.data.always_drop_at_least_one_view:
            # drop the clean view
            clean_dropout_rand = torch.rand(1)
            if clean_dropout_rand < self.cfg.optimization.hard_mining_proportion:
                # drop (optionally) one of the noisy views
                if self.cfg.optimization.noisy_dropout_proportion == 0.5:
                    # 2 noisy or 1 noisy with prob 0.0
                    rand_start = torch.randint(Cond, (1,))
                    idxs_to_keep = torch.arange(Cond+1)[1 + rand_start:]
                    noisy_idx_droped_out = torch.arange(Cond+1)[1 : 1 + rand_start]
                elif self.cfg.optimization.noisy_dropout_proportion == 1.0:
                    # 1 noisy
                    idxs_to_keep = torch.arange(Cond+1)[Cond:]
                    noisy_idx_droped_out = torch.arange(Cond+1)[1:Cond]
                elif self.cfg.optimization.noisy_dropout_proportion == 0.0:
                    # 2 noisy
                    idxs_to_keep = torch.arange(Cond+1)[1:]
                    noisy_idx_droped_out = torch.arange(Cond+1)[1:1]
                else:
                    raise NotImplementedError
                clean_idx_dropped_out = torch.arange(Cond+1)[:1]
            # keep the clean view
            else:
                noisy_dropout_rand = torch.rand(1)
                if noisy_dropout_rand < self.cfg.optimization.keep_clean_only_proportion:
                    idxs_to_keep = clean_idx
                else:
                    idxs_to_keep = torch.cat([clean_idx, noisy_idxs[:1]])   
        # do not drop out views
        else:
            idxs_to_keep = None

        if self.cfg.optimization.penalize == "seen":
            idxs_to_render = idxs_to_keep
        elif self.cfg.optimization.penalize == "all":
            idxs_to_render = torch.arange(Cond+1)
        elif self.cfg.optimization.penalize == "seen_and_one_more":
            if clean_dropout_rand < self.cfg.optimization.hard_mining_proportion:
                idxs_to_render = torch.cat([idxs_to_keep, clean_idx_dropped_out])
            else:
                idxs_to_render = torch.cat([idxs_to_keep, noisy_idxs[1:]])
        else:
            raise NotImplementedError
        
        return idxs_to_keep, idxs_to_render

    def p_losses(self, x_start, t, cond, noise = None, iteration = None):

        # sample noise for the noisy images
        b, Cond, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start = x_start.reshape(b*Cond, c, h, w), 
                          t = t.unsqueeze(1).expand(b, Cond).reshape(b*Cond,), 
                          noise = noise.reshape(b*Cond, c, h, w)
                          ).reshape(b, Cond, c, h, w)
        # view dropout
        idxs_to_keep, idxs_to_render = self.view_dropout(Cond)

        # optionally get self-conditioning
        if self.cfg.model.unet.self_condition:
            cond["x_self_condition"] = torch.zeros_like(x)
            if torch.rand(1) < 0.5:
                if self.cfg.model.unet.self_condition_images == "seen":
                    idxs_to_use_for_self_conditioning = idxs_to_keep
                else:
                    raise NotImplementedError
                # this forward pass has to render all images in the set
                if self.cfg.model.unet.self_condition_detach:
                    with torch.no_grad():
                        volume_model_out = self.model(x, t, cond,
                                                      idxs_to_keep=idxs_to_use_for_self_conditioning,
                                                      stratified_sampling=self.cfg.render.stratified_sampling
                                                      )
                else:   
                    volume_model_out = self.model(x, t, cond,
                                                  idxs_to_keep=idxs_to_use_for_self_conditioning,
                                                  stratified_sampling=self.cfg.render.stratified_sampling
                                                  )
                cond["x_self_condition"] = volume_model_out[:, cond["x_cond"].shape[1]:, ...]
        
        # model forward pass
        volume_model_out = self.model(x, t, cond,
                                      idxs_to_keep=idxs_to_keep,
                                      idxs_to_render=idxs_to_render,
                                      stratified_sampling=self.cfg.render.stratified_sampling)
        
        # compute non-reduced loss
        loss = self.loss_fn(volume_model_out[:, :, :3, ...], 
                            cond["training_imgs"][:, idxs_to_render, :3, ...], 
                            reduction = 'none')
        
        # weigh loss
        was_dropped_out = torch.logical_not(torch.isin(idxs_to_render, idxs_to_keep))
        if self.cfg.optimization.weigh_terms:
            if self.cfg.optimization.weigh_clean and 0 in idxs_to_keep:
                t_weighting = torch.cat([t.unsqueeze(1) * 0,
                                        t.unsqueeze(1).expand(b, Cond)], dim = 1)[:, idxs_to_render]
            else:
                t_weighting = t.unsqueeze(1).expand(b, len(idxs_to_render))
            if self.cfg.optimization.weigh_loss_unseen != 1.0:
                # was_dropped_out is in order of idxs_to_render
                loss[:, was_dropped_out, ...] *= self.cfg.optimization.weigh_loss_unseen
            if self.cfg.optimization.normalize_unseen_to_a_third and len(idxs_to_keep) == 1:
                loss[:, was_dropped_out, ...] *= 0.5
        loss = loss.reshape(b*len(idxs_to_render), -1)
        if self.cfg.optimization.weigh_terms:
            loss = loss * extract(self.loss_weight, 
                                      t_weighting.reshape(
                                        b*t_weighting.shape[1],), 
                                      loss.shape)
        return [loss.reshape(b, -1, loss.shape[1])[
                                :, was_dropped_out, :].mean(),
                loss.reshape(b, -1, loss.shape[1])[
                                :, torch.logical_not(was_dropped_out), :].mean()]
    
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, 
                          return_all_renders = False):

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        model_output = self.model(x, t, x_self_cond)
        model_output = maybe_clip(model_output)[:, :, :3, ...]
        if return_all_renders:
            return model_output

        assert self.objective == 'pred_x0', "Objective can only be the x0 formulation"
        # predict noise only for the noisy images
        x_start = model_output[:, x_self_cond["x_cond"].shape[1]:, :3, ...]
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)        
    
    @torch.no_grad()
    def vis_iteration(self, x_start, t, cond, noise = None, iteration = None,
                      split = 'training'):
        
        # for every pass through the model we want to show the inputs, outputs
        # and a red border if the image was dropped out
        vis_columns = []

        # sample noise for the noisy images
        b, Cond, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start = x_start.reshape(b*Cond, c, h, w), 
                          t = t.unsqueeze(1).expand(b, Cond).reshape(b*Cond,), 
                          noise = noise.reshape(b*Cond, c, h, w)
                          ).reshape(b, Cond, c, h, w)

        # view dropout
        idxs_to_keep, idxs_to_render = self.view_dropout(Cond)

        rows = []
        imgs_in = torch.cat([cond["x_cond"][0], x[0]], 
                            dim = 0)[:, :3, ...].permute(0, 2, 3, 1).cpu() * 0.5 + 0.5
        for img_idx in range(cond["x_cond"].shape[1] + Cond):
            img_in = imgs_in[img_idx].clone()
            if img_idx in idxs_to_keep:
                img_in[:5, :5, :] = 1.0
            else:
                img_in[:5, :5, :] = 0.0
            rows.append(img_in) 
        vis_columns.append(torch.vstack(rows))

        # optionally get self-conditioning
        if self.cfg.model.unet.self_condition:
            cond["x_self_condition"] = torch.zeros_like(x)
            # this forward pass has to render all images in the set
            volume_model_out, r_sil_out = self.model(x, t, cond,
                                            idxs_to_keep=idxs_to_keep,
                                            return_sil=True,
                                            stratified_sampling=self.cfg.render.stratified_sampling
                                            )
            # visualise output of self-conditioning pass: renders and silhouettes
            rows = []
            for img_idx in range(volume_model_out.shape[1]):
                rows.append(volume_model_out[0, img_idx, :3, ...].permute(1, 2, 0).cpu()* 0.5 + 0.5) 
            vis_columns.append(torch.vstack(rows))
            rows = []
            r_sil_out = r_sil_out.reshape(b, Cond+cond["x_cond"].shape[1], h, w, 1)
            for img_idx in range(r_sil_out.shape[1]):
                rows.append(r_sil_out[0, img_idx, ...].expand(h, w, 3).cpu()) 
            vis_columns.append(torch.vstack(rows))

            cond["x_self_condition"] = volume_model_out[:, cond["x_cond"].shape[1]:,
                                                        ...]
        
        # model forward pass
        volume_model_out, r_sil_out = self.model(x, t, cond,
                                        idxs_to_keep=idxs_to_keep,
                                        return_sil=True,
                                        stratified_sampling=self.cfg.render.stratified_sampling)
        rows = []
        was_dropped_out = torch.logical_not(torch.isin(torch.arange(imgs_in.shape[0]), 
                                                       idxs_to_keep))
        for img_idx in range(volume_model_out.shape[1]):
            img_out = volume_model_out[0, img_idx, :3, ...].permute(1, 2, 0).cpu()* 0.5 + 0.5
            if was_dropped_out[img_idx]:
                img_out[:5, :5, :] = 0.0
            else:
                img_out[:5, :5, :] = 1.0
            rows.append(img_out)
        vis_columns.append(torch.vstack(rows))
        rows = []
        r_sil_out = r_sil_out.reshape(b, Cond+cond["x_cond"].shape[1], h, w, 1)
        for img_idx in range(r_sil_out.shape[1]):
            rows.append(r_sil_out[0, img_idx, ...].expand(h, w, 3).cpu()) 
        vis_columns.append(torch.vstack(rows))

        im_log = wandb.Image(np.clip(torch.hstack(vis_columns).numpy(), 0.0, 1.0), caption="Vis")
        wandb.log({"{}_vis".format(split): im_log}, step=iteration)
    
    @torch.no_grad()
    def ddim_sample(self, shape, cond, return_all_timesteps = False,
                    render_spinning_volume = False, classifier_free_guidance_w = 0.0,
                    images_in_spinning_volume = 20):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        if self.cfg.model.unet.self_condition:
            cond["x_self_condition"] = torch.zeros_like(img)

        for time, time_next in tqdm.tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            if time_next < 0 and render_spinning_volume:
                N_vis = images_in_spinning_volume
                all_target_Rs_loop = []
                all_target_Ts_loop = []
                # get target rotations separately for each batch element - in CO3D
                # each sample is on a different radius
                if self.cfg.data.dataset_type == "minens":
                    elevation = 0.0
                    radius = 5.0
                elif self.cfg.data.dataset_type == "co3d":
                    radius = 2.1166
                    elevation = np.pi/6
                elif self.cfg.data.category == "cars":
                    elevation = np.pi / 6
                    radius = 1.3
                else:
                    elevation = np.pi / 6
                    radius = 2.0

                # if test Rs and Ts are provided, use them
                if "test_Rs" in cond.keys():
                    all_target_Ts_loop = cond["test_Ts"]
                    all_target_Rs_loop = cond["test_Rs"]
                    N_vis = cond["test_Rs"].shape[1]
                # otherwise hand-craft a loop around the object
                else:
                    #for b_idx in range(batch):
                    all_target_Rs_loop, all_target_Ts_loop = get_Ts_and_Rs_loop(batch, device, N_vis=N_vis, 
                                                                        radius=radius,
                                                                        elevation=elevation,
                                                                        cfg = self.cfg)
                    # all_target_Rs_loop.append(target_Rs_loop)
                    # all_target_Ts_loop.append(target_Ts_loop)
                    # all_target_Ts_loop = torch.cat(all_target_Ts_loop, dim=0)
                    # all_target_Rs_loop = torch.cat(all_target_Rs_loop, dim=0)
                cond_vis = {k: v for k, v in cond.items()}
                if classifier_free_guidance_w == -1.0:
                    uncond_cond = {}
                    for k, v in cond_vis.items():
                        if k == "x_self_condition":
                            uncond_cond[k] = v
                        else:
                            uncond_cond[k] = v[:, cond["x_cond"].shape[1]:, ...]
                    # override the conditioning dict with one without clean images
                    cond_vis = uncond_cond
                    cond_reference = uncond_cond
                else:
                    cond_reference = cond

                # append the rendering targets to the conditioning dict
                x_start_all = []
                # each render in the loop is done separately to manage memory usage
                for loop_idx in range(N_vis):
                    cond_vis["target_Rs"] = torch.cat([cond_reference["target_Rs"], 
                                                       all_target_Rs_loop[:, loop_idx:loop_idx+1]], dim=1)
                    cond_vis["target_Ts"] = torch.cat([cond_reference["target_Ts"], 
                                                       all_target_Ts_loop[:, loop_idx:loop_idx+1]], dim=1)
                    cond_vis["background"] = torch.cat([cond_reference["background"], 
                                                    cond["background"][:, :1, ...].expand(batch, 1, *shape[3:], 3)],
                                                    dim=1)
                    if self.cfg.data.dataset_type == "co3d":
                        cond_vis["focal_lengths"] = torch.cat([cond_reference["focal_lengths"], 
                                                        cond["focal_lengths"][:, -1:, ...].expand(batch, 1, *cond["principal_points"].shape[2:])], 
                                                        # use this focal for vis or for FID eval* 0.0+ 3.5615
                                                        dim=1)
                        cond_vis["principal_points"] = torch.cat([cond_reference["principal_points"], 
                                                        cond["principal_points"][:, -1:, ...].expand(batch, 1, *cond["principal_points"].shape[2:])], 
                                                        # use  * 0.0 for FID eval or for vis
                                                        dim=1)
                    x_start = []
                    
                    for b_idx in range(batch):
                        x_start_b = self.model_predictions(img[b_idx:b_idx+1, ...], 
                                time_cond[b_idx:b_idx+1, ...], 
                                {k:v[b_idx:b_idx+1, ...] for k, v in cond_vis.items()}, 
                                clip_x_start = True, 
                                return_all_renders = True)
                        x_start.append(x_start_b)
                    x_start = torch.cat(x_start, dim=0)
                    if loop_idx == 0:
                        x_start_all.append(x_start)
                    else:
                        x_start_all.append(x_start[:, -1:, ...])
                # contains renders from the viewpoints of: (0) x_cond, if classifier_free_guidance_w != -1.0
                # (1) x_in, noisy images in 
                # (2) viewpoints on a loop around the object
                x_start = torch.cat(x_start_all, dim=1)
                print(x_start.shape)
            else:
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond, 
                                                                 clip_x_start = True)

                if classifier_free_guidance_w != 0.0:
                    uncond_cond = {}
                    for k, v in cond.items():
                        if k == "x_self_condition":
                            uncond_cond[k] = v
                        else:
                            uncond_cond[k] = v[:, cond["x_cond"].shape[1]:, ...]
                    uncond_pred_noise, uncond_x_start, \
                        *_ = self.model_predictions(img, time_cond, 
                                                         uncond_cond, 
                                                         clip_x_start = True)
                    overall_pred_noise = (1+classifier_free_guidance_w) * pred_noise - \
                                         classifier_free_guidance_w * uncond_pred_noise
                    x_start = self.predict_start_from_noise(img, time_cond, overall_pred_noise)
                    x_start = torch.clamp(x_start, -1.0, 1.0)

                    pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

                if self.cfg.model.unet.self_condition:
                    # x_start was last calculated with the correct amount of classifier
                    # free guidance - this is the one that will be used for conditioning
                    cond["x_self_condition"] = x_start

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret
    
    def forward(self, img, *args, **kwargs):
        b, Cond, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

    def p_sample_loop(self):
        raise NotImplementedError