import torch
import torch.nn as nn
import torch.nn.functional as F

from . import get_cameras_from_data_dict
from .multi_image_unet import DDPMUNet, TriplaneUNet
from .renderer import PostActivatedVolumeRenderer
from .triplane_renderer import TriplaneRenderer
from .unet_parts import ResnetBlock2D, SinusoidalPosEmb
from .triplanes import Triplanes
from pytorch3d.structures import Volumes
from einops.layers.torch import Rearrange

class Reconstructor(nn.Module):
    def __new__(cls, cfg):
        if cfg.model.unet.volume_repr == "voxel":
            return VoxelReconstructor(cfg)
        elif cfg.model.unet.volume_repr == "triplanes":
            return TriplaneReconstructor(cfg)

class VoxelReconstructor(nn.Module):
    def __init__(self, cfg):
        super(VoxelReconstructor, self).__init__()
        print('Instantiated')
        self.cfg = cfg
        self.renderer = PostActivatedVolumeRenderer(cfg)

        if self.cfg.model.feature_extractor_2d.use:
            self.feature_extractor = FeatureExtractor2D(cfg)
        else:
            self.feature_extractor = IdentityWithTimestep()
        self.unprojector = ImageUnprojector(cfg)
        self.reconstructor = DDPMUNet(cfg)

        self._voxel_size = tuple(cfg.render.volume_extent_world * mult/ cfg.render.volume_size \
                         for mult in cfg.render.volume_size_mults)
        self._volume_translation = tuple(tr for tr in cfg.render.volume_offsets)

        # dummy attributes expected by GaussianDiffusion
        self.random_or_learned_sinusoidal_cond = False
        self.channels = 3
        self.self_condition = False

    def forward(self, imgs, timesteps, cond, 
                idxs_to_keep = None, idxs_to_render = None,
                return_sil = False, **kwargs):
        """
        Logic of the model:
        For image set in batch:
            For (image, camera, timestep) in idxs_to_keep:
                1. Unproject the image to volumes
                2. Pass the image through the encoder part of the U-Net
            Jointly decode volumes from N images into one volume per set.
            For camera in idxs_to_render:
                3. Render the volume to image from the given camera
        
        Implementation of the logic runs batches and encoding in parallel
                
        Args:
            images (torch.Tensor): [B, N_noisy_images, C, H, W]
            cond_data (dict): dictionary containing other data, i.e.
                camera poses (T, R), pose embedding for the image input
                and image background.
            timesteps (torch.Tensor): [B] timestep corresponding to the
                noisy images in the viewset. All noisy images are assumed
                to be from the same timestep.
            idxs_to_keep: idxs of images in a set used for reconstructing 
                the volume. If None, all images are used.
            idxs_to_render: idxs of cameras to render the volume from 
        """
        # ============ Building input images ============
        imgs = torch.cat([cond["x_cond"], imgs], dim=1)
        imgs = torch.cat([imgs, cond["pose_embed"]], dim=2)
        if self.cfg.model.unet.self_condition:
            self_conditioning = torch.cat([cond["x_cond"],
                                            cond["x_self_condition"]],
                                            dim=1)
            imgs = torch.cat([imgs, self_conditioning], dim=2)
        noisy_imgs_in = imgs.shape[1] - cond["x_cond"].shape[1]

        # ============ Building source cameras, images and timesteps
        if idxs_to_keep is None:
            idxs_to_keep = torch.arange(imgs.shape[1])
        source_cameras = get_cameras_from_data_dict(self.cfg, 
                                                    cond, 
                                                    imgs.device, 
                                                    idxs_to_keep)
        source_images = imgs[:, idxs_to_keep, ...]
        B, Cond, C, H, W = imgs.shape

        # ============ Preparing outputs in the target shape ============
        if idxs_to_render is None:
            idxs_to_render = torch.arange(cond["target_Rs"].shape[1])
        target_cameras = get_cameras_from_data_dict(self.cfg, 
                                                    cond, 
                                                    imgs.device,
                                                    idxs_to_render)
        Renders = len(idxs_to_render)
        # Expand timestep so that each image in a set has its own, 
        # including clean images in cond[x_cond]
        if noisy_imgs_in == 0:
            timesteps = torch.zeros((B, cond["x_cond"].shape[1]), 
                                    device=imgs.device)[:, idxs_to_keep, ...]
        else:
            timesteps = timesteps.unsqueeze(1).expand(B, noisy_imgs_in)
            timesteps = torch.cat([timesteps[:, :1].expand(B, cond["x_cond"].shape[1]) * 0,
                                   timesteps], dim=1)[..., idxs_to_keep]

        # ============ Image feature extraction ============
        source_images = self.feature_extractor(source_images, timesteps)

        # ============ Image unprojection ============
        volumes = self.unprojector(source_images, source_cameras)

        # ============ Volume reconstruction ============
        densities, colors = self.reconstructor(volumes, timesteps)
        
        # we need one volume per camera that we want to render from.
        # cameras are of shape B*Renders in the batch dimension
        densities = densities.unsqueeze(1)
        colors = colors.unsqueeze(1)
        densities = densities.expand(B, Renders, *densities.shape[2:])
        colors = colors.expand(B, Renders, *colors.shape[2:])
        densities = densities.reshape(B*Renders, *densities.shape[2:])
        colors = colors.reshape(B*Renders, *colors.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        volumes = Volumes(
            densities = densities,
            features = colors,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_sil = self.renderer(cameras=target_cameras,
                                     volumes=volumes
                                     )[0].split([3, 1], dim=-1)
        # Adding background and reshaping back to viewsets
        r_img = r_img + cond["background"][
            :, idxs_to_render, ...
            ].reshape(B*Renders, H, W, 3) * (1. - r_sil)
        r_img = r_img.permute(0, 3, 1, 2)

        r_img = r_img.reshape(B, Renders, 3, H, W) * 2 - 1
        
        if return_sil:
            return r_img, r_sil
        else:
            return r_img

class TriplaneReconstructor(nn.Module):
    def __init__(self, cfg):
        super(TriplaneReconstructor, self).__init__()
        self.cfg = cfg
        self.renderer = TriplaneRenderer(cfg)

        self.reconstructor = TriplaneUNet(cfg)

        self._voxel_size = tuple(cfg.render.volume_extent_world * mult / cfg.data.input_size[0] \
                         for mult in cfg.render.volume_size_mults)
        self._volume_translation = tuple(tr for tr in cfg.render.volume_offsets)

        # dummy attributes expected by GaussianDiffusion
        self.random_or_learned_sinusoidal_cond = False
        self.channels = 3
        self.self_condition = False

    def forward(self, imgs, timesteps, cond, 
                idxs_to_keep = None, idxs_to_render = None,
                return_sil = False, **kwargs):
        """
        1. Pass the image through a 2D U-Net
        2. Reshape into triplanes
        For camera in idxs_to_render:
        3. Render the triplane to image from the given camera
        
        Args:
            images (torch.Tensor): [B, N_noisy_images, C, H, W]
            cond_data (dict): dictionary containing other data, i.e.
                camera poses (T, R), pose embedding for the image input
                and image background.
            timesteps (torch.Tensor): [B] timestep corresponding to the
                noisy images in the viewset. All noisy images are assumed
                to be from the same timestep.
            idxs_to_keep: idxs of images in a set used for reconstructing 
                the volume. If None, all images are used.
            idxs_to_render: idxs of cameras to render the volume from 
        """
        # ============ Building input images ============
        imgs = torch.cat([cond["x_cond"], imgs], dim=1)
        # WARNING! No pose embedding here
        if self.cfg.model.unet.self_condition:
            self_conditioning = torch.cat([cond["x_cond"],
                                            cond["x_self_condition"]],
                                            dim=1)
            imgs = torch.cat([imgs, self_conditioning], dim=2)
        noisy_imgs_in = imgs.shape[1] - cond["x_cond"].shape[1]

        # ============ Building source cameras, images and timesteps
        if idxs_to_keep is None:
            idxs_to_keep = torch.arange(imgs.shape[1])

        B, Cond, C, H, W = imgs.shape

        # Expand timestep so that each image in a set has its own, 
        # including clean images in cond[x_cond]
        if noisy_imgs_in == 0:
            timesteps = torch.zeros((B, cond["x_cond"].shape[1]), 
                                    device=imgs.device)[:, idxs_to_keep, ...]
        else:
            timesteps = timesteps.unsqueeze(1).expand(B, noisy_imgs_in)
            timesteps = torch.cat([timesteps[:, :1].expand(B, cond["x_cond"].shape[1]) * 0,
                                   timesteps], dim=1)[..., idxs_to_keep]

        assert len(idxs_to_keep) == 1, "Only accepting one input image"
        # ============ Volume reconstruction ============
        triplane_features = self.reconstructor(imgs[:, idxs_to_keep, ...], 
                                               timesteps)
        
        # ============ Preparing outputs in the target shape ============
        if idxs_to_render is None:
            idxs_to_render = torch.arange(cond["target_Rs"].shape[1])
        target_cameras = get_cameras_from_data_dict(self.cfg, 
                                                    cond, 
                                                    imgs.device,
                                                    idxs_to_render)
        Renders = len(idxs_to_render)
        # we need one volume per camera that we want to render from.
        # cameras are of shape B*Renders in the batch dimension
        triplane_features = triplane_features.unsqueeze(1)
        triplane_features = triplane_features.expand(B, Renders, 
                                                     *triplane_features.shape[2:])
        triplane_features = triplane_features.reshape(B*Renders, 
                                                      *triplane_features.shape[2:])

        # Instantiate the Volumes object (densities and colors are already 5D)
        triplanes = Triplanes(
            features = triplane_features,
            voxel_size=self._voxel_size,
            volume_translation=self._volume_translation
        )

        # ============ Rendering ============
        r_img, r_sil = self.renderer(cameras=target_cameras,
                                     triplanes=triplanes
                                     )[0].split([3, 1], dim=-1)
        # Adding background and reshaping back to viewsets
        r_img = r_img + cond["background"][
            :, idxs_to_render, ...
            ].reshape(B*Renders, H, W, 3) * (1. - r_sil)
        r_img = r_img.permute(0, 3, 1, 2)

        r_img = r_img.reshape(B, Renders, 3, H, W) * 2 - 1
        
        if return_sil:
            return r_img, r_sil
        else:
            return r_img

class ImageUnprojector(nn.Module):

    def __init__(self, cfg):
        # only conditioning images should be passed to this network
        super().__init__()
        self.cfg = cfg
        
        sample_volume = Volumes(
            densities = torch.zeros([1, 1, self.cfg.model.volume_size,
                                           self.cfg.model.volume_size,
                                           self.cfg.model.volume_size]),
            features = torch.zeros([1, 3, self.cfg.model.volume_size,
                                          self.cfg.model.volume_size,
                                          self.cfg.model.volume_size]),
            voxel_size= tuple(cfg.render.volume_extent_world * mult/ cfg.model.volume_size \
                         for mult in cfg.render.volume_size_mults),
            volume_translation=tuple(tr for tr in cfg.render.volume_offsets)
        )
        
        grid_coordinates = sample_volume.get_coord_grid() # [B*Renders, D, H, W, 3]
        # all grids are the same shape so we can just take the first one
        grid_coordinates = grid_coordinates[0]
        self.register_buffer('grid_coordinates', grid_coordinates)

    def forward(self, images_kept, cameras):
        # takes in volume densities and colors as predicted by the
        # convolutional network. Aggregates unprojected features from
        # the conditioning images and outputs new densities and colors
        B, Cond, C, H, W = images_kept.shape
        Renders = cameras.T.shape[0] // B
        N_volumes = B * Cond
        H_vol = self.cfg.model.volume_size
        W_vol = self.cfg.model.volume_size
        D_vol = self.cfg.model.volume_size

        # project the locations on the voxel grid onto the conditioning images
        image_coordinates = cameras.transform_points_ndc(self.grid_coordinates.reshape(-1, 3)) # [B*renders, H*W*D, 2]
        # only keep the training images
        image_coordinates = image_coordinates.reshape(B, Renders, H_vol*W_vol*D_vol, 3)
        image_coordinates = image_coordinates[:, :Cond, ...]
        image_coordinates = image_coordinates.reshape(N_volumes, H_vol*W_vol*D_vol, 3)
        # image_coordinates have dim 3 but that does not mean that they are in
        # homogeneous coordinates, because they are in NDC coordinates. The last dimension
        # is the depth in the NDC volume but the first two are already the coordinates
        # in the image. So we only need to take the first two dimensions.
        image_coordinates = image_coordinates[..., :2] 
        # flip x and y coordinates because x and y in NDC are +ve for top left corner of the
        # image, while grid_sample expects (-1, -1) to correspond to the top left pixel
        image_coordinates *= -1 
        image_coordinates = image_coordinates.reshape(N_volumes, H_vol, W_vol, D_vol, 2)

        # gridsample image values
        gridsample_input = images_kept.reshape(B*Cond, C, H, W)
        # reshape the grid for gridsample to 4D because when input is 4D, 
        # output is expected to be 4D as well
        image_coordinates = image_coordinates.reshape(B*Cond,
                                                    H_vol,
                                                    W_vol,
                                                    D_vol,
                                                    2).reshape(B*Cond,
                                                                H_vol, 
                                                                W_vol*D_vol, 2)
        # use default align_corners=False because camera projection outputs 0, 0 for the top left corner
        # of the top left pixel 
        unprojected_colors = F.grid_sample(gridsample_input, image_coordinates, padding_mode="border", align_corners=False)
        # unprojected_colors is a slightly misleading name, it is actually the unprojected
        # features in the conditioning images, including the pose embedding.
        unprojected_colors = unprojected_colors.reshape(B*Cond, C, H_vol, W_vol, D_vol)
        unprojected_colors = unprojected_colors.reshape(B, Cond, *unprojected_colors.shape[1:])

        return unprojected_colors

class FeatureExtractor2D(nn.Module):
    """
    2D feature extract that also downsamples by a factor of 4
    after processing with 3 convolutional layers. 
    """
    def __init__(self, cfg, groups=8):
        super().__init__()
        self.cfg = cfg

        if self.cfg.model.feature_extractor_2d.pass_features != 'both':
            dim = cfg.model.unet.input_dim
        else:
            dim = cfg.model.unet.input_dim // 2

        input_dim = cfg.model.input_dim
        if self.cfg.model.unet.self_condition:
            input_dim += 3

        # ========== 2D feature extraction - 1 + 2 x N + 1 conv layers ==========
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
        )

        if self.cfg.model.feature_extractor_2d.pass_features != 'high_res':
            # ========== time embedding ==========
            time_dim = cfg.model.unet.input_dim * 4
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
            
            self.feature_extractor = nn.ModuleList(
                [ResnetBlock2D(dim, dim, time_emb_dim=time_dim) for i in 
                range(self.cfg.model.feature_extractor_2d.res_blocks)]
            )

            self.downsample = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
                nn.Conv2d(dim * 4, dim, 1)
            )

    def forward(self, viewset2d, t):
        # viewset: [B, Cond, C, H, W]
        # t: [B, Cond]
        B, Cond, C, H, W = viewset2d.shape

        viewset2d = viewset2d.reshape(B*Cond, C, H, W)
        viewset2d = self.init_conv(viewset2d)

        if self.cfg.model.feature_extractor_2d.pass_features != 'high_res':
            encoder_emb = self.time_mlp(t.reshape(B*Cond,))

            latents = [viewset2d]
            latents_sz = viewset2d.shape[2:]

            for layer in self.feature_extractor:
                viewset2d = layer(viewset2d, encoder_emb)
            viewset2d = self.downsample(viewset2d)

            if self.cfg.model.feature_extractor_2d.pass_features == 'both':
                latents.append(F.interpolate(
                    viewset2d,
                    latents_sz,
                    mode='nearest'
                ))
                viewset2d = torch.cat(latents, dim=1)

        viewset2d = viewset2d.reshape(B, Cond, *viewset2d.shape[1:])

        return viewset2d
    
class IdentityWithTimestep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return x