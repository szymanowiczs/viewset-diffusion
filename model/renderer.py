# Adapted from Pytorch3d

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Tuple, Union

from pytorch3d.renderer import (
    VolumeRenderer,
    CamerasBase,
    VolumeSampler,
    NDCMultinomialRaysampler
)

from pytorch3d.renderer.implicit.utils import (
    _validate_ray_bundle_variables, 
    ray_bundle_variables_to_ray_points
)

from pytorch3d.renderer.implicit.raysampling import (
    HeterogeneousRayBundle, 
    RayBundle
)
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod
)
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf
from pytorch3d.renderer.implicit.utils import RayBundle

from pytorch3d.structures import Volumes

class PostActivatedVolumeRenderer(VolumeRenderer):
    def __init__(self, cfg, sample_mode: str = "bilinear") -> None:
        """
        Overrides the renderer in Pytorch3D. Takes in non-activated volumes and
        non-activated features. Applies the activation before alpha compositing, 
        after the raymarcher has returned the interpolated volume values. 
        https://arxiv.org/abs/2111.11215
        """
        raysampler = NDCMultinomialRaysampler(
            image_width=cfg.data.input_size[0],
            image_height=cfg.data.input_size[1],
            n_pts_per_ray=cfg.render.n_pts_per_ray,
            min_depth=cfg.render.min_depth,
            max_depth=cfg.render.max_depth,
            stratified_sampling=cfg.render.stratified_sampling
        )

        # instantiate the standard ray marcher
        raymarcher = NeRFEmissionAbsorptionRaymarcher()

        super().__init__(raysampler=raysampler, raymarcher=raymarcher, sample_mode=sample_mode)
        self.cfg = cfg
        assert self.cfg.render.post_activation
        self.gain = cfg.render.n_pts_per_ray / (cfg.render.max_depth-cfg.render.min_depth)

    def density_activation_fn(self, sigma, lengths):
        dists = lengths[...,1:] - lengths[...,:-1]
        # In NeRF last appended length is 1e10 but in DVGO the last appended length is stepsize
        # https://github.com/sunset1995/DirectVoxGO/blob/341e1fc4e96efff146d42cd6f31b8199a3e536f7/lib/dvgo.py#LL309C1-L309C1
        dists = torch.cat([dists, torch.tensor([1/self.gain], device=dists.device).expand(dists[...,:1].shape)], -1)
        dists = dists[..., None] # last dimension of sigma is 1 - make it 1 for dists too

        noise = 0.
        if self.cfg.render.raw_noise_std > 0. and self.training:
            noise = torch.randn(sigma.shape, device=sigma.device) * self.cfg.render.raw_noise_std

        alpha = 1 - torch.exp(-F.softplus((sigma + noise) * self.gain - 6.0) * dists)

        return alpha

    def color_activation_fn(self, colors):
        return torch.sigmoid(colors) * 1.002 - 0.001

    def forward(self,  cameras: CamerasBase, volumes: Volumes):
        volumetric_function = MaskedVolumeSampler(volumes, sample_mode=self._sample_mode)

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        # use stratified sampling if specified in config and if
        # the model is in training mode        
        if self.training and self.cfg.render.stratified_sampling:
            stratified_sampling = True
        else:
            stratified_sampling = False 

        ray_bundle = self.renderer.raysampler(
            cameras=cameras, volumetric_function=volumetric_function,
            stratified_sampling=stratified_sampling
        )
        # ray_bundle.origins - minibatch x ... x 3
        # ray_bundle.directions - minibatch x ... x 3
        # ray_bundle.lengths - minibatch x ... x n_pts_per_ray
        # ray_bundle.xys - minibatch x ... x 2

        # given sampled rays, call the volumetric function that
        # evaluates the densities and features at the locations of the
        # ray points
        # pyre-fixme[23]: Unable to unpack `object` into 2 values.
        raw_sigma_coarse, rays_features, ray_mask_out = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras
        )
        # ray_densities - minibatch x ... x n_pts_per_ray x density_dim
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim

        rays_densities = self.density_activation_fn(raw_sigma_coarse, 
                                                    ray_bundle.lengths)
        rays_densities *= ray_mask_out

        if self.cfg.render.n_pts_per_ray_fine != 0:
            # sample new locations and override existing rays_densities and rays_features
            with torch.no_grad():
                rays_densities = rays_densities[..., 0] # get rid of the channel dimension
                absorption = _shifted_cumprod(
                    (1.0 + 1e-5) - rays_densities, shift=1
                )
                weights = rays_densities * absorption

                # compute bin edges - they are midpoints between samples apart
                # from the first and last sample which are the edges of the ray
                # weights then were sampled from midpoints between bin edges,
                # apart from the first and last sample which are the edges of the ray
                # this is technically incorrrect but given that the samples at the
                # edges should be transparent anyway, it should not matter
                bins = (ray_bundle.lengths[..., 1:] + ray_bundle.lengths[..., :-1]) / 2
                bins = torch.cat([ray_bundle.lengths[..., :1],
                                  bins, 
                                  ray_bundle.lengths[..., -1:]], -1)

                z_fine_samples = sample_pdf(
                    bins, weights, self.cfg.render.n_pts_per_ray_fine
                ).detach()

                # concatenate with the existing samples and sort
                new_lengths, _ = torch.sort(torch.cat(
                    [ray_bundle.lengths, z_fine_samples], -1), -1)
                ray_bundle = RayBundle(ray_bundle.origins,
                                       ray_bundle.directions,
                                       new_lengths,
                                       ray_bundle.xys)

            raw_sigma_fine, rays_features, ray_mask_out = volumetric_function(
                ray_bundle=ray_bundle, cameras=cameras
                )

            rays_densities = self.density_activation_fn(raw_sigma_fine, 
                                                        ray_bundle.lengths)

            rays_densities *= ray_mask_out

        rays_features = self.color_activation_fn(rays_features)

        # finally, march along the sampled rays to obtain the renders
        images = self.renderer.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
        )
        # images - minibatch x ... x (feature_dim + opacity_dim)

        return images, ray_bundle

class MaskedVolumeSampler(VolumeSampler):
    def __init__(self, volumes: Volumes, sample_mode: str = "bilinear") -> None:
        super().__init__(volumes, sample_mode=sample_mode)
    
    def forward(
        self, ray_bundle: Union[RayBundle, HeterogeneousRayBundle], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Overrides the Pytorch3D volumesampler which also
            returns the mask indicating if a sample fell in the volume or
            not. Samples outside volume will have opacity := 0
        """

        # take out the interesting parts of ray_bundle
        rays_origins_world = ray_bundle.origins
        rays_directions_world = ray_bundle.directions
        rays_lengths = ray_bundle.lengths

        # validate the inputs
        _validate_ray_bundle_variables(
            rays_origins_world, rays_directions_world, rays_lengths
        )
        if self._volumes.densities().shape[0] != rays_origins_world.shape[0]:
            raise ValueError("Input volumes have to have the same batch size as rays.")

        #########################################################
        # 1) convert the origins/directions to the local coords #
        #########################################################

        # origins are mapped with the world_to_local transform of the volumes
        rays_origins_local = self._volumes.world_to_local_coords(rays_origins_world)

        # obtain the Transform3d object that transforms ray directions to local coords
        directions_transform = self._get_ray_directions_transform()

        # transform the directions to the local coords
        rays_directions_local = directions_transform.transform_points(
            rays_directions_world.view(rays_lengths.shape[0], -1, 3)
        ).view(rays_directions_world.shape)

        ############################
        # 2) obtain the ray points #
        ############################

        # this op produces a fairly big tensor (minibatch, ..., n_samples_per_ray, 3)
        rays_points_local = ray_bundle_variables_to_ray_points(
            rays_origins_local, rays_directions_local, rays_lengths
        )

        ########################
        # 3) sample the volume #
        ########################

        rays_mask_out = torch.ones((*rays_points_local.shape[:-1], 1),
                                    device=rays_points_local.device, requires_grad=False,
                                    dtype=torch.bool)

        # generate the tensor for sampling
        volumes_densities = self._volumes.densities()
        dim_density = volumes_densities.shape[1]
        volumes_features = self._volumes.features()

        # reshape to a size which grid_sample likes
        rays_points_local_flat = rays_points_local.view(
            rays_points_local.shape[0], -1, 1, 1, 3
        )

        # run the grid sampler on the volumes densities
        rays_densities = torch.nn.functional.grid_sample(
            volumes_densities,
            rays_points_local_flat,
            align_corners=True,
            mode=self._sample_mode,
        )

        # permute the dimensions & reshape densities after sampling
        rays_densities = rays_densities.permute(0, 2, 3, 4, 1).view(
            *rays_points_local.shape[:-1], volumes_densities.shape[1]
        )

        # if features exist, run grid sampler again on the features densities
        if volumes_features is None:
            dim_feature = 0
            _, rays_features = rays_densities.split([dim_density, dim_feature], dim=-1)
        else:
            rays_features = torch.nn.functional.grid_sample(
                volumes_features,
                rays_points_local_flat,
                align_corners=True,
                mode=self._sample_mode,
            )

            # permute the dimensions & reshape features after sampling
            rays_features = rays_features.permute(0, 2, 3, 4, 1).view(
                *rays_points_local.shape[:-1], volumes_features.shape[1]
            )

        for dim_idx in [0, 1, 2]:
            outside_volume = torch.logical_or(rays_points_local[..., dim_idx:dim_idx+1] > 1,
                                              rays_points_local[..., dim_idx:dim_idx+1] < -1)
            rays_mask_out[outside_volume] = 0.0

        return rays_densities, rays_features, rays_mask_out

class NeRFEmissionAbsorptionRaymarcher(torch.nn.Module):
    """
    EA raymarcher.
    returned silhouette is calculated from sum of accumulated weights

    https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf.py#L302C1-L303C51
    """

    def __init__(self, surface_thickness: int = 1) -> None:
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        super().__init__()
        self.surface_thickness = surface_thickness

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.
        Returns:
            features_opacities: A tensor of shape `(..., feature_dim+1)`
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        # this line is different from pytorch3d implementation
        opacities = (weights[..., None] * 1).sum(dim=-2)
        # opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)

        return torch.cat((features, opacities), dim=-1)