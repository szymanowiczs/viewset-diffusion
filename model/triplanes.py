from typing import List, Optional, Tuple, Union

from pytorch3d.structures.volumes import VolumeLocator
import torch
from pytorch3d.transforms import Transform3d

_Scalar = Union[int, float]
_Vector = Union[torch.Tensor, Tuple[_Scalar, ...], List[_Scalar]]
_ScalarOrVector = Union[_Scalar, _Vector]

_VoxelSize = _ScalarOrVector
_Translation = _Vector

_TensorBatch = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]

class Triplanes:
    """
    This class provides functions for working with batches of triplanes
    of possibly varying spatial sizes.

    FEATURES

    The Triplanes class can be constructed from a 5D tensor of 'features'
    of size 'batch x 3 x feature_dim x height x width'. The dimension 
    with size 3 corresponds to the three planes which are reshaped and
    used for sampling.

    """
    def __init__(
        self,
        features: _TensorBatch,
        voxel_size: _VoxelSize = 1.0,
        volume_translation: _Translation = (0.0, 0.0, 0.0)
    ) -> None:
        assert len(features.shape) == 5, "Expected a 5D feature map"
        assert features.shape[1] == 3, "Expected a TRI-plane"
        assert features.shape[3] == features.shape[4], "Expected square triplanes"
        self._features = features
        self.device = features.device

        # assign a coordinate transformation member
        self.locator = VolumeLocator(
            batch_size=len(self._features),
            grid_sizes=(features.shape[3],
                        features.shape[3],
                        features.shape[3]),
            voxel_size=voxel_size,
            volume_translation=volume_translation,
            device=self.device,
        )

    def features(self):
        return self._features

    def get_local_to_world_coords_transform(self) -> Transform3d:
        """
        Return a Transform3d object that converts points in the
        the local coordinate frame of the volume to world coordinates.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **local_to_world_transform**: A Transform3d object converting
                points from local coordinates to the world coordinates.
        """
        return self.locator.get_local_to_world_coords_transform()

    def get_world_to_local_coords_transform(self) -> Transform3d:
        """
        Return a Transform3d object that converts points in the
        world coordinates to the local coordinate frame of the volume.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **world_to_local_transform**: A Transform3d object converting
                points from world coordinates to local coordinates.
        """
        return self.get_local_to_world_coords_transform().inverse()

    def world_to_local_coords(self, points_3d_world: torch.Tensor) -> torch.Tensor:
        return self.locator.world_to_local_coords(points_3d_world)