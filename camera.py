import math
import pyrr

import torch

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
)

import numpy as np
import random 

def rotate_cameras(camera_Rs, rot_aug, rot_axis):
    if rot_axis == "y":
        relative_rot_mat = torch.tensor([[torch.cos(rot_aug), 0, torch.sin(rot_aug)],
                                        [                 0, 1,                  0],
                                        [-torch.sin(rot_aug), 0, torch.cos(rot_aug)]]
                                        ).unsqueeze(0).expand(*camera_Rs.shape)
    elif rot_axis == "z":
        relative_rot_mat = torch.tensor([[torch.cos(rot_aug), torch.sin(rot_aug), 0],
                                        [-torch.sin(rot_aug), torch.cos(rot_aug), 0],
                                        [                 0,                  0, 1]]
                                        ).unsqueeze(0).expand(*camera_Rs.shape)
    camera_Rs = torch.bmm(relative_rot_mat, camera_Rs)
    return camera_Rs

def random_viewpoint(yaw=None, elevation=None):
    # can override random viewpoint with specifying yaw and elevation
    if yaw is None:
        yaw = random.uniform(0, 2 * math.pi)
    if elevation is None:
        elevation = random.uniform(-math.pi / 8, math.pi / 8)
    return get_viewpoint(yaw, elevation)

def translate_cameras(camera_Rs, camera_Ts, trans_aug):
    # camera_Rs: B x 3 x 3
    # camera_Ts: B x 3
    # trans_aug: 3
    relative_trans = torch.bmm(trans_aug.unsqueeze(0).expand(*camera_Ts.shape).unsqueeze(1),
                               camera_Rs).squeeze(1)
    return camera_Ts + relative_trans

def augment_cameras(camera_Rs, camera_Ts, rot_aug, trans_aug, rot_axis = "y"):
    if rot_aug is not None:
        camera_Rs = rotate_cameras(camera_Rs, rot_aug, rot_axis)
    if trans_aug is not None:
        camera_Ts = translate_cameras(camera_Rs, camera_Ts, trans_aug)

    return camera_Rs, camera_Ts

def get_cameras_from_data_dict(cfg, data, device, slicing_idxs = None):
    # slicing_idxs are the cameras that are kept in the batch
    if slicing_idxs is None:
        slicing_idxs = torch.arange(data["target_Rs"].shape[1])
    if cfg.data.dataset_type == "co3d":
        example_cameras = PerspectiveCameras(
                    R = data["target_Rs"][:, slicing_idxs].reshape([-1, 3, 3]),
                    T = data["target_Ts"][:, slicing_idxs].reshape([-1, 3]),
                    focal_length=data["focal_lengths"][:, slicing_idxs].reshape([-1, 2]),
                    principal_point=data["principal_points"][:, slicing_idxs].reshape([-1, 2]),
                    device=device
        )
    elif cfg.data.dataset_type == "skins" or cfg.data.dataset_type == "srn":
        example_cameras = FoVPerspectiveCameras(
                    R = data["target_Rs"][:, slicing_idxs].reshape([-1, 3, 3]), 
                    T = data["target_Ts"][:, slicing_idxs].reshape([-1, 3]),  
                    znear = cfg.render.znear,
                    zfar = cfg.render.zfar,
                    aspect_ratio = cfg.render.aspect_ratio,
                    fov = cfg.render.fov,
                    device = device,
                )
    return example_cameras

def get_viewpoint(yaw, elevation=0, radius=5, fov=15.0, cfg=None):
    if cfg is not None:
        if cfg.data.dataset_type == "srn":
            eye_location = [
                radius * math.cos(elevation) * math.cos(yaw),
                radius * math.cos(elevation) * math.sin(yaw),
                radius * math.sin(elevation),
                
            ]
            proj = pyrr.Matrix44.perspective_projection(
                fovy=fov, aspect=1, near=0.1, far=1000.0
            )
            look = pyrr.Matrix44.look_at(eye=eye_location, target=[0, 0, 0.0], up=[0, 0, 1])
            return proj, look

    eye_location = [
        radius * math.cos(elevation) * math.cos(yaw),
        radius * math.sin(elevation),
        radius * math.cos(elevation) * math.sin(yaw),
    ]
    proj = pyrr.Matrix44.perspective_projection(
        fovy=fov, aspect=1, near=0.1, far=1000.0
    )
    look = pyrr.Matrix44.look_at(eye=eye_location, target=[0, 0, 0.0], up=[0, 1, 0])
    return proj, look

def look_at_to_world_to_camera(look):
    # the camera-to-world rotation matrix in pytorch 3D has z facing away from the camera
    # opengl has the z vector facing into the camera and y upwards
    # to convert from pyrr lookat camera to pytorch we need to invert the z axis and
    # consequently the x axis to maintain right-handedness of the rotation convention
    # the transpose is taken so that the returned value is the world-to-camera matrix
    R = torch.from_numpy(np.array(look[:3, :3]) @ np.array([[-1, 0, 0],
                                                            [ 0, 1, 0],
                                                            [ 0, 0, -1]])).T
    # - T_lookat.T (transpose because T_lookat is a row vector) @ R_lookat.T 
    # is the position of the camera in world coordinates
    # last column of the world-to-camera matrix is therefore -R_world_to_camera @ T_world_coords
    T_world_coords = -torch.from_numpy(np.array(look[3:, :3]) @ np.array(look[:3, :3]).T) # row vector
    T = -R @ T_world_coords.T
    T = T[..., 0] # get rid of the last dimension for passing to camera constructor
    # throw in an assert check which uses the fact that the camera is looking at the origin and is 5 away
    # assert torch.abs(T[-1] - 5) < 1e-12 
    return R, T

@torch.no_grad()
def get_Ts_and_Rs_loop(batch_size, device, N_vis = 40, radius = 5,
                       elevation = np.pi / 6, cfg = None):
    max_yaw = 2 * np.pi
    max_elev = 0
    
    target_Rs = []
    target_Ts = []
    for i in range(N_vis):
        _, look = get_viewpoint(yaw = i * max_yaw / N_vis, elevation = elevation,
                                radius = radius, cfg = cfg)
        # Pytorch3D expects R and T to be world-to-camera transformation matrices
        R, T = look_at_to_world_to_camera(look)
        target_Rs.append(R.T[None, ])
        target_Ts.append(T[None, ])
    
    target_Rs = torch.cat(target_Rs, dim=0).unsqueeze(0).expand(batch_size, N_vis, 3, 3).to(device)
    target_Ts = torch.cat(target_Ts, dim=0).unsqueeze(0).expand(batch_size, N_vis, 3).to(device)

    return target_Rs, target_Ts
