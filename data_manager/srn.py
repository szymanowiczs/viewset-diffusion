import glob
import imageio
import os
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms

from . import augment_cameras
SHAPENET_DATASET_ROOT = None # Change this line
assert SHAPENET_DATASET_ROOT is not None, "Update the location of the SRN Shapenet Dataset"

class SRNDataset():
    def __init__(self, cfg,
                 convert_to_single_conditioning=False,
                 convert_to_double_conditioning=True,
                 dataset_name="train",
                 for_training=False
                 ):
        self.cfg = cfg
        self.dataset_name = dataset_name

        if for_training:
            self.for_training=True
        else:
            self.for_training=False

        self.original_size = 128
        if self.cfg.data.input_size[0] != self.original_size:
            # we will resize the images, adjust the focal length later in dataloading
            # we do not need to adjust the world size if focal length is adjusted
            self.resize_transform = transforms.Resize((self.cfg.data.input_size[0], 
                                                       self.cfg.data.input_size[1]))
        if convert_to_single_conditioning:
            self.no_imgs_per_example = 2
            self.source_img_idxs = [64]
        elif convert_to_double_conditioning:
            self.no_imgs_per_example = 3
            self.source_img_idxs = [64, 104]
        else:
            self.no_imgs_per_example = 4

        self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "srn_{}/srn_{}/{}_{}".format(cfg.data.category,
                                                                              cfg.data.category,
                                                                              cfg.data.category,
                                                                              dataset_name))

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        self.image_to_tensor = transforms.ToTensor()

        # SRN dataset is in convention x right, y down, z away from camera
        # Pytorch3D is in convention x left, y up, z away from the camera
        self._coord_trans = torch.diag(
            torch.tensor([-1, -1, 1, 1], dtype=torch.float32)
        )

        self.canonical_raydirs = self.get_camera_screen_unprojected()
        # focal field of view remains unchanged
        fov_focal = cfg.render.fov * 2 * np.pi / 360
        # focal length in pixels is adjusted with the data input size
        # Pytorch3D cameras created are FOV cameras with default
        # principal point at 0 so we do not need to adjust principal point
        self.focal = cfg.data.input_size[0] / (2 * np.tan(fov_focal / 2))

    def create_dataset(self, compressed_data_path):
        print("Dataset not found in .npz format, converting from .png and .txt files")
        os.mkdir(compressed_data_path)
        all_rgbs = {}
        all_poses = {}
        for index in tqdm(range(len(self.intrins))):
            example_id = os.path.basename(os.path.dirname(self.intrins[index]))
            all_rgbs[example_id] = []
            all_poses[example_id] = []
            intrin_path = self.intrins[index]
            dir_path = os.path.dirname(intrin_path)
        
            rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
            pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
            assert len(rgb_paths) == len(pose_paths), "Unequal number of paths"
            for rgb_path, pose_path in zip(rgb_paths, pose_paths):
                rgb = imageio.imread(rgb_path)[..., :3]
                rgb = self.image_to_tensor(rgb)
                pose = torch.from_numpy(
                    np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
                )
                all_rgbs[example_id].append(rgb)
                all_poses[example_id].append(pose)

            all_rgbs[example_id] = torch.stack(all_rgbs[example_id])
            all_poses[example_id] = torch.stack(all_poses[example_id])

        print("Dataset converted, saving to compressed format")
        # convert the data to numpy archives and save
        for dict_to_save, dict_name in zip([all_rgbs, all_poses],
                                           ["rgbs", "poses"]):
            np.savez(os.path.join(compressed_data_path, dict_name+".npz"),
                        **{k: v.numpy() for k, v in dict_to_save.items()})

    def read_dataset(self, compressed_data_path):
        self.all_rgbs = {k: torch.from_numpy(v) for k, v in 
            np.load(os.path.join(compressed_data_path, "rgbs.npz")).items()}
        self.all_poses = {k: torch.from_numpy(v) for k, v in 
            np.load(os.path.join(compressed_data_path, "poses.npz")).items()}

    def pose_to_target_Rs_and_Ts(self, pose):
        # pose is the camera to world matrix in column major order
        # Pytorch3D expects target R and T in row major order
        target_T = - pose[:3, :3].T @ pose[:3, 3]
        target_R = pose[:3, :3] # transpose for inverse and for
        # changing the major axis swap
        return target_R, target_T

    def get_camera_screen_unprojected(self):
        # Step 1. to encode ray direction
        # 1. generate a grid of x, y coordinates of every point in screen coordinates
        # 2. Invert direction: screen direction is +ve right, down, camera direction
        #  is +ve left, up
        Y, X = torch.meshgrid(-torch.linspace(0.5, self.cfg.data.input_size[1]-0.5, 
                                                    self.cfg.data.input_size[1]),
                              -torch.linspace(0.5, self.cfg.data.input_size[0]-0.5,
                                                    self.cfg.data.input_size[0]),
                              indexing='ij')
        Z = torch.ones_like(X)
        return X, Y, Z

    def get_raydir_embedding(self, camera_R, X, Y, Z):
        # Steps 3 and 4 to encode ray direction
        # 3. add the depth dimension and scale focal lengths
        raydirs_cam = torch.stack(((X - self.cfg.data.input_size[0]) / self.focal,
                                   (Y - self.cfg.data.input_size[0]) / self.focal,
                                    Z))
        # 4. convert from camera coordinates to world coordinates
        raydirs_cam = raydirs_cam / torch.norm(raydirs_cam, dim=0, keepdim=True) # 3 x H x W
        raydirs_cam = raydirs_cam.permute(1, 2, 0).reshape(-1, 3)
        # camera to world rotation matrix is camera_R.T. It assumes row-major order, i.e. that the
        # position vectors are row vectors so we post-multiply row vectors by the rotation matrix
        # camera position gets ignored because we want ray directions, not their end-points.
        raydirs_world = torch.matmul(raydirs_cam, camera_R.T.float())
        raydirs_world = raydirs_world.reshape(self.cfg.data.input_size[0], 
                                              self.cfg.data.input_size[1], 3).permute(2, 0, 1).float().unsqueeze(0)
        return raydirs_world

    def __len__(self):
        return len(self.intrins)

    def get_item_with_virtual_views(self, index, N_virtual_views):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))

        if self.dataset_name == "train":
            frame_idxs = torch.randint(0, len(self.all_rgbs[example_id]), 
                                       (self.no_imgs_per_example,))         
        else:
            self.load_example_id(example_id, intrin_path)
            frame_idxs = torch.cat([torch.tensor(self.source_img_idxs), 
                                    torch.tensor([96])
                                    ], dim=0)
        n_imgs_in_seq = len(self.all_rgbs[example_id])
        if N_virtual_views == 0:
            virtual_view_idxs = []
        elif N_virtual_views == 1:
            frame_idxs = torch.cat([torch.tensor(self.source_img_idxs), 
                                    torch.tensor([115])
                                    ], dim=0)
            virtual_view_idxs = [90]
        elif N_virtual_views == 3:
            frame_idxs = torch.cat([torch.tensor(self.source_img_idxs), 
                                    torch.tensor([64])
                                    ], dim=0)
            virtual_view_idxs = [90, 104, 115]
        else:
            virtual_view_idxs = [i for i in range(0, n_imgs_in_seq, n_imgs_in_seq // N_virtual_views)][:N_virtual_views]
        all_view_idxs = [*frame_idxs[:-1], *virtual_view_idxs, frame_idxs[-1]]
        all_view_idxs = torch.tensor(all_view_idxs)
        attributes = self.get_attributes_selected_sequence_and_frames(example_id, all_view_idxs)
        return self.rearange_order_for_diffusion(attributes)

    def get_item_for_testing(self, index, N_virtual_views):
        """
        Returns the conditioning images as the fixed source images from SRN
        The x_in images returned will be selected indexes on the archimedean spiral
        In addition to usual data passed by the dataloader this function also
        returns target_Rs_loop and target_Ts_loop. These are renders which will
        be needed for evaluating metrics.
        """
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())
            assert height == width
            assert height == 128
            assert focal == 131.25
        self.load_example_id(example_id, intrin_path)
        virtual_view_idxs = []
        if N_virtual_views == 2:
            virtual_view_idxs = [90, 115]
        elif N_virtual_views == 3:
            virtual_view_idxs = [90, 104, 120]
        elif N_virtual_views == 4:
            virtual_view_idxs = [30, 90, 104, 115]
        elif N_virtual_views == 5:
            virtual_view_idxs = [0, 20, 40, 80, 100]
        elif N_virtual_views == 6:
            virtual_view_idxs = [0, 15, 30, 45, 75, 90]
        cond_and_virtual_frame_idxs = self.source_img_idxs + [i for i in virtual_view_idxs]
        testing_frame_idxs = [i for i in range(len(self.all_rgbs[example_id]))
                              if i not in self.source_img_idxs] 
        attributes = self.get_attributes_selected_sequence_and_frames(example_id, 
                                                                      cond_and_virtual_frame_idxs)
        test_aug_imgs, test_Rs, test_Ts = self.get_attributes_selected_sequence_and_frames(example_id,
                                                                           testing_frame_idxs,
                                                                           for_testing=True)

        cond_and_virtual_views = self.rearange_order_for_diffusion(attributes)
        cond_and_virtual_views["test_imgs"] = test_aug_imgs[:, :3, ...]
        cond_and_virtual_views["test_Rs"] = test_Rs
        cond_and_virtual_views["test_Ts"] = test_Ts

        return cond_and_virtual_views

    def get_pose_embed(self, target_T, target_R):
        camera_dir_embed = self.get_raydir_embedding(target_R, *self.canonical_raydirs)
        # target_T is the translation from world to camera matrix
        # but we want the translation from camera to world matrix
        # otherwise all of them will be the same and that input will carry no information
        T_embedded = - torch.matmul(target_R, target_T.clone().detach())
        if self.cfg.data.normalize_pose:
            T_embedded /= 1.3
        camera_orig_embed = T_embedded[None, ..., None, None].expand(1, 3,
                                                            self.cfg.data.input_size[0],
                                                            self.cfg.data.input_size[1])        

        return camera_orig_embed, camera_dir_embed

    def get_attributes_selected_sequence_and_frames(self, example_id, frame_idxs,
                                                    for_testing = False):
        aug_img_in = []
        camera_Rs = []
        camera_Ts = []
        if self.cfg.data.rotation_augmentation and self.dataset_name=="train":
            rot_aug = torch.rand(1) * 2 * np.pi
        else:
            rot_aug = None
        if self.cfg.data.translation_augmentation!=0 and self.dataset_name=="train":
            trans_aug = torch.rand(3,) * 2 * self.cfg.data.translation_augmentation \
                - self.cfg.data.translation_augmentation
            trans_aug[2] = 0
        else:
            trans_aug = None

        for frame_idx in frame_idxs:
            img = self.all_rgbs[example_id][frame_idx].unsqueeze(0)
            if self.cfg.data.input_size[0] != self.original_size and not for_testing:
                img = self.resize_transform(img)
            pose = self.all_poses[example_id][frame_idx]
            pose = pose @ self._coord_trans
            target_R, target_T = self.pose_to_target_Rs_and_Ts(pose)

            if rot_aug is not None or trans_aug is not None:
                target_R, target_T = augment_cameras(target_R.unsqueeze(0),
                                                     target_T.unsqueeze(0),
                                                     rot_aug, trans_aug,
                                                     rot_axis="z")
                target_R = target_R.squeeze(0)
                target_T = target_T.squeeze(0)

            if not for_testing:
                camera_orig_embed, camera_dir_embed = self.get_pose_embed(target_T, target_R)
                aug_img_in.append(torch.cat([img, camera_orig_embed, camera_dir_embed],
                                            dim=1))
            else:
                aug_img_in.append(img)
            camera_Rs.append(target_R.unsqueeze(0))
            camera_Ts.append(target_T.unsqueeze(0))

        aug_img_in = torch.cat(aug_img_in, dim=0)
        camera_Rs = torch.cat(camera_Rs, dim=0)
        camera_Ts = torch.cat(camera_Ts, dim=0)
        return aug_img_in, camera_Rs, camera_Ts

    def rearange_order_for_diffusion(self, attributes):
        aug_img_in, camera_Rs, camera_Ts = attributes
        if self.dataset_name == "train":
            training_imgs = aug_img_in[:-1, :3, ...] 
            validation_imgs = aug_img_in[-1:, :3, ...]
            val_pose_embeds = aug_img_in[-1:, 3:, ...]
            input_order = torch.randperm(training_imgs.shape[0])
            x_in = aug_img_in[input_order[1:], :3, ...]
            x_cond = aug_img_in[input_order[:1], :3, ...]
            pose_embed = aug_img_in[input_order, 3:, ...]

            all_imgs_copy = training_imgs[input_order, ...] * 2 - 1
        else:
            cond_idxs = [i for i in range(len(self.source_img_idxs))]
            testing_idxs = [i for i in range(len(self.source_img_idxs), len(aug_img_in))]
            if not self.for_training:
                x_in = torch.zeros((aug_img_in.shape[0] - len(self.source_img_idxs), 
                                    3, *aug_img_in.shape[2:]))
            else:
                x_in = aug_img_in[testing_idxs, :3, ...]
            x_cond = aug_img_in[cond_idxs, :3, ...]
            validation_imgs = aug_img_in[testing_idxs, :3, ...]
            training_imgs = aug_img_in[cond_idxs, :3, ...]

            input_order = cond_idxs + testing_idxs
            val_pose_embeds = aug_img_in[testing_idxs, 3:, ...]
            pose_embed = aug_img_in[input_order, 3:, ...]

            all_imgs_copy = aug_img_in[input_order, :3, ...] * 2 - 1

        return {
                "training_imgs": all_imgs_copy,
                "validation_imgs": validation_imgs,
                "x_in": x_in,
                "x_cond": x_cond * 2 - 1,
                "pose_embed": pose_embed,
                "val_pose_embeds": val_pose_embeds,
                "target_Rs": camera_Rs[input_order],
                "target_Ts": camera_Ts[input_order],
                "background": all_imgs_copy.permute(0, 2, 3, 1) * 0.0 + 1.0
                }

    def load_example_id(self, example_id, intrin_path,
                        idxs_to_load=None):
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        assert len(rgb_paths) == len(pose_paths)

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_poses = {}
            self.all_above_0_z_ind = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_poses[example_id] = []
            self.all_above_0_z_ind[example_id] = []

            if idxs_to_load is not None:
                rgb_paths_load = [rgb_paths[i] for i in idxs_to_load]
                pose_paths_load = [pose_paths[i] for i in idxs_to_load]
            else:
                rgb_paths_load = rgb_paths
                pose_paths_load = pose_paths

            path_idx = 0
            for rgb_path, pose_path in zip(rgb_paths_load, pose_paths_load):
                rgb = imageio.imread(rgb_path)[..., :3]
                rgb = self.image_to_tensor(rgb)
                pose = torch.from_numpy(
                    np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
                )
                self.all_rgbs[example_id].append(rgb)
                self.all_poses[example_id].append(pose)
                if pose[2, 3] > 0:
                    self.all_above_0_z_ind[example_id].append(path_idx)
                path_idx += 1

            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])
            self.all_poses[example_id] = torch.stack(self.all_poses[example_id])

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        if self.dataset_name == "train":
            # Loads all frames in an example
            self.load_example_id(example_id, intrin_path)
            frame_idxs = [self.all_above_0_z_ind[
                example_id][i] for i in torch.randperm(
                len(self.all_above_0_z_ind[example_id])
                )[:self.no_imgs_per_example]]
        else:
            with open(intrin_path, "r") as intrinfile:
                lines = intrinfile.readlines()
                focal, cx, cy, _ = map(float, lines[0].split())
                height, width = map(int, lines[-1].split())
            assert height == width
            assert height == 128
            assert focal == 131.25
            # only read the frames that will be used for testing -> from 250 files to 3
            if self.for_training:
                load_idxs = self.source_img_idxs + [30, 100]
                frame_idxs = [0, 1, 2]
            else:
                load_idxs = None
                frame_idxs = self.source_img_idxs + [i for i in range(0, len(self.all_rgbs[example_id]), 10)
                                                     if i not in self.source_img_idxs]

            self.load_example_id(example_id, intrin_path, idxs_to_load=load_idxs)

        attributes = self.get_attributes_selected_sequence_and_frames(example_id, frame_idxs)
        return self.rearange_order_for_diffusion(attributes)