import torch
from torchvision import transforms
import numpy as np

import hydra
from omegaconf import DictConfig

import os
from tqdm import tqdm

from . import (
    EXCLUDE_SEQUENCE, 
    LOW_QUALITY_SEQUENCE, 
    CAMERAS_CLOSE_SEQUENCE, 
    CAMERAS_FAR_AWAY_SEQUENCE
)

from . import augment_cameras, normalize_sequence

from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.tools.config import expand_args_fields

CO3D_DATASET_ROOT = None # Change this line to where CO3D is downloaded
assert CO3D_DATASET_ROOT != None, "Please change CO3D path"

class CO3DDataset():
    def __init__(self, cfg,
                 convert_to_single_conditioning=False,
                 convert_to_double_conditioning=True,
                 dataset_name="train",
                 for_training=False):
        self.cfg = cfg
        self.dataset_name = dataset_name

        if for_training:
            self.for_training=True
        else:
            self.for_training=False

        # =============== Dataset parameters ===============
        if convert_to_single_conditioning:
            self.no_imgs_per_example = 2
        elif convert_to_double_conditioning:
            self.no_imgs_per_example = 3
        else:
            self.no_imgs_per_example = 4
        
        # =============== Dataset loading ===============
        data_path = os.path.join(self.cfg.data.data_path, 
                                 self.cfg.data.category)
        try:
            self.read_dataset(data_path, dataset_name)
        except:
            if not os.path.isdir(data_path):
                os.mkdir(data_path)
            print("building dataset from scratch at {}".format(data_path))
            self.create_dataset(cfg, data_path, "train")
            self.create_dataset(cfg, data_path, "val")
            self.create_dataset(cfg, data_path, "test")
            return 0

        self.preprocess_pose_embeddings()
        self.sequence_starts_from = torch.tensor(self.sequence_starts_from)

        # =============== Dataset order for evaluation ===============
        if dataset_name == "test" or dataset_name == "val":
            self.fixed_frame_idxs = torch.from_numpy(np.load(os.path.join(data_path,
                                                                          "fixed_frame_idxs_{}.npy".format(dataset_name))))
            if self.no_imgs_per_example == 2:
                self.fixed_frame_idxs = torch.cat([torch.cat([self.fixed_frame_idxs[:, :1], self.fixed_frame_idxs[:, -1:]], dim=1),
                                                   torch.cat([self.fixed_frame_idxs[:, 1:2], self.fixed_frame_idxs[:, -1:]], dim=1)], 
                                                   dim=0)
            elif self.no_imgs_per_example == 3:
                self.fixed_frame_idxs = torch.cat([self.fixed_frame_idxs[:, :2], self.fixed_frame_idxs[:, -1:]], dim=1)

    def select_testing_idxs(self):
        np_idxs_path = os.path.join(self.cfg.data.data_path,
                                    self.cfg.data.category,
                                        "random_order_of_sample_idxs.npy")
        sample_idx_order = np.load(np_idxs_path)[:100]
        self.fixed_frame_idxs = self.fixed_frame_idxs[sample_idx_order]

    def create_dataset(self, cfg, data_out_path, dataset_name):
        # run dataset creation 
        # check flagged sequences
        # copy over old validation split order

        # implement foreground augmentation
        # change training, validation and testing to have white backgrounds

        subset_name = "fewview_dev"

        expand_args_fields(JsonIndexDatasetMapProviderV2)
        dataset_map = JsonIndexDatasetMapProviderV2(
            category=cfg.data.category,
            subset_name=subset_name,
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=True,
            dataset_root=CO3D_DATASET_ROOT,
            dataset_JsonIndexDataset_args=DictConfig(
                {"remove_empty_masks": False, "load_point_clouds": True}
            ),
        ).get_dataset_map()

        self.created_dataset = dataset_map[dataset_name]

        # Exclude bad and low quality sequences
        if cfg.data.category in EXCLUDE_SEQUENCE.keys():
            valid_sequence_names = [k for k in self.created_dataset.seq_annots.keys() if k not in EXCLUDE_SEQUENCE[cfg.data.category]]
        else:
            valid_sequence_names = list(self.created_dataset.seq_annots.keys())
        if cfg.data.category in LOW_QUALITY_SEQUENCE.keys():
            valid_sequence_names = [k for k in valid_sequence_names if k not in LOW_QUALITY_SEQUENCE[cfg.data.category]]

        self.images_all_sequences = {}
        self.focal_lengths_all_sequences = {}
        self.principal_points_all_sequences = {}
        self.camera_Rs_all_sequences = {}
        self.camera_Ts_all_sequences = {}
        min_overall_distance = 100000
        max_overall_distance = 0
        sequences_that_need_checking = []
        for sequence_name in tqdm(valid_sequence_names):
            frame_idx_gen = self.created_dataset.sequence_indices_in_order(sequence_name)
            frame_idxs = []
            images_this_sequence = []
            focal_lengths_this_sequence = []
            principal_points_this_sequence = []

            while True:
                try:
                    frame_idx = next(frame_idx_gen)
                    frame_idxs.append(frame_idx)
                except StopIteration:
                    break

            for frame_idx in frame_idxs:
                frame = self.created_dataset[frame_idx]
                rgb = torch.cat([frame.image_rgb, frame.fg_probability], dim=0)
                assert frame.image_rgb.shape[1] == frame.image_rgb.shape[2], "Expected square images"
                assert rgb.shape[0] == 4, "Expected RGBA images, got {}".format(rgb.shape[0])
                # resizing_factor = self.cfg.data.input_size[0] / frame.image_rgb.shape[1]
                rgb = transforms.functional.resize(rgb,
                                                self.cfg.data.input_size[0],
                                                interpolation=transforms.InterpolationMode.BILINEAR)
                # cameras are in NDC convention so when resizing the image we do not need to change
                # the focal length or principal point
                focal_lengths_this_sequence.append(frame.camera.focal_length)
                principal_points_this_sequence.append(frame.camera.principal_point)                
                images_this_sequence.append(rgb.unsqueeze(0))
                
            self.images_all_sequences[sequence_name] = torch.cat(images_this_sequence,
                                                                 dim=0)
            self.focal_lengths_all_sequences[sequence_name] = torch.cat(focal_lengths_this_sequence,
                                                         dim=0)
            self.principal_points_all_sequences[sequence_name] = torch.cat(principal_points_this_sequence,
                                                                           dim=0)
            
            normalized_cameras, min_dist, max_dist, _, needs_checking = normalize_sequence(self.created_dataset, sequence_name,
                                                                                           self.cfg.render.volume_extent_world)
            if needs_checking:
                sequences_that_need_checking.append(str(sequence_name) + "\n")
            self.camera_Rs_all_sequences[sequence_name] = normalized_cameras.R
            self.camera_Ts_all_sequences[sequence_name] = normalized_cameras.T

            if min_dist < min_overall_distance:
                min_overall_distance = min_dist
            if max_dist > max_overall_distance:
                max_overall_distance = max_dist

        print("Min distance: ", min_overall_distance)
        print("Max distance: ", max_overall_distance)
        with open(os.path.join(data_out_path, "sequences_to_check_{}.txt".format(dataset_name)), "w+") as f:
            f.writelines(sequences_that_need_checking)
        # get the sequence names - this is what we will sample from
        self.sequence_names = [k for k in self.images_all_sequences.keys()]
        self.sequence_starts_from = [0]
        for i in range(1, len(self.sequence_names)+1):
            self.sequence_starts_from.append(self.sequence_starts_from[-1] + len(self.images_all_sequences[self.sequence_names[i-1]]))

        # convert the data to numpy archives and save
        for dict_to_save, dict_name in zip([self.images_all_sequences,
                                            self.focal_lengths_all_sequences,
                                            self.principal_points_all_sequences,
                                            self.camera_Rs_all_sequences,
                                            self.camera_Ts_all_sequences],
                                           ["images",
                                            "focal_lengths",
                                            "principal_points",
                                            "camera_Rs",
                                            "camera_Ts"]):
            np.savez(os.path.join(data_out_path, dict_name+"_{}.npz".format(dataset_name)),
                                  **{k: v.detach().cpu().numpy() for k, v in dict_to_save.items()})
        
        # If the dataset is being made for evaluation we need to fix the frame indices that are
        # passed in as one batch. Each batch should have 4 images - this will support 
        # 3-image testing, 2-image testing and 1-image testing. The images for each batch should
        # be from the same sequence. The frames should be selected randomly. The batches should
        # include as many images from every sequence as possible, sampling randomly without 
        # replacement.
        if dataset_name == "test" or dataset_name == "val":
            self.fixed_frame_idxs = []
            for sequence_name in self.sequence_names:
                sequence_length = len(self.images_all_sequences[sequence_name])
                # randomly permute the frame indices within the sequence and then split into
                # batches of 4
                frame_idxs = torch.randperm(sequence_length)
                frame_idxs = frame_idxs[:len(frame_idxs) // 4 * 4]
                frame_idxs = frame_idxs.view(-1, 4)
                self.fixed_frame_idxs.append(frame_idxs + 
                                             self.sequence_starts_from[self.sequence_names.index(sequence_name)])
            np.save(os.path.join(data_out_path, "fixed_frame_idxs_{}.npy".format(dataset_name)),
                    torch.cat(self.fixed_frame_idxs, dim=0).detach().cpu().numpy())

        return None

    def read_dataset(self, data_path, dataset_name):
    
        join_excluded_sequences = []
        for excluded_category_dict in [EXCLUDE_SEQUENCE, 
                                       LOW_QUALITY_SEQUENCE, 
                                       CAMERAS_FAR_AWAY_SEQUENCE, 
                                       CAMERAS_CLOSE_SEQUENCE]:
            if self.cfg.data.category in excluded_category_dict.keys():
                join_excluded_sequences = join_excluded_sequences + excluded_category_dict[self.cfg.data.category]
        # read the data from the npz archives
        self.images_all_sequences = {k: torch.from_numpy(v) for k, v in 
                                     np.load(os.path.join(data_path, "images_{}.npz".format(dataset_name))).items()
                                     if k not in join_excluded_sequences}
        self.focal_lengths_all_sequences = {k: torch.from_numpy(v) for k, v in
                                            np.load(os.path.join(data_path, "focal_lengths_{}.npz".format(dataset_name))).items()
                                            if k not in join_excluded_sequences}
        self.principal_points_all_sequences = {k: torch.from_numpy(v) for k, v in
                                               np.load(os.path.join(data_path, "principal_points_{}.npz".format(dataset_name))).items()
                                               if k not in join_excluded_sequences}
        self.camera_Rs_all_sequences = {k: torch.from_numpy(v) for k, v in
                                        np.load(os.path.join(data_path, "camera_Rs_{}.npz".format(dataset_name))).items()
                                        if k not in join_excluded_sequences}
        self.camera_Ts_all_sequences = {k: torch.from_numpy(v) for k, v in
                                        np.load(os.path.join(data_path, "camera_Ts_{}.npz".format(dataset_name))).items()
                                        if k not in join_excluded_sequences}

        min_overall_distance = 1000000
        max_overall_distance = 0

        for seq_name, camera_Ts in self.camera_Ts_all_sequences.items():
            camera_dists = torch.norm(camera_Ts, dim=1)
            if camera_dists.min() < min_overall_distance:
                min_overall_distance = camera_dists.min()
                min_dist_seq = seq_name
            if camera_dists.max() > max_overall_distance:
                max_overall_distance = camera_dists.max()
                max_dist_seq = seq_name

        print("Min distance: ", min_overall_distance)
        print("Min distance seq: ", min_dist_seq)
        print("Max distance: ", max_overall_distance)
        print("Max distance seq: ", max_dist_seq)

        self.sequence_names = [k for k in self.images_all_sequences.keys()]
        self.sequence_starts_from = [0]
        for i in range(1, len(self.sequence_names)+1):
            self.sequence_starts_from.append(self.sequence_starts_from[-1] + len(self.images_all_sequences[self.sequence_names[i-1]]))

    def get_camera_screen_unprojected(self):
        # Step 1. and 2 to encode ray direction
        # 1. generate a grid of x, y coordinates of every point in screen coordinates
        # 2. convert the grid to x, y coordinates in NDC coordinates
        # NDC coordinates are positive up and left, the image pixel matrix indexes increase down and right
        # so to go to NDC we need to invert the direction
        Y, X = torch.meshgrid(-(torch.linspace(0.5, self.cfg.data.input_size[1]-0.5, 
                                                    self.cfg.data.input_size[1]) * 2 / self.cfg.data.input_size[1] - 1),
                              -(torch.linspace(0.5, self.cfg.data.input_size[0]-0.5,
                                                    self.cfg.data.input_size[0]) * 2 / self.cfg.data.input_size[0] - 1),
                                indexing='ij')
        Z = torch.ones_like(X)
        return X, Y, Z

    def get_raydir_embedding(self, camera_R, principal_points, focal_lengths, X, Y, Z):
        # Steps 3 and 4 to encode ray direction
        # 3. add the depth dimension and scale focal lengths
        raydirs_cam = torch.stack(((X - principal_points[0]) / focal_lengths[0],
                                   (Y - principal_points[1]) / focal_lengths[1],
                                    Z))
        # 4. convert from camera coordinates to world coordinates
        raydirs_cam = raydirs_cam / torch.norm(raydirs_cam, dim=0, keepdim=True) # 3 x H x W
        raydirs_cam = raydirs_cam.permute(1, 2, 0).reshape(-1, 3)
        # camera to world rotation matrix is camera_R.T. It assumes row-major order, i.e. that the
        # position vectors are row vectors so we post-multiply row vectors by the rotation matrix
        # camera position gets ignored because we want ray directions, not their end-points.
        raydirs_world = torch.matmul(raydirs_cam, camera_R.T)
        raydirs_world = raydirs_world.reshape(self.cfg.data.input_size[0], 
                                              self.cfg.data.input_size[1], 3).permute(2, 0, 1).float().unsqueeze(0)
        return raydirs_world

    def preprocess_pose_embeddings(self):
        self.pose_orig_embed_all_sequences = {}
        self.pose_dir_embed_all_sequences = {}
        X, Y, Z = self.get_camera_screen_unprojected()

        for sequence_name in self.sequence_names:
            H, W = self.images_all_sequences[sequence_name].shape[2:]
            
            pose_orig_embed, pose_dir_embed = self.pose_embeddings_camera_sequence(
                self.camera_Rs_all_sequences[sequence_name],
                self.camera_Ts_all_sequences[sequence_name],
                H, W,
                self.principal_points_all_sequences[sequence_name],
                self.focal_lengths_all_sequences[sequence_name],
                X, Y, Z
            )

            self.pose_orig_embed_all_sequences[sequence_name] = pose_orig_embed
            self.pose_dir_embed_all_sequences[sequence_name] = pose_dir_embed

    def pose_embeddings_camera_sequence(self, camera_Rs, camera_Ts, H, W,
                                        principal_points, focal_lengths, X, Y, Z):
        pose_orig_embeds = []
        pose_dir_embeds = []
        for camera_idx in range(len(camera_Rs)):
            camera_R = camera_Rs[camera_idx]
            camera_T = camera_Ts[camera_idx]
            T_embedded = - torch.matmul(camera_R, camera_T.clone().detach())
            pose_orig_embeds.append(T_embedded[..., None, None].repeat(1, 1, H, W).float())
            # encode camera direction with the z-vector of the camera (away from the image)
            # z-vector in world coordinates is the third column of the rotation matrix
            assert self.cfg.data.encode_rays
            raydirs_world = self.get_raydir_embedding(camera_R, 
                                                      principal_points[camera_idx],
                                                      focal_lengths[camera_idx],
                                                      X, Y, Z)
            pose_dir_embeds.append(raydirs_world)

        pose_orig_embeds = torch.cat(pose_orig_embeds, dim=0)
        pose_dir_embeds = torch.cat(pose_dir_embeds, dim=0)

        return pose_orig_embeds, pose_dir_embeds

    def __len__(self):
        if hasattr(self, "fixed_frame_idxs"):
            return len(self.fixed_frame_idxs)
        else:
            return len(self.sequence_names)
        
    def __getitem__(self, idx):
        if hasattr(self, "fixed_frame_idxs"):
            frame_idxs = self.fixed_frame_idxs[idx]
            # for the sequence name need to find the sequence that the frame indices belong to
            # this is done by finding in which interval in self.sequence_starts_from the frame index falls
            sequence_name = self.sequence_names[torch.searchsorted(self.sequence_starts_from, frame_idxs[0], right=True)-1]
            # the first N-1 frames are conditioning, the last one is the target
            frame_idxs = frame_idxs - self.sequence_starts_from[self.sequence_names.index(sequence_name)]
        else:
            sequence_name = self.sequence_names[idx]
            frame_idxs = torch.randint(self.sequence_starts_from[idx],
                                       self.sequence_starts_from[idx+1],
                                       (self.no_imgs_per_example,)) - self.sequence_starts_from[idx]
        
        rot_aug = None
        trans_aug = None

        if self.cfg.data.rotation_augmentation and self.dataset_name=="train":
            rot_aug = torch.rand(1) * 2 * np.pi
        if self.cfg.data.translation_augmentation!=0 and self.dataset_name=="train":
            trans_aug = torch.rand(3,) * 2 * self.cfg.data.translation_augmentation \
                - self.cfg.data.translation_augmentation
            trans_aug[1] = 0

        attributes = self.get_attributes_selected_sequence_and_frames(sequence_name, frame_idxs,
                                                                      rot_aug, trans_aug)

        return self.rearange_order_for_diffusion(attributes)

    def get_attributes_selected_sequence_and_frames(self, sequence_name, frame_idxs,
                                                    rot_aug=None, trans_aug=None):
        rgbs = self.images_all_sequences[sequence_name][frame_idxs].clone()
        if rgbs.shape[1] == 4:
            # get rid of the background
            if self.cfg.data.white_background:
                bkgd = 1.0
            else:
                bkgd = 0.0
            rgbs = rgbs[:, :3, ...] * rgbs[:, 3:, ...] + bkgd * (1-rgbs[:, 3:, ...])

        rgbs = torch.clamp(rgbs, 0, 1)
        training_imgs = rgbs[:-1]
        validation_imgs = rgbs[-1:]

        camera_Rs = self.camera_Rs_all_sequences[sequence_name][frame_idxs].clone()
        camera_Ts = self.camera_Ts_all_sequences[sequence_name][frame_idxs].clone()

        principal_points = self.principal_points_all_sequences[sequence_name][frame_idxs]
        focal_lengths = self.focal_lengths_all_sequences[sequence_name][frame_idxs]

        if rot_aug is not None or trans_aug is not None:
            camera_Rs, camera_Ts = augment_cameras(camera_Rs=camera_Rs, camera_Ts=camera_Ts,
                                                   rot_aug=rot_aug, trans_aug=trans_aug)
            all_pose_orig_embeds, all_pose_dir_embeds = self.pose_embeddings_camera_sequence(
                camera_Rs, camera_Ts,
                rgbs.shape[2], rgbs.shape[3],
                principal_points, focal_lengths,
                *self.get_camera_screen_unprojected()
            )
            input_pose_dir_embeds = all_pose_dir_embeds[:-1]
            input_pose_orig_embeds = all_pose_orig_embeds[:-1]
            val_pose_dir_embeds = all_pose_dir_embeds[-1:]
            val_pose_orig_embeds = all_pose_orig_embeds[-1:]
        else:
            input_pose_orig_embeds = self.pose_orig_embed_all_sequences[sequence_name][frame_idxs][:-1]
            input_pose_dir_embeds = self.pose_dir_embed_all_sequences[sequence_name][frame_idxs][:-1]
            val_pose_orig_embeds = self.pose_orig_embed_all_sequences[sequence_name][frame_idxs][-1:]
            val_pose_dir_embeds = self.pose_dir_embed_all_sequences[sequence_name][frame_idxs][-1:]
        
        aug_img_in = torch.cat([training_imgs,
                                input_pose_orig_embeds,
                                input_pose_dir_embeds], dim=1)
        val_pose_embeds = torch.cat([val_pose_orig_embeds,
                                     val_pose_dir_embeds], dim=1)
        assert aug_img_in.shape[1] == 9

        return rgbs, training_imgs, validation_imgs, aug_img_in, val_pose_embeds, principal_points, focal_lengths, camera_Rs, camera_Ts

    def get_item_with_virtual_views(self, index, N_virtual_views):
        frame_idxs = self.fixed_frame_idxs[index]
        sequence_name = self.sequence_names[
            torch.searchsorted(self.sequence_starts_from, 
                               frame_idxs[0], right=True)-1]
        frame_idxs = frame_idxs - self.sequence_starts_from[self.sequence_names.index(sequence_name)]

        n_imgs_in_seq = self.sequence_starts_from[self.sequence_names.index(sequence_name)+1] - \
            self.sequence_starts_from[self.sequence_names.index(sequence_name)]

        if N_virtual_views == 0:
            virtual_view_idxs = []
        else:
            virtual_view_idxs = [i for i in range(0, n_imgs_in_seq, 
                                    n_imgs_in_seq // N_virtual_views)][:N_virtual_views]

        all_view_idxs = [*frame_idxs[:-1], *virtual_view_idxs, frame_idxs[-1]]
        all_view_idxs = torch.tensor(all_view_idxs)
        attributes = self.get_attributes_selected_sequence_and_frames(sequence_name, all_view_idxs)
        attributes_rearranged = self.rearange_order_for_diffusion(attributes)
        if N_virtual_views == -1: # means that N_noisy was 0, i.e. deterministic inference
            attributes_rearranged["pose_embed"] = attributes_rearranged["pose_embed"][:-1]
        
        return attributes_rearranged

    def get_item_for_testing(self, index, N_noisy):

        assert self.no_imgs_per_example == 2, "Expected conversion to single cond"
        ex_with_virtual_views = self.get_item_with_virtual_views(index, N_noisy-1)
        ex_with_virtual_views["test_imgs"] = ex_with_virtual_views["validation_imgs"]
        ex_with_virtual_views["test_Rs"] = ex_with_virtual_views["target_Rs"][-1:, ...]
        ex_with_virtual_views["test_Ts"] = ex_with_virtual_views["target_Ts"][-1:, ...]

        return ex_with_virtual_views

    def get_example_id(self, ex_idx):
        frame_idxs = self.fixed_frame_idxs[ex_idx]
        sequence_name = self.sequence_names[
            torch.searchsorted(self.sequence_starts_from, 
                               frame_idxs[0], right=True)-1]
        return str(sequence_name) + '_' + str(ex_idx)

    def get_sequence_virtual_views_only(self, sequence_idx, N_virtual_views):
        sequence_name = self.sequence_names[sequence_idx]
        n_imgs_in_seq = self.sequence_starts_from[self.sequence_names.index(sequence_name)+1] - \
            self.sequence_starts_from[self.sequence_names.index(sequence_name)]

        all_view_idxs = [i for i in range(0, n_imgs_in_seq, 
                                n_imgs_in_seq // N_virtual_views)][:N_virtual_views]

        all_view_idxs = torch.tensor(all_view_idxs)
        attributes = self.get_attributes_selected_sequence_and_frames(sequence_name, all_view_idxs)
        
        rgbs, _, _, aug_img_in, val_pose_embeds, principal_points, focal_lengths, camera_Rs, camera_Ts = \
            attributes
        
        if self.cfg.data.white_background:
            bkgd = 1.0
        else:
            bkgd = 0.0

        return {
                "x_in": torch.zeros_like(rgbs),
                "x_cond": torch.empty_like(rgbs)[:0, ...],
                "pose_embed":  torch.cat([aug_img_in[:, 3:, ...],
                                          val_pose_embeds], dim=0),
                "principal_points": principal_points,
                "focal_lengths": focal_lengths,
                "target_Rs": camera_Rs,
                "target_Ts": camera_Ts,
                "background": rgbs.permute(0, 2, 3, 1) * 0 + bkgd
                } 

    def rearange_order_for_diffusion(self, attributes):
        rgbs, training_imgs, validation_imgs, aug_img_in, val_pose_embeds, principal_points, focal_lengths, camera_Rs, camera_Ts = \
            attributes
        # In training the validation image is unused
        if self.dataset_name=="train" or self.for_training:
            # shuffle the input images
            input_order = torch.randperm(training_imgs.shape[0])
            x_in = aug_img_in[input_order[1:], :3, ...]
            x_cond = aug_img_in[input_order[:1], :3, ...]
            pose_embed = aug_img_in[input_order, 3:, ...]
            training_img_copy = training_imgs[input_order, ...]
        # In testing the 'validation' image is the target
        else:
            num_cond_views = self.no_imgs_per_example-1
            input_order = torch.arange(len(camera_Rs))
            x_in = torch.zeros((len(input_order) - num_cond_views, 3, *aug_img_in.shape[2:]))
            x_cond = aug_img_in[:num_cond_views, :3, ...]
            pose_embed = torch.cat([aug_img_in[:, 3:, ...], val_pose_embeds], dim=0)
            training_img_copy = training_imgs 

        if self.cfg.data.white_background:
            bkgd = 1.0
        else:
            bkgd = 0.0

        return {
                "training_imgs": training_img_copy * 2 - 1,
                "validation_imgs": validation_imgs,
                "x_in": x_in,
                "x_cond": x_cond * 2 - 1,
                "pose_embed": pose_embed,
                "val_pose_embeds": val_pose_embeds,
                "principal_points": principal_points[input_order],
                "focal_lengths": focal_lengths[input_order],
                "target_Rs": camera_Rs[input_order],
                "target_Ts": camera_Ts[input_order],
                "background": rgbs[input_order].permute(0, 2, 3, 1) * 0 + bkgd
                }