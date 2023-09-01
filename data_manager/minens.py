import torch
import numpy as np

from . import look_at_to_world_to_camera, get_viewpoint

import os

class QuadrupleDataset():
    def __init__(self, cfg, dataset_name="training",
                 convert_to_single_conditioning=False,
                 convert_to_double_conditioning=True,
                 for_training=False):

        self.images_by_character = []
        self.cfg = cfg
        self.for_training = for_training

        if dataset_name == "training":
            self.background = cfg.data.background
        else:
            self.background = "white"
            print("Warning: overriding background to white for validation and test sets")
        self.background_buffer = torch.ones(3, requires_grad=False).float()

        self.dataset_name = dataset_name
        training_image_path = os.path.join(cfg.data.data_path,
                                           "imgs_{}.npy".format(dataset_name))
        # ============ Load images ============
        images = np.load(training_image_path)
        self.images = images.astype(np.float32()) / 255
        if cfg.data.subset!=-1 and dataset_name=="training":
            self.images = self.images[:cfg.data.subset]
        self.images = torch.from_numpy(self.images)
        if self.background != "white":
            assert self.images.shape[-1] == 4, "Expected RGBA format when specifying different background"
        # put channel dimension ahead of height and width
        self.images = torch.permute(self.images, dims=[0, 1, 4, 2, 3])
        training_camera_path = os.path.join(cfg.data.data_path,
                                            "cameras_{}.npy".format(dataset_name))
        # ============ Load cameras and process ============
        cameras = np.load(training_camera_path)

        N_examples, N_imgs, C, self.H, self.W = self.images.shape

        self.rotations = []
        self.translations = []
        self.pose_orig_embed = []
        self.pose_dir_embed = []
        
        fov_focal = cfg.render.fov * 2 * np.pi / 360
        self.focal = cfg.data.input_size[0] / (2 * np.tan(fov_focal / 2))
        
        if convert_to_single_conditioning or convert_to_double_conditioning:
            images_reshaped = []

        print("Preparing the dataset of with {} examples...".format(N_examples))
        if self.cfg.data.encode_rays:
            self.X, self.Y, self.Z = self.get_camera_screen_unprojected()

        for i in range(N_examples):
            r_char, t_char, orig_embed, dir_embed = self.load_character_pose(cameras[i])
            # ============= Reshape data depending on conditioning =============
            if not convert_to_single_conditioning and not convert_to_double_conditioning:
                self.rotations.append(torch.cat(r_char).unsqueeze(0))
                self.translations.append(torch.cat(t_char).unsqueeze(0))
                if self.cfg.data.encode_pose:
                    self.pose_orig_embed.append(torch.cat(orig_embed).unsqueeze(0))
                    self.pose_dir_embed.append(torch.cat(dir_embed).unsqueeze(0))
            elif convert_to_double_conditioning:
                assert convert_to_single_conditioning==False, "Cannot convert to both single and double conditioning"
                self.rotations.append(torch.cat([r_char[0],
                                                 r_char[1],
                                                 r_char[-1]]).unsqueeze(0))
                self.translations.append(torch.cat([t_char[0],
                                                    t_char[1],
                                                    t_char[-1]]).unsqueeze(0))
                if self.cfg.data.encode_pose:
                    self.pose_orig_embed.append(torch.cat([orig_embed[0],
                                                           orig_embed[1],
                                                           orig_embed[-1]]).unsqueeze(0))
                    self.pose_dir_embed.append(torch.cat([dir_embed[0],
                                                          dir_embed[1],
                                                          dir_embed[-1]]).unsqueeze(0))
                images_reshaped.append(torch.cat([self.images[i, 0].unsqueeze(0),
                                                  self.images[i, 1].unsqueeze(0),
                                                  self.images[i, -1].unsqueeze(0)]).unsqueeze(0))
            else:
                max_chars = 2
                for j in range(max_chars):
                    self.rotations.append(torch.cat([r_char[j],
                                                     r_char[2]]).unsqueeze(0))
                    self.translations.append(torch.cat([t_char[j],
                                                        t_char[2]]).unsqueeze(0))
                    if self.cfg.data.encode_pose:
                        self.pose_orig_embed.append(torch.cat([orig_embed[j],
                                                            orig_embed[2]]).unsqueeze(0))
                        self.pose_dir_embed.append(torch.cat([dir_embed[j],
                                                            dir_embed[2]]).unsqueeze(0))
                    images_reshaped.append(torch.cat([self.images[i, j].unsqueeze(0),
                                                      self.images[i, 2].unsqueeze(0)]).unsqueeze(0))
        
        if convert_to_single_conditioning or convert_to_double_conditioning:
            self.images = torch.cat(images_reshaped)
        
        if self.cfg.data.encode_pose:
            self.pose_orig_embed = torch.cat(self.pose_orig_embed) 
            self.pose_dir_embed = torch.cat(self.pose_dir_embed) 
        self.rotations = torch.cat(self.rotations) 
        self.translations = torch.cat(self.translations) 

    def load_character_pose(self, cameras_char):
        rotations_this_character = []
        translations_this_character = []
        pose_orig_embed_this_character = []
        pose_dir_embed_this_character = []
        for c_idx in range(cameras_char.shape[0]):
            look_at = cameras_char[c_idx, ...]
            # R is in the convention that positions are column vectors
            R, T = look_at_to_world_to_camera(look_at)
            # transpose rotation because Pytorch3D expects pose matrices
            # in row major order
            rotations_this_character.append(R.T.unsqueeze(0))
            translations_this_character.append(T.unsqueeze(0))
            # T cam to world = - transpose(R world to cam) @ T 
            T_embedded = - torch.matmul(R.T, T.clone().detach())

            if self.cfg.data.normalize_pose:
                T_embedded /= 5

            if self.cfg.data.encode_pose:
                pose_orig_embed_this_character.append(
                    T_embedded[..., None, None].repeat(
                        1, 1, self.H,self. W).float())
                # encode camera direction with the z-vector of the camera (away from the image)
                # z-vector in world coordinates is the third column of the rotation matrix
                assert self.cfg.data.encode_rays
                pose_dir_embed_this_character.append(
                    self.get_raydir_embedding(
                        R.T, self.X, self.Y, self.Z))

        return (rotations_this_character, 
                translations_this_character, 
                pose_orig_embed_this_character, 
                pose_dir_embed_this_character)

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

    def get_item_with_virtual_views(self, idx, N_virtual_views):
        
        attributes_without_virtual_views = self.__getitem__(idx)
        if N_virtual_views==0:
            return attributes_without_virtual_views
        rotations_virtual = []
        translations_virtual = []
        pose_dir_embed_virtual = []
        pose_orig_embed_virtual = []            

        H, W = attributes_without_virtual_views["training_imgs"].shape[2:4]

        for idx in range(N_virtual_views):
            _, look = get_viewpoint(2 * np.pi * idx / N_virtual_views)
            
            R, T = look_at_to_world_to_camera(look)
            rotations_virtual.append(R.T.unsqueeze(0))
            translations_virtual.append(T.unsqueeze(0))
            pose_dir_embed_virtual.append(self.get_raydir_embedding(R.T, *self.get_camera_screen_unprojected()))

            T_embedded = - torch.matmul(R.T, T.clone().detach())

            if self.cfg.data.normalize_pose:
                T_embedded /= 5
            pose_orig_embed_virtual.append(T_embedded[..., None, None].repeat(1, 1, H, W).float())

        if N_virtual_views > 0:
            rotations_virtual = torch.cat(rotations_virtual, dim=0)
            translations_virtual = torch.cat(translations_virtual, dim=0)
            pose_embed_virtual = torch.cat([torch.cat(pose_orig_embed_virtual, dim=0),
                                            torch.cat(pose_dir_embed_virtual, dim=0)], dim=1)
            
            attributes_virtual = {"target_Rs": rotations_virtual,
                                "target_Ts": translations_virtual,
                                "pose_embed": pose_embed_virtual,
                                "background": attributes_without_virtual_views["background"][:1, 
                                                    ...].expand(N_virtual_views, 
                                                                *attributes_without_virtual_views["background"].shape[1:])}
            attributes_with_virtual_views = {}
            for k, v in attributes_without_virtual_views.items():
                if k in ["target_Rs", "target_Ts", "pose_embed", "background"]:
                    attributes_with_virtual_views[k] = torch.cat([attributes_without_virtual_views[k][:-1],
                                                                attributes_virtual[k],
                                                                attributes_without_virtual_views[k][-1:]],
                                                                dim=0)
                else:
                    attributes_with_virtual_views[k] = v

            return attributes_with_virtual_views
        else:
            if N_virtual_views == -1:
                attributes_without_virtual_views["pose_embed"] = attributes_without_virtual_views["pose_embed"][:-1]
            return attributes_without_virtual_views

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

    def __getitem__(self, idx):
        """
        Return an image pair of the same character. The input image camera pose is
        encoded via _embed vector. Target image camera pose is passed explicitly.
        """
        # need to apply background if images have alpha channel
        N_imgs, _, H, W = self.images[idx].shape
        if self.images.shape[2] == 4:
            alpha_mask = self.images[idx][:, 3:, ...].expand(N_imgs, 3, H, W)
            if self.background == "random":
                background = self.background_buffer * torch.rand(3).float()
            elif self.background == "white":
                background = self.background_buffer
            else:
                raise NotImplementedError
            background = background[None, :, None, None].expand(N_imgs, 3, H, W)
            imgs_with_background = self.images[idx][:, :3, ...] * alpha_mask + background * (1 - alpha_mask)
            training_imgs = imgs_with_background[:-1]
            validation_imgs = imgs_with_background[-1:]
        else:
            # TODO have a smarter way of implementing this
            background = self.images[idx][0, :, 0, 0][None, :, None, None].expand(N_imgs, 3, H, W)
            training_imgs = self.images[idx][:-1] # the first two images are training
            validation_imgs = self.images[idx][-1:] # the third image is validation

        input_pose_orig_embeds = self.pose_orig_embed[idx][:-1]
        input_pose_dir_embeds = self.pose_dir_embed[idx][:-1]

        target_Rs = self.rotations[idx] # by default render all 3 images
        target_Ts = self.translations[idx]

        # concatenate the images and their encodings along the channel dimension
        input_pose_orig_embeds = self.pose_orig_embed[idx][:-1]
        input_pose_dir_embeds = self.pose_dir_embed[idx][:-1]
        aug_img_in = torch.cat([training_imgs,
                                input_pose_orig_embeds,
                                input_pose_dir_embeds], dim=1)
        val_pose_orig_embeds = self.pose_orig_embed[idx][-1:]
        val_pose_dir_embeds = self.pose_dir_embed[idx][-1:]
        val_pose_embeds = torch.cat([val_pose_orig_embeds,
                                        val_pose_dir_embeds], dim=1)
        assert aug_img_in.shape[1] == 9

        if self.dataset_name=="training" or self.for_training:
            input_order = torch.randperm(training_imgs.shape[0])
            x_in = aug_img_in[input_order[1:], :3, ...]
            x_cond = aug_img_in[input_order[:1], :3, ...]
            pose_embed = aug_img_in[input_order, 3:, ...]
            training_img_copy = training_imgs[input_order, ...] * 2 - 1
        else:
            input_order = torch.arange(len(target_Rs))
            x_in = torch.zeros(1, 3, *aug_img_in.shape[2:])
            x_cond = aug_img_in[:, :3, ...]
            # last image is the target image we want to diffuse - append its pose embedding
            pose_embed = torch.cat([aug_img_in[:, 3:, ...], val_pose_embeds], dim=0)
            training_img_copy = training_imgs * 2 - 1

        return {
                "training_imgs": training_img_copy,
                "validation_imgs": validation_imgs,
                "x_in": x_in,
                "x_cond": x_cond * 2 - 1,
                "pose_embed": pose_embed,
                "val_pose_embeds": val_pose_embeds,
                "target_Rs": target_Rs[input_order],
                "target_Ts": target_Ts[input_order],
                "background": background[input_order].permute(0, 2, 3, 1)
                }

    def __len__(self):
        return len(self.images)
    
    def get_item_for_testing(self, ex_idx, N_noisy):
        assert self.images.shape[1] == 2, "Expected conversion to single cond"
        # ambiguous subset
        if ex_idx < 100:
            idx_in_dataset = self.ambiguous_idxs[ex_idx]
        # random subset
        else:
            idx_in_dataset = self.random_idxs[ex_idx-100]
        ex_with_virtual_views = self.get_item_with_virtual_views(idx_in_dataset, N_noisy-1)
        ex_with_virtual_views["test_imgs"] = ex_with_virtual_views["validation_imgs"]
        ex_with_virtual_views["test_Rs"] = ex_with_virtual_views["target_Rs"][-1:, ...]
        ex_with_virtual_views["test_Ts"] = ex_with_virtual_views["target_Ts"][-1:, ...]

        return ex_with_virtual_views
    
    def get_example_id(self, ex_idx):
        # ambiguous subset
        if ex_idx < 100:
            if not hasattr(self, "ambiguous_idxs"):
                self.ambiguous_idxs = np.load(os.path.join(self.cfg.data.data_path,
                    "ambiguous_pose.npy"))[:100]
            idx_in_dataset = self.ambiguous_idxs[ex_idx]
        # random subset            
        else:
            if not hasattr(self, "random_idxs"):
                self.random_idxs = np.load(os.path.join(self.cfg.data.data_path,
                    "random_order_of_sample_idxs.npy"))[:100]
            idx_in_dataset = self.random_idxs[ex_idx-100]
        return idx_in_dataset