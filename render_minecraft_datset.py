import glob
import os
import json

import moderngl
import numpy as np
import torch
from PIL import Image

from matplotlib import pyplot as plt

import minecraft
from camera import random_viewpoint
from utils import set_seed

import tqdm

def main():

    set_seed(0)
    # resolution at which the meshes are rasterized
    size = (256, 256)
    # resolution on which the images are saved
    save_size = (48, 48)

    out_dir = "data/minecraft"
    assert os.path.isdir(out_dir), "Dir {} does not exist, make sure .npy files are downloaded"

    # read data for training dataset
    with open(os.path.join(out_dir, "training_skins.txt"), "r") as f:
        training_skins = f.readlines()
    training_skins = [skin_name.split("\n")[0] for skin_name in training_skins]
    # TODO: change to the correct fnames
    training_cameras = np.load(os.path.join(out_dir, 
                            "cameras_training.npy"))
    training_poses = np.load(os.path.join(out_dir, 
                            "poses_training.npy"))

    # read data for validation dataset - uses training skins
    validation_cameras = np.load(os.path.join(out_dir, 
                            "cameras_ID_skin_OOD_pose_OOD_cond.npy"))
    validation_poses = np.load(os.path.join(out_dir, 
                            "poses_ID_skin_OOD_pose_OOD_cond.npy"))

    # read data for testing dataset
    with open(os.path.join(out_dir, "testing_skins.txt"), "r") as f:
        testing_skins = f.readlines()
    testing_skins = [skin_name.split("\n")[0] for skin_name in testing_skins]

    # TODO: change to the correct fnames
    testing_cameras = np.load(os.path.join(out_dir, 
                        "cameras_testing.npy"))
    testing_poses = np.load(os.path.join(out_dir, 
                        "poses_testing.npy"))
    testing_backgrounds = np.load(os.path.join(out_dir, 
                        "testing_bkgds.npy"))

    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    # ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    program = ctx.program(
        vertex_shader="""
        #version 330

        uniform mat4 Mvp;

        in vec3 in_pos;
        in vec2 in_uv;
        out vec2 uv;

        void main() {
            gl_Position = Mvp * vec4(in_pos, 1.0);
            uv = in_uv;
        }
        """,
        fragment_shader="""
        #version 330

        uniform sampler2D texture0;
        out vec4 color;
        in vec2 uv;

        void main() {
            vec4 sample = texture(texture0, uv);
            if (sample.a <= 0.0) {
                discard;
            }
            color = sample;
        }
        """,
    )

    fbo = ctx.framebuffer(
        ctx.renderbuffer(size),
        ctx.depth_renderbuffer(size),
    )
    fbo.use()

    training_imgs = []
    validation_imgs = []
    testing_imgs = []

    # =================== RENDERING TRAINING SET =======================
    for ex_idx in tqdm.tqdm(range(len(training_cameras))):
        
        training_imgs.append([])
        skin = Image.open(training_skins[ex_idx]).convert("RGBA")
        character = minecraft.character_for_skin(skin)

        skin_texture = ctx.texture(skin.size, 4, skin.tobytes())
        skin_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)       

        character.set_character_pose({k: v for k, v in zip(
            ["left_arm_pitch", "right_arm_pitch", "left_arm_roll",
             "right_arm_roll", "left_leg_pitch", "right_leg_pitch",
             "left_leg_roll", "right_leg_roll", "head_pitch",
             "head_roll", "head_yaw"],
             training_poses[ex_idx][0])}) 
        # [0] because the pose is saved multiple times for one character example 

        vertices, faces, uvs = character.get_mesh()
        geometry = ctx.buffer(torch.cat((0.03 * vertices, uvs), dim=-1)[faces, :].numpy())
        vao = ctx.vertex_array(program, [(geometry, "3f 2f", "in_pos", "in_uv")])

        for c_idx in range(training_cameras[ex_idx].shape[0]):
            # set camera
            proj, _ = random_viewpoint()
            look = training_cameras[ex_idx, c_idx]
            program["Mvp"].write((proj * look).astype("f4"))
            # clear buffer and render
            fbo.clear(1, 1, 1)
            skin_texture.use()
            vao.render(mode=moderngl.TRIANGLES)
            ctx.finish()

            img = Image.frombytes("RGBA", fbo.size, fbo.read(components=4))
            img.thumbnail(save_size, Image.ANTIALIAS)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = np.array(img)
            # add to the training images
            training_imgs[-1].append(img)

    # =================== RENDERING VALIDATION SET =======================
    for ex_idx in tqdm.tqdm(range(len(validation_cameras))):
        
        validation_imgs.append([])
        skin = Image.open(training_skins[ex_idx]).convert("RGBA")
        character = minecraft.character_for_skin(skin)

        skin_texture = ctx.texture(skin.size, 4, skin.tobytes())
        skin_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)       

        character.set_character_pose({k: v for k, v in zip(
            ["left_arm_pitch", "right_arm_pitch", "left_arm_roll",
             "right_arm_roll", "left_leg_pitch", "right_leg_pitch",
             "left_leg_roll", "right_leg_roll", "head_pitch",
             "head_roll", "head_yaw"],
                         validation_poses[ex_idx][0])})

        vertices, faces, uvs = character.get_mesh()
        geometry = ctx.buffer(torch.cat((0.03 * vertices, uvs), dim=-1)[faces, :].numpy())
        vao = ctx.vertex_array(program, [(geometry, "3f 2f", "in_pos", "in_uv")])

        for c_idx in range(validation_cameras[ex_idx].shape[0]):
            # set camera
            proj, _ = random_viewpoint()
            look = validation_cameras[ex_idx, c_idx]
            program["Mvp"].write((proj * look).astype("f4"))
            # clear buffer and render
            fbo.clear(1, 1, 1)
            skin_texture.use()
            vao.render(mode=moderngl.TRIANGLES)
            ctx.finish()
            img = Image.frombytes("RGBA", fbo.size, fbo.read(components=4))
            img.thumbnail(save_size, Image.ANTIALIAS)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = np.array(img)
            # add to the validation images
            validation_imgs[-1].append(img)

    # =================== RENDERING TESTING SET =======================
    for ex_idx in tqdm.tqdm(range(len(testing_cameras))):
        
        testing_imgs.append([])
        skin = Image.open(testing_skins[ex_idx]).convert("RGBA")
        character = minecraft.character_for_skin(skin)

        skin_texture = ctx.texture(skin.size, 4, skin.tobytes())
        skin_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)       

        character.set_character_pose({k: v for k, v in zip(
            ["left_arm_pitch", "right_arm_pitch", "left_arm_roll",
             "right_arm_roll", "left_leg_pitch", "right_leg_pitch",
             "left_leg_roll", "right_leg_roll", "head_pitch",
             "head_roll", "head_yaw"],
                         testing_poses[ex_idx][0])})

        vertices, faces, uvs = character.get_mesh()
        geometry = ctx.buffer(torch.cat((0.03 * vertices, uvs), dim=-1)[faces, :].numpy())
        vao = ctx.vertex_array(program, [(geometry, "3f 2f", "in_pos", "in_uv")])

        for c_idx in range(testing_cameras[ex_idx].shape[0]):
            # set camera
            proj, _ = random_viewpoint()
            look = testing_cameras[ex_idx, c_idx]
            program["Mvp"].write((proj * look).astype("f4"))
            # clear buffer and render
            fbo.clear(1, 1, 1)
            skin_texture.use()
            vao.render(mode=moderngl.TRIANGLES)
            ctx.finish()
            img = Image.frombytes("RGBA", fbo.size, fbo.read(components=4))
            img.thumbnail(save_size, Image.ANTIALIAS)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = np.array(img)

            mask = np.repeat(img[:, :, 3:], 3, axis=2).astype(np.float64)/255
            img = img.astype(np.float64)/255

            img = img[:, :, :3] * mask + testing_backgrounds[ex_idx] * np.ones_like(img[:, :, :3]) * (1-mask)
            img = np.round(img*255).astype(np.uint8)

            # add to the testing images
            testing_imgs[-1].append(img)

    # save sets to memory
    for set_name, image_list in zip(
            ["imgs_training", "imgs_ID_skin_OOD_pose_OOD_cond", "imgs_testing"],
            [training_imgs, validation_imgs, testing_imgs]
        ):
        for ex_idx, example in enumerate(image_list):
            image_list[ex_idx] = np.array(example)
        np.save(os.path.join(out_dir, set_name), np.array(image_list))

if __name__=="__main__":
    main()