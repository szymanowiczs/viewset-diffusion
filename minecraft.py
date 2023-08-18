from pytorch3d import transforms
from math import pi
import torch
import numpy as np
import random

class Character:
    def __init__(
        self, device="cpu", slim=False, second_layer=True, mirror=False, half_skin=False
    ):
        self.device = device
        self.slim = slim
        self.second_layer = second_layer
        self.mirror = mirror
        self.half_skin = half_skin
        self.left_arm_roll = 0
        self.left_arm_pitch = 0
        self.right_arm_roll = 0
        self.right_arm_pitch = 0
        self.left_leg_roll = 0
        self.left_leg_pitch = 0
        self.right_leg_roll = 0
        self.right_leg_pitch = 0
        self.head_roll = 0
        self.head_pitch = 0
        self.head_yaw = 0

        assert not half_skin or mirror

    def __str__(self):
        return f"Character(slim={self.slim},second_layer={self.second_layer},mirror={self.mirror},half_skin={self.half_skin})"

    def get_mesh(self):

        #      y
        #      ^
        #      |
        #      o--> x
        #     /
        #   z
        #
        #         +-------+-------+
        #         |19   18|23   22|
        #         |       |       |
        #         |  top  |bottom |
        #         |       |       |
        #         |16   17|20   21|
        # +-------+-------+-------+-------+
        # |3     2|7     6|11   10|15   14|
        # |       |       |       |       |
        # | right | front | left  | back  |
        # |       |       |       |       |
        # |0     1|4     5|8     9|12   13|
        # +-------+-------+-------+-------+

        def _make_box(width, height, depth, du, dv, expand=0, mirror=False):
            vertices = (
                torch.FloatTensor(
                    [
                        # right side
                        [-1, -1, -1],
                        [-1, -1, +1],
                        [-1, +1, +1],
                        [-1, +1, -1],
                        # front side
                        [-1, -1, +1],
                        [+1, -1, +1],
                        [+1, +1, +1],
                        [-1, +1, +1],
                        # left side
                        [+1, -1, +1],
                        [+1, -1, -1],
                        [+1, +1, -1],
                        [+1, +1, +1],
                        # back side
                        [+1, -1, -1],
                        [-1, -1, -1],
                        [-1, +1, -1],
                        [+1, +1, -1],
                        # top side
                        [-1, +1, +1],
                        [+1, +1, +1],
                        [+1, +1, -1],
                        [-1, +1, -1],
                        # bottom side
                        [+1, -1, +1],
                        [-1, -1, +1],
                        [-1, -1, -1],
                        [+1, -1, -1],
                    ]
                )
                * (expand + torch.FloatTensor([width, height, depth]))
                / 2
            )

            faces = torch.LongTensor(
                [
                    # right side
                    [0, 1, 2],
                    [2, 3, 0],
                    # front side
                    [4, 5, 6],
                    [6, 7, 4],
                    # front side
                    [8, 9, 10],
                    [10, 11, 8],
                    # back side
                    [12, 13, 14],
                    [14, 15, 12],
                    # top side
                    [16, 17, 18],
                    [18, 19, 16],
                    # bottom side
                    [20, 21, 22],
                    [22, 23, 20],
                ]
            )

            w, h, d = width, height, depth
            ox, oy = du, dv

            # With bilinear texture interpolation, we shift each texture by half a pixel.
            # Otherwise bilinear interpolation bleeds colors from nearby texture regions.
            # However, this also cuts half a pixel around the border, zooming the texture.
            # With nearest neighbour interpolation this should not be necessary.
            # e = 0.5
            e = 0

            uvs = torch.FloatTensor(
                [
                    # right side
                    [ox + 0 + e, oy + d + h - e],  # 0
                    [ox + d - e, oy + d + h - e],
                    [ox + d - e, oy + d + 0 + e],
                    [ox + 0 + e, oy + d + 0 + e],
                    # front side
                    [ox + d + 0 + e, oy + d + h - e],  # 4
                    [ox + d + w - e, oy + d + h - e],
                    [ox + d + w - e, oy + d + 0 + e],
                    [ox + d + 0 + e, oy + d + 0 + e],
                    # left side
                    [ox + d + w + 0 + e, oy + d + h - e],  # 8
                    [ox + d + w + d - e, oy + d + h - e],
                    [ox + d + w + d - e, oy + d + 0 + e],
                    [ox + d + w + 0 + e, oy + d + 0 + e],
                    # back side
                    [ox + d + w + d + 0 + e, oy + d + h - e],  # 12
                    [ox + d + w + d + w - e, oy + d + h - e],
                    [ox + d + w + d + w - e, oy + d + 0 + e],
                    [ox + d + w + d + 0 + e, oy + d + 0 + e],
                    # top side
                    [ox + d + 0 + e, oy + d - e],  # 16
                    [ox + d + w - e, oy + d - e],
                    [ox + d + w - e, oy + 0 + e],
                    [ox + d + 0 + e, oy + 0 + e],
                    # bottom side
                    [ox + d + w + 0 + e, oy + d - e],  # 20
                    [ox + d + w + w - e, oy + d - e],
                    [ox + d + w + w - e, oy + 0 + e],
                    [ox + d + w + 0 + e, oy + 0 + e],
                ]
            )

            if mirror:
                uvs[
                    [
                        8,  # swap side
                        9,
                        10,
                        11,
                        5,  # flip x
                        4,
                        7,
                        6,
                        0,  # swap side
                        1,
                        2,
                        3,
                        13,  # flip x
                        12,
                        15,
                        14,
                        17,  # flip x
                        16,
                        19,
                        18,
                        21,  # flip x
                        20,
                        23,
                        22,
                    ]
                ] = uvs.clone()

            uvs /= 64
            if self.half_skin:
                uvs[:, 1] *= 2

            return vertices, faces, uvs

        vertices = torch.FloatTensor(0, 3)
        faces = torch.LongTensor(0, 3)
        uvs = torch.FloatTensor(0, 2)

        def cat(v, f, u):
            nonlocal vertices
            nonlocal faces
            nonlocal uvs
            n = len(vertices)
            vertices = torch.cat((vertices, v), dim=0)
            faces = torch.cat((faces, f + n), dim=0)
            uvs = torch.cat((uvs, u), dim=0)

        def pose(v, roll, pitch, yaw, tx, ty):
            t1 = transforms.Rotate(
                transforms.euler_angles_to_matrix(
                    torch.FloatTensor([[roll, pitch, yaw]]) / 90 * pi, "ZXY"
                )
            )
            t2 = transforms.Translate([tx], [ty], [0])
            t = t1.compose(t2)
            return t.transform_points(v)

        def _make_head(du, dv, expand=0, mirror=False):
            v, f, u = _make_box(
                width=8, height=8, depth=8, du=du, dv=dv, expand=expand, mirror=mirror
            )
            v[:, 1] += 4  # joint at the base of the head
            v = pose(
                v,
                roll=self.head_roll,
                pitch=self.head_pitch,
                yaw=self.head_yaw,
                tx=0,
                ty=6,
            )
            cat(v, f, u)

        def _make_right_arm(du, dv, expand=0, mirror=False):
            v, f, u = _make_box(
                width=4 - self.slim,
                height=12,
                depth=4,
                du=du,
                dv=dv,
                expand=expand,
                mirror=mirror,
            )
            v[:, 1] -= 6  # joint at the top of the arm
            v = pose(
                v,
                roll=self.right_arm_roll,
                pitch=self.right_arm_pitch,
                yaw=0,
                tx=6 - self.slim,
                ty=6,
            )
            cat(v, f, u)

        def _make_left_arm(du, dv, expand=0, mirror=False):
            v, f, u = _make_box(
                width=4 - self.slim,
                height=12,
                depth=4,
                du=du,
                dv=dv,
                expand=expand,
                mirror=mirror,
            )
            v[:, 1] -= 6  # joint at the top of the arm
            v = pose(
                v,
                roll=self.left_arm_roll,
                pitch=self.left_arm_pitch,
                yaw=0,
                tx=-6 + self.slim,
                ty=6,
            )
            cat(v, f, u)

        def _make_torso(du, dv, expand=0, mirror=False):
            v, f, u = _make_box(
                width=8, height=12, depth=4, du=du, dv=dv, expand=expand, mirror=mirror
            )
            v = pose(v, roll=0, pitch=0, yaw=0, tx=0, ty=0)
            cat(v, f, u)

        def _make_right_leg(du, dv, expand=0, mirror=False):
            v, f, u = _make_box(
                width=4, height=12, depth=4, du=du, dv=dv, expand=expand, mirror=mirror
            )
            v[:, 1] -= 6  # joint at the top of the leg
            v = pose(
                v,
                roll=self.right_leg_roll,
                pitch=self.right_leg_pitch,
                yaw=0,
                tx=2,
                ty=-6,
            )
            cat(v, f, u)

        def _make_left_leg(du, dv, expand=0, mirror=False):
            v, f, u = _make_box(
                width=4, height=12, depth=4, du=du, dv=dv, expand=expand, mirror=mirror
            )
            v[:, 1] -= 6  # joint at the top of the leg
            v = pose(
                v,
                roll=self.left_leg_roll,
                pitch=self.left_leg_pitch,
                yaw=0,
                tx=-2,
                ty=-6,
            )
            cat(v, f, u)

        _make_head(du=0, dv=0, expand=0)
        _make_torso(du=16, dv=16, expand=0)
        _make_right_leg(du=0, dv=16, expand=0)
        _make_right_arm(du=40, dv=16, expand=0)
        if not self.mirror:
            _make_left_leg(du=16, dv=48, expand=0)
            _make_left_arm(du=32, dv=48, expand=0)
        else:
            _make_left_leg(du=0, dv=16, expand=0, mirror=True)
            _make_left_arm(du=40, dv=16, expand=0, mirror=True)

        if self.second_layer:
            _make_head(du=32, dv=0, expand=1)
        if self.second_layer and not self.half_skin:
            _make_torso(du=16, dv=32, expand=1)
            _make_left_leg(du=0, dv=48, expand=1)
            _make_right_arm(du=40, dv=32, expand=1)
            if not self.mirror:
                _make_right_leg(du=0, dv=32, expand=1)
                _make_left_arm(du=48, dv=48, expand=1)
            else:
                _make_right_leg(du=0, dv=48, expand=1, mirror=True)
                _make_left_arm(du=40, dv=32, expand=1, mirror=True)

        return vertices, faces, uvs

    def randomize_pose(self):
        self.left_arm_pitch = random.uniform(-20, 45)
        self.right_arm_pitch = random.uniform(-20, 45)
        self.left_arm_roll = random.uniform(0, 10)
        self.right_arm_roll = random.uniform(-10, 0)
        self.left_leg_pitch = random.uniform(-30, 30)
        self.right_leg_pitch = random.uniform(-30, 30)
        self.left_leg_roll = random.uniform(0, 10)
        self.right_leg_roll = random.uniform(-10, 0)
        self.head_pitch = random.uniform(-10, 10)
        self.head_roll = random.uniform(-5, -5)
        self.head_yaw = random.gauss(-10, 10)

    def get_character_pose(self):
        return {"left_arm_pitch": self.left_arm_pitch,
                "right_arm_pitch": self.right_arm_pitch,
                "left_arm_roll": self.left_arm_roll,
                "right_arm_roll" : self.right_arm_roll,
                "left_leg_pitch" : self.left_leg_pitch,
                "right_leg_pitch" : self.right_leg_pitch, 
                "left_leg_roll" : self.left_leg_roll, 
                "right_leg_roll" : self.right_leg_roll,
                "head_pitch" : self.head_pitch, 
                "head_roll" : self.head_roll,
                "head_yaw" : self.head_yaw}
    
    def set_character_pose(self, pose):
        self.left_arm_pitch = pose["left_arm_pitch"]
        self.right_arm_pitch = pose["right_arm_pitch"]
        self.left_arm_roll = pose["left_arm_roll"]
        self.right_arm_roll = pose["right_arm_roll"]
        self.left_leg_pitch = pose["left_leg_pitch"]
        self.right_leg_pitch = pose["right_leg_pitch"]
        self.left_leg_roll = pose["left_leg_roll"]
        self.right_leg_roll = pose["right_leg_roll"]
        self.head_pitch = pose["head_pitch"]
        self.head_roll = pose["head_roll"]
        self.head_yaw = pose["head_yaw"]


def character_for_skin(skin, device="cpu"):
    w, h = skin.size
    assert w in [32, 64, 128]
    assert h in [32, 64, 128]
    assert h == w or h == w//2
    half_skin = (h == w//2)
    skin_ = np.array(skin)
    slim = np.all(skin_[20:32, 54:56, 3] == 0)
    second_layer = np.any(skin_[32:48, :, 3] != 0) or np.any(skin_[0:16, 32:64, 3] != 0)
    return Character(
        slim=slim,
        mirror=half_skin,
        half_skin=half_skin,
        second_layer=second_layer,
        device=device,
    )
