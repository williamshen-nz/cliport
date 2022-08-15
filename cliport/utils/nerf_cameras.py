from typing import Dict, List, Tuple

import numpy as np
from pybullet_utils.transformations import quaternion_from_matrix

Quaternion = Tuple[float, float, float, float]  # xyzw quaternion


def normalize(vec: np.ndarray) -> np.ndarray:
    """Thanks Yen-Chen"""
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def look_at(camera_pos: np.ndarray, target_pos: np.ndarray) -> Quaternion:
    """
    Construct the rotation of a camera looking at a target.
    Assumes that the camera is pointing away from the z-axis.

    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    """
    assert len(camera_pos) == len(target_pos) == 3
    # Just hardcode tmp vector to fit this repo's rendering of camera
    tmp = np.float32([0, 0, -1])

    forward = normalize(camera_pos - target_pos)
    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    rotm = np.eye(4)
    rotm[:3, :3] = np.stack((right, up, forward), axis=-1)

    quat = quaternion_from_matrix(rotm)
    return quat


class NerfCameraParams:
    # Radius of the camera circle around origin in meters.
    radius: float = 0.7
    # Origin of workspace (x, y, z)
    workspace_origin: tuple = (0.5, 0, 0)
    # z-height of first image
    start_z: float = 1.0
    # z-height of the last image
    end_z: float = 0.25
    # Number of images/viewpoints
    num_images: int = 36
    # Number of circles around origin to take images from.
    num_circles: int = 2

    # Array of angles for the images
    angles = np.linspace(0, num_circles * 2 * np.pi, num_images)
    # Array of z heights for the images
    zs = np.linspace(start_z, end_z, num_images)

    @classmethod
    def get_camera_config(cls, image_size: Tuple, intrinsics: Tuple) -> List[Dict]:
        """Return the list of camera configurations as expected by PyBullet"""
        assert len(image_size) == 2 and len(intrinsics) == 9

        configs = []
        common_config = {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "target_position": cls.workspace_origin,
            "zrange": (0.01, 10.0),
            "noise": False,
        }

        for angle, z in zip(cls.angles, cls.zs):
            x = cls.radius * np.cos(angle) + cls.workspace_origin[0]
            y = cls.radius * np.sin(angle) + cls.workspace_origin[1]
            position = (x, y, z)
            rotation = look_at(
                camera_pos=np.array(cls.workspace_origin),
                target_pos=np.array(position),
            )

            configs.append(
                {
                    **common_config,
                    "position": position,
                    "rotation": rotation,
                }
            )
        return configs
