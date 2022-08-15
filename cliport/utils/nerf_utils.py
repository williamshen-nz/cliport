""" Modified from Yen-Chen's mira/demo_utils.py. """
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pybullet as p
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image

from cliport import Environment, RavensDataset


def _plot_images(color, depth, segm, label: str) -> None:
    """Plot color, depth and segmentation for debugging purposes."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(color)
    axs[0].set_title("Color")
    axs[1].imshow(depth)
    axs[1].set_title("Depth")
    axs[2].imshow(segm)
    axs[2].set_title("Segmentation")
    fig.suptitle(f"Camera {label}")
    plt.show()


def convert_pose(C2W: np.ndarray) -> np.ndarray:
    """Convert pose from camera frame to world frame. I don't think this comment is right"""
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def capture_nerf_cams(
    env: Environment,
    dataset: RavensDataset,
    seed: int,
    step: int,
    should_plot: bool = True,
) -> List[Dict]:
    """Capture image from each NeRF camera in the environment and compute transformation matrix."""
    assert env.task, "Task not set in Environment!"
    start_time = time.perf_counter()
    metadata = []

    # Directory for saving NeRF camera images
    nerf_image_path = os.path.join(dataset.nerf_dataset_path(seed, step), "images")
    if os.path.exists(nerf_image_path):
        raise RuntimeError(
            f"{nerf_image_path} already exists! Check you are handling episode steps uniquely."
        )
    os.makedirs(nerf_image_path)

    # Render images, compute transformation matrix and save to disk
    for idx, config in enumerate(env.nerf_cams):
        color, depth, segm = env.render_camera(config)
        if should_plot:
            # Warning: this is slow if there are many cameras
            _plot_images(color, depth, segm, label=str(idx))

        # Compute transformation matrix for COLMAP
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)

        # Camera pose in world frame????
        # TODO: figure out what's happening here
        c2w = np.eye(4)
        c2w[:3, :] = np.hstack((rotation, position))
        c2w = convert_pose(c2w)

        # Save image to disk
        image_name = f"{idx:06d}"
        image_path = os.path.join(nerf_image_path, f"{idx:06}.png")
        Image.fromarray(color).save(image_path, quality=100, subsampling=0)
        logger.debug(f"Saved image to {image_path}")

        metadata.append(
            {
                "file_path": f"./images/{image_name}",
                "transform_matrix": c2w.tolist(),
            }
        )

    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info(
        f"Time to capture {len(env.nerf_cams)} images for "
        f"seed={seed}, step={step}: {duration:.2f}s"
    )
    assert metadata, "No metadata found!"
    return metadata


def write_transforms_json(
    env: Environment,
    dataset: RavensDataset,
    seed: int,
    step: int,
    should_plot: bool,
    aabb_scale: int = 4,
) -> str:
    """
    Write transforms JSON in instant-ngp style (mini-ngp to be precise).

    Returns string with path to JSON file.
    """

    def get_field_from_cam_configs(field: str) -> Any:
        """
        Get a given field from the camera configs.
        Validates that the field is the same for all cameras.
        """
        all_fields = [config[field] for config in env.nerf_cams]
        assert all(field == all_fields[0] for field in all_fields)
        return all_fields[0]

    intrinsics = get_field_from_cam_configs("intrinsics")
    intrinsics = np.array(intrinsics).reshape(3, 3)
    image_size = get_field_from_cam_configs("image_size")

    # Form transforms dict
    transforms = dict(
        fl_x=intrinsics[0][0],
        fl_y=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2],
        w=image_size[1],
        h=image_size[0],
        aabb_scale=aabb_scale,
        scale=1.0,
        frames=capture_nerf_cams(env, dataset, seed, step, should_plot),
    )

    # Compute camera angles
    transforms["camera_angle_x"] = 2 * np.arctan(
        transforms["w"] / (2 * transforms["fl_x"])
    )
    transforms["camera_angle_y"] = 2 * np.arctan(
        transforms["h"] / (2 * transforms["fl_y"])
    )

    # Write to disk
    transforms_path = os.path.join(
        dataset.nerf_dataset_path(seed, step), "transforms.json"
    )
    with open(transforms_path, "w") as fp:
        json.dump(transforms, fp, indent=2)
        logger.info(f"Wrote transforms.json to {transforms_path}")

    return transforms_path


def write_nerf_data(**kwargs):
    return write_transforms_json(**kwargs)


# def write_nerf_data(path, env, act, t=0.3):
#     task = env.task
#
#     rolls = task.get_rolls()
#     pitchs = task.get_pitchs()
#     yaw = 0
#
#     pick_pos_idx = rolls.index(0) * len(pitchs) + pitchs.index(0)
#     _, p1_xyzw = act["pose1"]
#     eulerXYZ = utils.quatXYZW_to_eulerXYZ(p1_xyzw)
#     place_roll_idx = np.abs(-eulerXYZ[0] - np.array(rolls)).argmin()
#     place_pitch_idx = np.abs(-eulerXYZ[1] - np.array(pitchs)).argmin()
#     place_pos_idx = place_roll_idx * len(pitchs) + place_pitch_idx
#
#     # Create nerf-dataset dir
#     os.makedirs(path, exist_ok=True)
#     image_dir = os.path.join(path, "images")
#     os.makedirs(image_dir, exist_ok=True)
#     test_dir = os.path.join(path, "test")
#     os.makedirs(test_dir, exist_ok=True)
#
#     # Define the location for cameras to look at.
#     # Note(willshen): this differs from (0.25, 0, 0) in cameras.py
#     look_at = np.array([0.5, 0, 0])
#
#     # I think this is just generating multiple possible viewpoints for
#     # NeRF to reconstruct at test time. This is presumably orthographic.
#
#     # Write test cameras.
#     metadata = []
#     for idx_roll in range(len(rolls)):
#         for idx_pitch in range(len(pitchs)):
#             idx = idx_roll * len(pitchs) + idx_pitch
#
#             # Rotation.
#             c2w = np.eye(4)
#             c2w[2, 2] = -1
#             c2w[:3, :3] = c2w[:3, :3] @ utils.eulerXYZ_to_rotm(
#                 (rolls[idx_roll], pitchs[idx_pitch], yaw)
#             )
#
#             # Translation.
#             normal = c2w[:3, :3] @ np.array([0, 0, 1])
#             ray = -1 * normal
#             t = 0.3
#             cam_center = look_at + ray * t
#             c2w[:3, -1] = cam_center
#             c2w = utils.convert_pose(c2w)
#
#             # Convert the camera pose into OpenGL's format.
#             metadata.append(
#                 {
#                     "file_path": f"./images/{idx:06}.png",
#                     "transform_matrix": c2w.tolist(),
#                 }
#             )
#
#     transforms = {}
#     transforms["fl_x"] = 320.0
#     transforms["fl_y"] = 320.0
#     transforms["cx"] = 80.0
#     transforms["cy"] = 160.0
#     transforms["w"] = 160
#     transforms["h"] = 320
#     transforms["aabb_scale"] = 4
#     transforms["scale"] = 1.0
#     transforms["camera_angle_x"] = 2 * np.arctan(
#         transforms["w"] / (2 * transforms["fl_x"])
#     )
#     transforms["camera_angle_y"] = 2 * np.arctan(
#         transforms["h"] / (2 * transforms["fl_y"])
#     )
#     transforms["pick_pos_idx"] = int(pick_pos_idx)
#     transforms["place_pos_idx"] = int(place_pos_idx)
#     transforms["n_views"] = len(rolls) * len(pitchs)
#     transforms["frames"] = metadata
#
#     os.makedirs(path, exist_ok=True)
#     with open(os.path.join(test_dir, "transforms_test.json"), "w") as fp:
#         json.dump(transforms, fp, indent=2)
#
#     # Write train cameras.
#     metadata = []
#     i = 0
#     for config in env.nerf_cams:
#         color, depth, _ = env.render_camera(config)
#         intrinsics = np.array(config["intrinsics"]).reshape(3, 3)
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = p.getMatrixFromQuaternion(config["rotation"])
#         rotation = np.array(rotation).reshape(3, 3)
#         c2w = np.eye(4)
#         c2w[:3, :] = np.hstack((rotation, position))
#
#         def convert_pose(C2W):
#             flip_yz = np.eye(4)
#             flip_yz[1, 1] = -1
#             flip_yz[2, 2] = -1
#             C2W = np.matmul(C2W, flip_yz)
#             return C2W
#
#         c2w = convert_pose(c2w)
#
#         Image.fromarray(color).save(
#             os.path.join(image_dir, f"{i:06}.png"), quality=100, subsampling=0
#         )
#         # plt.imsave(os.path.join(color_dir, f"{i:06}.png"), obs['color'][i])
#         # np.save(os.path.join(depth_dir, f'{i:06}.npy'), depth)
#         metadata.append(
#             {
#                 "file_path": f"./images/{i:06}.png",
#                 "transform_matrix": c2w.tolist(),
#             }
#         )
#
#         i += 1
#
#     transforms = {}
#     transforms["fl_x"] = intrinsics[0][0]
#     transforms["fl_y"] = intrinsics[1][1]
#     transforms["cx"] = intrinsics[0][2]
#     transforms["cy"] = intrinsics[1][2]
#     transforms["w"] = config["image_size"][1]
#     transforms["h"] = config["image_size"][0]
#     transforms["aabb_scale"] = 4
#     transforms["scale"] = 1.0
#     transforms["camera_angle_x"] = 2 * np.arctan(
#         transforms["w"] / (2 * transforms["fl_x"])
#     )
#     transforms["camera_angle_y"] = 2 * np.arctan(
#         transforms["h"] / (2 * transforms["fl_y"])
#     )
#     transforms["frames"] = metadata
#
#     with open(os.path.join(path, "transforms.json"), "w") as fp:
#         json.dump(transforms, fp, indent=2)
