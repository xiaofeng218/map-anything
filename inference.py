import sys
import os

import argparse
import torch
import torchvision
import os.path as osp
import gzip
import json
import logging
from tqdm import tqdm

import cv2
import numpy as np
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/sam2"))

from mapanything.utils.viz import viser_wrapper
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.geometry import extri_to_homo, closed_form_pose_inverse
from mapanything.utils.wai.core import load_data, load_frame
from mapanything.utils.cropping import (
    rescale_image_and_other_optional_info,
    resize_with_nearest_interpolation_to_match_aspect_ratio,
)
from data_processing.viz_data import get_dataset_config


# inference of mapanything in co3d
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Path to the root directory",
        default="/data/hanxiaofeng/dataset/eth3d",
    )
    parser.add_argument(
        "--scene", type=str, help="Scene to visualize", default="courtyard"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "scannetpp",
            "blendedmvs",
            "eth3d",
            "megadepth",
            "spring",
            "mpsd",
            "ase",
            "tav2",
            "dl3dv",
            "unrealstereo4k",
            "mvs_synth",
            "paralleldomain4d",
            "sailvos3d",
            "dynamicreplica",
        ],
        default="eth3d",
        help="Dataset type to visualize",
    )
    parser.add_argument(
        "--depth_key", type=str, help="Key for depth data in the frame", default=None
    )
    parser.add_argument(
        "--load_skymask", action="store_true", help="Whether to load and apply sky mask"
    )
    parser.add_argument(
        "--local_frame",
        action="store_true",
        help="Whether to use local frame for visualization",
    )

    return parser

def get_data_from_wai(
    args,
    ids,
    depth_key="depth",
    load_skymask=False,
    confidence_key=None,
    confidence_thres=0,
):
    # Load the scene meta data
    scene_root = os.path.join(args.root_dir, args.scene)
    scene_meta = load_data(os.path.join(scene_root, "scene_meta.json"), "scene_meta")
    scene_frame_names = list(scene_meta["frame_names"].keys())
    scene_frame_names = [scene_frame_names[i] for i in ids]

    views = []
    for frame_idx, frame in enumerate(tqdm(scene_frame_names)):
        # Load the frame data
        if load_skymask:
            modalities = ["image", depth_key, "skymask"]
        else:
            modalities = ["image", depth_key]
        if confidence_key is not None:
            modalities.append(confidence_key)
        frame_data = load_frame(
            os.path.join(args.root_dir, args.scene),
            frame,
            modalities=modalities,
            scene_meta=scene_meta,
        )

        # Convert necessary data to numpy
        rgb_image = frame_data["image"].permute(1, 2, 0).numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)
        depth_data = frame_data[depth_key].numpy()
        intrinsics = frame_data["intrinsics"].numpy()

        # If depth is predicted, resize it to match the aspect ratio of the image
        # Then, resize the image and update intrinsics to match the resized predicted depth
        if "pred" in depth_key:
            # Get the dimensions of the original image
            img_h, img_w = rgb_image.shape[:2]

            # Resize depth to match image aspect ratio while ensuring that depth resolution doesn't increase
            depth_data, target_depth_h, target_depth_w = (
                resize_with_nearest_interpolation_to_match_aspect_ratio(
                    input_data=depth_data, img_h=img_h, img_w=img_w
                )
            )

            # Now resize the image and update intrinsics to match the resized depth
            rgb_image, _, intrinsics, _ = rescale_image_and_other_optional_info(
                image=rgb_image,
                output_resolution=(target_depth_w, target_depth_h),
                depthmap=None,
                camera_intrinsics=intrinsics,
            )
            rgb_image = np.array(rgb_image)

        # Mask depth if sky mask is loaded
        if load_skymask:
            mask_data = frame_data["skymask"].numpy().astype(int)
            mask_data = cv2.resize(
                mask_data,
                (depth_data.shape[1], depth_data.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            depth_data = np.where(mask_data, 0, depth_data)

        if confidence_key is not None:
            confidence_map = frame_data[confidence_key].numpy().astype(np.float32)
            confidence_mask = (confidence_map > confidence_thres).astype(int)
            confidence_mask = cv2.resize(
                confidence_mask,
                (depth_data.shape[1], depth_data.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            depth_data = np.where(confidence_mask, depth_data, 0)

        rgb_image = np.array(rgb_image)

        pose = frame_data["extrinsics"].numpy()

        view = {
            "img": torch.from_numpy(rgb_image), # (H, W, 3) - [0, 255]
            "intrinsics": torch.from_numpy(intrinsics), # (3, 3)
            "camera_poses": torch.from_numpy(pose), # (4, 4) in OpenCV cam2world convention
            "depth_z": torch.from_numpy(depth_data),
            "is_metric_scale": torch.tensor([True]), # (1,)
        }
        views.append(view)

    processed_views = preprocess_inputs(views)
    return processed_views

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Get dataset configuration
    config = get_dataset_config(args.dataset)

    # Override config with command line arguments if provided
    if args.root_dir != parser.get_default("root_dir"):
        config["root_dir"] = args.root_dir
    if args.scene != parser.get_default("scene"):
        config["scene"] = args.scene
    if args.depth_key is not None:
        config["depth_key"] = args.depth_key
    if args.load_skymask:
        config["load_skymask"] = True

    model = MapAnything.from_pretrained("facebook/map-anything", local_files_only=True).to('cuda')

    ids = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    processed_views = get_data_from_wai(
        args,
        ids,
        depth_key=config["depth_key"],
        load_skymask=config["load_skymask"],
        confidence_key=config["confidence_key"],
        confidence_thres=config["confidence_thres"],
    )

    exclude_infos = []#,'camera_poses' 'depth_z', 'intrinsics', 'is_metric_scale']

    input_views = []
    for view in processed_views:
        input_view = view.copy()
        for info in exclude_infos:
            input_view.pop(info)
        input_views.append(input_view)

    # Run inference with any combination of inputs
    predictions = model.infer(input_views)
    point_map = torch.stack([pred['pts3d'][0] for pred in predictions])
    depth = torch.stack([pred['depth_z'][0] for pred in predictions])
    mask = torch.stack([pred['mask'][0] for pred in predictions]).squeeze(-1)
    pose = torch.stack([pred['camera_poses'][0] for pred in predictions])
    conf = torch.stack([pred['conf'][0] for pred in predictions])
    images = torch.stack([pred['img_no_norm'][0] for pred in predictions])
    intr = torch.stack([pred['intrinsics'][0] for pred in predictions])


    input_depth = torch.stack([view['depth_z'][0] for view in processed_views]).to('cuda')
    input_mask = torch.stack([view['mask'][0] for view in predictions]).squeeze(-1)
    input_conf = torch.stack([view['conf'][0] for view in predictions])
    input_images = torch.stack([view['img_no_norm'][0] for view in predictions])
    input_intr = torch.stack([view['intrinsics'][0] for view in processed_views]).to('cuda')
    input_pose = torch.stack([view['camera_poses'][0] for view in processed_views]).to('cuda')
    first_pose_inv = closed_form_pose_inverse(input_pose[:1])  # [1, 4, 4]
    first_pose_inv = first_pose_inv.expand(input_pose.shape[0], -1, -1)  # [N, 4, 4]
    input_pose = first_pose_inv @ input_pose # c2c0


    preds = {
        "depth": torch.cat((depth.squeeze(-1), input_depth), dim=0),
        "images": torch.cat((images, input_images), dim=0),
        "depth_conf": torch.cat((conf, input_conf), dim=0),
        "extrinsic": torch.cat((pose, input_pose), dim=0), # c2w
        "intrinsic": torch.cat((intr, input_intr), dim=0),
        "mask": torch.cat((mask, input_mask), dim=0),
    }

    for key in preds.keys():
        if isinstance(preds[key], torch.Tensor):
            preds[key] = preds[key].cpu().numpy()  # remove batch dimension and convert to numpy

    viser_server = viser_wrapper(preds)#, use_point_map=True)
    print("Visualization complete")

