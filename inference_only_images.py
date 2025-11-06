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
from mapanything.utils.image import load_images
from mapanything.utils.geometry import closed_form_pose_inverse


# inference of mapanything in co3d
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default='examples/protein',
        help="Path to folder containing images for reconstruction",
    )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    input_views = load_images(args.image_folder)
    model = MapAnything.from_pretrained("facebook/map-anything", local_files_only=True).to('cuda')

    # Run inference with any combination of inputs
    predictions = model.infer(input_views)
    point_map = torch.stack([pred['pts3d'][0] for pred in predictions])
    depth = torch.stack([pred['depth_z'][0] for pred in predictions])
    pose = torch.stack([pred['camera_poses'][0] for pred in predictions])
    conf = torch.stack([pred['conf'][0] for pred in predictions])
    images = torch.stack([pred['img_no_norm'][0] for pred in predictions])
    mask = torch.stack([pred['mask'][0] for pred in predictions]).squeeze(-1)
    mask &= images.sum(dim=-1) > 1e-3
    intr = torch.stack([pred['intrinsics'][0] for pred in predictions])


    preds = {
        "depth": depth.squeeze(-1),
        "images": images,
        "depth_conf": conf,
        "extrinsic": pose, # c2w
        "intrinsic": intr,
        "mask": mask,
    }

    for key in preds.keys():
        if isinstance(preds[key], torch.Tensor):
            preds[key] = preds[key].cpu().numpy()  # remove batch dimension and convert to numpy

    viser_server = viser_wrapper(preds)#, use_point_map=True)
    print("Visualization complete")

