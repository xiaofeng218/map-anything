import argparse
import gzip
import json
import logging
import os
import os.path as osp
import sys

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/sam2")
)

from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import viser_wrapper


# inference of mapanything in co3d
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="apple")
    parser.add_argument("--seq_name", type=str, default="39_1752_5101")
    parser.add_argument("--data_dir", type=str, default="test_dataset/co3d_full")
    parser.add_argument("--data_anno_dir", type=str, default="test_dataset/co3d_anno")

    return parser.parse_args()


def save_video_frames_as_png(tensor: torch.Tensor, output_dir: str):
    """
    Save a [T, H, W, 3] float tensor (values in [0, 1]) as PNG images.

    Args:
        tensor (torch.Tensor): The video tensor with shape [T, H, W, 3].
        output_dir (str): Directory to save PNG frames.
    """
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError("Input tensor must have shape [T, H, W, 3].")

    os.makedirs(output_dir, exist_ok=True)

    # 转换为 [T, 3, H, W] 以符合torchvision格式
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)

    # 保存每一帧为 PNG 文件
    for t, frame in enumerate(tensor):
        filename = os.path.join(output_dir, f"frame_{t:04d}.png")
        torchvision.utils.save_image(frame, filename)

    print(f"✅ Saved {tensor.shape[0]} frames to '{output_dir}'.")


# co3d dataset
def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    """
    Reads a depth map from disk in either .exr or .png format. The .exr is loaded using OpenCV
    with the environment variable OPENCV_IO_ENABLE_OPENEXR=1. The .png is assumed to be a 16-bit
    PNG (converted from half float).

    Args:
        path (str):
            File path to the depth image. Must end with .exr or .png.
        scale_adjustment (float):
            A multiplier for adjusting the loaded depth values (default=1.0).

    Returns:
        np.ndarray:
            A float32 array (H, W) containing the loaded depth. Zeros or non-finite values
            may indicate invalid regions.

    Raises:
        ValueError:
            If the file extension is not supported.
    """
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16big_png_depth(path)
    else:
        raise ValueError(f'unsupported depth file name "{path}"')

    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0

    return d


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float32 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.
      - We reinterpret the bits as float16, then cast to float32.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float32 depth array of shape (H, W).
    """
    with PIL.Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
            Returns None if depth_map is None.
    """
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map


def get_data_from_co3d(co3d_dir, co3d_anno_dir, category, seq_name, ids):
    splits = ["train", "test"]
    metadata = None
    for split in splits:
        annotation_file = osp.join(co3d_anno_dir, f"{category}_{split}.jgz")
        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_file}")
            continue

        if seq_name in annotation.keys():
            metadata = annotation[seq_name]
            break

    assert metadata is not None

    annos = [metadata[i] for i in ids]

    views = []

    for anno in annos:
        filepath = anno["filepath"]

        image_path = osp.join(co3d_dir, filepath)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
        depth_map = read_depth(depth_path, 1.0)

        depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)

        extri_opencv, intri_opencv = anno["extri"], anno["intri"]

        image = torch.from_numpy(image)
        intri_opencv = torch.tensor(intri_opencv)
        extri_opencv = closed_form_pose_inverse(torch.tensor(extri_opencv)[None])[
            0
        ]  # c2w
        depth_map = torch.from_numpy(depth_map).float()

        view = {
            "img": image,  # (H, W, 3) - [0, 255]
            "intrinsics": intri_opencv,  # (3, 3)
            "camera_poses": extri_opencv,  # (4, 4) in OpenCV cam2world convention
            "depth_z": depth_map,  # (H, W)
            "is_metric_scale": torch.tensor([False], device=image.device),  # (1,)
        }
        views.append(view)

    processed_views = preprocess_inputs(views)
    return processed_views


if __name__ == "__main__":
    args = parse_args()

    model = MapAnything.from_pretrained(
        "facebook/map-anything", local_files_only=True
    ).to("cuda")

    ids = [1, 31, 42, 51, 71, 85, 93, 15]
    processed_views = get_data_from_co3d(
        args.data_dir, args.data_anno_dir, args.category, args.seq_name, ids
    )

    exclude_infos = []  # ['depth_z', 'camera_poses', 'intrinsics', 'is_metric_scale']

    input_views = []
    for view in processed_views:
        input_view = view.copy()
        for info in exclude_infos:
            input_view.pop(info)
        input_views.append(input_view)

    # Run inference with any combination of inputs
    predictions = model.infer(input_views)
    point_map = torch.stack([pred["pts3d"][0] for pred in predictions])
    depth = torch.stack([pred["depth_z"][0] for pred in predictions])
    mask = torch.stack([pred["mask"][0] for pred in predictions]).squeeze(-1)
    pose = torch.stack([pred["camera_poses"][0] for pred in predictions])
    conf = torch.stack([pred["conf"][0] for pred in predictions])
    images = torch.stack([pred["img_no_norm"][0] for pred in predictions])
    intr = torch.stack([pred["intrinsics"][0] for pred in predictions])

    input_depth = torch.stack([view["depth_z"][0] for view in processed_views]).to(
        "cuda"
    )
    input_mask = torch.stack([view["mask"][0] for view in predictions]).squeeze(-1)
    input_conf = torch.stack([view["conf"][0] for view in predictions])
    input_images = torch.stack([view["img_no_norm"][0] for view in predictions])
    input_intr = torch.stack([view["intrinsics"][0] for view in processed_views]).to(
        "cuda"
    )
    input_pose = torch.stack([view["camera_poses"][0] for view in processed_views]).to(
        "cuda"
    )
    first_pose_inv = closed_form_pose_inverse(input_pose[:1])  # [1, 4, 4]
    first_pose_inv = first_pose_inv.expand(input_pose.shape[0], -1, -1)  # [N, 4, 4]
    input_pose = first_pose_inv @ input_pose  # c2c0

    preds = {
        "depth": torch.cat((depth.squeeze(-1), input_depth), dim=0),
        "images": torch.cat((images, input_images), dim=0),
        "depth_conf": torch.cat((conf, input_conf), dim=0),
        "extrinsic": torch.cat((pose, input_pose), dim=0),  # c2w
        "intrinsic": torch.cat((intr, input_intr), dim=0),
        "mask": torch.cat((mask, input_mask), dim=0),
    }

    for key in preds.keys():
        if isinstance(preds[key], torch.Tensor):
            preds[key] = (
                preds[key].cpu().numpy()
            )  # remove batch dimension and convert to numpy

    viser_server = viser_wrapper(preds)  # , use_point_map=True)
    print("Visualization complete")
