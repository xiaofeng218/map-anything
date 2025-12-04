# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

from mapanything.utils.wai.camera import CAMERA_KEYS, gl2cv
from mapanything.utils.wai.core import load_data, store_data
from mapanything.utils.wai.scene_frame import get_scene_names

logger = logging.getLogger(__name__)


def get_original_scene_names(cfg):
    all_scene_names = []
    # hard-coding as DL3DV has exactly 11 splits from 1K to 11K
    for split_idx in range(1, 2):
        data_split = f"{split_idx}K"
        split_root = Path(cfg.original_root, data_split)
        scene_names_one_split = sorted(
            [
                f"{data_split}_{orig_scene_id}"
                for orig_scene_id in os.listdir(split_root)
                if os.path.isdir(os.path.join(split_root, orig_scene_id))
            ]
        )
        all_scene_names.extend(scene_names_one_split)
    scene_names = get_scene_names(cfg, scene_names=all_scene_names)
    return scene_names


def convert_scene(cfg, scene_name) -> None | tuple[str, str]:
    import cv2
    logger.info(f"Processing: {scene_name}")

    # scene_name is f"{split_name}_{scene_id}"
    dataset_name = cfg.get("dataset_name", "dl3dv")
    version = cfg.get("version", "0.1")
    source_scene_root = Path(cfg.original_root, scene_name.replace("_", "/"))

    # Sanity check
    if any([
        not Path(source_scene_root, "transforms.json").exists(),
        not Path(source_scene_root, "colmap").exists(),
        not Path(source_scene_root, "images").exists(),
    ]):
        raise RuntimeError(f"Expected required files missing in {source_scene_root}")

    transforms_fn = Path(source_scene_root, "transforms.json")
    out_path = Path(cfg.root) / scene_name
    meta = load_data(transforms_fn)
    frames = meta["frames"]

    # skip portrait images for now
    if meta["h"] > meta["w"]:
        return "data_issue", "Images are in portrait, not supported for now."

    # ✅ Detect actual image resolution from first frame image
    test_img = cv2.imread(str(Path(source_scene_root, frames[0]["file_path"])))
    real_h, real_w = test_img.shape[:2]

    # ✅ Extract original intrinsics and scale
    orig_fx = meta["fl_x"]
    orig_fy = meta["fl_y"]
    orig_cx = meta["cx"]
    orig_cy = meta["cy"]
    orig_w = meta["w"]
    orig_h = meta["h"]

    scale_x = real_w / orig_w
    scale_y = real_h / orig_h

    new_fx = orig_fx * scale_x
    new_fy = orig_fy * scale_y
    new_cx = orig_cx * scale_x
    new_cy = orig_cy * scale_y

    # ✅ Create target folder
    image_out_path = out_path / "images_distorted"
    os.makedirs(image_out_path, exist_ok=True)

    wai_frames = []
    for frame in frames:
        frame_name = Path(frame["file_path"]).stem
        wai_frame = {"frame_name": frame_name}

        org_transform_matrix = np.array(frame["transform_matrix"]).astype(np.float32)
        opencv_pose, gl2cv_cmat = gl2cv(org_transform_matrix, return_cmat=True)

        source_image_path = Path(source_scene_root, frame["file_path"])
        target_image_path = f"images_distorted/{frame_name}.png"

        # ✅ syslink distorted image files
        if not (out_path / target_image_path).exists():
            os.symlink(source_image_path, out_path / target_image_path)

        wai_frame["image_distorted"] = target_image_path
        wai_frame["file_path"] = target_image_path
        wai_frame["transform_matrix"] = opencv_pose.tolist()

        if "colmap_im_id" in frame:
            wai_frame["colmap_im_id"] = frame["colmap_im_id"]

        wai_frames.append(wai_frame)

    # ✅ Link colmap data
    if not (out_path / "colmap").exists():
        os.symlink(Path(source_scene_root, "colmap"), out_path / "colmap")

    # ✅ Build final scene_meta with updated intrinsics and size
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "version": version,
        "shared_intrinsics": True,
        "camera_model": meta["camera_model"],
        "camera_convention": "opencv",
        "scale_type": "colmap",

        # ✅ Updated intrinsics
        "fl_x": new_fx,
        "fl_y": new_fy,
        "cx": new_cx,
        "cy": new_cy,
        "w": real_w,
        "h": real_h,

        # ✅ Distortion params MUST remain tied to intrinsics
        "k1": meta.get("k1", 0.0),
        "k2": meta.get("k2", 0.0),
        "p1": meta.get("p1", 0.0),
        "p2": meta.get("p2", 0.0),
        "k3": meta.get("k3", 0.0),
    }

    scene_meta["frames"] = wai_frames
    scene_meta["frame_modalities"] = {
        "image_distorted": {"frame_key": "image_distorted", "format": "image"},
    }
    scene_meta["scene_modalities"] = {
        "colmap": {"scene_key": "colmap"}
    }

    # ✅ store transforms for pose correction
    applied_transform = np.array(meta["applied_transform"]).reshape(3, 4)
    applied_transform = np.vstack([applied_transform, [0, 0, 0, 1.0]])
    scene_meta["_applied_transform"] = applied_transform.tolist()
    scene_meta["_applied_transforms"] = {
        "opengl2opencv": gl2cv_cmat.tolist()
    }

    # ✅ Save final metadata
    store_data(out_path / "scene_meta_distorted.json", scene_meta, "scene_meta")

if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/dl3dv.yaml")
    convert_scenes_wrapper(
        convert_scene,
        cfg,
        # Need to use the dl3dv func to get original scenes names
        # as the subsets with 1K, 2K, 3K etc are DL3DV specific
        get_original_scene_names_func=get_original_scene_names,
    )
