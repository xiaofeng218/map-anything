import os
import gzip
import json
import logging
import numpy as np
from pathlib import Path
import cv2
from argconf import argconf_parse
from PIL import Image
import torch

from mapanything.utils.wai.core import load_data, store_data
from mapanything.utils.wai.camera import gl2cv   # 不会用到，但保持接口一致
from mapanything.utils.wai.scene_frame import get_scene_names
from wai_processing.utils.wrapper import convert_scenes_wrapper
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH

logger = logging.getLogger(__name__)


SEEN_CATEGORIES = [
    "apple","backpack","banana","baseballbat","baseballglove","bench",
    "bicycle","bottle","bowl","broccoli","cake","car","carrot","cellphone",
    "chair","cup","donut","hairdryer","handbag","hydrant","keyboard",
    "laptop","microwave","motorcycle","mouse","orange","parkingmeter","pizza",
    "plant","stopsign","teddybear","toaster","toilet","toybus","toyplane",
    "toytrain","toytruck","tv","umbrella","vase","wineglass",
]


# -------------------------------------------------------
# 1) 获取 CO3D 所有序列名
# -------------------------------------------------------
def get_original_scene_names(cfg):
    """
    CO3D 的组织方式是：
       category/seq_name/images/*.jpg

    annotation 存储于：
       {CO3D_ANNOTATION_DIR}/{category}_{train/test}.jgz
    """
    CO3D_ROOT = cfg.original_root  # Co3dDataset.CO3D_DIR
    CO3D_ANNOTATION_DIR = cfg.annotation_root

    split = ["train", "test"]

    all_seq_names = []

    for s in split:
        for cat in sorted(SEEN_CATEGORIES):
            ann_fn = Path(CO3D_ANNOTATION_DIR, f"{cat}_{s}.jgz")
            if not ann_fn.exists():
                continue

            with gzip.open(ann_fn, "r") as fin:
                ann = json.loads(fin.read())

            # ann 是 dict: seq_name -> list(frame_metadata)
            # 我们希望输出唯一 seq_name 列表
            for seq_name in ann.keys():
                # 存储格式保持 "apple_110_13051_23361"
                co3d_fn = Path(CO3D_ROOT, f"{cat}/{seq_name}")
                if not co3d_fn.exists():
                    continue
                all_seq_names.append(f"{cat}_{seq_name}")

    return sorted(all_seq_names)

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
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

def load_co3d_depth(path, scale_adjustment=1.0):
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

# -------------------------------------------------------
# 2) 转换单个 CO3D 序列为 WAI scene
# -------------------------------------------------------
def process_co3d_scene(cfg, scene_name):
    """
    scene_name 形如 "apple/110_13051_23361"
    Expected root directory structure for the raw co3d dataset:
    .
    └── co3d
        ├──apple
        │   ├── 110_13051_23361
        │   │   ├── depth_masks
        │   │   ├── depths
        │   │   ├── images
        │   │   └── masks
        │   ├── 189_20393_38136
        │   │   ├── depth_masks
        │   │   ├── depths
        │   │   ├── images
        │   │   └── masks
        │   ├── 540_79043_153212
        │   │   ├── depth_masks
        │   │   ├── depths
        │   │   ├── images
        │   │   └── masks
        │   ├── eval_batches
        │   └── set_lists
        ├──backpack
        │ ...
    """
    CO3D_ROOT = Path(cfg.original_root)  # Co3dDataset.CO3D_DIR
    OUT_ROOT = Path(cfg.root)
    CO3D_ANNOTATION_DIR = Path(cfg.annotation_root)
    category, seq = scene_name.split("_", 1)

    # ---------------- 查找 annotation ----------------
    ann = None
    for s in ["train", "test"]:
        ann_file = CO3D_ANNOTATION_DIR / f"{category}_{s}.jgz"

        with gzip.open(ann_file, "r") as fin:
            anno = json.loads(fin.read())

        if seq in anno:
            ann = anno
            break

    if ann is None:
        return "skipped", f"{seq} not found inside annotation {ann_file}"

    if len(ann[seq]) < cfg.min_num_images:
        return "skipped", "Too few images"
    step = max(1, min(len(ann[seq]) // 40, 6))
    frames_meta = ann[seq][::step]

    logger.info(f"Processing: {scene_name}  ({len(frames_meta)} frames)")

    out_path = OUT_ROOT / scene_name.replace("/", "_")
    img_out_dir = out_path / "images"
    os.makedirs(img_out_dir, exist_ok=True)

    # ---------------- 遍历所有帧 ----------------
    wai_frames = []

    for f in frames_meta:
        # f["filepath"] 类似: apple/110_13051_23361/images/frame_00000.jpg
        img_path = CO3D_ROOT / f["filepath"]
        if not img_path.exists():
            return "error", f"Missing image: {img_path}"

        frame_name = Path(f["filepath"]).stem
        target_relative_path = f"images/{frame_name}.png"
        target_path = out_path / target_relative_path

        # 建立软连接（不复制）
        if not target_path.exists():
            os.symlink(img_path, target_path)

        depth_path = str(img_path).replace("/images", "/depths") + ".geometric.png"
        depth_map = load_co3d_depth(depth_path, 1.0)
        H, W = depth_map.shape
        rel_depth_out_path = Path("depth") / (frame_name + ".exr")
        store_data(
            out_path / rel_depth_out_path,
            torch.tensor(depth_map),
            "depth",
        )

        world2cam_pose = np.eye(4)
        world2cam_pose[:3,:4] = np.array(f["extri"])
        cam2world_pose = np.linalg.inv(world2cam_pose)

        K = f["intri"]

        # 记录 WAI frame
        wai_frame = {
            "frame_name": frame_name,
            "image": target_relative_path,
            "file_path": target_relative_path,
            "depth": str(rel_depth_out_path),
            "transform_matrix": cam2world_pose.tolist(),
            "h": H,
            "w": W,
            "fl_x": float(K[0][0]),
            "fl_y": float(K[1][1]),
            "cx": float(K[0][2]),
            "cy": float(K[1][2]),
        }
        wai_frames.append(wai_frame)

    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "none", # co3d 无 metric-scale
        "frames": wai_frames,
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {
                "frame_key": "depth",
                "format": "depth",
            },
        },
        "scene_modalities": {},
    }

    store_data(out_path / "scene_meta.json", scene_meta, "scene_meta")
    return None

if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/co3d.yaml")

    convert_scenes_wrapper(
        process_co3d_scene,
        cfg,
        get_original_scene_names_func=get_original_scene_names
    )
