import os
import gzip
import json
import logging
import numpy as np
from pathlib import Path
import cv2
from argconf import argconf_parse

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
                # 存储格式保持 "apple/110_13051_23361"
                all_seq_names.append(f"{cat}/{s}/{seq_name}")

    return sorted(all_seq_names)



# -------------------------------------------------------
# 2) 转换单个 CO3D 序列为 WAI scene
# -------------------------------------------------------
def convert_scene(cfg, scene_name):
    """
    scene_name 形如 "apple/110_13051_23361"
    """

    CO3D_ROOT = Path(cfg.original_root)  # Co3dDataset.CO3D_DIR
    OUT_ROOT = Path(cfg.root)
    CO3D_ANNOTATION_DIR = Path(cfg.annotation_root)
    category, s, seq = scene_name.split("/")

    # ---------------- 查找 annotation ----------------
    ann_file = CO3D_ANNOTATION_DIR / f"{category}_{scene_meta}.jgz"
    if not ann_file.exists():
        return "error", f"Annotation missing for category {category}"

    with gzip.open(ann_file, "r") as fin:
        ann = json.loads(fin.read())

    if seq not in ann:
        return "error", f"{seq} not found inside annotation {ann_file}"

    frames_meta = ann[seq]
    if len(frames_meta) < cfg.min_num_images:
        return "skip", "Too few images"

    logger.info(f"Processing: {scene_name}  ({len(frames_meta)} frames)")

    out_path = OUT_ROOT / scene_name.replace("/", "_")
    img_out_dir = out_path / "images_distorted"
    os.makedirs(img_out_dir, exist_ok=True)

    # ---------------- 遍历所有帧 ----------------
    wai_frames = []
    real_w = None
    real_h = None

    for f in frames_meta:
        # f["filepath"] 类似: apple/110_13051_23361/images/frame_00000.jpg
        img_path = CO3D_ROOT / f["filepath"]
        if not img_path.exists():
            return "error", f"Missing image: {img_path}"

        # 读取图像，只为了提取实际分辨率
        if real_w is None:
            img = cv2.imread(str(img_path))
            real_h, real_w = img.shape[:2]

        frame_name = Path(f["filepath"]).stem
        target_relative_path = f"images_distorted/{frame_name}.png"
        target_path = out_path / target_relative_path

        # 建立软连接（不复制）
        if not target_path.exists():
            os.symlink(img_path, target_path)

        # 记录 WAI frame
        wai_frame = {
            "frame_name": frame_name,
            "image_distorted": target_relative_path,
            "file_path": target_relative_path,
            "transform_matrix": np.array(f["extri"]).tolist(),  # 已经是 opencv 4x4
            "intri": f["intri"],  # 保留原始 intrinsics
        }
        wai_frames.append(wai_frame)

    # ---------------- 相机内参 ----------------
    # CO3D intri = [fx, fy, cx, cy]
    intri0 = frames_meta[0]["intri"]
    fx, fy, cx, cy = intri0

    # ---------------- 输出 scene_meta ----------------
    scene_meta = {
        "scene_name": scene_name.replace("/", "_"),
        "dataset_name": "co3d",
        "version": "0.1",
        "shared_intrinsics": True,
        "camera_model": "OPENCV",
        "camera_convention": "opencv",
        "scale_type": "none",

        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": real_w,
        "h": real_h,

        # CO3D 无畸变参数，全部置 0
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,

        "frames": wai_frames,
        "frame_modalities": {
            "image_distorted": {"frame_key": "image_distorted", "format": "image"},
        },
        "scene_modalities": {},
    }

    store_data(out_path / "scene_meta_distorted.json", scene_meta, "scene_meta")
    return None



# -------------------------------------------------------
# 3) 主函数：与 dl3dv 一样使用 wrapper
# -------------------------------------------------------
if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/co3d.yaml")

    convert_scenes_wrapper(
        convert_scene,
        cfg,
        get_original_scene_names_func=get_original_scene_names
    )
