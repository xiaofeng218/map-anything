"""
Download MapAnything model weights locally.

Usage:
    python download_mapanything_model.py
"""

import os

os.environ["TORCH_HUB_DISABLE_AUTO_DOWNLOAD"] = "1"

import torch

from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model

# 避免 CUDA 内存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==== 本地配置 ====
high_level_config = {
    "path": "configs/train.yaml",
    "hf_model_name": "facebook/map-anything",
    "local_files_only": True,
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "checkpoint_name": "model.safetensors",
    "config_name": "config.json",
    "trained_with_amp": True,
    "trained_with_amp_dtype": "bf16",
    "data_norm_type": "dinov2",
    "patch_size": 14,
    "resolution": 518,
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = initialize_mapanything_model(high_level_config, device)


if __name__ == "__main__":
    main()
