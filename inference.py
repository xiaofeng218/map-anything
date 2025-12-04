import argparse
import os
import sys
import logging
log = logging.getLogger(__name__)

import torch
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from mapanything.train.training import train
from mapanything.utils.misc import StreamToLogger
from mapanything.train.losses import *  # noqa
from mapanything.utils.inference import loss_of_one_batch_multi_view

import rerun as rr
from tqdm import tqdm

from mapanything.datasets.base.base_dataset import view_name
from mapanything.utils.image import rgb

from mapanything.models import init_model

from mapanything.datasets.wai.dl3dv import DL3DVWAI
from mapanything.train.training import get_train_data_loader

def log_views(views, num):
    for view_idx in range(len(views)):
        image = rgb(
            views[view_idx]["img"], norm_type=views[view_idx]["data_norm_type"]
        )
        depthmap = views[view_idx]["depthmap"]
        pose = views[view_idx]["camera_pose"]
        intrinsics = views[view_idx]["camera_intrinsics"]
        pts3d = views[view_idx]["pts3d"]
        valid_mask = views[view_idx]["valid_mask"]
        if "non_ambiguous_mask" in views[view_idx]:
            non_ambiguous_mask = views[view_idx]["non_ambiguous_mask"]
        else:
            non_ambiguous_mask = None
        if "prior_depth_along_ray" in views[view_idx]:
            prior_depth_along_ray = views[view_idx]["prior_depth_along_ray"]
        else:
            prior_depth_along_ray = None

        rr.set_time_sequence("stable_time", sequence=num)
        base_name = f"world/view_{view_idx}"
        pts_name = f"world/view_{view_idx}_pointcloud"
        # Log camera info and loaded data
        height, width = image.shape[0], image.shape[1]
        rr.log(
            base_name,
            rr.Transform3D(
                translation=pose[:3, 3],
                mat3x3=pose[:3, :3],
            ),
        )
        rr.log(
            f"{base_name}/pinhole",
            rr.Pinhole(
                image_from_camera=intrinsics,
                height=height,
                width=width,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rr.log(
            f"{base_name}/pinhole/rgb",
            rr.Image(image),
        )
        rr.log(
            f"{base_name}/pinhole/depth",
            rr.DepthImage(depthmap),
        )
        if prior_depth_along_ray is not None:
            rr.log(
                f"prior_depth_along_ray_{view_idx}",
                rr.DepthImage(prior_depth_along_ray),
            )
        if non_ambiguous_mask is not None:
            rr.log(
                f"{base_name}/pinhole/non_ambiguous_mask",
                rr.SegmentationImage(non_ambiguous_mask.astype(int)),
            )
        # Log points in 3D
        filtered_pts = pts3d[valid_mask]
        filtered_pts_col = image[valid_mask]
        rr.log(
            pts_name,
            rr.Points3D(
                positions=filtered_pts.reshape(-1, 3),
                colors=filtered_pts_col.reshape(-1, 3),
            ),
        )

def collate_fn(batch):
    collateds = []
    for view_idx in range(len(batch[0])):
        collated = {}
        for key in batch[0][view_idx].keys():
            values = [d[view_idx][key] for d in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], np.ndarray):
                # 将 numpy array 转成 tensor 再堆叠
                collated[key] = torch.stack([torch.tensor(v) for v in values], dim=0)
            else:
                collated[key] = values  # 保持 list
                print(key)
        collateds.append(collated)
    return collateds

def inference(cfg):
    train_data_loader = get_train_data_loader(
        dataset=cfg.dataset.train_dataset,
        num_workers=cfg.dataset.num_workers,
        max_num_of_imgs_per_gpu=4,
    )

    if cfg.visual.viz:
        rr.script_setup(cfg.visual, "DL3DV_Dataloader")
        rr.set_time_sequence("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # sampled_indices = [0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # TODO:目前还没有加载模型的权重，可以考虑加上去
    model = init_model(cfg.model.model_str, cfg.model.model_config)
    model.to(device)

    train_criterion = eval(cfg.loss.train_criterion).to(device)

    if hasattr(train_data_loader.dataset, "set_epoch"): # 在遍历dataloader前要手动设置一下epoch
        train_data_loader.dataset.set_epoch(0)

    for data_iter_step, batch in enumerate(train_data_loader):
        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            train_criterion,
            device,
            use_amp=bool(cfg.train_params.amp),
            amp_dtype=cfg.train_params.amp_dtype,
            ret="loss",
        )

        loss, loss_details = loss_tuple  # criterion returns two values

    # for num, idx in enumerate(tqdm(sampled_indices)):
    #     views = dataset[idx]
    #     assert len(views) == cfg.data.num_of_views
    #     sample_name = f"{idx}"
    #     for view_idx in range(cfg.data.num_of_views):
    #         sample_name += f" {view_name(views[view_idx])}"
    #     print(sample_name)

    #     if cfg.visual.viz:
    #         log_views(views, num)

    #     batch = collate_fn([views])

    #     loss_tuple = loss_of_one_batch_multi_view(
    #         batch,
    #         model,
    #         train_criterion,
    #         device,
    #         use_amp=bool(cfg.train_params.amp),
    #         amp_dtype=cfg.train_params.amp_dtype,
    #         ret="loss",
    #     )

    #     loss, loss_details = loss_tuple  # criterion returns two values



@hydra.main(version_base=None, config_path="configs", config_name="inference")
def execute_training(cfg: DictConfig):
    """
    Execute the training process with the provided configuration.

    cfg.data:
        cfg (DictConfig): Configuration object loaded by Hydra
    """
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Run the training
    inference(cfg)

if __name__ == "__main__":
     execute_training()
