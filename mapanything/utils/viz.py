# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for visualization
"""

from argparse import ArgumentParser, Namespace
from distutils.util import strtobool

import numpy as np
import rerun as rr
import trimesh
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import time
import threading
from typing import List

from mapanything.utils.hf_utils.viz import image_mesh
from mapanything.utils.geometry import depthmap_to_absolute_camera_coordinates, closed_form_pose_inverse


def log_posed_rgbd_data_to_rerun(
    image, depthmap, pose, intrinsics, base_name, mask=None
):
    """
    Log camera and image data to Rerun visualization tool.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image to be logged
    depthmap : numpy.ndarray
        Depth map corresponding to the image
    pose : numpy.ndarray
        4x4 camera pose matrix with rotation (3x3) and translation (3x1)
    intrinsics : numpy.ndarray
        Camera intrinsic matrix
    base_name : str
        Base name for the logged entities in Rerun
    mask : numpy.ndarray, optional
        Optional segmentation mask for the depth image
    """
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
    if mask is not None:
        rr.log(
            f"{base_name}/pinhole/depth_mask",
            rr.SegmentationImage(mask),
        )


def str2bool(v):
    return bool(strtobool(v))


def script_add_rerun_args(parser: ArgumentParser) -> None:
    """
    Add common Rerun script arguments to `parser`.

    Change Log from https://github.com/rerun-io/rerun/blob/29eb8954b08e59ff96943dc0677f46f7ea4ea734/rerun_py/rerun_sdk/rerun/script_helpers.py#L65:
        - Added default portforwarding url for ease of use
        - Update parser types

    Parameters
    ----------
    parser : ArgumentParser
        The parser to add arguments to.

    Returns
    -------
    None
    """
    parser.add_argument(
        "--headless",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Don't show GUI",
    )
    parser.add_argument(
        "--connect",
        dest="connect",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Connect to an external viewer",
    )
    parser.add_argument(
        "--serve",
        dest="serve",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Serve a web viewer (WARNING: experimental feature)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="rerun+http://127.0.0.1:2004/proxy",
        help="Connect to this HTTP(S) URL",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save data to a .rrd file at this path"
    )
    parser.add_argument(
        "-o",
        "--stdout",
        dest="stdout",
        action="store_true",
        help="Log data to standard output, to be piped into a Rerun Viewer",
    )


def init_rerun_args(
    headless=True,
    connect=True,
    serve=False,
    url="rerun+http://127.0.0.1:2004/proxy",
    save=None,
    stdout=False,
) -> Namespace:
    """
    Initialize common Rerun script arguments.

    Parameters
    ----------
    headless : bool, optional
        Don't show GUI, by default True
    connect : bool, optional
        Connect to an external viewer, by default True
    serve : bool, optional
        Serve a web viewer (WARNING: experimental feature), by default False
    url : str, optional
        Connect to this HTTP(S) URL, by default rerun+http://127.0.0.1:2004/proxy
    save : str, optional
        Save data to a .rrd file at this path, by default None
    stdout : bool, optional
        Log data to standard output, to be piped into a Rerun Viewer, by default False

    Returns
    -------
    Namespace
        The parsed arguments.
    """
    rerun_args = Namespace()
    rerun_args.headless = headless
    rerun_args.connect = connect
    rerun_args.serve = serve
    rerun_args.url = url
    rerun_args.save = save
    rerun_args.stdout = stdout

    return rerun_args


def predictions_to_glb(
    predictions,
    as_mesh=True,
) -> trimesh.Scene:
    """
    Converts predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (V, H, W, 3)
            - images: Input images (V, H, W, 3)
            - final_masks: Validity masks (V, H, W)
        as_mesh (bool): Represent the data as a mesh instead of point cloud (default: True)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud/mesh and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # Get the world frame points and images from the predictions
    pred_world_points = predictions["world_points"]
    images = predictions["images"]
    final_masks = predictions["final_masks"]

    # Get the points and rgb
    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    if as_mesh:
        # 原来的 mesh 处理逻辑保持不变
        for frame_idx in range(pred_world_points.shape[0]):
            H, W = pred_world_points.shape[1:3]
            frame_points = pred_world_points[frame_idx]
            frame_final_mask = final_masks[frame_idx]
            if images.ndim == 4 and images.shape[1] == 3:  # NCHW
                frame_image = np.transpose(images[frame_idx], (1, 2, 0))
            else:
                frame_image = images[frame_idx]
            frame_image *= 255

            faces, vertices, vertex_colors = image_mesh(
                frame_points * np.array([1, -1, 1], dtype=np.float32),
                frame_image / 255.0,
                mask=frame_final_mask,
                tri=True,
                return_indices=False,
            )
            vertices = vertices * np.array([1, -1, 1], dtype=np.float32)

            frame_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=(vertex_colors * 255).astype(np.uint8),
                process=False,
            )
            scene_3d.add_geometry(frame_mesh)
    else:
        # 改写非 mesh 部分：逐帧保存点云
        for frame_idx in range(pred_world_points.shape[0]):
            frame_points = pred_world_points[frame_idx].reshape(-1, 3)
            frame_mask = final_masks[frame_idx].reshape(-1)
            frame_points = frame_points[frame_mask]

            # 获取对应的颜色
            if images.ndim == 4 and images.shape[1] == 3:  # NCHW
                frame_image = np.transpose(images[frame_idx], (1, 2, 0))
            else:
                frame_image = images[frame_idx]
            frame_colors = frame_image.reshape(-1, 3)[frame_mask]

            # 创建单帧点云对象
            point_cloud = trimesh.PointCloud(vertices=frame_points, colors=frame_colors)
            scene_3d.add_geometry(point_cloud, node_name=f"frame_{frame_idx}")

    # Apply 180° rotation around X-axis to fix orientation (upside-down issue)
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    scene_3d.apply_transform(rotation_matrix_x)

    return scene_3d


def generate_ply_bytes(points, colors):
    # generate binary ply object bytes from the point cloud and their color
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    header = "\n".join(header).encode("ascii") + b"\n"

    colors_uint8 = safe_color_conversion(colors)

    # Pack data into binary format
    data = np.empty(len(points), dtype=[
        ("xyz", np.float32, 3),
        ("rgb", np.uint8, 3),
    ])
    data["xyz"] = points
    data["rgb"] = colors_uint8

    return header + data.tobytes()

def safe_color_conversion(colors):
    # If colors are in float format (normalized)
    if colors.dtype in [np.float32, np.float64]:
        # Handle two common normalization ranges
        if colors.min() >= 0 and colors.max() <= 1:
            # 0 to 1 range
            colors_uint8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
        elif colors.min() >= -1 and colors.max() <= 1:
            # -1 to 1 range (common in some frameworks)
            colors_uint8 = np.clip((colors + 1) * 127.5, 0, 255).astype(np.uint8)
        else:
            # Unexpected range, try linear scaling
            colors_min, colors_max = colors.min(), colors.max()
            colors_uint8 = np.clip(
                ((colors - colors_min) / (colors_max - colors_min)) * 255,
                0, 255
            ).astype(np.uint8)
    else:
        # Already in uint8 or similar integer format
        colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)

    return colors_uint8


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 0.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4), #c2w
                "intrinsic": (S, 3, 3),
                "mask": (S, H, W)
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    if "world_points" in pred_dict:
        world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
        conf_map = pred_dict["world_points_conf"]  # (S, H, W)
    else:
        world_points_map = None
        conf_map = None
        use_point_map = False

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        depth_map = pred_dict["depth"]  # (S, H, W)
        depth_conf = pred_dict["depth_conf"]  # (S, H, W)
        world_points, valid_mask = depthmap_to_absolute_camera_coordinates(depth_map, intrinsics_cam, extrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map
        valid_mask = np.ones_like(colors[..., 0], dtype=bool)


    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images
    if images.shape[1] == 3:
        colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)

    if "mask" in pred_dict:
        mask = pred_dict["mask"] & valid_mask  # (S, H, W)
    else:
        mask = valid_mask

    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    mask_flat = mask.reshape(-1)

    cam_to_world_mat = extrinsics_cam  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points[mask_flat], axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox(
        "Show Cameras",
        initial_value=True,
    )

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent",
        min=0,
        max=100,
        step=0.1,
        initial_value=init_conf_threshold,
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + ["first_half"] + ["second_half"] + [str(i) for i in range(S)],
        initial_value="All",
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    init_conf_mask &= mask_flat
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # @debug
    #save_point_cloud_ply("generate_point_cloud.ply", points_centered, colors_flat)


    with server.gui.add_folder("Export Options", expand_by_default=False):
        button_download_ply = server.gui.add_button("Download PLY")

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, H, W, 3)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            # Choose a color for the camera
            if img_id >= S / 2:
                cam_color = (255, 0, 0)
            else:
                cam_color = (0, 255, 0)
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=1.0,
                color=cam_color,
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        elif gui_frame_selector.value == "first_half":
            frame_mask = frame_indices < S / 2
        elif gui_frame_selector.value == "second_half":
            frame_mask = frame_indices >= S / 2
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask & mask_flat

        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

        # === 新增逻辑：控制相机 frustum 显示 ===
        selected_val = gui_frame_selector.value

        # 先全部隐藏
        for f in frames:
            f.visible = False
        for fr in frustums:
            fr.visible = False

        if selected_val == "All":
            # 显示全部
            for f in frames:
                f.visible = True
            for fr in frustums:
                fr.visible = True
        elif selected_val == "first_half":
            for i in range(S // 2):
                frames[i].visible = True
                frustums[i].visible = True
        elif selected_val == "second_half":
            for i in range(S // 2, S):
                frames[i].visible = True
                frustums[i].visible = True
        else:
            # 选择了具体帧号
            selected_idx = int(selected_val)
            if 0 <= selected_idx < len(frames):
                frames[selected_idx].visible = True
                frustums[selected_idx].visible = True

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    @button_download_ply.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        if client is None:
            print("No client connected; skipping download.")
            return

        # Generate PLY and send to client
        try:
            ply_bytes = generate_ply_bytes(point_cloud.points, point_cloud.colors)
            client.send_file_download("pointcloud.ply", ply_bytes)
        except Exception as e:
            print(f"Failed to generate PLY: {e}")

        public_url = server.request_share_url()
        return server

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, colors)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server
