import cv2
import os
import numpy as np

def sample_frames(video_path, output_dir, num_frames=10):
    # 创建保存帧的文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    # 计算均匀采样的帧索引
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    print(f"采样帧索引: {frame_indices}")

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"保存帧: {frame_filename}")
        else:
            print(f"无法读取帧: {idx}")

    cap.release()
    print("采样完成！")

# 使用示例
video_path = "ego_teaser_trim.mp4"       # 视频路径
output_dir = "sampled_frames"          # 保存帧的文件夹
sample_frames(video_path, output_dir, num_frames=15)
