conda create -n mapanything python=3.12 -y
conda activate mapanything
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install -e ".[all]"
pre-commit install

# 下载hf-cli，让服务器登陆huggingface
pip install hf-cli
sudo apt install git-lfs
hf auth login


# 下载 mapanything 模型
python load_model.py

# 将safetensors 转换成 pth 文件
python test_rerun.py

# 配置数据集预处理操作
cd data_processing/wai_processing/
pip install -e .[all]
pip install -e .[moge] # install with moge support

# 下载数据划分meta（train、test、split划分）
hf download --repo-type dataset facebook/map-anything --include "mapanything_dataset_metadata" --local-dir /data/dataset/mapanything/

# 下载数据集eth3d
python data_processing/wai_processing/download_scripts/download_eth3d.py /data/dataset/eth3d/
python -m wai_processing.scripts.conversion.eth3d \
          original_root="/data/dataset/eth3d/eth3d_raw" \
          root="/data/dataset/eth3d/eth3d_WAI"
# Run covisibility
python -m wai_processing.scripts.covisibility \
          data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml \
          root="/data/dataset/eth3d/eth3d_WAI"
# Run moge
python -m wai_processing.scripts.run_moge \
          root="/data/dataset/eth3d/eth3d_WAI" \
          batch_size=1 # MoGe stage doesn't support nested tensors


# 下载 dl3dv
# hf download --repo-type dataset DL3DV/DL3DV-ALL-960P --include "1K/01*" --local-dir /data/dataset/DL3DV_960P_1K_00/
# hf download --repo-type dataset DL3DV/DL3DV-ALL-ColmapCache --include "1K/01*" --local-dir /data/dataset/DL3DV_960P_1K_00_Cache/
wget https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/scripts/download.py
python data_processing/wai_processing/download_scripts/download_dl3dv.py --odir /data/dataset/dl3dv/DL3DV_960P_1K_00/ --subset 1K --resolution 960P --file_type images+poses --clean_cache --hash 67f97041185afa2812133dde64b55790d788de8d191ba1a54c392561868190bc
python data_processing/wai_processing/download_scripts/download_dl3dv.py --odir /data/dataset/dl3dv/DL3DV_960P_1K_00_colmap/ --subset 1K --resolution 960P --file_type colmap_cache --clean_cache --hash 67f97041185afa2812133dde64b55790d788de8d191ba1a54c392561868190bc
# 注意！！要将images-4修改为images, 把colmap挪到和images的同一个文件夹下
# Run a conversion script (dataset specific)
python -m wai_processing.scripts.conversion.dl3dv\
          original_root=/data/dataset/dl3dv/DL3DV_960P_1K_00 \
          root=/data/dataset/dl3dv/DL3DV_960P_1K_00_WAI
# Run undistortion (modalities can be dataset specific)
python -m wai_processing.scripts.undistort \
          data_processing/wai_processing/configs/undistortion/default.yaml \
          root=/data/dataset/dl3dv/DL3DV_960P_1K_00_WAI
# Run moge
python -m wai_processing.scripts.run_moge \
          root=/data/dataset/dl3dv/DL3DV_960P_1K_00_WAI
# Run mvsanywhere（only for dl3dv）
python -m wai_processing.scripts.run_mvsanywhere \
          root=/data/dataset/dl3dv/DL3DV_960P_1K_00_WAI
# Run covisibility
python -m wai_processing.scripts.covisibility \
          data_processing/wai_processing/configs/covisibility/covisibility_pred_depth_mvsa.yaml \
          root=/data/dataset/dl3dv/DL3DV_960P_1K_00_WAI
# Get depth consistency confidence (for e.g., useful for mvsanywhere)
python -m wai_processing.scripts.depth_consistency_confidence \
          data_processing/wai_processing/configs/depth_consistency_confidence/depth_consistency_confidence_mvsa.yaml \
          root=/data/dataset/dl3dv/DL3DV_960P_1K_00_WAI

# 下载 UnrealStereo4K
hf download --repo-type dataset fabiotosi92/UnrealStereo4K --include 00000.zip --local-dir /data/dataset/unrealstereo4k/

# 下载ase(Aria Synthetic Environments)
python3 /data/dataset/ase/projectaria_tools/projects/AriaSyntheticEnvironment/aria_synthetic_environments_downloader.py --set train --scene-ids 0-9 --cdn-file /data/dataset/ase/aria_download_uls.json --output-dir /data/dataset/ase/ase_train_chunk_000 --unzip True
# 下面的命令参考：data_processing/wai_processing/configs/launch/ase.yaml
python -m wai_processing.scripts.conversion.ase\
          original_root=/data/dataset/ase/ase_train_chunk_000 \
          root=/data/dataset/data_WAI/ase
python -m wai_processing.scripts.covisibility \
          data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml \
          root=/data/dataset/data_WAI/ase

# co3d
python -m wai_processing.scripts.conversion.co3d\
          original_root=/data/dataset/co3d/co3d-subset \
          root=/data/dataset/data_WAI/co3d
python -m wai_processing.scripts.covisibility \
          data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml \
          root=/data/dataset/data_WAI/co3d
python -m wai_processing.scripts.run_moge \
          root=/data/dataset/data_WAI/co3d
# 生成train/val划分
python -m data_processing.aggregate_scene_names --wai_root /data/dataset/data_WAI/ --output_dir /data/dataset/mapanything_meta --datasets co3d
# rr可视化目标数据集
rerun --web-viewer
python -m mapanything.datasets.wai.co3d --viz --connect

# 推理
python inference.py
