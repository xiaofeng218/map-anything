hf download --repo-type dataset DL3DV/DL3DV-ALL-480P --include "00*" --local-dir /data/hanxiaofeng/dataset/DL3DV_480P_1K_00/

# Run a conversion script (dataset specific)
python -m wai_processing.scripts.conversion.dl3dv\
          original_root=/data/hanxiaofeng/dataset/DL3DV_480P_1K_00 \
          root=/data/hanxiaofeng/dataset/DL3DV_480P_1K_00_WAI

# Run mvsanywhere（only for dl3dv）
python -m wai_processing.scripts.run_mvsanywhere \
          root=/data/hanxiaofeng/dataset/DL3DV_480P_1K_00_WAI

# Get depth consistency confidence (for e.g., useful for mvsanywhere)
python -m wai_processing.scripts.depth_consistency_confidence \
          data_processing/wai_processing/configs/depth_consistency_confidence/depth_consistency_confidence_mvsa.yaml \
          root=/data/hanxiaofeng/dataset/DL3DV_480P_1K_00_WAI

# Run moge
python -m wai_processing.scripts.run_moge \
          root=/data/hanxiaofeng/dataset/DL3DV_480P_1K_00_WAI

# Run covisibility
python -m wai_processing.scripts.covisibility \
          data_processing/wai_processing/configs/covisibility/covisibility_pred_depth_mvsa.yaml \
          root=/data/hanxiaofeng/dataset/DL3DV_480P_1K_00_WAI
