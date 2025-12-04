from safetensors.torch import load_file
import torch

# 1. 加载 safetensors 文件（权重会是一个 dict）
weights = load_file("/data/hub/models--facebook--map-anything/snapshots/6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a/model.safetensors")
# weights = torch.load("checkpoint/model.pth")
print(weights.keys())

delete_keys = []
for k in weights.keys():
    if "dense_head" in k:
        delete_keys.append(k)

for k in delete_keys:
    del weights[k]

checkpoint = {"model": weights}

# 2. 保存成 PyTorch 原生格式
torch.save(checkpoint, "checkpoint/model.pth")
