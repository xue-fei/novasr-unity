import torch

ckpt = torch.load("pytorch_model_v2.bin", map_location='cpu')
total_params = sum(p.numel() for p in ckpt.values())
print(f"参数量: {total_params} (~{total_params * 4 / 1024:.1f} KB)")