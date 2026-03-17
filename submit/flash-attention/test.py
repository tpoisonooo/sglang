import sys
sys.modules['flash_attn_2_cuda'] = type(sys)('flash_attn_2_cuda')
import torch
from flash_attn.cute import flash_attn_func

q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device='cuda')
out = flash_attn_func(q, k, v, causal=True)
print(f'Output shape: {out[0].shape}, max: {out[0].max():.4f}')
print('Forward pass OK!')
