"""Fused RMSNorm + RoPE for MiniCPM.

This module provides a fused Triton implementation for:
1. Q/K RMSNorm (per-head, head_dim=128)
2. RoPE application

Fixed parameters for MiniCPM:
- num_heads: 32
- head_dim: 128
- num_kv_heads: 32
- hidden_size: 4096 (32 * 128)

Optimized for Blackwell architecture (RTX 6000D).
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# Fixed parameters for MiniCPM
NUM_HEADS = 32
HEAD_DIM = 128
NUM_KV_HEADS = 32
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 4096

# Kernel configurations for Blackwell architecture (RTX 6000D)
# Format: (num_warps, num_stages)
# Optimized with num_stages=4 for better memory latency hiding
BLACKWELL_KERNEL_CONFIGS = {
    # For head_dim=128, use 4 warps and 4 stages for optimal occupancy on Blackwell
    128: (4, 4),
}

# Default config for other head dimensions
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 4


@triton.jit
def _fused_rms_norm_rope_kernel_optimized(
    q_ptr,
    k_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    out_q_ptr,
    out_k_ptr,
    q_stride,
    k_stride,
    out_q_stride,
    out_k_stride,
    cos_sin_stride,
    eps: tl.constexpr,
):
    """Optimized Fused RMSNorm + RoPE kernel for MiniCPM.
    
    Each block processes one head of one token.
    Grid: (num_tokens, num_heads + num_kv_heads)
    
    Optimizations:
    1. Load x and w in two halves (64 elements each) - no redundant loads
    2. Compute RMS from both halves directly
    3. num_stages=4 for better memory latency hiding on Blackwell
    
    Fixed parameters (as tl.constexpr):
    - NUM_HEADS = 32
    - NUM_KV_HEADS = 32
    - HEAD_DIM = 128
    - HALF_HEAD_DIM = 64
    """
    # Define constants
    NUM_HEADS: tl.constexpr = 32
    NUM_KV_HEADS: tl.constexpr = 32
    HEAD_DIM: tl.constexpr = 128
    HALF_HEAD_DIM: tl.constexpr = 64
    
    # 2D program IDs
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Determine if processing Q or K
    is_q = head_idx < NUM_HEADS
    
    if is_q:
        actual_head_idx = head_idx
        in_ptr = q_ptr
        in_stride = q_stride
        out_ptr = out_q_ptr
        out_stride = out_q_stride
        norm_weight_ptr = q_norm_weight_ptr
    else:
        actual_head_idx = head_idx - NUM_HEADS
        if actual_head_idx >= NUM_KV_HEADS:
            return
        in_ptr = k_ptr
        in_stride = k_stride
        out_ptr = out_k_ptr
        out_stride = out_k_stride
        norm_weight_ptr = k_norm_weight_ptr
    
    # Compute input offset: [num_tokens, num_heads * head_dim]
    in_base = token_idx * in_stride + actual_head_idx * HEAD_DIM
    
    # Compute output offset: [num_tokens, num_heads, head_dim]
    out_base = token_idx * out_stride + actual_head_idx * HEAD_DIM
    
    # Load position for RoPE
    position = tl.load(positions_ptr + token_idx)
    
    # Get cos/sin for this position
    cos_ptr = cos_sin_cache_ptr + position * cos_sin_stride
    sin_ptr = cos_ptr + HALF_HEAD_DIM
    
    # Create indices for first and second half
    half_idx = tl.arange(0, HALF_HEAD_DIM)
    
    # Load input in two halves (optimization: no redundant loads)
    x1 = tl.load(in_ptr + in_base + half_idx).to(tl.float32)
    x2 = tl.load(in_ptr + in_base + half_idx + HALF_HEAD_DIM).to(tl.float32)
    
    # Load weights in two halves
    w1 = tl.load(norm_weight_ptr + half_idx).to(tl.float32)
    w2 = tl.load(norm_weight_ptr + half_idx + HALF_HEAD_DIM).to(tl.float32)
    
    # Compute RMS using both halves (no need to load full x)
    sum_sq1 = tl.sum(x1 * x1, axis=0)
    sum_sq2 = tl.sum(x2 * x2, axis=0)
    mean_sq = (sum_sq1 + sum_sq2) / HEAD_DIM
    rms = tl.rsqrt(mean_sq + eps)
    
    # Apply RMSNorm to both halves
    x1_norm = x1 * rms * w1
    x2_norm = x2 * rms * w2
    
    # Load cos and sin
    cos_vals = tl.load(cos_ptr + half_idx)
    sin_vals = tl.load(sin_ptr + half_idx)
    
    # Apply RoPE rotation: y1 = x1 * cos - x2 * sin, y2 = x2 * cos + x1 * sin
    y1 = x1_norm * cos_vals - x2_norm * sin_vals
    y2 = x2_norm * cos_vals + x1_norm * sin_vals
    
    # Store results
    tl.store(out_ptr + out_base + half_idx, y1.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + out_base + half_idx + HALF_HEAD_DIM, y2.to(out_ptr.dtype.element_ty))


class FusedRMSNormRoPE(torch.nn.Module):
    """Fused RMSNorm + RoPE for MiniCPM.
    
    This fuses the following operations:
    1. Q/K RMSNorm (per-head normalization with head_dim=128)
    2. RoPE application
    
    Fixed parameters:
    - num_heads: 32
    - head_dim: 128
    - num_kv_heads: 32
    
    Input:
    - q: [num_tokens, num_heads * head_dim] = [num_tokens, 4096]
    - k: [num_tokens, num_kv_heads * head_dim] = [num_tokens, 4096]
    - positions: [num_tokens]
    - q_norm_weight: [head_dim]
    - k_norm_weight: [head_dim]
    - cos_sin_cache: [max_position, head_dim]
    
    Output:
    - q: [1, num_tokens, num_heads, head_dim]
    - k: [1, num_tokens, num_kv_heads, head_dim]
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused RMSNorm + RoPE.
        
        Args:
            q: [num_tokens, 4096] query tensor
            k: [num_tokens, 4096] key tensor
            positions: [num_tokens] position indices
            q_norm_weight: [128] Q normalization weight
            k_norm_weight: [128] K normalization weight
            cos_sin_cache: [max_position, 128] cos/sin cache
            
        Returns:
            q: [1, num_tokens, 32, 128] normalized and rotated query
            k: [1, num_tokens, 32, 128] normalized and rotated key
        """
        num_tokens = q.shape[0]
        
        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        
        # Output tensors with shape [num_tokens, num_heads, head_dim]
        out_q = torch.empty(num_tokens, NUM_HEADS, HEAD_DIM, dtype=q.dtype, device=q.device)
        out_k = torch.empty(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=k.dtype, device=k.device)
        
        # Launch kernel with 2D grid
        grid = (num_tokens, NUM_HEADS + NUM_KV_HEADS)
        
        # Get kernel config for Blackwell architecture
        if HEAD_DIM in BLACKWELL_KERNEL_CONFIGS:
            num_warps, num_stages = BLACKWELL_KERNEL_CONFIGS[HEAD_DIM]
        else:
            num_warps = DEFAULT_NUM_WARPS
            num_stages = DEFAULT_NUM_STAGES
        
        _fused_rms_norm_rope_kernel_optimized[grid](
            q,
            k,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            out_q,
            out_k,
            q.stride(0),
            k.stride(0),
            out_q.stride(0),  # stride between tokens
            out_k.stride(0),  # stride between tokens
            cos_sin_cache.stride(0),
            eps=self.eps,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        
        # Reshape to [1, num_tokens, num_heads, head_dim]
        out_q = out_q.unsqueeze(0)
        out_k = out_k.unsqueeze(0)
        
        return out_q, out_k


def fused_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional API for fused RMSNorm + RoPE.
    
    Args:
        q: [num_tokens, 4096] query tensor
        k: [num_tokens, 4096] key tensor
        positions: [num_tokens] position indices
        q_norm_weight: [128] Q normalization weight
        k_norm_weight: [128] K normalization weight
        cos_sin_cache: [max_position, 128] cos/sin cache
        eps: epsilon for RMSNorm
        
    Returns:
        q: [1, num_tokens, 32, 128] normalized and rotated query
        k: [1, num_tokens, 32, 128] normalized and rotated key
    """
    num_tokens = q.shape[0]
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    
    # Output tensors with shape [num_tokens, num_heads, head_dim]
    out_q = torch.empty(num_tokens, NUM_HEADS, HEAD_DIM, dtype=q.dtype, device=q.device)
    out_k = torch.empty(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=k.dtype, device=k.device)
    
    # Launch kernel with 2D grid
    grid = (num_tokens, NUM_HEADS + NUM_KV_HEADS)
    
    # Get kernel config for Blackwell architecture
    if HEAD_DIM in BLACKWELL_KERNEL_CONFIGS:
        num_warps, num_stages = BLACKWELL_KERNEL_CONFIGS[HEAD_DIM]
    else:
        num_warps = DEFAULT_NUM_WARPS
        num_stages = DEFAULT_NUM_STAGES
    
    _fused_rms_norm_rope_kernel_optimized[grid](
        q,
        k,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        out_q,
        out_k,
        q.stride(0),
        k.stride(0),
        out_q.stride(0),
        out_k.stride(0),
        cos_sin_cache.stride(0),
        eps=eps,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    # Reshape to [1, num_tokens, num_heads, head_dim]
    out_q = out_q.unsqueeze(0)
    out_k = out_k.unsqueeze(0)
    
    return out_q, out_k
