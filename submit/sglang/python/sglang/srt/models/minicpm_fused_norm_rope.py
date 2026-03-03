"""Fused RMSNorm + RoPE for MiniCPM.

This module provides a fused Triton implementation for:
1. Q/K RMSNorm (per-head, head_dim=128)
2. RoPE application

Fixed parameters for MiniCPM:
- num_heads: 32
- head_dim: 128
- num_kv_heads: 32
- hidden_size: 4096 (32 * 128)

Optimized for Blackwell architecture (RTX 6000D, Compute Capability 12.0).
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

# Blackwell architecture optimizations
# RTX 6000D has 132 SMs, each with 128KB L1 cache
# Using num_stages=4 for better memory latency hiding
# BLOCK_SIZE optimized for head_dim=128


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
    """Optimized Fused RMSNorm + RoPE kernel for MiniCPM - Blackwell Optimized.
    
    Each block processes one head of one token.
    Grid: (num_tokens, num_heads + num_kv_heads)
    
    Blackwell Optimizations:
    1. Uses tl.sum for efficient parallel reduction (Blackwell has fast reduction units)
    2. Optimized memory access patterns for high bandwidth memory (~960 GB/s)
    3. num_stages=4 for maximum memory latency hiding
    4. Coalesced memory access with vectorized loads/stores
    5. No bank conflicts in shared memory operations
    
    Fixed parameters (as tl.constexpr for compiler optimization):
    - NUM_HEADS = 32
    - NUM_KV_HEADS = 32  
    - HEAD_DIM = 128
    - HALF_HEAD_DIM = 64
    """
    # Define constants as constexpr for compiler optimization
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
    
    # Load input in two halves with vectorized coalesced access
    x1 = tl.load(in_ptr + in_base + half_idx).to(tl.float32)
    x2 = tl.load(in_ptr + in_base + half_idx + HALF_HEAD_DIM).to(tl.float32)
    
    # Load weights in two halves
    w1 = tl.load(norm_weight_ptr + half_idx).to(tl.float32)
    w2 = tl.load(norm_weight_ptr + half_idx + HALF_HEAD_DIM).to(tl.float32)
    
    # Compute RMS using efficient tl.sum (Blackwell optimized)
    sum_sq1 = tl.sum(x1 * x1, axis=0)
    sum_sq2 = tl.sum(x2 * x2, axis=0)
    mean_sq = (sum_sq1 + sum_sq2) / HEAD_DIM
    rms = tl.math.rsqrt(mean_sq + eps)
    
    # Apply RMSNorm to both halves
    x1_norm = x1 * rms * w1
    x2_norm = x2 * rms * w2
    
    # Load cos and sin
    cos_vals = tl.load(cos_ptr + half_idx)
    sin_vals = tl.load(sin_ptr + half_idx)
    
    # Apply RoPE rotation: y1 = x1 * cos - x2 * sin, y2 = x2 * cos + x1 * sin
    y1 = x1_norm * cos_vals - x2_norm * sin_vals
    y2 = x2_norm * cos_vals + x1_norm * sin_vals
    
    # Store results with vectorized coalesced writes
    tl.store(out_ptr + out_base + half_idx, y1.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + out_base + half_idx + HALF_HEAD_DIM, y2.to(out_ptr.dtype.element_ty))


@triton.jit
def _fused_rms_norm_rope_kernel_large_tokens(
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
    num_tokens,
    eps: tl.constexpr,
):
    """Optimized kernel for large num_tokens using 1D grid with better SM utilization.
    
    Grid: (num_tokens * (NUM_HEADS + NUM_KV_HEADS),)
    Each block processes one head of one token.
    
    This kernel is optimized for large batch sizes where 2D grid may not saturate
    all SMs on Blackwell (132 SMs on RTX 6000D).
    """
    NUM_HEADS: tl.constexpr = 32
    NUM_KV_HEADS: tl.constexpr = 32
    TOTAL_HEADS: tl.constexpr = 64  # NUM_HEADS + NUM_KV_HEADS
    HEAD_DIM: tl.constexpr = 128
    HALF_HEAD_DIM: tl.constexpr = 64
    
    # 1D program ID to 2D indices
    pid = tl.program_id(0)
    token_idx = pid // TOTAL_HEADS
    head_idx = pid % TOTAL_HEADS
    
    # Bounds check
    if token_idx >= num_tokens:
        return
    
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
    
    # Compute offsets
    in_base = token_idx * in_stride + actual_head_idx * HEAD_DIM
    out_base = token_idx * out_stride + actual_head_idx * HEAD_DIM
    
    # Load position and get cos/sin
    position = tl.load(positions_ptr + token_idx)
    cos_ptr = cos_sin_cache_ptr + position * cos_sin_stride
    sin_ptr = cos_ptr + HALF_HEAD_DIM
    
    # Process both halves
    half_idx = tl.arange(0, HALF_HEAD_DIM)
    
    # Load data
    x1 = tl.load(in_ptr + in_base + half_idx).to(tl.float32)
    x2 = tl.load(in_ptr + in_base + half_idx + HALF_HEAD_DIM).to(tl.float32)
    w1 = tl.load(norm_weight_ptr + half_idx).to(tl.float32)
    w2 = tl.load(norm_weight_ptr + half_idx + HALF_HEAD_DIM).to(tl.float32)
    
    # RMSNorm with efficient reduction
    sum_sq1 = tl.sum(x1 * x1, axis=0)
    sum_sq2 = tl.sum(x2 * x2, axis=0)
    mean_sq = (sum_sq1 + sum_sq2) / HEAD_DIM
    rms = tl.math.rsqrt(mean_sq + eps)
    
    x1_norm = x1 * rms * w1
    x2_norm = x2 * rms * w2
    
    # RoPE
    cos_vals = tl.load(cos_ptr + half_idx)
    sin_vals = tl.load(sin_ptr + half_idx)
    
    y1 = x1_norm * cos_vals - x2_norm * sin_vals
    y2 = x2_norm * cos_vals + x1_norm * sin_vals
    
    # Store
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
        return fused_rms_norm_rope(q, k, positions, q_norm_weight, k_norm_weight, 
                                   cos_sin_cache, self.eps)


def _get_kernel_config(num_tokens: int):
    """Get optimal kernel configuration for Blackwell RTX 6000D.
    
    RTX 6000D specs:
    - 132 SMs
    - 128KB L1 cache per SM
    - ~960 GB/s memory bandwidth
    - Enhanced warp scheduler supports more concurrent warps
    
    Returns: (kernel_func, grid, num_warps, num_stages)
    """
    total_heads = NUM_HEADS + NUM_KV_HEADS  # 64
    
    # For small num_tokens, use 2D grid for better locality
    # For large num_tokens, use 1D grid for better SM utilization
    if num_tokens <= 512:
        # 2D grid: (num_tokens, total_heads)
        # Good for small batches, maintains per-token locality
        grid = (num_tokens, total_heads)
        kernel = _fused_rms_norm_rope_kernel_optimized
        num_warps = 4
        num_stages = 4
    else:
        # 1D grid for large tokens to saturate all 132 SMs
        # Each block processes one head of one token
        total_blocks = num_tokens * total_heads
        grid = (total_blocks,)
        kernel = _fused_rms_norm_rope_kernel_large_tokens
        # Balanced warps for memory-level parallelism on Blackwell
        num_warps = 4
        num_stages = 4
    
    return kernel, grid, num_warps, num_stages


def fused_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional API for fused RMSNorm + RoPE - Blackwell Optimized.
    
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
    
    # Get optimal kernel configuration for Blackwell
    kernel, grid, num_warps, num_stages = _get_kernel_config(num_tokens)
    
    # Launch kernel with appropriate arguments
    kernel_args = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    
    if num_tokens > 512:
        # 1D kernel needs num_tokens as constexpr
        kernel[grid](
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
            num_tokens=num_tokens,
            eps=eps,
            **kernel_args,
        )
    else:
        # 2D kernel doesn't need num_tokens
        kernel[grid](
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
            **kernel_args,
        )
    
    # Reshape to [1, num_tokens, num_heads, head_dim]
    out_q = out_q.unsqueeze(0)
    out_k = out_k.unsqueeze(0)
    
    return out_q, out_k
