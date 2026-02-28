# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused RMSNorm + RoPE kernel for MiniCPM.

This module provides a fused Triton kernel that combines:
1. Per-head RMSNorm on Q and K
2. Neox-style RoPE (Rotary Position Embedding)
3. Layout transformation to [1, seq_len, num_heads, head_dim]

All operations are fused into a single kernel launch to minimize
memory bandwidth usage and kernel launch overhead.
"""

import math
import torch
import triton
import triton.language as tl


# Predefined kernel configurations for common head_dim values
# Each config: (BLOCK_HEAD_DIM, num_warps, num_stages)
KERNEL_CONFIGS = {
    64: (64, 2, 3),    # head_dim=64: 64 threads, 2 warps, 3 stages
    128: (128, 4, 3),  # head_dim=128: 128 threads, 4 warps, 3 stages
}


@triton.jit
def fused_rms_norm_rope_kernel(
    # Input pointers
    q_ptr,
    k_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    # Output pointers
    q_out_ptr,
    k_out_ptr,
    # Dimensions
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    max_position,
    eps,
    # Strides
    stride_q_seq,
    stride_q_head,
    stride_k_seq,
    stride_k_head,
    stride_out_seq,
    stride_out_head,
    stride_cache_pos,
    # Block size (constexpr)
    BLOCK_HEAD_DIM: tl.constexpr,
):
    """Fused RMSNorm + RoPE kernel.
    
    Each block processes one (seq_pos, head) pair.
    Grid: (seq_len, num_heads + num_kv_heads) - 2D grid for better occupancy
    """
    # 2D grid: (seq_len, total_heads)
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Determine if this is Q or K
    if head_idx < num_heads:
        # Q head
        in_ptr = q_ptr
        out_ptr = q_out_ptr
        norm_weight_ptr = q_norm_weight_ptr
        stride_seq = stride_q_seq
        stride_head = stride_q_head
        actual_head_idx = head_idx
    else:
        # K head
        in_ptr = k_ptr
        out_ptr = k_out_ptr
        norm_weight_ptr = k_norm_weight_ptr
        stride_seq = stride_k_seq
        stride_head = stride_k_head
        actual_head_idx = head_idx - num_heads
    
    # Compute offsets
    in_offset = seq_idx * stride_seq + actual_head_idx * stride_head
    out_offset = seq_idx * stride_out_seq + actual_head_idx * stride_out_head
    
    # Load position for this sequence element
    pos = tl.load(positions_ptr + seq_idx)
    
    # Create offsets for head_dim
    offs = tl.arange(0, BLOCK_HEAD_DIM)
    mask = offs < head_dim
    
    # Load norm weight (per-head, same for all positions in head)
    norm_w = tl.load(norm_weight_ptr + offs, mask=mask).to(tl.float32)
    
    # Load input and convert to float32
    x = tl.load(in_ptr + in_offset + offs, mask=mask).to(tl.float32)
    
    # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    x_sq = x * x
    mean_sq = tl.sum(x_sq) / head_dim
    rms = tl.rsqrt(mean_sq + eps)
    x_normed = x * rms * norm_w
    
    # Load cos and sin from cache
    half_dim = head_dim // 2
    cache_row_offset = pos * stride_cache_pos
    
    # Compute offset within the cache row for cos and sin
    cos_sin_offs = tl.where(offs < half_dim, offs, offs - half_dim)
    
    # Load cos from first half of cache row
    cos = tl.load(cos_sin_cache_ptr + cache_row_offset + cos_sin_offs, mask=mask)
    # Load sin from second half of cache row
    sin = tl.load(cos_sin_cache_ptr + cache_row_offset + half_dim + cos_sin_offs, mask=mask)
    
    # Neox-style RoPE rotation
    other_offs = tl.where(offs < half_dim, offs + half_dim, offs - half_dim)
    
    # Load the value from the other half of the same head
    x_other = tl.load(in_ptr + in_offset + other_offs, mask=mask).to(tl.float32)
    x_other_normed = x_other * rms * norm_w
    
    # Construct rotated: negate the first half
    rotated = tl.where(offs < half_dim, -x_other_normed, x_other_normed)
    
    # Apply RoPE: x * cos + rotated * sin
    x_rope = x_normed * cos + rotated * sin
    
    # Convert back to original dtype and store
    x_out = x_rope.to(in_ptr.dtype.element_ty)
    tl.store(out_ptr + out_offset + offs, x_out, mask=mask)


def fused_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + RoPE operation.
    
    Args:
        q: [seq_len, num_heads * head_dim] bf16/fp16
        k: [seq_len, num_kv_heads * head_dim] bf16/fp16
        positions: [seq_len] int32/int64
        cos_sin_cache: [max_position, head_dim] fp32
        q_norm_weight: [head_dim] fp32
        k_norm_weight: [head_dim] fp32
        eps: RMSNorm epsilon
    
    Returns:
        q_out: [1, seq_len, num_heads, head_dim]
        k_out: [1, seq_len, num_kv_heads, head_dim]
    """
    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    
    seq_len, q_hidden = q.shape
    _, k_hidden = k.shape
    
    head_dim = q_norm_weight.shape[0]
    num_heads = q_hidden // head_dim
    num_kv_heads = k_hidden // head_dim
    max_position = cos_sin_cache.shape[0]
    
    # Ensure positions is contiguous and int32
    if positions.dtype != torch.int32:
        positions = positions.to(torch.int32)
    positions = positions.contiguous()
    
    # Ensure cos_sin_cache is contiguous
    cos_sin_cache = cos_sin_cache.contiguous()
    
    # Output tensors
    q_out = torch.empty(1, seq_len, num_heads, head_dim, dtype=q.dtype, device=q.device)
    k_out = torch.empty(1, seq_len, num_kv_heads, head_dim, dtype=k.dtype, device=k.device)
    
    # Get kernel configuration based on head_dim
    if head_dim in KERNEL_CONFIGS:
        BLOCK_HEAD_DIM, num_warps, num_stages = KERNEL_CONFIGS[head_dim]
    else:
        # Fallback for other head_dim values
        BLOCK_HEAD_DIM = triton.next_power_of_2(head_dim)
        num_warps = 4
        num_stages = 2
    
    # 2D grid: (seq_len, num_heads + num_kv_heads)
    grid = (seq_len, num_heads + num_kv_heads)
    
    fused_rms_norm_rope_kernel[grid](
        q, k, positions, cos_sin_cache, q_norm_weight, k_norm_weight,
        q_out, k_out,
        seq_len, num_heads, num_kv_heads, head_dim, max_position, eps,
        q.stride(0), head_dim,
        k.stride(0), head_dim,
        num_heads * head_dim, head_dim,
        cos_sin_cache.stride(0),
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return q_out, k_out
