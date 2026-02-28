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
"""Fused output processing kernel for MiniCPM Lightning Attention.

This module provides a fused Triton kernel that combines:
1. RMSNorm on the output (full hidden_size)
2. Output gate: sigmoid(z) + multiply with o

Benchmark results (vs torch.compile baseline):
- ~10x speedup across all hidden sizes (512-4096) and seq lengths (8-2048)
- Latency: ~0.021ms vs ~0.21ms for baseline
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_output_kernel(
    # Input pointers
    o_ptr,              # [seq_len, hidden_size] - attention output (already reshaped)
    z_ptr,              # [seq_len, hidden_size] - precomputed z_proj output
    norm_weight_ptr,    # [hidden_size]
    # Output pointers
    out_ptr,            # [seq_len, hidden_size]
    # Dimensions
    hidden_size,
    eps,
    # Strides
    stride_o_seq,
    stride_z_seq,
    stride_out_seq,
    # Block size (constexpr)
    BLOCK_SIZE: tl.constexpr,
):
    """Fused output processing kernel.
    
    This kernel fuses:
    1. RMSNorm (full hidden_size)
    2. Gate: o * sigmoid(z)
    
    Grid: (seq_len,) - 1D grid, each block handles one sequence position
    Each block loops over hidden_size in chunks.
    """
    seq_idx = tl.program_id(0)
    
    # Compute base offset for this sequence position
    o_base = seq_idx * stride_o_seq
    z_base = seq_idx * stride_z_seq
    out_base = seq_idx * stride_out_seq
    
    # First pass: compute sum of squares for RMS
    num_blocks = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    sum_sq = 0.0
    
    for block_idx in range(num_blocks):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        
        o_val = tl.load(o_ptr + o_base + offs, mask=mask, other=0.0).to(tl.float32)
        x_sq = o_val * o_val
        sum_sq += tl.sum(x_sq)
    
    # Compute RMS
    mean_sq = sum_sq / hidden_size
    rms = tl.rsqrt(mean_sq + eps)
    
    # Second pass: apply RMSNorm and gate
    for block_idx in range(num_blocks):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        
        # Load o
        o_val = tl.load(o_ptr + o_base + offs, mask=mask).to(tl.float32)
        
        # Load norm weight
        norm_w = tl.load(norm_weight_ptr + offs, mask=mask).to(tl.float32)
        
        # Apply RMSNorm
        o_val = o_val * rms * norm_w
        
        # Load z and apply sigmoid gate
        z_val = tl.load(z_ptr + z_base + offs, mask=mask).to(tl.float32)
        gate = tl.sigmoid(z_val)
        o_val = o_val * gate
        
        # Store output
        tl.store(out_ptr + out_base + offs, o_val.to(out_ptr.dtype.element_ty), mask=mask)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _get_launch_config(hidden_size: int):
    """Get kernel launch configuration based on hidden_size.
    
    Returns (BLOCK_SIZE, num_warps, num_stages)
    
    Note: Configurations adjusted for Blackwell consumer GPUs (RTX 6000D).
    Original values were optimized for H800 Hopper datacenter GPU with many SMs.
    Reduced block sizes and num_warps to improve occupancy on consumer GPUs.
    """
    if hidden_size <= 512:
        # Small hidden_size: process in one block
        BLOCK_SIZE = max(128, _next_power_of_2(hidden_size))
        num_warps = min(BLOCK_SIZE // 32, 4)  # Reduced max warps from 8 to 4
        num_stages = 3
    elif hidden_size <= 2048:
        # Medium hidden_size: process in one block
        BLOCK_SIZE = _next_power_of_2(hidden_size)
        num_warps = min(BLOCK_SIZE // 32, 4)  # Reduced max warps from 8 to 4
        num_stages = 3
    else:
        # Large hidden_size: use 1024 threads (reduced from 2048 for Blackwell)
        BLOCK_SIZE = 1024
        num_warps = 4  # Reduced from 8 to 4 for better occupancy on consumer GPUs
        num_stages = 2
    
    return BLOCK_SIZE, num_warps, num_stages


def fused_output_processing(
    o: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused output processing operation.
    
    Fuses RMSNorm and sigmoid gate into a single kernel.
    
    Args:
        o: [seq_len, hidden_size] bf16/fp16 - attention output (already reshaped)
        z: [seq_len, hidden_size] bf16/fp16 - precomputed z_proj output
        norm_weight: [hidden_size] fp32 - RMSNorm weight
        eps: RMSNorm epsilon
    
    Returns:
        out: [seq_len, hidden_size] bf16/fp16
    """
    # Ensure inputs are contiguous
    o = o.contiguous()
    z = z.contiguous()
    
    seq_len = o.shape[0]
    hidden_size = o.shape[1]
    
    # Output tensor
    out = torch.empty(seq_len, hidden_size, dtype=o.dtype, device=o.device)
    
    # Get launch configuration
    BLOCK_SIZE, num_warps, num_stages = _get_launch_config(hidden_size)
    
    # 1D grid: one block per sequence position
    grid = (seq_len,)
    
    fused_output_kernel[grid](
        o, z, norm_weight, out,
        hidden_size, eps,
        o.stride(0),
        z.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return out
