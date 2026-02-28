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
"""Fused MLP kernel for MiniCPM.

This module provides highly optimized fused operation for:
    output = residual + input * scale

Input: 2D tensor with shape [seq_len, 4096], dtype = bfloat16

Optimized for NVIDIA GPUs (Ampere, Ada, Hopper architectures).

Usage:
    >>> from minicpm_fused_mlp import fused_scale_add
    >>> output = fused_scale_add(input_tensor, residual, scale=0.125)
    
    # Or use specific version:
    >>> output = fused_scale_add(input_tensor, residual, scale, version="ultra")
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Triton Kernels - Implementation Details
# =============================================================================
# Each kernel uses different optimization techniques:
# 
# 1. basic: Simple 1D grid, good for large contiguous memory
# 2. ultra: 16x vectorization, one block per row, best for seq_len >= 1024
# 3. pipeline: Software pipelining, overlaps memory and compute
# 4. persistent: Grid-stride loop, reduces kernel launch overhead for large batches
# =============================================================================


@triton.jit
def _fused_scale_add_basic_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    scale,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Basic kernel: 1D grid, each thread processes BLOCK_SIZE elements.
    
    Best for: Large contiguous memory regions where coalescing is automatic.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    r = tl.load(residual_ptr + offsets, mask=mask)
    result = r + x * scale
    tl.store(output_ptr + offsets, result, mask=mask)

    # x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    # r = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)
    # result = r + x * scale
    # tl.store(output_ptr + offsets, result.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_scale_add_ultra_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    scale,
    seq_len: tl.constexpr,
    HIDDEN_DIM: tl.constexpr = 4096,
):
    """Ultra-optimized kernel: 16x vectorization, one block per row.
    
    Optimization techniques:
    - 16x vectorization (256 threads * 16 = 4096 elements per row)
    - Perfect memory coalescing (consecutive threads access consecutive addresses)
    - FP32 accumulation for numerical stability
    - Static loop unrolling via tl.static_range
    
    Best for: seq_len >= 1024, HIDDEN_DIM = 4096
    """
    pid = tl.program_id(axis=0)
    
    BLOCK_SIZE: tl.constexpr = 256
    UNROLL: tl.constexpr = 16
    
    base = pid * HIDDEN_DIM
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Each thread processes UNROLL elements
    for i in tl.static_range(UNROLL):
        idx = i * BLOCK_SIZE + offsets
        off = base + idx
        
        # Load as bf16, compute as fp32, store as bf16
        x = tl.load(input_ptr + off).to(tl.float32)
        r = tl.load(residual_ptr + off).to(tl.float32)
        result = r + x * scale
        tl.store(output_ptr + off, result.to(tl.bfloat16))


@triton.jit
def _fused_scale_add_pipeline_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    scale,
    seq_len: tl.constexpr,
    HIDDEN_DIM: tl.constexpr = 4096,
):
    """Pipelined kernel: software pipelining for better memory-compute overlap.
    
    Optimization techniques:
    - Prefetch first iteration
    - Overlap computation of current iteration with memory ops of next
    - Reduces memory latency stalls
    
    Best for: Large seq_len where memory latency matters
    """
    pid = tl.program_id(axis=0)
    
    row_off = pid * HIDDEN_DIM
    tid = tl.arange(0, 256)
    
    # Prefetch first
    x_prev = tl.load(input_ptr + row_off + tid).to(tl.float32)
    r_prev = tl.load(residual_ptr + row_off + tid).to(tl.float32)
    
    for i in tl.static_range(1, 16):
        # Compute previous
        result_prev = r_prev + x_prev * scale
        
        # Load next (overlaps with previous compute)
        idx_next = i * 256 + tid
        x_next = tl.load(input_ptr + row_off + idx_next).to(tl.float32)
        r_next = tl.load(residual_ptr + row_off + idx_next).to(tl.float32)
        
        # Store previous result
        tl.store(output_ptr + row_off + (i - 1) * 256 + tid, result_prev.to(tl.bfloat16))
        
        # Move next to previous
        x_prev = x_next
        r_prev = r_next
    
    # Final store
    result_prev = r_prev + x_prev * scale
    tl.store(output_ptr + row_off + 15 * 256 + tid, result_prev.to(tl.bfloat16))


@triton.jit
def _fused_scale_add_persistent_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    scale,
    seq_len: tl.constexpr,
    HIDDEN_DIM: tl.constexpr = 4096,
    ROWS_PER_BLOCK: tl.constexpr = 4,
):
    """Persistent kernel: processes multiple rows per block via grid-stride loop.
    
    Optimization techniques:
    - Grid-stride loop for flexible work distribution
    - Fewer blocks launched, reducing kernel launch overhead
    - Better GPU occupancy for large seq_len
    
    Best for: Very large seq_len (>= 4096) to reduce launch overhead
    """
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    
    total_rows = seq_len
    rows_per_grid = num_blocks * ROWS_PER_BLOCK
    
    # Grid-stride loop
    for row_base in range(pid * ROWS_PER_BLOCK, total_rows, rows_per_grid):
        for row_idx in tl.static_range(ROWS_PER_BLOCK):
            row = row_base + row_idx
            if row < total_rows:
                row_off = row * HIDDEN_DIM
                tid = tl.arange(0, 256)
                
                for i in tl.static_range(16):
                    idx = i * 256 + tid
                    off = row_off + idx
                    
                    x = tl.load(input_ptr + off).to(tl.float32)
                    r = tl.load(residual_ptr + off).to(tl.float32)
                    result = r + x * scale
                    tl.store(output_ptr + off, result.to(tl.bfloat16))


# =============================================================================
# Autotuned Kernel
# =============================================================================
# Automatically selects best num_warps and num_stages based on seq_len

fused_scale_add_autotune = triton.autotune(
    configs=[
        # Small seq_len: minimize launch overhead
        triton.Config(kwargs={}, num_warps=4, num_stages=1),
        triton.Config(kwargs={}, num_warps=8, num_stages=1),
        
        # Medium seq_len: balance
        triton.Config(kwargs={}, num_warps=4, num_stages=2),
        triton.Config(kwargs={}, num_warps=8, num_stages=2),
        triton.Config(kwargs={}, num_warps=16, num_stages=2),
        
        # Large seq_len: maximize memory bandwidth
        triton.Config(kwargs={}, num_warps=8, num_stages=4),
        triton.Config(kwargs={}, num_warps=16, num_stages=4),
        triton.Config(kwargs={}, num_warps=32, num_stages=4),
    ],
    key=["seq_len"],
)


@triton.jit
def _fused_scale_add_autotuned_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    scale,
    seq_len: tl.constexpr,
    HIDDEN_DIM: tl.constexpr = 4096,
):
    """Autotuned kernel - best configuration selected at runtime."""
    pid = tl.program_id(axis=0)
    
    tid = tl.arange(0, 256)
    row_off = pid * HIDDEN_DIM
    
    for i in tl.static_range(16):
        idx = i * 256 + tid
        off = row_off + idx
        
        x = tl.load(input_ptr + off).to(tl.float32)
        r = tl.load(residual_ptr + off).to(tl.float32)
        result = r + x * scale
        tl.store(output_ptr + off, result.to(tl.bfloat16))


_fused_scale_add_autotuned = fused_scale_add_autotune(_fused_scale_add_autotuned_kernel)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def fused_scale_add_basic(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Basic triton version - flattens to 1D and processes."""
    output = torch.empty_like(input_tensor)
    seq_len, hidden_dim = input_tensor.shape
    assert hidden_dim == 4096, f"Expected hidden_dim=4096, got {hidden_dim}"
    
    n_elements = seq_len * hidden_dim
    BLOCK_SIZE = 1024
    
    _fused_scale_add_basic_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        input_tensor, residual, output, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def fused_scale_add_ultra(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Ultra-optimized version - one block per row, 16x vectorization."""
    output = torch.empty_like(input_tensor)
    seq_len, hidden_dim = input_tensor.shape
    assert hidden_dim == 4096
    
    grid = (seq_len,)
    _fused_scale_add_ultra_kernel[grid](
        input_tensor, residual, output, scale, seq_len=seq_len,
        num_warps=8, num_stages=2,
    )
    return output


def fused_scale_add_pipeline(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Pipelined version - software pipelining for better overlap."""
    output = torch.empty_like(input_tensor)
    seq_len, hidden_dim = input_tensor.shape
    assert hidden_dim == 4096
    
    grid = (seq_len,)
    _fused_scale_add_pipeline_kernel[grid](
        input_tensor, residual, output, scale, seq_len=seq_len,
        num_warps=8, num_stages=4,
    )
    return output


def fused_scale_add_persistent(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
    rows_per_block: int = 4,
) -> torch.Tensor:
    """Persistent kernel - grid-stride loop for large batches."""
    output = torch.empty_like(input_tensor)
    seq_len, hidden_dim = input_tensor.shape
    assert hidden_dim == 4096
    
    num_sms = torch.cuda.get_device_properties(input_tensor.device).multi_processor_count
    grid_size = min(seq_len, num_sms * 4)
    
    _fused_scale_add_persistent_kernel[(grid_size,)](
        input_tensor, residual, output, scale, seq_len=seq_len,
        ROWS_PER_BLOCK=rows_per_block, num_warps=8, num_stages=2,
    )
    return output


def fused_scale_add_autotuned(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Autotuned version - automatically selects best config."""
    output = torch.empty_like(input_tensor)
    seq_len, hidden_dim = input_tensor.shape
    assert hidden_dim == 4096
    
    grid = lambda meta: (seq_len,)
    _fused_scale_add_autotuned[grid](
        input_tensor, residual, output, scale, seq_len=seq_len,
    )
    return output


# =============================================================================
# Main Entry Point
# =============================================================================

def fused_scale_add(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """Fused scale and add: output = residual + input * scale
    
    This operation is commonly used in transformer MLP layers with residual
    connections, where the MLP output is scaled before adding to the residual.
    
    Args:
        input_tensor: [seq_len, 4096] - typically MLP output, bfloat16
        residual: [seq_len, 4096] - residual connection, bfloat16
        scale: float - scale factor (e.g., 0.125 for 1/sqrt(64))
        version: Kernel version selection:
            - "auto": Automatically select based on seq_len (recommended)
            - "basic": Simple 1D kernel, best for large contiguous memory
            - "ultra": 16x vectorized, one block per row
            - "pipeline": Software pipelined for memory-compute overlap
            - "persistent": Grid-stride loop for large batches
            - "autotuned": Runtime autotuned configuration
            - "pytorch": PyTorch native implementation
    
    Returns:
        output: [seq_len, 4096] = residual + input_tensor * scale, bfloat16
        
    Example:
        >>> import torch
        >>> from minicpm_fused_mlp import fused_scale_add
        >>> 
        >>> input_tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        >>> residual = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        >>> output = fused_scale_add(input_tensor, residual, scale=0.125)
        >>> 
        >>> # Or use specific version:
        >>> output = fused_scale_add(input_tensor, residual, 0.125, version="ultra")
    """
    # Validate inputs
    assert input_tensor.is_cuda, "Input must be on CUDA"
    assert residual.is_cuda, "Residual must be on CUDA"
    assert input_tensor.dtype == torch.bfloat16, f"Input must be bf16, got {input_tensor.dtype}"
    assert residual.dtype == torch.bfloat16, f"Residual must be bf16, got {residual.dtype}"
    assert input_tensor.shape == residual.shape, f"Shape mismatch: {input_tensor.shape} vs {residual.shape}"

    input_tensor = input_tensor.contiguous()
    residual = residual.contiguous()

    seq_len, hidden_dim = input_tensor.shape
    assert hidden_dim == 4096, f"Expected hidden_dim=4096, got {hidden_dim}"
    

    # return residual + input_tensor * scale

    # Based on benchmarks:
    # - Small seq_len: PyTorch has lower overhead
    # - Medium seq_len: ultra version is well-balanced
    # - Large seq_len: basic version has best memory bandwidth
    # if seq_len <= 512:
    #     return residual + input_tensor * scale
    # elif seq_len <= 2048:
    #     return fused_scale_add_ultra(input_tensor, residual, scale)
    # else:
    return fused_scale_add_basic(input_tensor, residual, scale)

# Backward compatibility
def fused_scale_add_default(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Default implementation using auto-selected kernel."""
    return fused_scale_add(input_tensor, residual, scale, version="auto")


# Keep original function name for compatibility
fused_scale_add_original = fused_scale_add_default
