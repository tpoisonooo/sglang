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
"""Optimized RMSNorm kernels for MiniCPM with different seq_len strategies.

This module provides optimized RMSNorm implementations that select different
kernels based on sequence length:
- Very short seq (<= 8): Single-pass Triton kernel (loads data once)
- Short seq (9-64): Multi-pass Triton kernel with fixed config
- Medium/Long seq (> 64): sgl_kernel (FlashInfer CUDA implementation)
  Including very long sequences like 32000+

All kernels assume fixed head_dim=128 and num_heads=32 (hidden_size=4096).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Import sgl_kernel as one of the implementation strategies
try:
    from sgl_kernel import rmsnorm as sgl_rmsnorm, fused_add_rmsnorm as sgl_fused_add_rmsnorm
    from sgl_kernel.utils import is_arch_support_pdl
    SGL_KERNEL_AVAILABLE = True
except ImportError:
    SGL_KERNEL_AVAILABLE = False


# Fixed dimensions for MiniCPM
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 4096


# =============================================================================
# Kernel 1: Very Short Sequence - Single Pass (seq_len <= 8)
# =============================================================================

@triton.jit
def rmsnorm_single_pass_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass RMSNorm kernel - loads input data only once.
    
    Optimized for very short sequences where BLOCK_SIZE >= N.
    Each block handles one row.
    """
    row = tl.program_id(0)
    
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row
    
    # Single load of input data and weight
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute RMS
    x_masked = tl.where(mask, x, 0.0)
    sum_sq = tl.sum(x_masked * x_masked, axis=0)
    mean_sq = sum_sq / N
    rms = tl.rsqrt(mean_sq + eps)
    
    # Apply normalization and store
    out = x * rms * w
    tl.store(out_row + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# Kernel 2: Short Sequence (9 <= seq_len <= 64)
# =============================================================================

@triton.jit
def rmsnorm_short_seq_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm kernel optimized for short sequences.
    
    Uses multi-pass approach with fixed BLOCK_SIZE=128.
    Maximizes parallelism by using many small blocks.
    """
    row = tl.program_id(0)
    
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row
    
    # First pass: compute sum of squares
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        _sum_sq += x * x
    
    sum_sq = tl.sum(_sum_sq, axis=0)
    mean_sq = sum_sq / N
    rms = tl.rsqrt(mean_sq + eps)
    
    # Second pass: apply normalization and weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
        out = x * rms * w
        tl.store(out_row + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# Kernel 3: Long Sequence - Persistent Kernel (seq_len > 512)
# =============================================================================

@triton.jit
def rmsnorm_long_seq_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    N: tl.constexpr,
    T: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """Persistent RMSNorm kernel optimized for long sequences.
    
    Each program handles multiple rows (ROWS_PER_PROG) to amortize
    kernel launch overhead.
    """
    start_row = tl.program_id(0) * ROWS_PER_PROG
    
    for row_idx in range(ROWS_PER_PROG):
        row = start_row + row_idx
        if row < T:
            x_row = x_ptr + row * stride_row
            out_row = out_ptr + row * stride_row
            
            # Compute sum of squares
            _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
                _sum_sq += x * x
            
            sum_sq = tl.sum(_sum_sq, axis=0)
            mean_sq = sum_sq / N
            rms = tl.rsqrt(mean_sq + eps)
            
            # Apply normalization and weight
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                x = tl.load(x_row + cols, mask=mask).to(tl.float32)
                w = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
                out = x * rms * w
                tl.store(out_row + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# Helper Functions
# =============================================================================

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _get_single_pass_config(hidden_size: int) -> Tuple[int, int, int]:
    """Get config for single-pass kernel (BLOCK_SIZE >= hidden_size).
    
    Returns (BLOCK_SIZE, num_warps, num_stages)
    """
    BLOCK_SIZE = _next_power_of_2(hidden_size)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    
    if BLOCK_SIZE <= 128:
        num_warps = 4
        num_stages = 3
    elif BLOCK_SIZE <= 512:
        num_warps = 4
        num_stages = 3
    else:
        num_warps = 8
        num_stages = 2
    
    return BLOCK_SIZE, num_warps, num_stages


def _get_long_seq_config(hidden_size: int) -> Tuple[int, int, int, int]:
    """Get config for long sequence persistent kernel.
    
    Returns (BLOCK_SIZE, ROWS_PER_PROG, num_warps, num_stages)
    """
    if hidden_size <= 2048:
        BLOCK_SIZE = 1024
        ROWS_PER_PROG = 4
        num_warps = 8
        num_stages = 2
    else:
        BLOCK_SIZE = 2048
        ROWS_PER_PROG = 8
        num_warps = 8
        num_stages = 2
    
    return BLOCK_SIZE, ROWS_PER_PROG, num_warps, num_stages


# =============================================================================
# Main RMSNorm Functions - Strategy Pattern Based on seq_len
# =============================================================================

def minicpm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MiniCPM optimized RMSNorm with seq_len-aware implementation selection.
    
    Implementation strategy:
    - seq_len <= 8: Single-pass Triton kernel (minimize memory access)
    - 9 <= seq_len <= 64: Multi-pass Triton kernel (maximize parallelism)
    - seq_len > 64: sgl_kernel (FlashInfer CUDA, best for medium/long seq)
      Fallback to persistent Triton if sgl_kernel not available
    
    Args:
        x: [seq_len, hidden_size] input tensor
        weight: [hidden_size] RMSNorm weight
        eps: epsilon for numerical stability
    
    Returns:
        out: [seq_len, hidden_size] normalized output
    """
    x = x.contiguous()
    seq_len, hidden_size = x.shape
    out = torch.empty_like(x)
    
    # Strategy 1: Very short sequence - single pass Triton
    if seq_len <= 8:
        BLOCK_SIZE, num_warps, num_stages = _get_single_pass_config(hidden_size)
        grid = (seq_len,)
        rmsnorm_single_pass_kernel[grid](
            x, weight, out,
            x.stride(0),
            N=hidden_size,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    # Strategy 2: Short sequence - multi-pass Triton with fixed config
    elif seq_len <= 64:
        BLOCK_SIZE = 128
        num_warps = 4
        num_stages = 3
        grid = (seq_len,)
        rmsnorm_short_seq_kernel[grid](
            x, weight, out,
            x.stride(0),
            N=hidden_size,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    # Strategy 3: Medium/Long sequence - use sgl_kernel (FlashInfer CUDA)
    # sgl_kernel is faster for all seq_len > 64 based on benchmarks
    elif SGL_KERNEL_AVAILABLE:
        enable_pdl = is_arch_support_pdl()
        sgl_rmsnorm(x, weight, eps, out=out, enable_pdl=enable_pdl)
    
    # Strategy 4: Fallback to persistent Triton if sgl_kernel not available
    else:
        BLOCK_SIZE, ROWS_PER_PROG, num_warps, num_stages = _get_long_seq_config(hidden_size)
        grid = (triton.cdiv(seq_len, ROWS_PER_PROG),)
        
        rmsnorm_long_seq_kernel[grid](
            x, weight, out,
            x.stride(0),
            N=hidden_size,
            T=seq_len,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
            ROWS_PER_PROG=ROWS_PER_PROG,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    return out


def minicpm_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MiniCPM optimized Fused Add + RMSNorm with seq_len-aware implementation.
    
    Implementation strategy:
    - seq_len <= 64: PyTorch add + Triton RMSNorm
    - seq_len > 64: sgl_kernel fused_add_rmsnorm (if available)
      Fallback to PyTorch add + Triton RMSNorm if sgl_kernel not available
    
    Args:
        x: [seq_len, hidden_size] input tensor
        residual: [seq_len, hidden_size] residual tensor
        weight: [hidden_size] RMSNorm weight
        eps: epsilon for numerical stability
    
    Returns:
        out: [seq_len, hidden_size] normalized output
        residual_out: [seq_len, hidden_size] x + residual
    """
    x = x.contiguous()
    residual = residual.contiguous()
    seq_len, hidden_size = x.shape
    
    # Strategy 1: Short sequence - PyTorch add + Triton RMSNorm
    if seq_len <= 64:
        residual_out = x + residual
        out = minicpm_rmsnorm(residual_out, weight, eps)
        return out, residual_out
    
    # Strategy 2: Medium/Long sequence - sgl_kernel fused_add_rmsnorm
    elif SGL_KERNEL_AVAILABLE:
        x_copy = x.clone()
        residual_copy = residual.clone()
        enable_pdl = is_arch_support_pdl()
        sgl_fused_add_rmsnorm(x_copy, residual_copy, weight, eps, enable_pdl=enable_pdl)
        return x_copy, residual_copy
    
    # Strategy 3: Fallback to PyTorch add + Triton RMSNorm
    else:
        residual_out = x + residual
        out = minicpm_rmsnorm(residual_out, weight, eps)
        return out, residual_out


# =============================================================================
# PyTorch Module Wrapper
# =============================================================================

class MiniCPMRMSNorm(nn.Module):
    """MiniCPM RMSNorm module with seq_len-aware optimized kernels.
    
    Automatically selects the best implementation based on input sequence length:
    - Very short (<=8): Single-pass Triton (minimize memory access)
    - Short (9-64): Multi-pass Triton (maximize parallelism)
    - Medium/Long (>64): sgl_kernel (FlashInfer CUDA, best for all longer sequences)
      Including very long sequences like 32000+
    
    Args:
        hidden_size: Size of the hidden dimension (default: 4096 for MiniCPM)
        eps: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [seq_len, hidden_size] input tensor
            residual: Optional [seq_len, hidden_size] residual for fused add
        
        Returns:
            If residual is None: normalized output
            If residual is provided: (normalized output, x + residual)
        """
        if residual is not None:
            return minicpm_fused_add_rmsnorm(x, residual, self.weight, self.eps)
        else:
            return minicpm_rmsnorm(x, self.weight, self.eps)
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


# =============================================================================
# Benchmark and Testing
# =============================================================================

def _pytorch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PyTorch reference implementation."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x


def _pytorch_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for fused add + rmsnorm."""
    x = x + residual
    residual_out = x.clone()
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x, residual_out


def benchmark():
    """Benchmark MiniCPM RMSNorm against PyTorch reference."""
    import time
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    device = torch.device("cuda")
    hidden_size = HIDDEN_SIZE
    
    # Test different sequence lengths
    seq_lens = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    print("=" * 100)
    print("Benchmarking MiniCPM RMSNorm")
    print(f"Hidden size: {hidden_size}")
    print(f"SGL Kernel available: {SGL_KERNEL_AVAILABLE}")
    print("=" * 100)
    
    # Benchmark RMSNorm
    print("\n" + "-" * 100)
    print("RMSNorm (without residual)")
    print("-" * 100)
    print(f"{'Seq Len':>10} {'Strategy':>20} {'MiniCPM (ms)':>18} {'PyTorch (ms)':>15} {'Speedup':>12}")
    print("-" * 100)
    
    for seq_len in seq_lens:
        # Create test data
        x = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
        weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
        
        # Determine strategy
        if seq_len <= 8:
            strategy = "Triton single-pass"
        elif seq_len <= 64:
            strategy = "Triton multi-pass"
        else:
            strategy = "sgl_kernel" if SGL_KERNEL_AVAILABLE else "Triton persistent"
        
        # Adjust iterations for long sequences
        n_iters = 100 if seq_len <= 2048 else 20
        
        # Warmup
        for _ in range(10):
            _ = minicpm_rmsnorm(x, weight)
            _ = _pytorch_rmsnorm(x, weight)
        
        torch.cuda.synchronize()
        
        # Benchmark MiniCPM
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = minicpm_rmsnorm(x, weight)
        torch.cuda.synchronize()
        minicpm_time = (time.perf_counter() - start) / n_iters * 1000
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = _pytorch_rmsnorm(x, weight)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / n_iters * 1000
        
        speedup = torch_time / minicpm_time if minicpm_time > 0 else 1.0
        
        print(f"{seq_len:>10} {strategy:>20} {minicpm_time:>18.4f} {torch_time:>15.4f} {speedup:>11.2f}x")
    
    # Benchmark Fused Add + RMSNorm
    print("\n" + "-" * 100)
    print("Fused Add + RMSNorm")
    print("-" * 100)
    print(f"{'Seq Len':>10} {'Strategy':>20} {'MiniCPM (ms)':>18} {'PyTorch (ms)':>15} {'Speedup':>12}")
    print("-" * 100)
    
    for seq_len in seq_lens:
        # Create test data
        x = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
        residual = torch.randn_like(x)
        weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
        
        # Determine strategy
        if seq_len <= 64:
            strategy = "PyTorch+Triton"
        else:
            strategy = "sgl_kernel" if SGL_KERNEL_AVAILABLE else "PyTorch+Triton"
        
        # Adjust iterations for long sequences
        n_iters = 100 if seq_len <= 2048 else 20
        
        # Warmup
        for _ in range(10):
            _ = minicpm_fused_add_rmsnorm(x.clone(), residual, weight)
            _ = _pytorch_fused_add_rmsnorm(x.clone(), residual, weight)
        
        torch.cuda.synchronize()
        
        # Benchmark MiniCPM
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = minicpm_fused_add_rmsnorm(x.clone(), residual, weight)
        torch.cuda.synchronize()
        minicpm_time = (time.perf_counter() - start) / n_iters * 1000
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = _pytorch_fused_add_rmsnorm(x.clone(), residual, weight)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / n_iters * 1000
        
        speedup = torch_time / minicpm_time if minicpm_time > 0 else 1.0
        
        print(f"{seq_len:>10} {strategy:>20} {minicpm_time:>18.4f} {torch_time:>15.4f} {speedup:>11.2f}x")
    
    print("\n" + "=" * 100)
    print("Benchmark completed!")
    print("=" * 100)


def verify_correctness():
    """Verify correctness of implementations against PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping verification")
        return True
    
    device = torch.device("cuda")
    hidden_size = HIDDEN_SIZE
    seq_lens = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    eps = 1e-6
    
    print("=" * 80)
    print("Verifying MiniCPM RMSNorm Correctness")
    print("=" * 80)
    
    all_passed = True
    
    for seq_len in seq_lens:
        # Test data
        x = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
        weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
        
        # PyTorch reference
        pt_out = _pytorch_rmsnorm(x, weight, eps)
        
        # MiniCPM implementation
        minicpm_out = minicpm_rmsnorm(x, weight, eps)
        
        # Compare
        match = torch.allclose(pt_out, minicpm_out, atol=1e-3, rtol=1e-3)
        max_diff = torch.max(torch.abs(pt_out - minicpm_out)).item()
        
        status = "✅" if match else "❌"
        if not match:
            all_passed = False
        
        print(f"{status} seq_len={seq_len:4d}: max diff = {max_diff:.6f}")
    
    # Verify Fused Add + RMSNorm
    print("\n" + "-" * 80)
    print("Verifying Fused Add + RMSNorm")
    print("-" * 80)
    
    for seq_len in seq_lens:
        x = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device)
        residual = torch.randn_like(x)
        weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
        
        # PyTorch reference
        pt_out, pt_res = _pytorch_fused_add_rmsnorm(x.clone(), residual, weight, eps)
        
        # MiniCPM implementation
        minicpm_out, minicpm_res = minicpm_fused_add_rmsnorm(x.clone(), residual, weight, eps)
        
        # Compare
        out_match = torch.allclose(pt_out, minicpm_out, atol=1e-3, rtol=1e-3)
        res_match = torch.allclose(pt_res, minicpm_res, atol=1e-3, rtol=1e-3)
        
        max_diff_out = torch.max(torch.abs(pt_out - minicpm_out)).item()
        max_diff_res = torch.max(torch.abs(pt_res - minicpm_res)).item()
        
        status = "✅" if (out_match and res_match) else "❌"
        if not (out_match and res_match):
            all_passed = False
        
        print(f"{status} seq_len={seq_len:4d}: out diff = {max_diff_out:.6f}, res diff = {max_diff_res:.6f}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MiniCPM RMSNorm benchmark and verification")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--verify", action="store_true", help="Run correctness verification")
    parser.add_argument("--all", action="store_true", help="Run both benchmark and verification")
    
    args = parser.parse_args()
    
    if args.all or (not args.benchmark and not args.verify):
        args.benchmark = True
        args.verify = True
    
    if args.verify:
        verify_correctness()
        print()
    
    if args.benchmark:
        benchmark()
