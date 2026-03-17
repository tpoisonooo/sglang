"""
Test script for chunk_simple_gla kernel - Blackwell 6000D Optimized.

This file contains test inputs and benchmarks for comparing:
1. Original FLA implementation
2. Blackwell-optimized forward-only implementation
"""

import torch
import math
import sys
import os

# Add test_chunk_gla to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_chunk_gla import (
    chunk_simple_gla_blackwell,
    chunk_simple_gla_original,
    is_blackwell,
)


# =============================================================================
# Test Inputs (from test_fuse/test_chunk_gla.py)
# =============================================================================

def build_slope_tensor(nheads: int) -> torch.Tensor:
    """Build ALiBi slope tensor."""
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(nheads))
    return slopes


def get_test_inputs(
    seq_len: int = 1024,
    batch: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
):
    """Generate test inputs for chunk_simple_gla.
    
    Args:
        seq_len: Sequence length
        batch: Batch size
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type for tensors
        device: Device to place tensors on
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (q, k, v, g_gamma, scale)
    """
    torch.manual_seed(seed)
    
    # Query, Key, Value tensors
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    
    # Build g_gamma using ALiBi slopes
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device=device, dtype=torch.float32) * 0.01
    
    # Scale factor
    scale = head_dim ** -0.5
    
    return q, k, v, g_gamma, scale


# Default test parameters
DEFAULT_TEST_CONFIGS = [
    {"seq_len": 16},
    {"seq_len": 32},
    {"seq_len": 64},
    {"seq_len": 128},
    {"seq_len": 256},
    {"seq_len": 512},
    {"seq_len": 1024},
    {"seq_len": 2048},
    {"seq_len": 4096},
]

# Default tensor shapes
DEFAULT_BATCH = 1
DEFAULT_NUM_HEADS = 32
DEFAULT_HEAD_DIM = 128
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_DEVICE = "cuda"


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_blackwell(q, k, v, g_gamma, scale, warmup=10, iterations=50):
    """Benchmark Blackwell-optimized implementation."""
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        _, _ = chunk_simple_gla_blackwell(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        o, ht = chunk_simple_gla_blackwell(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms, o, ht


def benchmark_original(q, k, v, g_gamma, scale, warmup=10, iterations=50):
    """Benchmark original FLA implementation."""
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        _, _ = chunk_simple_gla_original(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        o, ht = chunk_simple_gla_original(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms, o, ht


def compare_accuracy(o_blackwell, o_original, ht_blackwell, ht_original):
    """Compare accuracy between implementations."""
    output_diff = (o_blackwell - o_original).abs().max().item()
    state_diff = (ht_blackwell - ht_original).abs().max().item()
    
    output_scale = o_original.abs().mean().item() + 1e-6
    state_scale = ht_original.abs().mean().item() + 1e-6
    output_rel_diff = ((o_blackwell - o_original).abs() / output_scale).max().item()
    state_rel_diff = ((ht_blackwell - ht_original).abs() / state_scale).max().item()
    
    output_mae = (o_blackwell - o_original).abs().mean().item()
    state_mae = (ht_blackwell - ht_original).abs().mean().item()
    
    has_nan = (torch.isnan(o_blackwell).any() or torch.isnan(o_original).any() or 
               torch.isnan(ht_blackwell).any() or torch.isnan(ht_original).any())
    
    return {
        'output_abs_diff': output_diff,
        'output_rel_diff': output_rel_diff,
        'state_rel_diff': state_rel_diff,
        'output_mae': output_mae,
        'state_mae': state_mae,
        'has_nan': has_nan,
    }


# =============================================================================
# Main Test
# =============================================================================

def main():
    """Test and benchmark chunk_simple_gla implementations."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return
    
    # GPU Info
    print("=" * 90)
    print("Chunk Simple GLA Test - Blackwell 6000D Optimized")
    print("=" * 90)
    
    print(f"\nGPU Information:")
    print(f"  Device: {torch.cuda.get_device_name()}")
    major, minor = torch.cuda.get_device_capability()
    print(f"  Compute Capability: {major}.{minor}")
    print(f"  Is Blackwell: {is_blackwell()}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    import triton
    print(f"Triton version: {triton.__version__}")
    
    # Test parameters
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    num_heads = 32
    head_dim = 128
    
    print("\n" + "=" * 90)
    print("Correctness Test")
    print("=" * 90)
    
    for config in DEFAULT_TEST_CONFIGS[:5]:  # Test shorter sequences first
        seq_len = config["seq_len"]
        q, k, v, g_gamma, scale = get_test_inputs(seq_len=seq_len)
        
        # Run both implementations
        o_blackwell, ht_blackwell = chunk_simple_gla_blackwell(
            q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
        )
        o_original, ht_original = chunk_simple_gla_original(
            q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
        )
        
        # Compare
        diff_info = compare_accuracy(o_blackwell, o_original, ht_blackwell, ht_original)
        
        if diff_info['has_nan']:
            status = "NAN"
        elif diff_info['output_mae'] > 0.5 or diff_info['state_mae'] > 1.0:
            status = "HIGH_ERR"
        elif diff_info['output_mae'] > 0.05 or diff_info['state_mae'] > 0.1:
            status = "WARN"
        else:
            status = "PASS"
        
        print(f"Seq Len {seq_len:4d}: Output MAE={diff_info['output_mae']:.2e}, "
              f"State MAE={diff_info['state_mae']:.2e} [{status}]")
    
    print("\n" + "=" * 90)
    print("Performance Benchmark")
    print("=" * 90)
    print(f"{'Seq Len':<10} {'Blackwell (ms)':<15} {'Original (ms)':<15} {'Speedup':<10} {'Status':<10}")
    print("-" * 90)
    
    for config in DEFAULT_TEST_CONFIGS:
        seq_len = config["seq_len"]
        q, k, v, g_gamma, scale = get_test_inputs(seq_len=seq_len)
        
        try:
            # Benchmark Blackwell
            t_blackwell, o_blackwell, ht_blackwell = benchmark_blackwell(
                q, k, v, g_gamma, scale, warmup=5, iterations=20
            )
            
            # Benchmark Original
            t_original, o_original, ht_original = benchmark_original(
                q, k, v, g_gamma, scale, warmup=5, iterations=20
            )
            
            # Check accuracy
            diff_info = compare_accuracy(o_blackwell, o_original, ht_blackwell, ht_original)
            
            if diff_info['has_nan']:
                status = "NAN"
            elif diff_info['output_mae'] > 0.05:
                status = "WARN"
            else:
                status = "PASS"
            
            speedup = t_original / t_blackwell if t_blackwell > 0 else 0
            
            print(f"{seq_len:<10} {t_blackwell:>13.3f}   {t_original:>13.3f}   "
                  f"{speedup:>8.2f}x  {status}")
            
        except Exception as e:
            print(f"{seq_len:<10} {'ERROR':<15} {'ERROR':<15} {'N/A':<10} {str(e)[:20]}")
    
    print("\n" + "=" * 90)
    print("Blackwell 6000D Optimizations Applied:")
    print("  - Forward-only (no backward overhead)")
    print("  - Larger tile sizes: BK/BV up to 128 (128KB shared mem)")
    print("  - Optimized warp distribution for 156 SMs")
    print("  - Chunk size optimized for GDDR7 bandwidth")
    print("  - Pipeline stages tuned for sm_120")
    print("=" * 90)


if __name__ == "__main__":
    main()
