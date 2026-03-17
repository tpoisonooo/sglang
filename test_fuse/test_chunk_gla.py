"""
Benchmark script for chunk_simple_gla kernel on Blackwell (RTX 6000D).

Compares:
1. Our optimized implementation (sglang) - with internal chunk_size selection
2. FLA official implementation - with internal chunk_size selection
"""

import torch
import time
import sys
import math

sys.path.insert(0, '/root/soar2026/python')

# Our implementation
from sglang.srt.layers.attention.minicpm_chunk_gla import (
    chunk_simple_gla as our_chunk_simple_gla,
    IS_NVIDIA_BLACKWELL,
    get_optimal_chunk_size,
)

# FLA official implementation
from fla.ops.simple_gla import chunk_simple_gla as fla_chunk_simple_gla


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


def benchmark_our_impl(q, k, v, g_gamma, scale, warmup=25, iterations=200):
    """Benchmark our optimized chunk_simple_gla implementation."""
    torch.cuda.synchronize()
    
    # Clear L2 cache before benchmarking for consistent results
    cache_size = 256 * 1024 * 1024  # 256MB cache flush
    _cache_tensor = torch.empty(cache_size, dtype=torch.uint8, device=q.device)
    
    for _ in range(warmup):
        _, _ = our_chunk_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        o, ht = our_chunk_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    del _cache_tensor
    
    return elapsed_ms, o, ht


def benchmark_fla_impl(q, k, v, g_gamma, scale, warmup=25, iterations=200):
    """Benchmark FLA official chunk_simple_gla implementation."""
    torch.cuda.synchronize()
    
    # Clear L2 cache before benchmarking for consistent results
    cache_size = 256 * 1024 * 1024  # 256MB cache flush
    _cache_tensor = torch.empty(cache_size, dtype=torch.uint8, device=q.device)
    
    for _ in range(warmup):
        _, _ = fla_chunk_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        o, ht = fla_chunk_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    del _cache_tensor
    
    return elapsed_ms, o, ht


def compare_accuracy(o_our, o_fla, ht_our, ht_fla):
    """Compare accuracy between our and FLA implementations."""
    output_diff = (o_our - o_fla).abs().max().item()
    state_diff = (ht_our - ht_fla).abs().max().item()
    
    output_scale = o_fla.abs().mean().item() + 1e-6
    state_scale = ht_fla.abs().mean().item() + 1e-6
    output_rel_diff = ((o_our - o_fla).abs() / output_scale).max().item()
    state_rel_diff = ((ht_our - ht_fla).abs() / state_scale).max().item()
    
    output_mae = (o_our - o_fla).abs().mean().item()
    state_mae = (ht_our - ht_fla).abs().mean().item()
    
    has_nan = (torch.isnan(o_our).any() or torch.isnan(o_fla).any() or 
               torch.isnan(ht_our).any() or torch.isnan(ht_fla).any())
    
    return {
        'output_abs_diff': output_diff,
        'output_rel_diff': output_rel_diff,
        'state_rel_diff': state_rel_diff,
        'output_mae': output_mae,
        'state_mae': state_mae,
        'has_nan': has_nan,
    }


def main():
    print("=" * 90)
    print("Chunk Simple GLA Benchmark - Comparing Our Implementation vs FLA Official")
    print("=" * 90)
    
    # GPU Info
    print(f"\nGPU Information:")
    print(f"  Device: {torch.cuda.get_device_name()}")
    major, minor = torch.cuda.get_device_capability()
    print(f"  Compute Capability: {major}.{minor}")
    print(f"  Is Blackwell: {IS_NVIDIA_BLACKWELL}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    import triton
    print(f"Triton version: {triton.__version__}")
    
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    num_heads = 32
    head_dim = 128
    
    # Build g_gamma
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device=device, dtype=torch.float32) * 0.01
    scale = head_dim ** -0.5
    
    # Test different sequence lengths
    seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    print("\n" + "=" * 90)
    print("Performance & Accuracy Comparison")
    print("(Both implementations use internal chunk_size selection)")
    print("=" * 90)
    print(f"{'Seq Len':<10} {'Chunk Size':<12} {'Our (ms)':<12} {'FLA (ms)':<12} {'Speedup':<10} "
          f"{'Out MAE':<12} {'State MAE':<12} {'Status':<10}")
    print("-" * 90)
    
    for seq_len in seq_lens:
        torch.manual_seed(42)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        
        chunk_size = get_optimal_chunk_size(seq_len)
        
        try:
            # Benchmark our implementation
            t_our, o_our, ht_our = benchmark_our_impl(
                q, k, v, g_gamma, scale, warmup=5, iterations=20
            )
            
            # Benchmark FLA implementation
            t_fla, o_fla, ht_fla = benchmark_fla_impl(
                q, k, v, g_gamma, scale, warmup=5, iterations=20
            )
            
            # Compare accuracy
            diff_info = compare_accuracy(o_our, o_fla, ht_our, ht_fla)
            speedup = t_fla / t_our
            
            # Determine status
            if diff_info['has_nan']:
                status = "NAN"
            elif diff_info['output_mae'] > 0.5 or diff_info['state_mae'] > 1.0:
                status = "HIGH_ERR"
            elif diff_info['output_mae'] > 0.05 or diff_info['state_mae'] > 0.1:
                status = "WARN"
            else:
                status = "PASS"
            
            print(f"{seq_len:<10} {chunk_size:<12} {t_our:>10.3f}   {t_fla:>10.3f}   {speedup:>8.2f}x   "
                  f"{diff_info['output_mae']:>10.2e}   {diff_info['state_mae']:>10.2e}   {status}")
            
        except Exception as e:
            print(f"{seq_len:<10} {chunk_size:<12} {'ERROR':<12} {'ERROR':<12} {'N/A':<10} "
                  f"{'N/A':<12} {'N/A':<12} {'FAILED':<10}")
            print(f"  Error: {str(e)[:60]}")
    
    # Memory bandwidth analysis for seq_len=1024
    print("\n" + "=" * 90)
    print("Memory Bandwidth Analysis (seq_len=1024)")
    print("=" * 90)
    
    seq_len = 1024
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    
    # Calculate memory traffic
    q_size = q.numel() * 2  # bf16 = 2 bytes
    k_size = k.numel() * 2
    v_size = v.numel() * 2
    o_size = v.numel() * 2  # output same size as v
    total_bytes = q_size + k_size + v_size + o_size
    
    chunk_size = get_optimal_chunk_size(seq_len)
    
    # Our implementation
    t_our, _, _ = benchmark_our_impl(q, k, v, g_gamma, scale, warmup=10, iterations=50)
    our_bandwidth_gbps = (total_bytes / (t_our / 1000)) / 1e9
    
    # FLA implementation
    t_fla, _, _ = benchmark_fla_impl(q, k, v, g_gamma, scale, warmup=10, iterations=50)
    fla_bandwidth_gbps = (total_bytes / (t_fla / 1000)) / 1e9
    
    print(f"Memory Traffic: {total_bytes / 1e6:.2f} MB")
    print(f"\nOur Implementation:")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Time: {t_our:.3f} ms")
    print(f"  Memory Bandwidth: {our_bandwidth_gbps:.1f} GB/s")
    
    print(f"\nFLA Implementation:")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Time: {t_fla:.3f} ms")
    print(f"  Memory Bandwidth: {fla_bandwidth_gbps:.1f} GB/s")
    
    print(f"\nRTX 6000D Theoretical: ~960 GB/s")
    print(f"Our Efficiency: {our_bandwidth_gbps / 960 * 100:.1f}%")
    print(f"FLA Efficiency: {fla_bandwidth_gbps / 960 * 100:.1f}%")
    
    # Summary
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print("Our Implementation:")
    if IS_NVIDIA_BLACKWELL:
        print(f"  - Blackwell optimized with autotune")
        print(f"  - More warps options (8, 16) and stages (2-4) for better occupancy")
    print(f"  - Increased autotune warmup (20) and rep (50) for accurate selection")
    print(f"  - Same chunk_size selection logic as FLA (internal)")
    print(f"  - Dead code elimination for unused decay computations")
    print(f"  - Boundary check elimination when dimensions are aligned")
    print(f"  - tl.assume() hints for better compiler optimization")
    
    print("\nFLA Official Implementation:")
    print(f"  - Generic implementation supporting all features")
    print(f"  - Standard chunk_size selection")
    print(f"  - Full autograd support")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)
    
    main()
