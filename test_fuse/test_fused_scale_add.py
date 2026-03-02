"""
Benchmark script for fused_scale_add on Blackwell (RTX 6000D).

Compares:
1. PyTorch native implementation
2. Blackwell-optimized Triton kernel
"""

import torch
import time
import sys

sys.path.insert(0, '/root/soar2026/python')

from sglang.srt.models.minicpm_fused_scale_add import (
    fused_scale_add,
    fused_scale_add_blackwell,
    fused_scale_add_basic,
    get_gpu_info,
)


def benchmark_kernel(func, input_tensor, residual, scale, warmup=10, iterations=100):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = func(input_tensor, residual, scale)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        _ = func(input_tensor, residual, scale)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms


def main():
    print("=" * 70)
    print("Fused Scale Add Benchmark - Blackwell (RTX 6000D) Optimization")
    print("=" * 70)
    
    # GPU Info
    gpu_info = get_gpu_info()
    print(f"\nGPU Information:")
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")
    
    import triton
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
    
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = 4096
    scale = 0.125
    
    # Test configurations
    test_configs = [
        {"seq_len": 128, "desc": "Small (<=512, PyTorch native)"},
        {"seq_len": 512, "desc": "Medium-small (<=512, PyTorch native)"},
        {"seq_len": 1024, "desc": "Medium (Blackwell optimized)"},
        {"seq_len": 2048, "desc": "Medium-large (Blackwell optimized)"},
        {"seq_len": 4096, "desc": "Large (Blackwell optimized)"},
        {"seq_len": 8192, "desc": "Very large (Basic kernel)"},
        {"seq_len": 16384, "desc": "Huge (Basic kernel)"},
    ]
    
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"{'Seq Len':<10} {'Description':<35} {'PyTorch (ms)':<15} {'Blackwell (ms)':<15} {'Speedup':<10}")
    print("-" * 90)
    
    for config in test_configs:
        seq_len = config["seq_len"]
        desc = config["desc"]
        
        # Generate inputs
        torch.manual_seed(42)
        input_tensor = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        residual = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        
        # Benchmark PyTorch native
        pytorch_time = benchmark_kernel(
            lambda i, r, s: r + i * s,
            input_tensor, residual, scale,
            warmup=10, iterations=50
        )
        
        # Benchmark Blackwell fused
        blackwell_time = benchmark_kernel(
            fused_scale_add,
            input_tensor, residual, scale,
            warmup=10, iterations=50
        )
        
        speedup = pytorch_time / blackwell_time
        
        print(f"{seq_len:<10} {desc:<35} {pytorch_time*1000:>12.3f}   {blackwell_time*1000:>12.3f}   {speedup:>7.2f}x")
    
    # Detailed kernel comparison for medium seq_len
    print("\n" + "=" * 70)
    print("Detailed Kernel Comparison (seq_len=4096)")
    print("=" * 70)
    
    seq_len = 4096
    torch.manual_seed(42)
    input_tensor = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
    residual = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
    
    kernels = [
        ("PyTorch Native", lambda i, r, s: r + i * s),
        ("fused_scale_add (auto)", fused_scale_add),
        ("fused_scale_add_blackwell", fused_scale_add_blackwell),
        ("fused_scale_add_basic", fused_scale_add_basic),
    ]
    
    print(f"{'Kernel':<35} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20}")
    print("-" * 70)
    
    total_bytes = seq_len * hidden_dim * 2 * 3  # 2 bytes per bf16, 3 tensors (read x2, write x1)
    
    for name, kernel in kernels:
        try:
            elapsed = benchmark_kernel(kernel, input_tensor, residual, scale, warmup=10, iterations=100)
            bandwidth = (total_bytes / (elapsed / 1000)) / 1e9  # GB/s
            print(f"{name:<35} {elapsed*1000:>12.3f}   {bandwidth:>16.1f}")
        except Exception as e:
            print(f"{name:<35} ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("Memory Bandwidth Analysis")
    print("=" * 70)
    
    # Theoretical memory bandwidth for RTX 6000D
    theoretical_bw = 960  # GB/s (approximate for RTX 6000D)
    print(f"Theoretical memory bandwidth (RTX 6000D): ~{theoretical_bw} GB/s")
    print(f"Achievable typically: ~80-90% of theoretical")
    
    print("\n" + "=" * 70)
    print("Correctness Verification")
    print("=" * 70)
    
    all_passed = True
    for seq_len in [128, 512, 1024, 2048, 4096, 8192]:
        torch.manual_seed(42)
        input_tensor = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        residual = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        
        ref_out = residual + input_tensor * scale
        fused_out = fused_scale_add(input_tensor, residual, scale)
        
        diff = (fused_out - ref_out).abs().max().item()
        passed = diff < 1e-2
        all_passed = all_passed and passed
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  seq_len={seq_len:>5}: max_diff={diff:.6e} {status}")
    
    print(f"\nOverall: {'All tests PASSED!' if all_passed else 'Some tests FAILED!'}")


if __name__ == "__main__":
    main()
