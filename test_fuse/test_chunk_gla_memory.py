"""
Benchmark script for chunk_simple_gla kernel - Memory Usage Comparison

Compares GPU memory usage between:
1. Our optimized implementation (sglang)
2. FLA official implementation
"""

import torch
import time
import sys
import math
import gc

sys.path.insert(0, '/root/soar2026/python')

# Our implementation
from sglang.srt.layers.attention.minicpm_chunk_gla import (
    chunk_simple_gla as our_chunk_simple_gla,
    chunk_simple_gla_fwd as our_chunk_simple_gla_fwd,
    IS_NVIDIA_BLACKWELL,
    DEFAULT_CHUNK_SIZE,
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


def get_gpu_memory_info():
    """Get current GPU memory info in MiB."""
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    return allocated, reserved, max_allocated


def reset_peak_memory_stats():
    """Reset peak memory stats."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def benchmark_memory(impl_name, fn, *args, **kwargs):
    """Benchmark memory usage for a function."""
    # Clean up before measurement
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Reset peak memory stats
    reset_peak_memory_stats()
    
    # Record memory before
    mem_before = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Run the function
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Record memory after
    mem_after = torch.cuda.memory_allocated() / 1024 / 1024
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Calculate memory usage
    delta_mem = mem_after - mem_before
    
    return {
        'name': impl_name,
        'mem_before_mb': mem_before,
        'mem_after_mb': mem_after,
        'delta_mem_mb': delta_mem,
        'peak_mem_mb': peak_mem,
        'result': result,
    }


def main():
    print("=" * 100)
    print("Chunk Simple GLA Memory Benchmark - Comparing Our Implementation vs FLA Official")
    print("=" * 100)
    
    # GPU Info
    print(f"\nGPU Information:")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"  Is Blackwell: {IS_NVIDIA_BLACKWELL}")
    print(f"  Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GiB")
    
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
    
    # Test configurations with increasing sequence lengths
    test_configs = [
        {"seq_len": 1024, "our_chunk_size": 128},
        {"seq_len": 4096, "our_chunk_size": 128},
        {"seq_len": 8192, "our_chunk_size": 128},
        {"seq_len": 16384, "our_chunk_size": 128},
        {"seq_len": 32768, "our_chunk_size": 128},
        {"seq_len": 65536, "our_chunk_size": 128},
    ]
    
    print("\n" + "=" * 100)
    print("Memory Usage Comparison (Measured at Runtime)")
    print("=" * 100)
    print(f"{'Seq Len':<10} {'Impl':<12} {'Before(MB)':<12} {'After(MB)':<12} {'Delta(MB)':<12} {'Peak(MB)':<12} {'Status':<10}")
    print("-" * 100)
    
    for config in test_configs:
        seq_len = config["seq_len"]
        our_chunk_size = config["our_chunk_size"]
        
        # Estimate input size
        input_size_mb = (batch * seq_len * num_heads * head_dim * 3 * 2) / 1024 / 1024  # bf16 = 2 bytes
        
        torch.manual_seed(42)
        try:
            q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
            k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
            v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
            
            # Benchmark our implementation
            our_result = benchmark_memory(
                "Our",
                our_chunk_simple_gla_fwd,
                q, k, v,
                g_gamma=g_gamma,
                scale=scale,
                chunk_size=our_chunk_size
            )
            
            # Clean up our result
            del our_result['result']
            torch.cuda.empty_cache()
            
            # Benchmark FLA implementation
            fla_result = benchmark_memory(
                "FLA",
                fla_chunk_simple_gla,
                q, k, v,
                g_gamma=g_gamma,
                scale=scale,
                output_final_state=True
            )
            
            # Clean up FLA result
            del fla_result['result']
            torch.cuda.empty_cache()
            
            # Print results
            print(f"{seq_len:<10} {'Ours':<12} {our_result['mem_before_mb']:>10.1f}   {our_result['mem_after_mb']:>10.1f}   "
                  f"{our_result['delta_mem_mb']:>10.1f}   {our_result['peak_mem_mb']:>10.1f}   {'OK':<10}")
            print(f"{'':<10} {'FLA':<12} {fla_result['mem_before_mb']:>10.1f}   {fla_result['mem_after_mb']:>10.1f}   "
                  f"{fla_result['delta_mem_mb']:>10.1f}   {fla_result['peak_mem_mb']:>10.1f}   {'OK':<10}")
            
            # Print comparison
            mem_diff = our_result['peak_mem_mb'] - fla_result['peak_mem_mb']
            if mem_diff > 0:
                print(f"{'':<10} {'=> Ours uses':<12} {mem_diff:>10.1f} MB MORE than FLA")
            else:
                print(f"{'':<10} {'=> Ours uses':<12} {abs(mem_diff):>10.1f} MB LESS than FLA")
            print()
            
            # Clean up tensors
            del q, k, v
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"{seq_len:<10} {'N/A':<12} {'OOM':<12} {'OOM':<12} {'N/A':<12} {'N/A':<12} {'OOM':<10}")
            print(f"  Error: CUDA OOM at seq_len={seq_len}")
            # Clean up any remaining tensors
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"{seq_len:<10} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {str(e)[:10]:<10}")
    
    print("\n" + "=" * 100)
    print("Memory Analysis Summary")
    print("=" * 100)
    print(f"Input tensor size per seq_len:")
    print(f"  - 1024:  ~{(1024 * num_heads * head_dim * 3 * 2) / 1024 / 1024:.1f} MB")
    print(f"  - 32768: ~{(32768 * num_heads * head_dim * 3 * 2) / 1024 / 1024:.1f} MB")
    print(f"  - 65536: ~{(65536 * num_heads * head_dim * 3 * 2) / 1024 / 1024:.1f} MB")
    
    print("\nNotes:")
    print("  - 'Before' = Memory allocated before kernel execution")
    print("  - 'After'  = Memory allocated after kernel execution")  
    print("  - 'Delta'  = Additional memory allocated during execution")
    print("  - 'Peak'   = Maximum memory allocated during execution (includes intermediate tensors)")
    print("\nBenchmark completed!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)
    
    main()
