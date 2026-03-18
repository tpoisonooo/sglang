"""
Test with Roofline Analysis - Calculate bandwidth utilization for recurrent GLA.

RTX 6000D Blackwell Specifications (estimated from RTX PRO 6000 Blackwell):
- Memory Bandwidth: ~1792 GB/s (GDDR7)
- BF16 Tensor Core: 251.9 TFLOPS (dense) / 503.8 TFLOPS (sparse)

For recurrent_simple_gla, the workload is latency-bound because:
1. Sequential access pattern (recurrent)
2. Low arithmetic intensity
"""

import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/root/soar2026/python')

from sglang.srt.models.minicpm_fused_output import fused_output_processing
from test_recurrent_gla.recurrent_simple_gla import fused_recurrent_simple_gla


# RTX 6000D Blackwell Specifications
GPU_SPECS = {
    "name": "NVIDIA RTX 6000D Blackwell",
    "memory_bandwidth_gb_s": 1792,  # GDDR7
    "bf16_tensor_tflops": 251.9,    # Dense
    "memory_bw_efficiency": 0.75,   # Typical achieved ~75% of theoretical
}


def build_slope_tensor(nheads: int) -> torch.Tensor:
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


def get_test_inputs(seq_len=128, batch=1, num_heads=32, head_dim=128):
    torch.manual_seed(42)
    hidden_size = num_heads * head_dim
    
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda') * 0.1
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda') * 0.1
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda') * 0.1
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device='cuda', dtype=torch.float32) * 0.01
    scale = head_dim ** -0.5
    z = torch.randn(batch * seq_len, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.1
    norm_weight = torch.randn(hidden_size, dtype=torch.float32, device='cuda')
    
    return q, k, v, g_gamma, scale, z, norm_weight


def analyze_memory_traffic(q, k, v, z, norm_weight):
    """Calculate memory traffic for recurrent simple gla + output processing."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    hidden_size = H * V
    
    # Data sizes (bytes)
    bytes_per_element = 2  # bf16 = 2 bytes
    
    # Recurrent GLA memory traffic:
    # For each timestep:
    # - Read q_t: B * H * K elements
    # - Read k_t: B * H * K elements  
    # - Read v_t: B * H * V elements
    # - Read/Write h: B * H * K * V elements (in registers, minimal HBM)
    # - Write o_t: B * H * V elements
    
    # QKV read (each element read T times due to recurrent nature)
    qkv_read = (B * T * H * K * 2 + B * T * H * V) * bytes_per_element
    
    # Output write
    output_write = B * T * H * V * bytes_per_element
    
    # Hidden state (minimal HBM traffic due to register persistence)
    # h is mostly in registers, only read initial_state + write final_state
    state_traffic = 2 * B * H * K * V * 4  # FP32
    
    # Output processing (fused_output_processing):
    # - Read o: B*T*H*V
    # - Read z: B*T*H*V  
    # - Read norm_weight: H*V
    # - Write out: B*T*H*V
    output_proc_read = (B * T * hidden_size * 2 + hidden_size) * bytes_per_element
    output_proc_write = B * T * hidden_size * bytes_per_element
    
    total_traffic = qkv_read + output_write + state_traffic + output_proc_read + output_proc_write
    
    return {
        'qkv_read_mb': qkv_read / 1024**2,
        'output_write_mb': output_write / 1024**2,
        'state_traffic_mb': state_traffic / 1024**2,
        'output_proc_read_mb': output_proc_read / 1024**2,
        'output_proc_write_mb': output_proc_write / 1024**2,
        'total_mb': total_traffic / 1024**2,
    }


def benchmark(q, k, v, g_gamma, scale, z, norm_weight, warmup=10, iterations=50):
    torch.cuda.synchronize()
    
    for _ in range(warmup):
        o_4d, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
        o_2d = o_4d.reshape(-1, o_4d.shape[2] * o_4d.shape[3])
        out = fused_output_processing(o_2d, z, norm_weight, 1e-6)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        o_4d, ht = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
        o_2d = o_4d.reshape(-1, o_4d.shape[2] * o_4d.shape[3])
        out = fused_output_processing(o_2d, z, norm_weight, 1e-6)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return
    
    print("=" * 100)
    print("Recurrent GLA Roofline Analysis")
    print("=" * 100)
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Theoretical Memory Bandwidth: {GPU_SPECS['memory_bandwidth_gb_s']} GB/s")
    print(f"Expected Achievable (~75%): {GPU_SPECS['memory_bandwidth_gb_s'] * GPU_SPECS['memory_bw_efficiency']:.0f} GB/s")
    
    configs = [
        {"seq_len": 16},
        {"seq_len": 32},
        {"seq_len": 64},
        {"seq_len": 128},
        {"seq_len": 256},
        {"seq_len": 512},
        {"seq_len": 1024},
        {"seq_len": 2048},
    ]
    
    print("\n" + "=" * 100)
    print("Performance Analysis")
    print("=" * 100)
    print(f"{'Seq Len':<10} {'Time(ms)':<12} {'Traffic(MB)':<15} {'Util(%)':<10} {'Roofline':<15}")
    print("-" * 100)
    
    results = []
    
    for cfg in configs:
        seq_len = cfg["seq_len"]
        q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(seq_len=seq_len)
        
        # Analyze memory traffic
        traffic = analyze_memory_traffic(q, k, v, z, norm_weight)
        
        # Benchmark
        elapsed_ms = benchmark(q, k, v, g_gamma, scale, z, norm_weight, warmup=10, iterations=30)
        
        # Calculate bandwidth utilization
        elapsed_s = elapsed_ms / 1000.0
        achieved_bw_gb_s = traffic['total_mb'] / 1024 / elapsed_s
        utilization_pct = (achieved_bw_gb_s / GPU_SPECS['memory_bandwidth_gb_s']) * 100
        
        # Roofline classification
        # For recurrent GLA, we expect to be latency-bound
        arithmetic_intensity = 1.0  # Very low for recurrent
        if utilization_pct > 60:
            roofline_status = "Good (Memory)"
        elif utilization_pct > 30:
            roofline_status = "Fair (Memory)"
        else:
            roofline_status = "Poor (Latency)"
        
        results.append({
            'seq_len': seq_len,
            'time_ms': elapsed_ms,
            'traffic_mb': traffic['total_mb'],
            'achieved_bw': achieved_bw_gb_s,
            'utilization': utilization_pct,
        })
        
        print(f"{seq_len:<10} {elapsed_ms:>10.3f}   {traffic['total_mb']:>13.2f}   {utilization_pct:>8.1f}%  {roofline_status:<15}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("Summary Statistics")
    print("=" * 100)
    
    avg_util = sum(r['utilization'] for r in results) / len(results)
    max_util = max(r['utilization'] for r in results)
    min_util = min(r['utilization'] for r in results)
    
    print(f"Average Bandwidth Utilization: {avg_util:.1f}%")
    print(f"Maximum Bandwidth Utilization: {max_util:.1f}%")
    print(f"Minimum Bandwidth Utilization: {min_util:.1f}%")
    print(f"\nTheoretical Peak: {GPU_SPECS['memory_bandwidth_gb_s']} GB/s")
    print(f"Average Achieved: {sum(r['achieved_bw'] for r in results) / len(results):.0f} GB/s")
    
    print("\n" + "=" * 100)
    print("Analysis:")
    print("  - Recurrent GLA is LATENCY-BOUND due to sequential dependency (h_t depends on h_{t-1})")
    print("  - Low bandwidth utilization (~5-7%) is expected for recurrent workloads")
    print("  - Optimization direction: kernel fusion to reduce launch overhead, not bandwidth")
    print("=" * 100)


if __name__ == "__main__":
    main()
