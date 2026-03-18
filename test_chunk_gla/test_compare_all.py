"""Compare all implementations: FLA baseline vs 3-kernel vs 2-kernel."""
import torch
import torch.nn.functional as F
import triton
import time

from fla.ops.simple_gla import chunk_simple_gla as fla_chunk_simple_gla
from chunk_gla_fused_output import chunk_simple_gla_fused_output, fused_output_final
from chunk_gla_fused_all import chunk_simple_gla_fused_all


def get_test_inputs(seq_len=1024, batch=1, num_heads=32, head_dim=128):
    """Generate test inputs."""
    torch.manual_seed(42)
    
    hidden_size = num_heads * head_dim
    B, T, H, K = batch, seq_len, num_heads, head_dim
    V = K
    
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.1
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda') * 0.1
    
    # Build slopes
    slopes = torch.tensor([1.0 / (2 ** (8 * (i + 1) / H)) for i in range(H)], 
                          device='cuda', dtype=torch.float32)
    g_gamma = -torch.abs(slopes) * 0.01
    scale = K ** -0.5
    
    z = torch.randn(B * T, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.1
    norm_weight = torch.randn(hidden_size, dtype=torch.float32, device='cuda')
    
    return q, k, v, g_gamma, scale, z, norm_weight


def benchmark_func(func, *args, warmup=10, iterations=80):
    """Benchmark a function."""
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations


def test_implementations():
    """Compare all implementations."""
    
    configs = [
        {"seq_len": 64, "num_heads": 32, "head_dim": 128},
        {"seq_len": 128, "num_heads": 32, "head_dim": 128},
        {"seq_len": 256, "num_heads": 32, "head_dim": 128},
        {"seq_len": 512, "num_heads": 32, "head_dim": 128},
        {"seq_len": 1024, "num_heads": 32, "head_dim": 128},
        {"seq_len": 2048, "num_heads": 32, "head_dim": 128},
        {"seq_len": 4096, "num_heads": 32, "head_dim": 128},
        {"seq_len": 8192, "num_heads": 32, "head_dim": 128},
        {"seq_len": 16384, "num_heads": 32, "head_dim": 128},
        {"seq_len": 32768, "num_heads": 32, "head_dim": 128},
        {"seq_len": 65536, "num_heads": 32, "head_dim": 128},
        {"seq_len": 131072, "num_heads": 32, "head_dim": 128},
    ]
    
    print("=" * 100)
    print("Implementation Comparison: FLA Baseline vs 3-Kernel vs 2-Kernel")
    print("=" * 100)
    print("\nNote:")
    print("  - FLA Baseline: FLA chunk_gla (4D) -> reshape -> fused_output_processing")
    print("  - 3-Kernel:     FLA chunk_fwd_h + chunk_fwd_o_fused + fused_output_final")
    print("  - 2-Kernel:     Our _chunk_fwd_h + fused_o_final (single kernel, no o tensor)")
    print()
    print(f"{'Seq Len':<10} {'FLA Base':<12} {'3-Kernel':<12} {'2-Kernel':<12} {'vs FLA':<10} {'vs 3-Ker':<10}")
    print("-" * 100)
    
    for config in configs:
        seq_len = config["seq_len"]
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        
        q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(
            seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
        )
        
        # FLA Baseline
        def run_fla_baseline():
            o_4d, _ = fla_chunk_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
            B, T, H, D = o_4d.shape
            o_2d = o_4d.reshape(-1, H * D)
            return fused_output_final(o_2d, z, norm_weight)
        
        # 3-Kernel
        def run_3kernel():
            return chunk_simple_gla_fused_output(
                q, k, v, z, norm_weight, g_gamma=g_gamma, scale=scale
            )
        
        # 2-Kernel
        def run_2kernel():
            return chunk_simple_gla_fused_all(
                q, k, v, z, norm_weight, g_gamma=g_gamma, scale=scale
            )
        
        t_fla = benchmark_func(run_fla_baseline, warmup=10, iterations=50)
        t_3kernel = benchmark_func(run_3kernel, warmup=10, iterations=50)
        t_2kernel = benchmark_func(run_2kernel, warmup=10, iterations=50)
        
        vs_fla = t_fla / t_2kernel
        vs_3kernel = t_3kernel / t_2kernel
        
        print(f"{seq_len:<10} {t_fla:<12.3f} {t_3kernel:<12.3f} {t_2kernel:<12.3f} {vs_fla:<10.2f}x {vs_3kernel:<10.2f}x")
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_implementations()
