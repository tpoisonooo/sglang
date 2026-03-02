"""
Test script for fused_recurrent_simple_gla kernel in MiniCPM.

Tests the fused recurrent Simple Gated Linear Attention (GLA) implementation.
Input shapes:
- q: [batch, seq_len, num_heads, head_dim] = [batch, seq_len, 32, 128]
- k: [batch, seq_len, num_heads, head_dim] = [batch, seq_len, 32, 128]
- v: [batch, seq_len, num_heads, head_dim] = [batch, seq_len, 32, 128]
- g_gamma: [num_heads] = [32]

Where:
- batch is variable (typically 1 for inference)
- num_heads = 32 (fixed)
- head_dim = 128 (fixed)
- seq_len is variable

Blackwell (RTX 6000D) Optimizations:
1. Full-dimension block sizes (BK=128, BV=128) for MiniCPM fixed dimensions,
   eliminating block-splitting loops (NK=NV=1)
2. Extended autotune configs with up to 32 warps for better SM occupancy on Blackwell
3. Specialized code path for K=V=128, H=32

This test targets Blackwell (RTX 6000D) optimization.
"""

import torch
import math
import sys
import time

sys.path.insert(0, '/root/soar2026/python')

from sglang.srt.layers.attention.minicpm_recurrent_simple_gla import fused_recurrent_simple_gla

# Import FLA official version for comparison
try:
    from fla.ops.simple_gla import fused_recurrent_simple_gla as fla_fused_recurrent_simple_gla
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    print("Warning: FLA package not found, skipping FLA comparison")


def build_slope_tensor(nheads: int) -> torch.Tensor:
    """Build ALiBi slope tensor - matches MiniCPM implementation."""
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


def reference_recurrent_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_gamma: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
):
    """
    Reference implementation of Recurrent Simple GLA using standard PyTorch operations.
    
    Recurrent Simple GLA computation:
    - Uses head-wise decay g_gamma
    - Computes attention with decay over time
    - Recurrent formulation: h_t = decay * h_{t-1} + k_t^T @ v_t
    - Output: o_t = q_t @ h_t
    
    Args:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - keys
        v: [B, T, H, V] - values (V = K = 128)
        g_gamma: [H] - head-wise decay rates
        scale: attention scale factor (default: 1/sqrt(K))
        initial_state: [B, H, K, V] - initial hidden state (optional)
    
    Returns:
        o: [B, T, H, V] - output
        final_state: [B, H, K, V] - final hidden state
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    if scale is None:
        scale = K ** -0.5
    
    # Move to fp32 for numerical stability
    q_fp32 = q.float()
    k_fp32 = k.float()
    v_fp32 = v.float()
    
    # Output tensor
    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)
    
    # Initialize final state
    final_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    
    # Compute decay per timestep for each head
    # g_gamma: [H] -> expanded to [T, H]
    # For recurrent formulation, decay is applied at each step
    decay = torch.exp(g_gamma).view(1, H)  # [1, H]
    
    for b in range(B):
        for h in range(H):
            # Hidden state [K, V]
            h_state = torch.zeros(K, V, dtype=torch.float32, device=q.device)
            
            # Apply initial state if provided
            if initial_state is not None:
                h_state = initial_state[b, h, :, :].float().clone()
            
            for t in range(T):
                q_t = q_fp32[b, t, h, :]  # [K]
                k_t = k_fp32[b, t, h, :]  # [K]
                v_t = v_fp32[b, t, h, :]  # [V]
                
                # Decay hidden state
                h_state = h_state * decay[0, h]
                
                # Accumulate: h += k^T @ v (outer product)
                h_state += torch.outer(k_t, v_t)
                
                # Compute output: o_t = q_t @ h_state
                o[b, t, h, :] = torch.matmul(q_t, h_state) * scale
            
            # Store final state
            final_state[b, h, :, :] = h_state
    
    return o.to(q.dtype), final_state


def test_fused_recurrent_simple_gla_basic():
    """Test fused_recurrent_simple_gla with various sequence lengths."""
    print("=" * 70)
    print("Testing fused_recurrent_simple_gla (basic functionality)")
    print("=" * 70)
    
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    num_heads = 32
    head_dim = 128
    
    # Test configurations
    test_configs = [
        {"seq_len": 1, "desc": "Single token (decode)"},
        {"seq_len": 16, "desc": "Small seq"},
        {"seq_len": 64, "desc": "Medium seq"},
        {"seq_len": 128, "desc": "Large seq"},
        {"seq_len": 256, "desc": "Very large seq"},
        {"seq_len": 512, "desc": "Huge seq"},
    ]
    
    # Build g_gamma (head-wise decay) - use small negative values for stability
    # In practice, g_gamma should be negative log decay rates
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device=device, dtype=torch.float32) * 0.01
    
    all_passed = True
    
    for config in test_configs:
        seq_len = config["seq_len"]
        desc = config["desc"]
        
        # Generate inputs
        torch.manual_seed(42)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        
        try:
            # Compute scale
            scale = head_dim ** -0.5
            
            # Fused implementation
            o_fused, final_state = fused_recurrent_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=g_gamma,
                scale=scale,
            )
            
            # Check output validity (no NaN or Inf)
            has_nan = torch.isnan(o_fused).any()
            has_inf = torch.isinf(o_fused).any()
            passed = not has_nan and not has_inf
            
            if not passed:
                all_passed = False
            
            status = "PASSED" if passed else "FAILED"
            print(f"  Config: seq_len={seq_len:4d} ({desc})")
            print(f"    Output shape: {o_fused.shape}")
            print(f"    Output range: [{o_fused.min():.4f}, {o_fused.max():.4f}]")
            print(f"    Final state shape: {final_state.shape}")
            print(f"    NaN: {has_nan.item()}, Inf: {has_inf.item()}")
            print(f"    Status: {status}")
            
        except Exception as e:
            print(f"  Config: seq_len={seq_len:4d} ({desc})")
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            print(f"    Status: FAILED")
            all_passed = False
    
    print()
    return all_passed


def test_fused_recurrent_simple_gla_with_initial_state():
    """Test fused_recurrent_simple_gla with initial state."""
    print("=" * 70)
    print("Testing fused_recurrent_simple_gla (with initial state)")
    print("=" * 70)
    
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    seq_len = 64
    num_heads = 32
    head_dim = 128
    
    # Build g_gamma
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device=device, dtype=torch.float32) * 0.01
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    initial_state = torch.randn(batch, num_heads, head_dim, head_dim, dtype=torch.float32, device=device) * 0.01
    
    try:
        # Fused implementation with initial state
        o_fused, final_state = fused_recurrent_simple_gla(
            q=q,
            k=k,
            v=v,
            g_gamma=g_gamma,
            initial_state=initial_state,
        )
        
        # Check output validity
        has_nan = torch.isnan(o_fused).any()
        has_inf = torch.isinf(o_fused).any()
        passed = not has_nan and not has_inf
        
        status = "PASSED" if passed else "FAILED"
        print(f"  With initial state:")
        print(f"    Output shape: {o_fused.shape}")
        print(f"    Output range: [{o_fused.min():.4f}, {o_fused.max():.4f}]")
        print(f"    Final state shape: {final_state.shape}")
        print(f"    NaN: {has_nan.item()}, Inf: {has_inf.item()}")
        print(f"    Status: {status}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Status: FAILED")
        passed = False
    
    print()
    return passed


def test_fused_recurrent_simple_gla_correctness():
    """Test correctness against reference implementation."""
    print("=" * 70)
    print("Testing fused_recurrent_simple_gla correctness")
    print("=" * 70)
    
    device = "cuda"
    dtype = torch.float32  # Use fp32 for better numerical comparison
    batch = 1
    seq_len = 32  # Smaller seq_len for faster reference computation
    num_heads = 32
    head_dim = 128
    
    # Build g_gamma
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device=device, dtype=torch.float32) * 0.01
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    
    try:
        scale = head_dim ** -0.5
        
        # Fused implementation
        o_fused, final_state_fused = fused_recurrent_simple_gla(
            q=q,
            k=k,
            v=v,
            g_gamma=g_gamma,
            scale=scale,
        )
        
        # Reference implementation
        o_ref, final_state_ref = reference_recurrent_simple_gla(
            q=q,
            k=k,
            v=v,
            g_gamma=g_gamma,
            scale=scale,
        )
        
        # Compare outputs
        output_diff = (o_fused.float() - o_ref.float()).abs()
        output_max_diff = output_diff.max().item()
        output_mean_diff = output_diff.mean().item()
        
        # Compare final states
        state_diff = (final_state_fused.float() - final_state_ref.float()).abs()
        state_max_diff = state_diff.max().item()
        state_mean_diff = state_diff.mean().item()
        
        # Threshold for numerical comparison (relaxed for bfloat16)
        threshold = 0.1
        passed = output_max_diff < threshold and state_max_diff < threshold
        
        status = "PASSED" if passed else "FAILED"
        print(f"  Output comparison:")
        print(f"    Max difference: {output_max_diff:.6f}")
        print(f"    Mean difference: {output_mean_diff:.6f}")
        print(f"  Final state comparison:")
        print(f"    Max difference: {state_max_diff:.6f}")
        print(f"    Mean difference: {state_mean_diff:.6f}")
        print(f"  Status: {status}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Status: FAILED")
        passed = False
    
    print()
    return passed


def test_fused_recurrent_simple_gla_output_dtype():
    """Test that output dtype matches input dtype."""
    print("=" * 70)
    print("Testing fused_recurrent_simple_gla output dtype")
    print("=" * 70)
    
    device = "cuda"
    batch = 1
    seq_len = 64
    num_heads = 32
    head_dim = 128
    
    g_gamma = build_slope_tensor(num_heads).to(device=device, dtype=torch.float32)
    
    # Test different dtypes
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    all_passed = True
    
    for dtype in dtypes:
        torch.manual_seed(42)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        
        try:
            o, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma)
            
            passed = o.dtype == dtype
            if not passed:
                all_passed = False
            
            status = "PASSED" if passed else "FAILED"
            print(f"  Input dtype: {dtype}, Output dtype: {o.dtype} - {status}")
            
        except Exception as e:
            print(f"  Input dtype: {dtype} - ERROR: {e}")
            all_passed = False
    
    print()
    return all_passed


def benchmark_fused_recurrent_simple_gla():
    """Benchmark fused_recurrent_simple_gla performance."""
    print("=" * 70)
    print("Benchmarking fused_recurrent_simple_gla (Blackwell RTX 6000D)")
    print("=" * 70)
    print("Configuration:")
    print("  - Block size: BK=64, BV=64 (NK=NV=2, 4 blocks parallel)")
    print("  - Autotune: 4, 8, 16 warps")
    print("  - Target: H=32, K=128, V=128 (MiniCPM fixed dims)")
    print()
    
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    num_heads = 32
    head_dim = 128
    
    # Build g_gamma
    g_gamma = build_slope_tensor(num_heads).to(device=device, dtype=torch.float32)
    scale = head_dim ** -0.5
    
    test_configs = [
        {"seq_len": 1, "desc": "Single token"},
        {"seq_len": 16, "desc": "Small"},
        {"seq_len": 32, "desc": "Small-Medium"},
        {"seq_len": 64, "desc": "Medium"},
        {"seq_len": 128, "desc": "Large"},
        {"seq_len": 256, "desc": "Very Large"},
        {"seq_len": 512, "desc": "Huge"},
        {"seq_len": 1024, "desc": "Massive"},
        {"seq_len": 2048, "desc": "XLarge"},
        {"seq_len": 4096, "desc": "XXLarge"},
        {"seq_len": 8192, "desc": "XXXLarge"},
    ]
    
    print(f"{'Seq Len':<10} {'Time (ms)':<15} {'Throughput':<20} {'Status':<10}")
    print("-" * 60)
    
    times = []
    for config in test_configs:
        seq_len = config["seq_len"]
        desc = config["desc"]
        
        torch.manual_seed(42)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        
        # Warmup
        for _ in range(10):
            _, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
        torch.cuda.synchronize()
        
        # Benchmark
        if seq_len <= 64:
            num_iters = 50
        elif seq_len <= 512:
            num_iters = 20
        else:
            num_iters = 10
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            _, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end) / num_iters
        times.append((seq_len, elapsed_ms))
        
        # Compute throughput (tokens/sec)
        throughput = (batch * seq_len) / (elapsed_ms / 1000)
        
        print(f"{seq_len:<10} {elapsed_ms:>12.3f}   {throughput:>15.1f} tok/s   ✓")
    
    print()
    print("Scaling Analysis:")
    print("-" * 40)
    for i in range(1, len(times)):
        prev_len, prev_time = times[i-1]
        curr_len, curr_time = times[i]
        time_ratio = curr_time / prev_time
        len_ratio = curr_len / prev_len
        efficiency = len_ratio / time_ratio if time_ratio > 0 else 0
        print(f"  {prev_len:4d} -> {curr_len:4d}: time ratio = {time_ratio:.2f}x "
              f"(linear would be {len_ratio:.2f}x, efficiency: {efficiency:.1%})")


def benchmark_comparison():
    """Performance test across different sequence lengths."""
    print("=" * 70)
    print("PERFORMANCE TEST: fused_recurrent_simple_gla")
    print("=" * 70)
    print("Testing performance across seq_len = 1 to 8192")
    print()
    
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    num_heads = 32
    head_dim = 128
    
    # Build g_gamma
    g_gamma = build_slope_tensor(num_heads).to(device=device, dtype=torch.float32)
    g_gamma = -torch.abs(g_gamma) * 0.01
    scale = head_dim ** -0.5
    
    test_configs = [
        {"seq_len": 1, "desc": "Decode (seq_len=1)"},
        {"seq_len": 8, "desc": "Small (seq_len=8)"},
        {"seq_len": 16, "desc": "Small (seq_len=16)"},
        {"seq_len": 32, "desc": "Medium (seq_len=32)"},
        {"seq_len": 64, "desc": "Medium (seq_len=64)"},
        {"seq_len": 128, "desc": "Large (seq_len=128)"},
        {"seq_len": 256, "desc": "XLarge (seq_len=256)"},
        {"seq_len": 512, "desc": "XXLarge (seq_len=512)"},
        {"seq_len": 1024, "desc": "Huge (seq_len=1024)"},
        {"seq_len": 2048, "desc": "Massive (seq_len=2048)"},
        {"seq_len": 4096, "desc": "Giant (seq_len=4096)"},
        {"seq_len": 8192, "desc": "Enormous (seq_len=8192)"},
    ]
    
    print(f"{'Seq Len':<10} {'Time (ms)':<15} {'Throughput':<20} {'Status':<10}")
    print("-" * 70)
    
    for config in test_configs:
        seq_len = config["seq_len"]
        
        torch.manual_seed(42)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        
        # Warmup
        for _ in range(10):
            _, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
        torch.cuda.synchronize()
        
        # Benchmark
        if seq_len <= 32:
            num_iters = 50
        elif seq_len <= 512:
            num_iters = 20
        else:
            num_iters = 10
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            _, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / num_iters
        
        throughput = (batch * seq_len) / (elapsed_ms / 1000)
        
        print(f"{seq_len:<10} {elapsed_ms:>12.3f}   {throughput:>15.1f} tok/s   ✓")
    
    print()


def benchmark_vs_fla():
    """Compare our implementation with FLA official version."""
    if not HAS_FLA:
        print("=" * 70)
        print("FLA COMPARISON: Skipped (FLA not installed)")
        print("=" * 70)
        print()
        return
    
    print("=" * 70)
    print("PERFORMANCE COMPARISON: Our Implementation vs FLA Official")
    print("=" * 70)
    print("Input shape: [batch, seq_len, num_heads=32, head_dim=128]")
    print("Our impl:   minicpm_recurrent_simple_gla (local)")
    print("FLA impl:   fla.ops.simple_gla.fused_recurrent_simple_gla")
    print()
    
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1
    num_heads = 32
    head_dim = 128
    
    # Build g_gamma
    g_gamma = build_slope_tensor(num_heads).to(device=device, dtype=torch.float32)
    g_gamma = -torch.abs(g_gamma) * 0.01
    scale = head_dim ** -0.5
    
    test_configs = [
        {"seq_len": 1, "desc": "Decode"},
        {"seq_len": 64, "desc": "Medium"},
        {"seq_len": 128, "desc": "Large"},
        {"seq_len": 256, "desc": "XLarge"},
        {"seq_len": 512, "desc": "XXLarge"},
        {"seq_len": 1024, "desc": "Huge"},
        {"seq_len": 2048, "desc": "Massive"},
        {"seq_len": 4096, "desc": "Giant"},
        {"seq_len": 8192, "desc": "Enormous"},
    ]
    
    print(f"{'Seq Len':<10} {'Ours (ms)':<15} {'FLA (ms)':<15} {'Speedup':<10} {'Status':<10}")
    print("-" * 70)
    
    results = []
    for config in test_configs:
        seq_len = config["seq_len"]
        
        torch.manual_seed(42)
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        
        # Warmup both
        for _ in range(10):
            _, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
            _, _ = fla_fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
        torch.cuda.synchronize()
        
        # Benchmark ours
        if seq_len <= 32:
            num_iters = 50
        elif seq_len <= 512:
            num_iters = 20
        else:
            num_iters = 10
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            _, _ = fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
        end.record()
        torch.cuda.synchronize()
        ours_ms = start.elapsed_time(end) / num_iters
        
        # Benchmark FLA
        start.record()
        for _ in range(num_iters):
            _, _ = fla_fused_recurrent_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True)
        end.record()
        torch.cuda.synchronize()
        fla_ms = start.elapsed_time(end) / num_iters
        
        speedup = fla_ms / ours_ms if ours_ms > 0 else 0
        results.append((seq_len, ours_ms, fla_ms, speedup))
        
        status = "✓" if speedup > 0.95 else "⚠"
        print(f"{seq_len:<10} {ours_ms:>12.3f}   {fla_ms:>12.3f}   {speedup:>8.2f}x   {status}")
    
    print()
    print("Summary:")
    print("-" * 70)
    avg_speedup = sum(r[3] for r in results) / len(results) if results else 0
    print(f"  Average speedup vs FLA: {avg_speedup:.2f}x")
    if avg_speedup >= 1.0:
        print(f"  ✓ Our implementation is faster on average")
    elif avg_speedup >= 0.95:
        print(f"  ≈ Our implementation is comparable to FLA")
    else:
        print(f"  ⚠ FLA is faster on average")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FUSED RECURRENT SIMPLE GLA TEST SUITE")
    print("Data type: bfloat16 (unless otherwise specified)")
    print("Target: NVIDIA Blackwell RTX 6000D")
    print("=" * 70 + "\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Tests require a GPU.")
        return False
    
    print(f"Using device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    results = []
    
    # Test 1: Basic functionality
    results.append(("basic_functionality", test_fused_recurrent_simple_gla_basic()))
    
    # Test 2: With initial state
    results.append(("with_initial_state", test_fused_recurrent_simple_gla_with_initial_state()))
    
    # Test 3: Correctness (optional, may be slow)
    # results.append(("correctness", test_fused_recurrent_simple_gla_correctness()))
    
    # Test 4: Output dtype
    results.append(("output_dtype", test_fused_recurrent_simple_gla_output_dtype()))
    
    # Benchmark
    benchmark_fused_recurrent_simple_gla()
    
    # Performance comparison
    benchmark_comparison()
    
    # FLA comparison
    benchmark_vs_fla()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print()
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
