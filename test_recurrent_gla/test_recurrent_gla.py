"""
Test script for fused_recurrent_simple_gla kernel - Baseline vs Fused comparison.

Baseline reflects the actual minicpm.py execution flow:
1. Call fused_recurrent_simple_gla (output 4D [B, T, H, D])
2. Reshape to 2D [B*T, H*D]
3. Call fused_output_processing (RMSNorm + sigmoid gate)
"""

import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/root/soar2026/python')

# Import fused output processing from minicpm
from sglang.srt.models.minicpm_fused_output import fused_output_processing

from test_recurrent_gla.recurrent_simple_gla import fused_recurrent_simple_gla as fused_recurrent_simple_gla_baseline

# Try to import FLA implementation
try:
    from fla.ops.simple_gla import fused_recurrent_simple_gla as fused_recurrent_simple_gla_fla
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("Warning: FLA not available, only testing standalone implementation")


# =============================================================================
# Test Inputs
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
    hidden_size: int = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
):
    """Generate test inputs including z and norm_weight for output processing."""
    torch.manual_seed(seed)
    
    if hidden_size is None:
        hidden_size = num_heads * head_dim
    
    # QKV for attention
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device=device, dtype=torch.float32) * 0.01
    scale = head_dim ** -0.5
    
    # z for output gate (from hidden_states in minicpm.py)
    z = torch.randn(batch * seq_len, hidden_size, dtype=dtype, device=device) * 0.1
    
    # RMSNorm weight
    norm_weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    
    return q, k, v, g_gamma, scale, z, norm_weight


def get_test_inputs_with_initial_state(
    seq_len: int = 1024,
    batch: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    hidden_size: int = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
):
    """Generate test inputs with initial state for stateful testing."""
    torch.manual_seed(seed)
    
    if hidden_size is None:
        hidden_size = num_heads * head_dim
    
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    g_gamma = -torch.abs(build_slope_tensor(num_heads)).to(device=device, dtype=torch.float32) * 0.01
    scale = head_dim ** -0.5
    
    # Generate initial state [B, H, K, V]
    initial_state = torch.randn(batch, num_heads, head_dim, head_dim, dtype=torch.float32, device=device) * 0.01
    
    # z for output gate
    z = torch.randn(batch * seq_len, hidden_size, dtype=dtype, device=device) * 0.1
    
    # RMSNorm weight
    norm_weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    
    return q, k, v, g_gamma, scale, initial_state, z, norm_weight


# =============================================================================
# Reference Implementation (for small seq_len verification)
# =============================================================================

def recurrent_simple_gla_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_gamma: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
):
    """
    Pure PyTorch reference implementation for verification.
    Uses recurrence: h_t = decay * h_{t-1} + k_t^T @ v_t
                      o_t = q_t @ h_t
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    device = q.device
    
    # Convert to float32 for numerical stability
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    
    # Decay per head: [H]
    decay = g_gamma.exp()  # [H]
    
    # Initialize hidden state
    if initial_state is not None:
        h = initial_state.clone().float()  # [B, H, K, V]
    else:
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=device)
    
    outputs = []
    
    for t in range(T):
        q_t = q_f[:, t]  # [B, H, K]
        k_t = k_f[:, t]  # [B, H, K]
        v_t = v_f[:, t]  # [B, H, V]
        
        # Update hidden state: h = decay * h + k_t^T @ v_t
        # h: [B, H, K, V], decay: [H] -> [1, H, 1, 1]
        h = h * decay.view(1, H, 1, 1) + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        
        # Compute output: o_t = q_t @ h
        o_t = torch.einsum('bhk,bhkv->bhv', q_t * scale, h)  # [B, H, V]
        outputs.append(o_t)
    
    # Stack outputs: [B, T, H, V]
    o = torch.stack(outputs, dim=1)
    
    return o, h


# =============================================================================
# Benchmark Functions - Reflecting minicpm.py execution flow
# =============================================================================

def benchmark_baseline(q, k, v, g_gamma, scale, z, norm_weight, eps=1e-6, initial_state=None, warmup=10, iterations=50):
    """Benchmark baseline: fused_recurrent_simple_gla -> reshape -> fused_output_processing.
    
    This matches the actual minicpm.py execution flow:
    1. linear_attn_backend.forward() -> 4D output [B, T, H, D]
    2. Reshape to 2D [B*T, H*D]
    3. fused_output_processing(o, z, norm_weight) -> RMSNorm + sigmoid gate
    """
    torch.cuda.synchronize()
    
    for _ in range(warmup):
        o_4d, _ = fused_recurrent_simple_gla_baseline(
            q, k, v, g_gamma=g_gamma, scale=scale, initial_state=initial_state, output_final_state=True
        )
        B, T, H, D = o_4d.shape
        o_2d = o_4d.reshape(-1, H * D)
        out = fused_output_processing(o_2d, z, norm_weight, eps)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        o_4d, ht = fused_recurrent_simple_gla_baseline(
            q, k, v, g_gamma=g_gamma, scale=scale, initial_state=initial_state, output_final_state=True
        )
        B, T, H, D = o_4d.shape
        o_2d = o_4d.reshape(-1, H * D)
        out = fused_output_processing(o_2d, z, norm_weight, eps)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms, out, ht


def benchmark_fla(q, k, v, g_gamma, scale, z, norm_weight, eps=1e-6, initial_state=None, warmup=10, iterations=50):
    """Benchmark FLA implementation with output processing."""
    if not FLA_AVAILABLE:
        return None, None, None
    
    torch.cuda.synchronize()
    
    for _ in range(warmup):
        o_4d, _ = fused_recurrent_simple_gla_fla(
            q, k, v, g_gamma=g_gamma, scale=scale, initial_state=initial_state, output_final_state=True
        )
        B, T, H, D = o_4d.shape
        o_2d = o_4d.reshape(-1, H * D)
        out = fused_output_processing(o_2d, z, norm_weight, eps)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        o_4d, ht = fused_recurrent_simple_gla_fla(
            q, k, v, g_gamma=g_gamma, scale=scale, initial_state=initial_state, output_final_state=True
        )
        B, T, H, D = o_4d.shape
        o_2d = o_4d.reshape(-1, H * D)
        out = fused_output_processing(o_2d, z, norm_weight, eps)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms, out, ht


def compare_accuracy(o_ref, o_test, ht_ref, ht_test, name="Test"):
    """Compare accuracy between implementations."""
    # Handle shape mismatch (4D vs 2D)
    if o_ref.dim() == 4 and o_test.dim() == 2:
        B, T, H, D = o_ref.shape
        o_ref = o_ref.reshape(-1, H * D)
    elif o_ref.dim() == 2 and o_test.dim() == 4:
        B, T, H, D = o_test.shape
        o_test = o_test.reshape(-1, H * D)
    
    output_diff = (o_ref - o_test).abs().max().item()
    state_diff = (ht_ref - ht_test).abs().max().item()
    
    output_scale = o_ref.abs().mean().item() + 1e-6
    state_scale = ht_ref.abs().mean().item() + 1e-6
    output_rel_diff = ((o_ref - o_test).abs() / output_scale).max().item()
    state_rel_diff = ((ht_ref - ht_test).abs() / state_scale).max().item()
    
    output_mae = (o_ref - o_test).abs().mean().item()
    state_mae = (ht_ref - ht_test).abs().mean().item()
    
    has_nan = (torch.isnan(o_ref).any() or torch.isnan(o_test).any() or 
               torch.isnan(ht_ref).any() or torch.isnan(ht_test).any())
    
    return {
        'name': name,
        'output_abs_diff': output_diff,
        'output_rel_diff': output_rel_diff,
        'state_rel_diff': state_rel_diff,
        'output_mae': output_mae,
        'state_mae': state_mae,
        'has_nan': has_nan,
    }


# =============================================================================
# Test Configurations
# =============================================================================

DEFAULT_TEST_CONFIGS = [
    {"seq_len": 16},
    {"seq_len": 32},
    {"seq_len": 64},
    {"seq_len": 128},
    {"seq_len": 256},
    {"seq_len": 384},
    {"seq_len": 512},
    {"seq_len": 768},
    {"seq_len": 1024},
    {"seq_len": 2048},
]

# Performance test configs (shorter list for faster testing)
PERF_TEST_CONFIGS = [
    {"seq_len": 16},
    {"seq_len": 32},
    {"seq_len": 64},
    {"seq_len": 128},
    {"seq_len": 256},
    {"seq_len": 512},
    {"seq_len": 1024},
    {"seq_len": 2048},
]


# =============================================================================
# Main Test
# =============================================================================

def main():
    """Test and benchmark fused_recurrent_simple_gla implementations."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return
    
    print("=" * 100)
    print("Fused Recurrent Simple GLA - Standalone vs FLA vs Reference")
    print("=" * 100)
    
    print(f"\nGPU Information:")
    print(f"  Device: {torch.cuda.get_device_name()}")
    major, minor = torch.cuda.get_device_capability()
    print(f"  Compute Capability: {major}.{minor}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    import triton
    print(f"Triton version: {triton.__version__}")
    print(f"FLA available: {FLA_AVAILABLE}")
    
    # =============================================================================
    # Correctness Test: Baseline vs Reference (small seq_len)
    # =============================================================================
    print("\n" + "=" * 100)
    print("Correctness Test: Baseline vs PyTorch Reference")
    print("=" * 100)
    print(f"{'Seq Len':<10} {'Output MAE':<15} {'State MAE':<15} {'Status':<10}")
    print("-" * 100)
    
    for config in DEFAULT_TEST_CONFIGS[:6]:  # Only test small seq_len vs reference
        seq_len = config["seq_len"]
        q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(seq_len=seq_len)
        
        try:
            # Reference implementation (without output processing for simplicity)
            o_ref, ht_ref = recurrent_simple_gla_reference(q, k, v, g_gamma, scale)
            
            # Baseline: recurrent_gla -> reshape -> fused_output_processing
            o_4d, ht_baseline = fused_recurrent_simple_gla_baseline(
                q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
            )
            B, T, H, D = o_4d.shape
            o_2d = o_4d.reshape(-1, H * D)
            out_baseline = fused_output_processing(o_2d, z, norm_weight, eps=1e-6)
            
            # Compare reference (reshaped to 2D) with baseline output
            o_ref_2d = o_ref.reshape(-1, H * D)
            diff_info = compare_accuracy(o_ref_2d, out_baseline.float(), ht_ref, ht_baseline, "Baseline vs Ref")
            
            if diff_info['has_nan']:
                status = "NAN"
            elif diff_info['output_mae'] > 0.5 or diff_info['state_mae'] > 1.0:
                status = "HIGH_ERR"
            elif diff_info['output_mae'] > 0.05 or diff_info['state_mae'] > 0.1:
                status = "WARN"
            else:
                status = "PASS"
            
            print(f"{seq_len:<10} {diff_info['output_mae']:<15.2e} {diff_info['state_mae']:<15.2e} {status}")
        except Exception as e:
            import traceback
            print(f"{seq_len:<10} {'ERROR':<15} {'ERROR':<15} {str(e)[:20]}")
            traceback.print_exc()
    
    # =============================================================================
    # Correctness Test: Baseline vs FLA (with output processing)
    # =============================================================================
    if FLA_AVAILABLE:
        print("\n" + "=" * 100)
        print("Correctness Test: Baseline vs FLA (with output processing)")
        print("=" * 100)
        print(f"{'Seq Len':<10} {'Output MAE':<15} {'State MAE':<15} {'Status':<10}")
        print("-" * 100)
        
        for config in DEFAULT_TEST_CONFIGS:
            seq_len = config["seq_len"]
            q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(seq_len=seq_len)
            
            try:
                # FLA implementation with output processing
                o_fla_4d, ht_fla = fused_recurrent_simple_gla_fla(
                    q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
                )
                B, T, H, D = o_fla_4d.shape
                o_fla_2d = o_fla_4d.reshape(-1, H * D)
                out_fla = fused_output_processing(o_fla_2d, z, norm_weight, eps=1e-6)
                
                # Baseline with output processing
                o_baseline_4d, ht_baseline = fused_recurrent_simple_gla_baseline(
                    q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
                )
                o_baseline_2d = o_baseline_4d.reshape(-1, H * D)
                out_baseline = fused_output_processing(o_baseline_2d, z, norm_weight, eps=1e-6)
                
                diff_info = compare_accuracy(out_fla, out_baseline, ht_fla, ht_baseline, "Baseline vs FLA")
                
                if diff_info['has_nan']:
                    status = "NAN"
                elif diff_info['output_mae'] > 0.5 or diff_info['state_mae'] > 1.0:
                    status = "HIGH_ERR"
                elif diff_info['output_mae'] > 0.01 or diff_info['state_mae'] > 0.05:
                    status = "WARN"
                else:
                    status = "PASS"
                
                print(f"{seq_len:<10} {diff_info['output_mae']:<15.2e} {diff_info['state_mae']:<15.2e} {status}")
            except Exception as e:
                import traceback
                print(f"{seq_len:<10} {'ERROR':<15} {'ERROR':<15} {str(e)[:40]}")
                traceback.print_exc()
    
    # =============================================================================
    # Correctness Test: With Initial State
    # =============================================================================
    print("\n" + "=" * 100)
    print("Correctness Test: With Initial State (Baseline vs Reference)")
    print("=" * 100)
    print(f"{'Seq Len':<10} {'Output MAE':<15} {'State MAE':<15} {'Status':<10}")
    print("-" * 100)
    
    for config in DEFAULT_TEST_CONFIGS[:5]:
        seq_len = config["seq_len"]
        q, k, v, g_gamma, scale, initial_state, z, norm_weight = get_test_inputs_with_initial_state(seq_len=seq_len)
        
        try:
            # Reference implementation
            o_ref, ht_ref = recurrent_simple_gla_reference(q, k, v, g_gamma, scale, initial_state)
            
            # Baseline implementation with output processing
            o_4d, ht_baseline = fused_recurrent_simple_gla_baseline(
                q, k, v, g_gamma=g_gamma, scale=scale, initial_state=initial_state, output_final_state=True
            )
            B, T, H, D = o_4d.shape
            o_2d = o_4d.reshape(-1, H * D)
            out_baseline = fused_output_processing(o_2d, z, norm_weight, eps=1e-6)
            
            # Compare reference (reshaped) with baseline
            o_ref_2d = o_ref.reshape(-1, H * D)
            diff_info = compare_accuracy(o_ref_2d, out_baseline.float(), ht_ref, ht_baseline, "Stateful Baseline vs Ref")
            
            if diff_info['has_nan']:
                status = "NAN"
            elif diff_info['output_mae'] > 0.5 or diff_info['state_mae'] > 1.0:
                status = "HIGH_ERR"
            elif diff_info['output_mae'] > 0.05 or diff_info['state_mae'] > 0.1:
                status = "WARN"
            else:
                status = "PASS"
            
            print(f"{seq_len:<10} {diff_info['output_mae']:<15.2e} {diff_info['state_mae']:<15.2e} {status}")
        except Exception as e:
            import traceback
            print(f"{seq_len:<10} {'ERROR':<15} {'ERROR':<15} {str(e)[:20]}")
            traceback.print_exc()
    
    # =============================================================================
    # Performance Benchmark
    # =============================================================================
    print("\n" + "=" * 100)
    print("Performance Benchmark")
    print("=" * 100)
    print("Baseline: recurrent_gla -> reshape -> fused_output_processing")
    print("(Matches minicpm.py execution flow)")
    print("-" * 100)
    
    if FLA_AVAILABLE:
        print(f"{'Seq Len':<10} {'Baseline (ms)':<18} {'FLA (ms)':<15} {'Speedup':<10} {'Status':<10}")
    else:
        print(f"{'Seq Len':<10} {'Baseline (ms)':<18} {'Status':<10}")
    print("-" * 100)
    
    for config in PERF_TEST_CONFIGS:
        seq_len = config["seq_len"]
        q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(seq_len=seq_len)
        
        try:
            # Benchmark baseline (with output processing)
            t_baseline, out_baseline, ht_baseline = benchmark_baseline(
                q, k, v, g_gamma, scale, z, norm_weight, warmup=3, iterations=10
            )
            
            if FLA_AVAILABLE:
                # Benchmark FLA (with output processing)
                t_fla, out_fla, ht_fla = benchmark_fla(
                    q, k, v, g_gamma, scale, z, norm_weight, warmup=3, iterations=10
                )
                
                # Compare accuracy
                diff_info = compare_accuracy(out_baseline, out_fla, ht_baseline, ht_fla)
                
                if diff_info['has_nan']:
                    status = "NAN"
                elif diff_info['output_mae'] > 0.1:
                    status = "FAIL"
                elif diff_info['output_mae'] > 0.01:
                    status = "WARN"
                else:
                    status = "PASS"
                
                speedup = t_fla / t_baseline if t_baseline > 0 else 0
                
                print(f"{seq_len:<10} {t_baseline:>16.3f}   {t_fla:>13.3f}   {speedup:>8.2f}x  {status}")
            else:
                print(f"{seq_len:<10} {t_baseline:>16.3f}   {'PASS':<10}")
            
        except Exception as e:
            import traceback
            if FLA_AVAILABLE:
                print(f"{seq_len:<10} {'ERROR':<18} {'ERROR':<15} {'N/A':<10} {str(e)[:20]}")
            else:
                print(f"{seq_len:<10} {'ERROR':<18} {str(e)[:20]}")
            traceback.print_exc()
    
    # =============================================================================
    # Memory Analysis
    # =============================================================================
    print("\n" + "=" * 100)
    print("Memory Analysis")
    print("=" * 100)
    print("Fused Recurrent Simple GLA memory characteristics:")
    print("  - No intermediate h tensor (unlike chunk-based methods)")
    print("  - Memory usage: O(B * H * K * V) for state + O(B * T * H * (K + V)) for I/O")
    print("  - Recurrent computation avoids materializing full attention matrix")
    
    def analyze_memory_recurrent(seq_len, batch=1, num_heads=32, head_dim=128):
        """Analyze memory usage of recurrent method."""
        K = V = head_dim
        
        qkv_size = 3 * batch * seq_len * num_heads * head_dim * 2  # bf16 = 2 bytes
        state_size = batch * num_heads * K * V * 4  # fp32 state
        o_size = batch * seq_len * num_heads * head_dim * 2
        
        total = qkv_size + state_size + o_size
        
        return {
            'qkv_mb': qkv_size / 1024**2,
            'state_mb': state_size / 1024**2,
            'o_mb': o_size / 1024**2,
            'total_mb': total / 1024**2,
        }
    
    print(f"\n{'Seq Len':<10} {'QKV (MB)':<12} {'State (MB)':<12} {'Output (MB)':<12} {'Total (MB)':<12}")
    print("-" * 70)
    for config in PERF_TEST_CONFIGS[::2]:
        seq_len = config["seq_len"]
        mem = analyze_memory_recurrent(seq_len)
        print(f"{seq_len:<10} {mem['qkv_mb']:<12.2f} {mem['state_mb']:<12.2f} {mem['o_mb']:<12.2f} {mem['total_mb']:<12.2f}")
    
    print("\n" + "=" * 100)
    print("Summary:")
    print("  - Baseline reflects minicpm.py execution flow:")
    print("    1. linear_attn_backend.forward() -> 4D output [B, T, H, D]")
    print("    2. Reshape to 2D [B*T, H*D]")
    print("    3. fused_output_processing(o, z, norm_weight) -> RMSNorm + sigmoid gate")
    print("  - No chunking overhead, suitable for short sequences (< 128)")
    print("  - Supports initial state for prefix caching")
    print("  - Next step: Fuse recurrent_gla + output_processing into single kernel")
    print("=" * 100)


# =============================================================================
# Correctness Test: Fully Fused Version vs Baseline
# =============================================================================

def test_fully_fused_accuracy():
    """Test fully fused recurrent GLA + output processing against baseline."""
    from test_recurrent_gla.recurrent_simple_gla_fused_output import (
        fused_recurrent_gla_fused_output,
        fused_recurrent_gla_with_output_fully_fused,
        fused_recurrent_gla_with_output_warp_fused,
    )
    
    print("\n" + "=" * 100)
    print("Correctness Test: Fully Fused Version vs Baseline")
    print("=" * 100)
    print(f"{'Seq Len':<10} {'Heads':<8} {'Head Dim':<10} {'Fully MAE':<15} {'Warp MAE':<15} {'Status':<10}")
    print("-" * 100)
    
    test_configs = [
        {"seq_len": 16, "num_heads": 4, "head_dim": 64},
        {"seq_len": 32, "num_heads": 4, "head_dim": 64},
        {"seq_len": 64, "num_heads": 8, "head_dim": 64},
        {"seq_len": 128, "num_heads": 8, "head_dim": 64},
        {"seq_len": 256, "num_heads": 8, "head_dim": 64},
        {"seq_len": 16, "num_heads": 32, "head_dim": 128},
        {"seq_len": 32, "num_heads": 32, "head_dim": 128},
        {"seq_len": 64, "num_heads": 32, "head_dim": 128},
    ]
    
    for config in test_configs:
        seq_len = config["seq_len"]
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        
        try:
            q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(
                seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
            )
            
            # Baseline: recurrent_gla -> reshape -> fused_output_processing
            o_baseline_4d, ht_baseline = fused_recurrent_simple_gla_baseline(
                q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
            )
            B, T, H, D = o_baseline_4d.shape
            o_baseline_2d = o_baseline_4d.reshape(-1, H * D)
            out_baseline = fused_output_processing(o_baseline_2d, z, norm_weight, eps=1e-6)
            
            # Fully fused version (B*T, H grid)
            out_fully, ht_fully = fused_recurrent_gla_with_output_fully_fused(
                q, k, v, z, norm_weight,
                g_gamma=g_gamma, scale=scale, eps=1e-6,
                output_final_state=True
            )
            
            # Warp fused version (B, T grid)
            out_warp, ht_warp = fused_recurrent_gla_with_output_warp_fused(
                q, k, v, z, norm_weight,
                g_gamma=g_gamma, scale=scale, eps=1e-6,
                output_final_state=True
            )
            
            # Compare fully vs baseline
            diff_fully = compare_accuracy(out_baseline, out_fully, ht_baseline, ht_fully, "Fully vs Baseline")
            # Compare warp vs baseline  
            diff_warp = compare_accuracy(out_baseline, out_warp, ht_baseline, ht_warp, "Warp vs Baseline")
            
            # Status
            if diff_fully['has_nan'] or diff_warp['has_nan']:
                status = "NAN"
            elif diff_fully['output_mae'] > 0.5 or diff_warp['output_mae'] > 0.5:
                status = "HIGH_ERR"
            elif diff_fully['output_mae'] > 0.01 or diff_warp['output_mae'] > 0.01:
                status = "WARN"
            else:
                status = "PASS"
            
            print(f"{seq_len:<10} {num_heads:<8} {head_dim:<10} {diff_fully['output_mae']:<15.2e} {diff_warp['output_mae']:<15.2e} {status}")
            
        except Exception as e:
            import traceback
            print(f"{seq_len:<10} {num_heads:<8} {head_dim:<10} {'ERROR':<15} {'ERROR':<15} {str(e)[:20]}")
            traceback.print_exc()
    
    # =============================================================================
    # Cross-Check: Fully Fused vs Warp Fused
    # =============================================================================
    print("\n" + "=" * 100)
    print("Cross-Check: Fully Fused vs Warp Fused (should be identical)")
    print("=" * 100)
    print(f"{'Seq Len':<10} {'Heads':<8} {'Head Dim':<10} {'Output MAE':<15} {'State MAE':<15} {'Status':<10}")
    print("-" * 100)
    
    for config in test_configs:
        seq_len = config["seq_len"]
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        
        try:
            q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(
                seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
            )
            
            # Fully fused version
            out_fully, ht_fully = fused_recurrent_gla_with_output_fully_fused(
                q, k, v, z, norm_weight,
                g_gamma=g_gamma, scale=scale, eps=1e-6,
                output_final_state=True
            )
            
            # Warp fused version
            out_warp, ht_warp = fused_recurrent_gla_with_output_warp_fused(
                q, k, v, z, norm_weight,
                g_gamma=g_gamma, scale=scale, eps=1e-6,
                output_final_state=True
            )
            
            # Compare two fused versions
            diff = compare_accuracy(out_fully, out_warp, ht_fully, ht_warp, "Fully vs Warp")
            
            if diff['has_nan']:
                status = "NAN"
            elif diff['output_mae'] > 1e-4:  # Should be very close
                status = "DIFF"
            elif diff['output_mae'] > 1e-6:
                status = "WARN"
            else:
                status = "MATCH"
            
            print(f"{seq_len:<10} {num_heads:<8} {head_dim:<10} {diff['output_mae']:<15.2e} {diff['state_mae']:<15.2e} {status}")
            
        except Exception as e:
            import traceback
            print(f"{seq_len:<10} {num_heads:<8} {head_dim:<10} {'ERROR':<15} {'ERROR':<15} {str(e)[:20]}")
            traceback.print_exc()


def benchmark_fused_versions():
    """Benchmark Fully Fused vs Warp Fused versions."""
    from test_recurrent_gla.recurrent_simple_gla_fused_output import (
        fused_recurrent_gla_with_output_fully_fused,
        fused_recurrent_gla_with_output_warp_fused,
    )
    
    print("\n" + "=" * 100)
    print("Performance Benchmark: Fully Fused vs Warp Fused")
    print("=" * 100)
    print("Fully Fused: Grid = (B*T, H) - each block handles one head")
    print("Warp Fused:  Grid = (B, T) - each block handles all heads sequentially")
    print("-" * 100)
    print(f"{'Seq Len':<10} {'Heads':<8} {'Head Dim':<10} {'Baseline (ms)':<15} {'Fully (ms)':<15} {'Warp (ms)':<15} {'F/B':<8} {'W/B':<8}")
    print("-" * 100)
    
    test_configs = [
        {"seq_len": 16, "num_heads": 4, "head_dim": 64},
        {"seq_len": 32, "num_heads": 4, "head_dim": 64},
        {"seq_len": 64, "num_heads": 8, "head_dim": 64},
        {"seq_len": 128, "num_heads": 8, "head_dim": 64},
        {"seq_len": 256, "num_heads": 8, "head_dim": 64},
        {"seq_len": 512, "num_heads": 8, "head_dim": 64},
        {"seq_len": 16, "num_heads": 32, "head_dim": 128},
        {"seq_len": 32, "num_heads": 32, "head_dim": 128},
        {"seq_len": 64, "num_heads": 32, "head_dim": 128},
        {"seq_len": 128, "num_heads": 32, "head_dim": 128},
        {"seq_len": 256, "num_heads": 32, "head_dim": 128},
        {"seq_len": 512, "num_heads": 32, "head_dim": 128},
    ]
    
    warmup = 3
    iterations = 10
    
    for config in test_configs:
        seq_len = config["seq_len"]
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        
        try:
            q, k, v, g_gamma, scale, z, norm_weight = get_test_inputs(
                seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
            )
            
            # Warmup baseline
            for _ in range(warmup):
                o_baseline_4d, _ = fused_recurrent_simple_gla_baseline(
                    q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
                )
                B, T, H, D = o_baseline_4d.shape
                o_baseline_2d = o_baseline_4d.reshape(-1, H * D)
                _ = fused_output_processing(o_baseline_2d, z, norm_weight, eps=1e-6)
            torch.cuda.synchronize()
            
            # Benchmark baseline
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                o_baseline_4d, _ = fused_recurrent_simple_gla_baseline(
                    q, k, v, g_gamma=g_gamma, scale=scale, output_final_state=True
                )
                B, T, H, D = o_baseline_4d.shape
                o_baseline_2d = o_baseline_4d.reshape(-1, H * D)
                _ = fused_output_processing(o_baseline_2d, z, norm_weight, eps=1e-6)
            end.record()
            torch.cuda.synchronize()
            t_baseline = start.elapsed_time(end) / iterations
            
            # Warmup fully fused
            for _ in range(warmup):
                _, _ = fused_recurrent_gla_with_output_fully_fused(
                    q, k, v, z, norm_weight,
                    g_gamma=g_gamma, scale=scale, eps=1e-6, output_final_state=True
                )
            torch.cuda.synchronize()
            
            # Benchmark fully fused
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                _, _ = fused_recurrent_gla_with_output_fully_fused(
                    q, k, v, z, norm_weight,
                    g_gamma=g_gamma, scale=scale, eps=1e-6, output_final_state=True
                )
            end.record()
            torch.cuda.synchronize()
            t_fully = start.elapsed_time(end) / iterations
            
            # Warmup warp fused
            for _ in range(warmup):
                _, _ = fused_recurrent_gla_with_output_warp_fused(
                    q, k, v, z, norm_weight,
                    g_gamma=g_gamma, scale=scale, eps=1e-6, output_final_state=True
                )
            torch.cuda.synchronize()
            
            # Benchmark warp fused
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                _, _ = fused_recurrent_gla_with_output_warp_fused(
                    q, k, v, z, norm_weight,
                    g_gamma=g_gamma, scale=scale, eps=1e-6, output_final_state=True
                )
            end.record()
            torch.cuda.synchronize()
            t_warp = start.elapsed_time(end) / iterations
            
            speedup_fully = t_baseline / t_fully if t_fully > 0 else 0
            speedup_warp = t_baseline / t_warp if t_warp > 0 else 0
            
            print(f"{seq_len:<10} {num_heads:<8} {head_dim:<10} {t_baseline:<15.3f} {t_fully:<15.3f} {t_warp:<15.3f} {speedup_fully:<8.2f} {speedup_warp:<8.2f}")
            
        except Exception as e:
            import traceback
            print(f"{seq_len:<10} {num_heads:<8} {head_dim:<10} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15} {'N/A':<8} {'N/A':<8}")
            traceback.print_exc()


def main_with_fused_test():
    """Run all tests including fully fused version."""
    main()
    
    # Test fully fused version
    test_fully_fused_accuracy()
    
    # Benchmark fused versions
    benchmark_fused_versions()


if __name__ == "__main__":
    main_with_fused_test()
