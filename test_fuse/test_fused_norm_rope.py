"""Test correctness of Fused RMSNorm + RoPE for MiniCPM.

Fixed parameters:
- num_heads: 32
- head_dim: 128
- num_kv_heads: 32
- hidden_size: 4096

This tests the fused kernel against the ground truth implementation:
1. Q/K RMSNorm (per-head)
2. RoPE application
3. Reshape to [1, num_tokens, num_heads, head_dim]
"""

import torch
import torch.nn as nn
import sys
import math

# Add python path
sys.path.insert(0, '/root/soar2026/python')

from sglang.srt.models.minicpm_fused_norm_rope import (
    fused_rms_norm_rope,
    FusedRMSNormRoPE,
    NUM_HEADS,
    HEAD_DIM,
    NUM_KV_HEADS,
)

# Import RoPE for cos_sin_cache generation
from sglang.srt.models.minicpm_rope import MiniCPMRoPEBaseline


def pytorch_rms_norm(x, weight, eps=1e-6):
    """PyTorch RMSNorm implementation.
    
    Args:
        x: [..., hidden_size] input tensor
        weight: [hidden_size] norm weight
        eps: epsilon
    
    Returns:
        normalized x
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x


def _rotate_neox(x):
    """Rotate half the hidden dims (Neox style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding (Neox style).
    
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    
    # Neox style: split in half
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def ground_truth_fused_norm_rope(
    q, k, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps=1e-6
):
    """Ground truth implementation matching minicpm.py logic.
    
    From minicpm.py:
    1. Q/K RMSNorm (per-head, head_dim=128)
    2. RoPE application
    3. Reshape to [1, num_tokens, num_heads, head_dim]
    """
    num_tokens = q.shape[0]
    
    # Step 1: RMSNorm per head
    # q: [num_tokens, num_heads * head_dim] -> [num_tokens * num_heads, head_dim]
    q_reshaped = q.reshape(-1, HEAD_DIM)
    k_reshaped = k.reshape(-1, HEAD_DIM)
    
    q_normed = pytorch_rms_norm(q_reshaped, q_norm_weight, eps)
    k_normed = pytorch_rms_norm(k_reshaped, k_norm_weight, eps)
    
    # Step 2: Reshape back
    q_normed = q_normed.reshape(num_tokens, NUM_HEADS, HEAD_DIM)
    k_normed = k_normed.reshape(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    
    # Step 3: Apply RoPE (matching minicpm.py logic)
    # Get cos/sin for positions
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    # Apply RoPE
    q_rot = _apply_rotary_emb(q_normed, cos, sin)
    k_rot = _apply_rotary_emb(k_normed, cos, sin)
    
    # Step 4: Unsqueeze to [1, num_tokens, num_heads, head_dim]
    q_out = q_rot.unsqueeze(0)
    k_out = k_rot.unsqueeze(0)
    
    return q_out, k_out


def test_fused_norm_rope_correctness():
    """Test fused kernel against ground truth."""
    print("=" * 80)
    print("Testing Fused RMSNorm + RoPE Correctness")
    print("=" * 80)
    print(f"Fixed parameters:")
    print(f"  num_heads: {NUM_HEADS}")
    print(f"  head_dim: {HEAD_DIM}")
    print(f"  num_kv_heads: {NUM_KV_HEADS}")
    print("=" * 80)
    
    device = torch.device("cuda")
    max_position = 8192
    rope_theta = 10000.0
    eps = 1e-6
    
    # Create RoPE baseline for cos_sin_cache
    rope_baseline = MiniCPMRoPEBaseline(
        max_position_embeddings=max_position,
        rope_theta=rope_theta,
    ).cuda()
    
    # Test cases covering different sequence lengths
    test_cases = [
        ("Single token", 1),
        ("Short sequence", 8),
        ("Medium sequence", 64),
        ("Long sequence", 256),
        ("Very long sequence", 1024),
        ("Batch decode", 16),
        ("Prefill", 2048),
    ]
    
    all_passed = True
    
    # Tolerance: bf16/fp16 needs higher tolerance due to different computation order
    # Note: bf16 has limited precision (~2-3 decimal digits)
    # The fused kernel does RMSNorm and RoPE in a single pass which may have
    # different rounding behavior compared to separate operations
    bf16_tolerance = 2e-1  # 0.2 tolerance for bf16
    fp32_tolerance = 1e-3
    
    print(f"\n{'Test Case':<25} {'Tokens':>10} {'Dtype':>8} {'Q Diff':>12} {'K Diff':>12} {'Status':>10}")
    print("-" * 80)
    
    for name, num_tokens in test_cases:
        for dtype in [torch.bfloat16, torch.float32]:
            dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
            tolerance = bf16_tolerance if dtype == torch.bfloat16 else fp32_tolerance
            
            # Create test data
            torch.manual_seed(42)
            q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, dtype=dtype, device=device)
            k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=dtype, device=device)
            positions = torch.randint(0, max_position, (num_tokens,), device=device)
            
            # Norm weights
            q_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
            k_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
            
            # Ground truth
            q_gt, k_gt = ground_truth_fused_norm_rope(
                q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
                rope_baseline.cos_sin_cache, eps
            )
            
            # Fused kernel
            q_fused, k_fused = fused_rms_norm_rope(
                q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
                rope_baseline.cos_sin_cache.to(dtype), eps
            )
            
            # Compare
            q_diff = torch.max(torch.abs(q_gt - q_fused)).item()
            k_diff = torch.max(torch.abs(k_gt - k_fused)).item()
            
            passed = (q_diff < tolerance) and (k_diff < tolerance)
            status = "PASS ✓" if passed else "FAIL ✗"
            if not passed:
                all_passed = False
            
            print(f"{name:<25} {num_tokens:>10} {dtype_str:>8} {q_diff:>12.6e} {k_diff:>12.6e} {status:>10}")
    
    print("-" * 80)
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 80)
    
    return all_passed


def test_module_interface():
    """Test FusedRMSNormRoPE module interface."""
    print("\n" + "=" * 80)
    print("Testing FusedRMSNormRoPE Module Interface")
    print("=" * 80)
    
    device = torch.device("cuda")
    max_position = 8192
    rope_theta = 10000.0
    eps = 1e-6
    num_tokens = 128
    
    # Create module
    fused_module = FusedRMSNormRoPE(eps=eps).cuda()
    
    # Create RoPE baseline for cos_sin_cache
    rope_baseline = MiniCPMRoPEBaseline(
        max_position_embeddings=max_position,
        rope_theta=rope_theta,
    ).cuda()
    
    all_passed = True
    bf16_tolerance = 2e-1  # 0.2 tolerance for bf16
    fp32_tolerance = 1e-3
    
    print(f"\n{'Test':<45} {'Dtype':>8} {'Max Diff':>15} {'Status':>10}")
    print("-" * 80)
    
    for dtype in [torch.bfloat16, torch.float32]:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        tolerance = bf16_tolerance if dtype == torch.bfloat16 else fp32_tolerance
        
        # Create test data
        torch.manual_seed(42)
        q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, dtype=dtype, device=device)
        k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=dtype, device=device)
        positions = torch.arange(num_tokens, device=device)
        q_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        k_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        
        # Module forward
        q_module, k_module = fused_module(
            q, k, positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache.to(dtype)
        )
        
        # Functional equivalent
        q_func, k_func = fused_rms_norm_rope(
            q, k, positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache.to(dtype), eps
        )
        
        # Compare
        q_diff = torch.max(torch.abs(q_module - q_func)).item()
        k_diff = torch.max(torch.abs(k_module - k_func)).item()
        max_diff = max(q_diff, k_diff)
        
        passed = max_diff < tolerance
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_passed = False
        
        print(f"{'Module vs Functional':<45} {dtype_str:>8} {max_diff:>15.6e} {status:>10}")
    
    # Test output shape
    q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.arange(num_tokens, device=device)
    q_norm_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    q_out, k_out = fused_module(
        q, k, positions, q_norm_weight, k_norm_weight,
        rope_baseline.cos_sin_cache.to(torch.bfloat16)
    )
    
    q_shape_correct = q_out.shape == (1, num_tokens, NUM_HEADS, HEAD_DIM)
    k_shape_correct = k_out.shape == (1, num_tokens, NUM_KV_HEADS, HEAD_DIM)
    
    passed = q_shape_correct and k_shape_correct
    status = "PASS ✓" if passed else "FAIL ✗"
    if not passed:
        all_passed = False
    
    print(f"{'Output shape check':<45} {'bf16':>8} {'-':>15} {status:>10}")
    print(f"  Q shape: {q_out.shape} (expected: (1, {num_tokens}, {NUM_HEADS}, {HEAD_DIM}))")
    print(f"  K shape: {k_out.shape} (expected: (1, {num_tokens}, {NUM_KV_HEADS}, {HEAD_DIM}))")
    
    print("-" * 80)
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 80)
    
    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 80)
    print("Testing Edge Cases")
    print("=" * 80)
    
    device = torch.device("cuda")
    max_position = 8192
    rope_theta = 10000.0
    eps = 1e-6
    
    rope_baseline = MiniCPMRoPEBaseline(
        max_position_embeddings=max_position,
        rope_theta=rope_theta,
    ).cuda()
    
    all_passed = True
    bf16_tolerance = 2e-1  # 0.2 tolerance for bf16
    fp32_tolerance = 1e-3
    
    print(f"\n{'Test Case':<35} {'Dtype':>8} {'Q Diff':>12} {'K Diff':>12} {'Status':>10}")
    print("-" * 80)
    
    # Test 1: Single token
    for dtype in [torch.bfloat16, torch.float32]:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        tolerance = bf16_tolerance if dtype == torch.bfloat16 else fp32_tolerance
        
        num_tokens = 1
        q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, dtype=dtype, device=device)
        k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=dtype, device=device)
        positions = torch.tensor([0], device=device)
        q_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        k_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        
        q_gt, k_gt = ground_truth_fused_norm_rope(
            q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache, eps
        )
        
        q_fused, k_fused = fused_rms_norm_rope(
            q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache.to(dtype), eps
        )
        
        q_diff = torch.max(torch.abs(q_gt - q_fused)).item()
        k_diff = torch.max(torch.abs(k_gt - k_fused)).item()
        
        passed = (q_diff < tolerance) and (k_diff < tolerance)
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_passed = False
        
        print(f"{'Single token':<35} {dtype_str:>8} {q_diff:>12.6e} {k_diff:>12.6e} {status:>10}")
    
    # Test 2: Large values
    for dtype in [torch.bfloat16, torch.float32]:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        tolerance = bf16_tolerance if dtype == torch.bfloat16 else fp32_tolerance
        
        num_tokens = 64
        q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, dtype=dtype, device=device) * 10
        k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=dtype, device=device) * 10
        positions = torch.randint(0, max_position, (num_tokens,), device=device)
        q_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        k_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        
        q_gt, k_gt = ground_truth_fused_norm_rope(
            q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache, eps
        )
        
        q_fused, k_fused = fused_rms_norm_rope(
            q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache.to(dtype), eps
        )
        
        q_diff = torch.max(torch.abs(q_gt - q_fused)).item()
        k_diff = torch.max(torch.abs(k_gt - k_fused)).item()
        
        passed = (q_diff < tolerance) and (k_diff < tolerance)
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_passed = False
        
        print(f"{'Large values (x10)':<35} {dtype_str:>8} {q_diff:>12.6e} {k_diff:>12.6e} {status:>10}")
    
    # Test 3: Small values
    for dtype in [torch.bfloat16, torch.float32]:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        tolerance = bf16_tolerance if dtype == torch.bfloat16 else fp32_tolerance
        
        num_tokens = 64
        q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, dtype=dtype, device=device) * 0.01
        k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=dtype, device=device) * 0.01
        positions = torch.randint(0, max_position, (num_tokens,), device=device)
        q_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        k_norm_weight = torch.randn(HEAD_DIM, dtype=dtype, device=device)
        
        q_gt, k_gt = ground_truth_fused_norm_rope(
            q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache, eps
        )
        
        q_fused, k_fused = fused_rms_norm_rope(
            q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
            rope_baseline.cos_sin_cache.to(dtype), eps
        )
        
        q_diff = torch.max(torch.abs(q_gt - q_fused)).item()
        k_diff = torch.max(torch.abs(k_gt - k_fused)).item()
        
        passed = (q_diff < tolerance) and (k_diff < tolerance)
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_passed = False
        
        print(f"{'Small values (x0.01)':<35} {dtype_str:>8} {q_diff:>12.6e} {k_diff:>12.6e} {status:>10}")
    
    # Test 4: Zero input (fp32 only)
    num_tokens = 64
    q = torch.zeros(num_tokens, NUM_HEADS * HEAD_DIM, dtype=torch.float32, device=device)
    k = torch.zeros(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32, device=device)
    positions = torch.randint(0, max_position, (num_tokens,), device=device)
    q_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32, device=device)
    k_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32, device=device)
    
    q_gt, k_gt = ground_truth_fused_norm_rope(
        q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
        rope_baseline.cos_sin_cache, eps
    )
    
    q_fused, k_fused = fused_rms_norm_rope(
        q.clone(), k.clone(), positions, q_norm_weight, k_norm_weight,
        rope_baseline.cos_sin_cache.to(torch.float32), eps
    )
    
    q_diff = torch.max(torch.abs(q_gt - q_fused)).item()
    k_diff = torch.max(torch.abs(k_gt - k_fused)).item()
    
    passed = (q_diff < fp32_tolerance) and (k_diff < fp32_tolerance)
    status = "PASS ✓" if passed else "FAIL ✗"
    if not passed:
        all_passed = False
    
    print(f"{'Zero input':<35} {'fp32':>8} {q_diff:>12.6e} {k_diff:>12.6e} {status:>10}")
    
    print("-" * 80)
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MiniCPM Fused RMSNorm + RoPE Test Suite")
    print("=" * 80)
    print(f"\nFixed Parameters:")
    print(f"  num_heads: {NUM_HEADS}")
    print(f"  head_dim: {HEAD_DIM}")
    print(f"  num_kv_heads: {NUM_KV_HEADS}")
    print(f"\nTolerance:")
    print(f"  fp32: 1e-3")
    print(f"  bf16: 2e-1 (higher due to rounding differences in fused kernel)")
    
    results = []
    
    results.append(("Fused Norm + RoPE Correctness", test_fused_norm_rope_correctness()))
    results.append(("Module Interface", test_module_interface()))
    results.append(("Edge Cases", test_edge_cases()))
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:<40} {status}")
    print("=" * 80)
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)
