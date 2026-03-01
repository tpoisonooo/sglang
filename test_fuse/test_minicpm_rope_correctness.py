"""Test correctness of MiniCPM Triton RoPE implementation."""

import torch
import sys

# Add python path
sys.path.insert(0, '/data/khj/workspace/sglang/python')

from sglang.srt.models.minicpm_rope import MiniCPMRoPE, MiniCPMRoPEBaseline


def test_correctness():
    """Test that triton implementation matches baseline."""
    print("=" * 70)
    print("Testing correctness of MiniCPM RoPE implementations")
    print("=" * 70)
    
    # Parameters
    num_heads = 32
    head_dim = 128
    num_kv_heads = 32
    max_position = 8192
    rope_theta = 10000.0
    
    # Create modules
    triton_rope = MiniCPMRoPE(
        max_position_embeddings=max_position,
        rope_theta=rope_theta,
    ).cuda()
    
    baseline_rope = MiniCPMRoPEBaseline(
        max_position_embeddings=max_position,
        rope_theta=rope_theta,
    ).cuda()
    
    # Copy cache to ensure same values
    triton_rope.cos_sin_cache.copy_(baseline_rope.cos_sin_cache)
    
    # Test cases
    test_cases = [
        ("Small batch", 1, 128),
        ("Medium batch", 8, 128),
        ("Large batch", 32, 128),
        ("Single token", 1, 1),
        ("Long sequence", 2, 4096),
        ("Batch=4, Seq=1024", 4, 1024),
        ("Batch=16, Seq=64", 16, 64),
    ]
    
    all_passed = True
    tolerance = 1e-4
    
    for name, batch_size, seq_len in test_cases:
        print(f"\nTest: {name} (batch={batch_size}, seq_len={seq_len})")
        
        num_tokens = batch_size * seq_len
        
        # Create inputs
        torch.manual_seed(42)
        q = torch.randn(num_tokens, num_heads * head_dim, dtype=torch.float32, device="cuda")
        k = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=torch.float32, device="cuda")
        positions = torch.randint(0, max_position, (num_tokens,), device="cuda")
        
        # Baseline
        q_baseline = q.clone()
        k_baseline = k.clone()
        q_out_base, k_out_base = baseline_rope(positions, q_baseline, k_baseline)
        
        # Triton
        q_triton = q.clone()
        k_triton = k.clone()
        q_out_triton, k_out_triton = triton_rope(positions, q_triton, k_triton)
        
        # Compare
        q_diff = torch.abs(q_out_base - q_out_triton).max().item()
        k_diff = torch.abs(k_out_base - k_out_triton).max().item()
        
        q_pass = q_diff < tolerance
        k_pass = k_diff < tolerance
        
        status = "PASS" if (q_pass and k_pass) else "FAIL"
        print(f"  Q max diff: {q_diff:.2e} {'✓' if q_pass else '✗'}")
        print(f"  K max diff: {k_diff:.2e} {'✓' if k_pass else '✗'}")
        print(f"  Result: {status}")
        
        if not (q_pass and k_pass):
            all_passed = False
            # Print more details for debugging
            print(f"  Q baseline first 10: {q_out_base[0, :10]}")
            print(f"  Q triton first 10: {q_out_triton[0, :10]}")
    
    print("\n" + "=" * 70)
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 70)
    
    return all_passed


def test_gradient():
    """Test gradient flow through triton implementation."""
    print("\n" + "=" * 70)
    print("Testing gradient flow")
    print("=" * 70)
    
    num_tokens = 64
    num_heads = 32
    head_dim = 128
    num_kv_heads = 32
    max_position = 8192
    
    triton_rope = MiniCPMRoPE(
        max_position_embeddings=max_position,
        rope_theta=10000.0,
    ).cuda()
    
    # Create inputs with gradient
    q = torch.randn(num_tokens, num_heads * head_dim, dtype=torch.float32, device="cuda", requires_grad=True)
    k = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=torch.float32, device="cuda", requires_grad=True)
    positions = torch.arange(num_tokens, device="cuda")
    
    # Forward pass
    q_out, k_out = triton_rope(positions, q, k)
    
    # Backward pass
    loss = (q_out.sum() + k_out.sum())
    loss.backward()
    
    # Check gradients exist and are not NaN
    q_grad_valid = q.grad is not None and not torch.isnan(q.grad).any()
    k_grad_valid = k.grad is not None and not torch.isnan(k.grad).any()
    
    print(f"Q gradient valid: {'✓' if q_grad_valid else '✗'}")
    print(f"K gradient valid: {'✓' if k_grad_valid else '✗'}")
    
    if q_grad_valid and k_grad_valid:
        print(f"Gradient test: PASS ✓")
        return True
    else:
        print(f"Gradient test: FAIL ✗")
        return False


if __name__ == "__main__":
    correctness_passed = test_correctness()
    gradient_passed = test_gradient()
    
    print("\n" + "=" * 70)
    if correctness_passed and gradient_passed:
        print("All tests completed successfully! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)
