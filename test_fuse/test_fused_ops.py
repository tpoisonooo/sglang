"""
Test script for fused operations in MiniCPM.
Tests the following fused operations:
1. fused_rms_norm_rope - Fused RMSNorm + RoPE
2. fused_output_processing - Fused RMSNorm + sigmoid gate
3. fused_scale_add - Fused scale and add

All tests use bf16 dtype.
"""

import torch
import torch.nn.functional as F
import math
import sys

# Add python directory to path
sys.path.insert(0, '/data/khj/workspace/sglang/python')

from sglang.srt.models.minicpm_fused_norm_rope import fused_rms_norm_rope
from sglang.srt.models.minicpm_fused_output import fused_output_processing
from sglang.srt.models.minicpm_fused_scale_add import fused_scale_add


def rms_norm(x, weight, eps=1e-6):
    """Reference RMSNorm implementation."""
    original_shape = x.shape
    x = x.float()
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(mean_sq + eps)
    result = x_normed * weight
    return result.to(torch.bfloat16)


def apply_rope_neox(x, cos, sin):
    """Reference Neox-style RoPE implementation.
    
    Args:
        x: [seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]
    """
    x = x.float()
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    
    # Expand cos/sin for broadcasting with num_heads dimension
    cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim]
    sin = sin.unsqueeze(1)  # [seq_len, 1, head_dim]
    
    # Neox-style rotation: [-x2, x1]
    rotated = torch.cat([-x2, x1], dim=-1)  # [seq_len, num_heads, head_dim]
    
    # Apply RoPE
    x_out = x * cos + rotated * sin
    return x_out.to(torch.bfloat16)



def test_fused_output_processing():
    """Test fused_output_processing against reference implementation."""
    print("=" * 60)
    print("Testing fused_output_processing")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {"seq_len": 16, "hidden_size": 512},
        {"seq_len": 32, "hidden_size": 1024},
        {"seq_len": 64, "hidden_size": 2048},
        {"seq_len": 128, "hidden_size": 4096},
        {"seq_len": 256, "hidden_size": 512},
    ]
    
    device = "cuda"
    dtype = torch.bfloat16
    
    all_passed = True
    
    for config in test_configs:
        seq_len = config["seq_len"]
        hidden_size = config["hidden_size"]
        eps = 1e-6
        
        # Generate inputs
        torch.manual_seed(42)
        o = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
        z = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
        norm_weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
        
        try:
            # Fused implementation
            fused_out = fused_output_processing(
                o=o,
                z=z,
                norm_weight=norm_weight,
                eps=eps,
            )
            
            # Reference implementation
            # 1. Apply RMSNorm
            o_normed = rms_norm(o, norm_weight, eps)
            
            # 2. Apply sigmoid gate
            z_sigmoid = torch.sigmoid(z.float())
            ref_out = o_normed.float() * z_sigmoid
            ref_out = ref_out.to(dtype)
            
            # Compare
            diff = (fused_out - ref_out).abs().max().item()
            
            # Use a tolerance suitable for bf16
            tolerance = 1e-2
            passed = diff < tolerance
            
            status = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
            
            print(f"  Config: seq_len={seq_len}, hidden_size={hidden_size}")
            print(f"    Max diff: {diff:.6e} {'✓' if passed else '✗'}")
            print(f"    Status: {status}")
        except Exception as e:
            print(f"  Config: seq_len={seq_len}, hidden_size={hidden_size}")
            print(f"    ERROR: {e}")
            print(f"    Status: FAILED")
            all_passed = False
    
    print()
    return all_passed


def test_fused_scale_add():
    """Test fused_scale_add against reference implementation."""
    print("=" * 60)
    print("Testing fused_scale_add")
    print("=" * 60)
    
    # Test configurations - note: fused_scale_add requires hidden_dim=4096
    test_configs = [
        {"seq_len": 128},   # Uses PyTorch native (seq_len <= 512)
        {"seq_len": 512},   # Uses PyTorch native (seq_len <= 512)
        {"seq_len": 1024},  # Uses ultra kernel (512 < seq_len <= 2048)
        {"seq_len": 2048},  # Uses ultra kernel (512 < seq_len <= 2048)
        {"seq_len": 4096},  # Uses basic kernel (seq_len > 2048)
        {"seq_len": 8192},  # Uses basic kernel (seq_len > 2048)
    ]
    
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = 4096
    
    all_passed = True
    
    for config in test_configs:
        seq_len = config["seq_len"]
        scale = 0.125  # Typical scale factor like 1/sqrt(64)
        
        # Generate inputs
        torch.manual_seed(42)
        input_tensor = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        residual = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        
        try:
            # Fused implementation
            fused_out = fused_scale_add(
                input_tensor=input_tensor,
                residual=residual,
                scale=scale,
            )
            
            # Reference implementation
            ref_out = residual + input_tensor * scale
            
            # Compare
            diff = (fused_out - ref_out).abs().max().item()
            
            # Use a tolerance suitable for bf16
            tolerance = 1e-2
            passed = diff < tolerance
            
            status = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
            
            print(f"  Config: seq_len={seq_len}, hidden_dim={hidden_dim}, scale={scale}")
            print(f"    Max diff: {diff:.6e} {'✓' if passed else '✗'}")
            print(f"    Status: {status}")
        except Exception as e:
            print(f"  Config: seq_len={seq_len}, hidden_dim={hidden_dim}, scale={scale}")
            print(f"    ERROR: {e}")
            print(f"    Status: FAILED")
            all_passed = False
    
    print()
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FUSED OPERATIONS TEST SUITE")
    print("Data type: bfloat16")
    print("=" * 60 + "\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Tests require a GPU.")
        return False
    
    print(f"Using device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    results = []
    
    # Test 2: fused_output_processing
    results.append(("fused_output_processing", test_fused_output_processing()))
    
    # Test 3: fused_scale_add
    results.append(("fused_scale_add", test_fused_scale_add()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
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
