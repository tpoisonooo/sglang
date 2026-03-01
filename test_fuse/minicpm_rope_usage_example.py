"""Usage example for MiniCPM Triton RoPE.

This file shows how to integrate the Triton-based RoPE into MiniCPM model.
"""

# =============================================================================
# Option 1: Direct replacement in MiniCPMAttention (recommended)
# =============================================================================

# In minicpm.py, replace the get_rope import and usage:

"""
# Original imports:
from sglang.srt.layers.rotary_embedding import get_rope

# New imports:
from sglang.srt.models.minicpm_rope import MiniCPMRoPE


class MiniCPMAttention(nn.Module):
    def __init__(...):
        ...
        if self.attn_use_rope:
            # Original code:
            # self.rotary_emb = get_rope(
            #     self.head_dim,
            #     rotary_dim=self.head_dim,
            #     max_position=max_position_embeddings,
            #     base=rope_theta,
            #     rope_scaling=rope_scaling,
            # )
            
            # New code (when num_heads=32, head_dim=128, num_kv_heads=32):
            if (self.num_heads == 32 and 
                self.head_dim == 128 and 
                self.num_kv_heads == 32):
                self.rotary_emb = MiniCPMRoPE(
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                )
            else:
                # Fallback to original for other configurations
                self.rotary_emb = get_rope(
                    self.head_dim,
                    rotary_dim=self.head_dim,
                    max_position=max_position_embeddings,
                    base=rope_theta,
                    rope_scaling=rope_scaling,
                )
        ...
    
    def forward(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.attn_use_rope:
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
            # This works the same way as before
            q, k = self.rotary_emb(positions, q, k)
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        attn_output = self.attn(q, k, v, forward_batch)
        ...
"""


# =============================================================================
# Option 2: Direct usage (functional API)
# =============================================================================

def example_direct_usage():
    """Example of using the functional API directly."""
    import torch
    from sglang.srt.models.minicpm_rope import apply_minicpm_rope, MiniCPMRoPEBaseline
    
    # Parameters
    num_tokens = 128
    num_heads = 32
    head_dim = 128
    num_kv_heads = 32
    max_position = 8192
    
    # Create cache
    baseline = MiniCPMRoPEBaseline(
        max_position_embeddings=max_position,
        rope_theta=10000.0,
    ).cuda()
    
    # Create inputs
    q = torch.randn(num_tokens, num_heads * head_dim, dtype=torch.float32, device="cuda")
    k = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, max_position, (num_tokens,), device="cuda")
    
    # Apply RoPE using functional API
    q_rotated, k_rotated = apply_minicpm_rope(
        positions, q, k, baseline.cos_sin_cache
    )
    
    print(f"Q shape: {q_rotated.shape}")  # [128, 4096]
    print(f"K shape: {k_rotated.shape}")  # [128, 4096]


# =============================================================================
# Option 3: Using the module class
# =============================================================================

def example_module_usage():
    """Example of using the MiniCPMRoPE module class."""
    import torch
    from sglang.srt.models.minicpm_rope import MiniCPMRoPE
    
    # Create RoPE module
    rope = MiniCPMRoPE(
        max_position_embeddings=8192,
        rope_theta=10000.0,
    ).cuda()
    
    # Create inputs
    num_tokens = 128
    q = torch.randn(num_tokens, 32 * 128, dtype=torch.float32, device="cuda")
    k = torch.randn(num_tokens, 32 * 128, dtype=torch.float32, device="cuda")
    positions = torch.arange(num_tokens, device="cuda")
    
    # Apply RoPE
    q_rotated, k_rotated = rope(positions, q, k)
    
    print(f"Q shape: {q_rotated.shape}")  # [128, 4096]
    print(f"K shape: {k_rotated.shape}")  # [128, 4096]


# =============================================================================
# Performance comparison
# =============================================================================

def benchmark():
    """Benchmark the Triton implementation vs baseline."""
    import torch
    import time
    from sglang.srt.models.minicpm_rope import MiniCPMRoPE, MiniCPMRoPEBaseline
    
    # Create modules
    triton_rope = MiniCPMRoPE(max_position_embeddings=8192, rope_theta=10000.0).cuda()
    baseline_rope = MiniCPMRoPEBaseline(max_position_embeddings=8192, rope_theta=10000.0).cuda()
    
    # Test configurations
    configs = [
        ("Decode (batch=1, seq=1)", 1, 1),
        ("Decode (batch=16, seq=1)", 16, 1),
        ("Prefill (batch=1, seq=512)", 1, 512),
        ("Prefill (batch=1, seq=2048)", 1, 2048),
    ]
    
    print("Performance Benchmark:")
    print("=" * 60)
    
    for name, batch_size, seq_len in configs:
        num_tokens = batch_size * seq_len
        
        q = torch.randn(num_tokens, 4096, dtype=torch.float32, device="cuda")
        k = torch.randn(num_tokens, 4096, dtype=torch.float32, device="cuda")
        positions = torch.randint(0, 8192, (num_tokens,), device="cuda")
        
        # Warmup
        for _ in range(10):
            _ = triton_rope(positions, q.clone(), k.clone())
            _ = baseline_rope(positions, q.clone(), k.clone())
        
        torch.cuda.synchronize()
        
        # Benchmark baseline
        start = time.perf_counter()
        for _ in range(100):
            _ = baseline_rope(positions, q.clone(), k.clone())
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) / 100 * 1000
        
        # Benchmark triton
        start = time.perf_counter()
        for _ in range(100):
            _ = triton_rope(positions, q.clone(), k.clone())
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / 100 * 1000
        
        speedup = baseline_time / triton_time
        print(f"{name}:")
        print(f"  Baseline: {baseline_time:.3f} ms")
        print(f"  Triton:   {triton_time:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print()


if __name__ == "__main__":
    print("Running examples...\n")
    
    print("=" * 60)
    print("Example 1: Direct usage")
    print("=" * 60)
    example_direct_usage()
    
    print("\n" + "=" * 60)
    print("Example 2: Module usage")
    print("=" * 60)
    example_module_usage()
    
    print("\n" + "=" * 60)
    print("Example 3: Benchmark")
    print("=" * 60)
    benchmark()
