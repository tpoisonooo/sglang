"""Triton-based RoPE (Rotary Position Embedding) for MiniCPM.

This module provides a specialized triton implementation for RoPE with fixed parameters:
- num_heads: 32
- head_dim: 128
- num_kv_heads: 32

Usage:
    # Replace the original RoPE in MiniCPM:
    
    # Original code:
    # self.rotary_emb = get_rope(
    #     self.head_dim,
    #     rotary_dim=self.head_dim,
    #     max_position=max_position_embeddings,
    #     base=rope_theta,
    #     rope_scaling=rope_scaling,
    # )
    # q, k = self.rotary_emb(positions, q, k)
    
    # New code:
    from minicpm_rope import MiniCPMRoPE
    
    self.rotary_emb = MiniCPMRoPE(
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    q, k = self.rotary_emb(positions, q, k)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# Fixed parameters for MiniCPM
NUM_HEADS = 32
HEAD_DIM = 128
NUM_KV_HEADS = 32
ROTARY_DIM = HEAD_DIM  # Full rotary dimension


@triton.jit
def _minicpm_rope_forward_kernel(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride,
    k_stride,
    cos_sin_stride,
):
    """Triton kernel for MiniCPM RoPE (Neox-style).
    
    Uses 2D grid: (num_tokens, total_heads) where total_heads = NUM_HEADS + NUM_KV_HEADS = 64
    
    Fixed parameters (as tl.constexpr for triton):
    - NUM_HEADS = 32
    - NUM_KV_HEADS = 32
    - HEAD_DIM = 128
    - HALF_HEAD_DIM = 64 (each rotation pair uses 2 elements)
    
    RoPE formula (Neox-style):
        For each pair (x1, x2) at dimension d:
        y1 = x1 * cos(d) - x2 * sin(d)
        y2 = x2 * cos(d) + x1 * sin(d)
    
    Args:
        q_ptr: Query tensor pointer [num_tokens, NUM_HEADS * HEAD_DIM]
        k_ptr: Key tensor pointer [num_tokens, NUM_KV_HEADS * HEAD_DIM]
        cos_sin_cache_ptr: Cos/Sin cache pointer [max_position, HEAD_DIM]
        positions_ptr: Position indices pointer [num_tokens]
        q_stride: Stride for q tensor (usually NUM_HEADS * HEAD_DIM)
        k_stride: Stride for k tensor (usually NUM_KV_HEADS * HEAD_DIM)
        cos_sin_stride: Stride for cos_sin_cache (usually HEAD_DIM)
    """
    # Define constants as tl.constexpr for triton
    NUM_HEADS: tl.constexpr = 32
    NUM_KV_HEADS: tl.constexpr = 32
    HEAD_DIM: tl.constexpr = 128
    HALF_HEAD_DIM: tl.constexpr = 64
    
    # 2D program IDs
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Determine if processing Q or K based on head_idx
    # First NUM_HEADS (0-31) are Q, next NUM_KV_HEADS (32-63) are K
    is_q = head_idx < NUM_HEADS
    
    if is_q:
        actual_head_idx = head_idx
        base_ptr = q_ptr
        stride = q_stride
    else:
        actual_head_idx = head_idx - NUM_HEADS
        if actual_head_idx >= NUM_KV_HEADS:
            return
        base_ptr = k_ptr
        stride = k_stride
    
    # Load position for this token
    position = tl.load(positions_ptr + token_idx)
    
    # Get cos/sin cache for this position
    # cos_sin_cache shape: [max_position, head_dim]
    # cos is first half (indices 0-63), sin is second half (indices 64-127)
    cos_ptr = cos_sin_cache_ptr + position * cos_sin_stride
    sin_ptr = cos_ptr + HALF_HEAD_DIM
    
    # Base offset for this head in the token's data
    base_offset = token_idx * stride + actual_head_idx * HEAD_DIM
    
    # Process all dimension pairs (64 pairs for 128-dim head)
    # Each thread processes one pair (x1, x2)
    dim_idx = tl.arange(0, HALF_HEAD_DIM)
    
    # Load cos and sin values for all dimensions
    cos_vals = tl.load(cos_ptr + dim_idx)
    sin_vals = tl.load(sin_ptr + dim_idx)
    
    # Load x1 (first half) and x2 (second half) of the head
    x1_offset = base_offset + dim_idx
    x2_offset = base_offset + HALF_HEAD_DIM + dim_idx
    
    x1 = tl.load(base_ptr + x1_offset)
    x2 = tl.load(base_ptr + x2_offset)
    
    # Apply Neox-style rotation:
    # y1 = x1 * cos - x2 * sin
    # y2 = x2 * cos + x1 * sin
    y1 = x1 * cos_vals - x2 * sin_vals
    y2 = x2 * cos_vals + x1 * sin_vals
    
    # Store results back
    tl.store(base_ptr + x1_offset, y1)
    tl.store(base_ptr + x2_offset, y2)


class MiniCPMRoPE(torch.nn.Module):
    """Triton-based RoPE for MiniCPM with fixed parameters.
    
    This is a drop-in replacement for the standard RoPE used in MiniCPM.
    
    Fixed parameters:
    - num_heads: 32
    - head_dim: 128
    - num_kv_heads: 32
    - rotary_dim: 128 (full head_dim)
    - is_neox_style: True
    
    Equivalent to:
        from sglang.srt.layers.rotary_embedding import get_rope
        self.rotary_emb = get_rope(
            head_dim=128,
            rotary_dim=128,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        q, k = self.rotary_emb(positions, q, k)
    
    Args:
        max_position_embeddings: Maximum sequence length (default: 8192)
        rope_theta: Base for rotary embeddings (default: 10000.0)
        rope_scaling: RoPE scaling configuration (default: None)
    """
    
    def __init__(
        self,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: dict = None,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.head_dim = HEAD_DIM
        self.rotary_dim = HEAD_DIM
        
        # Compute cos/sin cache
        cache = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
    
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequency for RoPE."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )
        return inv_freq
    
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute cos and sin cache for all positions."""
        inv_freq = self._compute_inv_freq()
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.
        
        Args:
            positions: [num_tokens] position indices (int64)
            query: [num_tokens, num_heads * head_dim] = [num_tokens, 4096]
            key: [num_tokens, num_kv_heads * head_dim] = [num_tokens, 4096]
            
        Returns:
            query: [num_tokens, num_heads * head_dim] rotated query (in-place)
            key: [num_tokens, num_kv_heads * head_dim] rotated key (in-place)
        """
        num_tokens = positions.shape[0]
        
        # Ensure tensors are contiguous and on the same device
        query = query.contiguous()
        key = key.contiguous()
        cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
        
        # Launch triton kernel with 2D grid
        # Grid: (num_tokens, NUM_HEADS + NUM_KV_HEADS) = (num_tokens, 64)
        # Each block processes one head of one token
        grid = (num_tokens, NUM_HEADS + NUM_KV_HEADS)
        
        _minicpm_rope_forward_kernel[grid](
            query,
            key,
            cos_sin_cache,
            positions,
            query.stride(0),
            key.stride(0),
            cos_sin_cache.stride(0),
        )
        
        return query, key


class MiniCPMRoPEBaseline(torch.nn.Module):
    """Baseline PyTorch implementation of RoPE for MiniCPM.
    
    This matches the behavior of RotaryEmbedding.forward_native for comparison
    and testing purposes.
    """
    
    def __init__(
        self,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: dict = None,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.head_dim = HEAD_DIM
        self.rotary_dim = HEAD_DIM
        
        # Compute cos/sin cache
        cache = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
    
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequency."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )
        return inv_freq
    
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute cos and sin cache."""
        inv_freq = self._compute_inv_freq()
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
    
    def _rotate_neox(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims (Neox style)."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding.
        
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
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors (baseline implementation).
        
        Args:
            positions: [num_tokens] position indices
            query: [num_tokens, num_heads * head_dim]
            key: [num_tokens, num_kv_heads * head_dim]
            
        Returns:
            query: [num_tokens, num_heads * head_dim] rotated query
            key: [num_tokens, num_kv_heads * head_dim] rotated key
        """
        num_tokens = positions.shape[0]
        
        # Get cos/sin for positions
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # Process query
        query_shape = query.shape
        query = query.view(num_tokens, NUM_HEADS, self.head_dim)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self._apply_rotary_emb(query_rot, cos, sin)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        
        # Process key
        key_shape = key.shape
        key = key.view(num_tokens, NUM_KV_HEADS, self.head_dim)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self._apply_rotary_emb(key_rot, cos, sin)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        
        return query, key


def apply_minicpm_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional API for MiniCPM RoPE.
    
    This is a stateless function that applies RoPE using the provided cache.
    
    Args:
        positions: [num_tokens] position indices
        query: [num_tokens, 4096] (32 heads * 128 dim)
        key: [num_tokens, 4096] (32 heads * 128 dim)
        cos_sin_cache: [max_position, 128] cos/sin cache
        
    Returns:
        Rotated query and key tensors (in-place modification)
    """
    num_tokens = positions.shape[0]
    
    query = query.contiguous()
    key = key.contiguous()
    cos_sin_cache = cos_sin_cache.to(query.device, dtype=query.dtype)
    
    grid = (num_tokens, NUM_HEADS + NUM_KV_HEADS)
    
    _minicpm_rope_forward_kernel[grid](
        query,
        key,
        cos_sin_cache,
        positions,
        query.stride(0),
        key.stride(0),
        cos_sin_cache.stride(0),
    )
    
    return query, key
