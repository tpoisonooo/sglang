"""
Autotuned Chunk GLA implementation.

Automatically selects between 2-kernel and 3-kernel based on sequence length
to achieve optimal performance across all sizes.
"""
import torch
from typing import Optional, Tuple

from chunk_gla_fused_output import chunk_simple_gla_fused_output, chunk_gla_fused_output
from chunk_gla_fused_all import chunk_simple_gla_fused_all, chunk_gla_fused_all


def chunk_simple_gla_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Autotuned Chunk GLA with RMSNorm and gate.
    
    Automatically selects best kernel implementation based on sequence length:
    - T <= 512: Uses 2-kernel (~1.4-1.7x speedup)
    - T == 1024: Uses 3-kernel (avoids occupancy dip)
    - T >= 2048: Uses 2-kernel (~1.07-1.16x speedup)
    
    Args:
        q, k, v: [B, T, H, D] - QKV tensors
        z: [B*T, H*V] - gate input
        norm_weight: [H*V] - RMSNorm weight
        g, g_gamma: decay parameters (optional)
        scale: attention scale (default: K^-0.5)
        initial_state: initial recurrent state
        output_final_state: whether to return final state
        cu_seqlens: cumulative sequence lengths for varlen
        eps: RMSNorm epsilon
        
    Returns:
        out: [B*T, H*V] - final output
        ht: final state (if output_final_state=True)
    """
    B, T = q.shape[0], q.shape[1]
    
    # Heuristic: T=1024 has occupancy issues with 2-kernel
    use_3kernel = (T == 1024)
    
    if use_3kernel:
        return chunk_simple_gla_fused_output(
            q, k, v, z, norm_weight,
            g=g, g_gamma=g_gamma, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            eps=eps
        )
    else:
        return chunk_simple_gla_fused_all(
            q, k, v, z, norm_weight,
            g=g, g_gamma=g_gamma, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            eps=eps
        )


def chunk_gla_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Autotuned Chunk GLA with full state support.
    
    Same as chunk_simple_gla_autotuned but with configurable chunk_size.
    """
    B, T = q.shape[0], q.shape[1]
    
    # Heuristic: T=1024 has occupancy issues with 2-kernel
    use_3kernel = (T == 1024)
    
    if use_3kernel:
        return chunk_gla_fused_output(
            q, k, v, z, norm_weight,
            g=g, g_gamma=g_gamma, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            eps=eps
        )
    else:
        return chunk_gla_fused_all(
            q, k, v, z, norm_weight,
            g=g, g_gamma=g_gamma, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            eps=eps
        )


# Alias for convenience
chunk_simple_gla = chunk_simple_gla_autotuned
chunk_gla = chunk_gla_autotuned
