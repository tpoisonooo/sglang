"""
Autotuned Chunk GLA implementation.

Uses the proven-correct 3-kernel approach for all sequence lengths.
(2-kernel approach has numerical issues and is disabled)
"""
import torch
from typing import Optional, Tuple

from chunk_gla_fused_output import chunk_simple_gla_fused_output, chunk_gla_fused_output


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
    Chunk GLA with RMSNorm and gate.
    
    Uses the proven-correct 3-kernel approach:
    1. chunk_fwd_h (FLA): compute hidden states
    2. chunk_fwd_o_fused: compute output in 2D layout
    3. fused_output_final: apply RMSNorm + sigmoid gate
    
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
    # Always use the proven-correct 3-kernel implementation
    return chunk_simple_gla_fused_output(
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
    Chunk GLA with full state support.
    
    Same as chunk_simple_gla_autotuned but with configurable chunk_size.
    Uses the proven-correct 3-kernel approach.
    """
    # Always use the proven-correct 3-kernel implementation
    return chunk_gla_fused_output(
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
