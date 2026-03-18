"""
Fused Chunk GLA with Output Processing (RMSNorm + Gate)

Based on FLA's chunk_fwd_kernel_o, fused with:
1. Reshape from [B, T, H, V] to [B*T, H*V] 
2. RMSNorm
3. Sigmoid gate: o * sigmoid(z)

This is implemented as a two-kernel approach:
1. chunk_fwd_kernel_o_fused: Computes o and stores in 2D layout
2. kernel_fused_output_final: RMSNorm + gate (per-sequence position)
"""

import torch
import triton
import triton.language as tl
from typing import Optional

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


# =============================================================================
# Kernel 1: Modified chunk_fwd_o with 2D output
# =============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_fused(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,  # 2D output: [B*T, H*V]
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,  # Batch size for 2D output
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Modified chunk_fwd_kernel_o with 2D output [B*T, H*V].
    
    This is the first kernel in the fused pipeline.
    Output is in 2D layout ready for RMSNorm + gate.
    """
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # Input offsets (4D layout)
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale

    # Store to 2D output [B*T, H*V]
    # Row stride: H*V, Col stride: 1
    o_row_stride = H * V
    o_col_stride = 1
    o_row_offset = bos + i_t * BT
    o_col_offset = i_h * V + i_v * BV
    
    p_o = tl.make_block_ptr(
        o,
        (B * T, H * V),
        (o_row_stride, o_col_stride),
        (o_row_offset, o_col_offset),
        (BT, BV),
        (1, 0)
    )
    
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Kernel 2: RMSNorm + Gate (per sequence position)
# =============================================================================

@triton.jit
def kernel_fused_output_final(
    o_ptr,           # [B*T, H*V] - attention output
    z_ptr,           # [B*T, H*V] - gate input
    norm_weight_ptr, # [H*V] - RMSNorm weight
    out_ptr,         # [B*T, H*V] - final output
    hidden_size,
    eps,
    stride_o_seq,
    stride_z_seq,
    stride_out_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second kernel: RMSNorm + sigmoid gate per sequence position.
    
    Grid: (B*T,) - one block per sequence position
    """
    seq_idx = tl.program_id(0)
    
    # Base offsets
    o_base = seq_idx * stride_o_seq
    z_base = seq_idx * stride_z_seq
    out_base = seq_idx * stride_out_seq
    
    # First pass: compute sum of squares for RMS
    num_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_idx in range(num_blocks):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        
        o_val = tl.load(o_ptr + o_base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += o_val * o_val
    
    total_sum_sq = tl.sum(sum_sq, axis=0)
    mean_sq = total_sum_sq / hidden_size
    rms = tl.math.rsqrt(mean_sq + eps)
    
    # Second pass: apply RMSNorm and gate
    for block_idx in range(num_blocks):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        
        # Load and apply RMSNorm
        o_val = tl.load(o_ptr + o_base + offs, mask=mask, other=0.0).to(tl.float32)
        norm_w = tl.load(norm_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        
        o_val = o_val * rms * norm_w
        
        # Match reference: convert to bf16 and back
        o_val = o_val.to(out_ptr.dtype.element_ty).to(tl.float32)
        
        # Load z and apply sigmoid gate
        z_val = tl.load(z_ptr + z_base + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(z_val)
        o_val = o_val * gate
        
        # Store
        tl.store(out_ptr + out_base + offs, o_val.to(out_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def chunk_fwd_o_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Wrapper for chunk_fwd_kernel_o_fused with 2D output."""
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    if scale is None:
        scale = K ** -0.5

    # 2D output: [B*T, H*V]
    o = v.new_empty(B * T, H * V)
    
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    
    chunk_fwd_kernel_o_fused[grid](
        q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma, o=o,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, B=B, H=H, K=K, V=V, BT=BT,
    )
    return o


def fused_output_final(
    o: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Wrapper for kernel_fused_output_final (RMSNorm + gate)."""
    seq_len, hidden_size = o.shape
    
    out = torch.empty_like(o)
    
    # Choose block size
    BLOCK_SIZE = 1024 if hidden_size >= 1024 else 512
    
    grid = (seq_len,)
    
    kernel_fused_output_final[grid](
        o, z, norm_weight, out,
        hidden_size, eps,
        o.stride(0), z.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# =============================================================================
# Main Interface: Fused Chunk GLA + Output Processing
# =============================================================================

from fla.ops.common.chunk_h import chunk_fwd_h


def chunk_gla_fused_output(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Chunk GLA + Output Processing.
    
    Two-kernel fusion:
    1. chunk_fwd_h (from FLA): Compute hidden states
    2. chunk_fwd_o_fused: Compute output in 2D layout
    3. fused_output_final: RMSNorm + sigmoid gate
    
    Args:
        q, k, v: [B, T, H, D] - QKV tensors
        z: [B*T, H*V] - gate input
        norm_weight: [H*V] - RMSNorm weight
        g, g_gamma: decay parameters
        scale: attention scale
        initial_state: initial recurrent state
        output_final_state: whether to output final state
        cu_seqlens: cumulative sequence lengths for varlen
        chunk_size: chunk size for computation
        eps: RMSNorm epsilon
    
    Returns:
        out: [B*T, H*V] - final output after RMSNorm and gate
        ht: [B, H, K, V] - final state (if output_final_state=True)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    # Step 1: Compute hidden states (using FLA's kernel)
    h, ht = chunk_fwd_h(
        k=k, v=v, g=g, g_gamma=g_gamma,
        gk=None, gv=None,
        h0=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    )
    
    # Step 2: Compute output with 2D layout
    o = chunk_fwd_o_fused(
        q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma,
        scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    )
    
    # Step 3: RMSNorm + gate
    out = fused_output_final(o, z, norm_weight, eps)
    
    return out, ht


@torch.compiler.disable
def chunk_simple_gla_fused_output(
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
    head_first: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Main interface matching FLA's API style."""
    if head_first:
        raise DeprecationWarning("head_first is deprecated.")
    
    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    T = q.shape[1]
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))
    
    out, final_state = chunk_gla_fused_output(
        q=q, k=k, v=v, z=z, norm_weight=norm_weight,
        g=g, g_gamma=g_gamma, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, eps=eps,
    )
    
    return out, final_state
