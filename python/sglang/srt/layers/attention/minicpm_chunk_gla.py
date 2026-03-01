"""
MiniCPM Chunk Simple GLA Implementation (Inference-Only, Simplified)

This module contains a simplified chunk_simple_gla implementation for inference only.
- No backward pass support
- output_final_state is always True

Input shapes (from fla.shape log):
- q: [batch, seq_len, num_heads, head_dim] = [1, seq_len, 32, 128]
- k: [batch, seq_len, num_heads, head_dim] = [1, seq_len, 32, 128]
- v: [batch, seq_len, num_heads, head_dim] = [1, seq_len, 32, 128]
- g_gamma: [num_heads] = [32]

Where batch=1 (fixed), num_heads=32 (fixed), head_dim=128 (fixed), seq_len is variable.
"""

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Try to import from fla.utils, provide fallback if not available
try:
    from fla.utils import autocast_custom_fwd, input_guard
    from fla.ops.utils import prepare_chunk_offsets, prepare_chunk_indices
    from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem
    FLA_UTILS_AVAILABLE = True
except ImportError:
    FLA_UTILS_AVAILABLE = False
    # Define minimal fallback implementations
    def input_guard(func):
        return func
    
    def autocast_custom_fwd(func):
        return func
    
    def check_shared_mem(arch=None, device_idx=None):
        # Assume modern GPU with good shared memory
        return True
    
    autotune_cache_kwargs = {}
    IS_NVIDIA_HOPPER = False


# ============================================================================
# Configuration Constants
# ============================================================================

# Block sizes for chunk processing - tuned for head_dim=128
BKV_LIST = [32, 64] if check_shared_mem() else [16, 32]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]

# Default chunk size - can be tuned for performance
DEFAULT_CHUNK_SIZE = 64


# ============================================================================
# Triton Kernels for chunk_fwd_h (hidden state computation)
# ============================================================================

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    # Note: STORE_FINAL_STATE is always True for inference
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    # STORE_FINAL_STATE is always True, no need for constexpr
    IS_VARLEN: tl.constexpr,
):
    """
    Forward kernel for computing hidden states h.
    
    This kernel computes the recurrent hidden states for the chunk-based GLA.
    For each chunk, it:
    1. Loads the initial state (if provided)
    2. Processes tokens within the chunk, accumulating k^T @ v
    3. Applies decay (g_gamma for Simple GLA)
    4. Stores intermediate states and final state (always stored for inference)
    
    Shapes:
        k: [B, T, H, K] - keys
        v: [B, T, H, V] - values  
        h: [B, NS, H, K, V] - intermediate hidden states (NS = num splits)
        g_gamma: [H] - head-wise decay rates
        h0: [N, H, K, V] - initial states (N = num sequences)
        ht: [N, H, K, V] - final states (always output)
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV] - accumulator for hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // NTS
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        if i_t % NTS == 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        # scalar decay using g_gamma (Simple GLA path)
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_h *= tl.exp(b_g_last)
            b_v = (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= tl.exp(b_g_last)
            b_v = (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay (not used in Simple GLA)
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_h *= tl.exp(b_gk_last)[:, None]

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay for values (not used in Simple GLA)
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_h *= tl.exp(b_gv_last)[None, :]

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * tl.exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        # h += k^T @ v
        b_h += tl.dot(b_k, b_v)

    # Always store final state for inference
    p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Triton Kernels for chunk_fwd_o (output computation)
# ============================================================================

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
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
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
    Forward kernel for computing output o.
    
    This kernel computes the attention output for each chunk using:
    o = softmax(q @ k^T) @ v + q @ h
    
    Where h is the accumulated hidden state from previous chunks.
    
    Shapes:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - keys
        v: [B, T, H, V] - values
        h: [B, NS, H, K, V] - intermediate hidden states
        g_gamma: [H] - head-wise decay rates
        o: [B, T, H, V] - output
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

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    # o = (q @ h + softmax(q @ k^T) @ v) * scale
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Python wrapper functions
# ============================================================================

def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    h0: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    split_size: Optional[int] = None,
    states_in_fp32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute forward hidden states for chunk-based GLA.
    
    Note: This is an inference-only implementation. output_final_state is always True.
    
    Args:
        k: Keys of shape [B, T, H, K]
        v: Values of shape [B, T, H, V]
        g: Forget gates of shape [B, T, H] (optional)
        g_gamma: Log decay of shape [H] for head-wise decay (optional)
        gk: Key gates of shape [B, T, H, K] (optional, not used in Simple GLA)
        gv: Value gates of shape [B, T, H, V] (optional, not used in Simple GLA)
        h0: Initial state of shape [N, H, K, V] (optional)
        cu_seqlens: Cumulative sequence lengths for variable-length inputs
        chunk_size: Size of each chunk (default 64)
        split_size: Size of each split for storing intermediate states
        states_in_fp32: Whether to store states in fp32
        
    Returns:
        h: Intermediate hidden states of shape [B, NS, H, K, V]
        ht: Final state of shape [N, H, K, V] (always returned)
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    
    # N: the actual number of sequences in the batch
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        if FLA_UTILS_AVAILABLE:
            split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        else:
            # Fallback implementation
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            split_offsets = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device),
                torch.cumsum((seqlens + BS - 1) // BS, dim=0)
            ])
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float)  # Always output final state
    
    def grid(meta): 
        return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    
    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return h, ht


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Compute output for chunk-based GLA.
    
    Args:
        q: Queries of shape [B, T, H, K]
        k: Keys of shape [B, T, H, K]
        v: Values of shape [B, T, H, V]
        h: Intermediate hidden states of shape [B, NS, H, K, V]
        g: Forget gates of shape [B, T, H] (optional)
        g_gamma: Log decay of shape [H] (optional)
        scale: Scale factor for attention scores
        cu_seqlens: Cumulative sequence lengths for variable-length inputs
        chunk_size: Size of each chunk
        
    Returns:
        o: Output of shape [B, T, H, V]
    """
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    
    if FLA_UTILS_AVAILABLE:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    else:
        # Fallback: create chunk indices
        if cu_seqlens is not None:
            chunk_indices = []
            for i in range(len(cu_seqlens) - 1):
                bos, eos = cu_seqlens[i].item(), cu_seqlens[i+1].item()
                for t in range((eos - bos + BT - 1) // BT):
                    chunk_indices.append([i, t])
            chunk_indices = torch.tensor(chunk_indices, dtype=torch.int32, device=q.device)
        else:
            chunk_indices = None
    
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K ** -0.5

    o = torch.empty_like(v)
    
    def grid(meta): 
        return (triton.cdiv(V, meta['BV']), NT, B * H)
    
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for chunk_simple_gla (inference-only).
    
    Note: output_final_state is always True for inference.
    
    Args:
        q: Queries of shape [B, T, H, K]
        k: Keys of shape [B, T, H, K]
        v: Values of shape [B, T, H, V]
        g: Forget gates of shape [B, T, H] (optional)
        g_gamma: Log decay of shape [H] for Simple GLA (optional)
        scale: Scale factor for attention scores
        initial_state: Initial state of shape [N, H, K, V]
        cu_seqlens: Cumulative sequence lengths for variable-length inputs
        chunk_size: Size of each chunk
        
    Returns:
        o: Output of shape [B, T, H, V]
        ht: Final state of shape [N, H, K, V] (always returned)
    """
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return o, ht


# ============================================================================
# Simplified Inference-Only Function (No Autograd, No Backward)
# ============================================================================

@torch.compiler.disable
def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Chunk-based Simple Gated Linear Attention (Simple GLA) - Inference Only.
    
    This is a simplified inference-only implementation:
    - No backward pass support
    - output_final_state is always True (returns final state)
    
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        g_gamma (torch.Tensor):
            Log decay of shape `[H]`.
            Head-wise data-independent decay is used if `g_gamma` is provided.
            Only one of `g` or `g_gamma` should be provided.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` (always returned for inference).
            
    Note:
        For MiniCPM models with the following fixed dimensions:
        - batch_size = 1
        - num_heads (H) = 32
        - head_dim (K = V) = 128
        - seq_len (T) = variable (depends on input)
        
        The implementation uses:
        - chunk_size = 64 (default, auto-tuned based on sequence length)
        - g_gamma for head-wise decay (Simple GLA variant)
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )
    
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
        )
    
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    
    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    # Auto-determine chunk size based on sequence length
    T = q.shape[1]
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))
    
    # Direct forward call (no autograd)
    o, final_state = chunk_simple_gla_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    
    return o.to(q.dtype), final_state


# ============================================================================
# Optimization Notes for Blackwell (B100/B200) GPUs
# ============================================================================
"""
OPTIMIZATION GUIDE FOR BLACKWELL ARCHITECTURE (Compute Capability 10.0+)

The following optimizations are recommended for Blackwell GPUs without modifying
the internal kernel implementations:

1. CHUNK SIZE TUNING
   - Current default: 64
   - For Blackwell with 6000D (likely high-end), try: 128 or 256
   - Larger chunks reduce kernel launch overhead and improve tensor core utilization
   - Trade-off: Larger chunks use more shared memory per SM

2. BLOCK SIZE (BK, BV) TUNING
   - Current: [32, 64] for K=128, V=128
   - For Blackwell, try: [64, 128] or [128, 128]
   - Blackwell has increased shared memory capacity (up to 228KB per SM)
   - Larger blocks improve data reuse and reduce global memory traffic

3. NUM_WARPS and NUM_STAGES
   - Current: [1, 2, 4, 8] warps, [2, 3, 4] stages
   - For Blackwell, try: [4, 8] warps (higher occupancy)
   - Increase num_stages to [3, 4, 5] for better latency hiding
   - Blackwell's improved async copy engines benefit from more stages

4. FP8 SUPPORT (Future)
   - Blackwell has native FP8 tensor cores
   - When FP8 is supported in PyTorch/Triton, quantize q, k, v to FP8
   - Expected speedup: ~2x for memory-bound operations

5. PERSISTENT KERNELS (Advanced)
   - For very long sequences, consider persistent kernel patterns
   - Keep intermediate states (h) in shared memory across chunks
   - Requires careful occupancy calculation for Blackwell's 128 SMs

6. CUDA GRAPH CAPTURE
   - The current implementation supports CUDA graphs via @torch.compiler.disable
   - For decode phase (seq_len=1), capture CUDA graphs for common batch sizes
   - Blackwell benefits more from graphs due to lower kernel launch overhead

7. MEMORY LAYOUT
   - Ensure q, k, v are contiguous in memory for coalesced access
   - For variable-length sequences, use cu_seqlens with batch_size=1
   - Consider using torch.compile() with "reduce-overhead" mode for the wrapper

8. SPECIFIC RECOMMENDATIONS FOR MINICPM SHAPES
   - Input: [1, seq_len, 32, 128] where seq_len varies
   - For seq_len < 128: Use fused_recurrent mode (already implemented)
   - For seq_len >= 128: Use chunk mode with these Blackwell-specific configs:
     
     chunk_size = 128  (instead of 64)
     BK = 128          (full head_dim)
     BV = 128          (full head_dim)
     num_warps = 8
     num_stages = 4

9. AUTO-TUNING STRATEGY
   - The @triton.autotune decorator already handles this
   - For production, run autotune once and cache the best configs
   - Clear cache when upgrading Triton or CUDA versions

10. MULTI-GPU SCALING (6000D likely has multiple GPUs)
    - Use tensor parallelism across heads (H=32 splits nicely)
    - Each GPU handles H/tp_size heads
    - All-reduce the output o across GPUs
"""
