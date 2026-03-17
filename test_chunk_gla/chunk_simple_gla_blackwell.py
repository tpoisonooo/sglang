# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for NVIDIA Blackwell RTX 6000D - Forward Only Optimized Version
#
# Hardware Target: NVIDIA Blackwell RTX 6000D
# - Compute Capability: sm_120 (12.0)
# - SMs: 156
# - Shared Memory: 128 KB per SM
# - Memory: GDDR7, ~1568 GB/s
# - L2 Cache: 128 MB
#
# Optimizations for Blackwell:
# 1. Larger tile sizes (BK/BV up to 128) leveraging 128KB shared memory
# 2. Optimized num_warps/num_stages for sm_120
# 3. Optimized chunk_size selection for GDDR7 bandwidth
# 4. Removed backward kernels (not needed)
# 5. FP4 support preparation (for future)

import warnings
from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_offsets, prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs


# ============================================================================
# Blackwell 6000D Specific Configurations
# ============================================================================

# Blackwell has 128KB shared memory per SM, allowing larger tiles
# BK/BV can be up to 128 for K,V dimensions
BLACKWELL_BK_LIST = [64, 128]  # Key dimension tile sizes
BLACKWELL_BV_LIST = [64, 128]  # Value dimension tile sizes

# Warp and stage configurations optimized for sm_120
# Blackwell has improved scheduler, can benefit from different warp distribution
BLACKWELL_NUM_WARPS = [4, 8]  # Focus on higher occupancy
BLACKWELL_NUM_STAGES = [2, 3, 4]  # Pipeline stages

# Chunk size selection for Blackwell GDDR7
# Higher bandwidth allows larger chunks for better compute efficiency
BLACKWELL_CHUNK_SIZES = [64, 128]  # Preferred chunk sizes


def is_blackwell() -> bool:
    """Check if running on Blackwell architecture (sm_120)."""
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] == 12  # sm_120


def get_optimal_chunk_size_blackwell(seq_len: int) -> int:
    """
    Optimized chunk size selection for Blackwell GDDR7.
    
    Blackwell has higher memory bandwidth (GDDR7), so we can use
    larger chunks to improve compute efficiency without being
    memory-bound.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Optimal chunk size for Blackwell
    """
    if not is_blackwell():
        # Fallback to standard sizing
        return min(64, max(16, triton.next_power_of_2(seq_len)))
    
    # For Blackwell, use larger chunks for better compute utilization
    if seq_len <= 16:
        return 16
    elif seq_len <= 32:
        return 32
    elif seq_len <= 64:
        return 64
    else:
        # For longer sequences, use 128 for better compute efficiency
        # GDDR7 bandwidth can sustain the larger working set
        return 128


# ============================================================================
# Kernel: chunk_fwd_h - Compute hidden states h (Blackwell Optimized)
# ============================================================================

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config(
            {'BK': BK, 'BV': BV}, 
            num_warps=num_warps, 
            num_stages=num_stages
        )
        for BK in BLACKWELL_BK_LIST
        for BV in BLACKWELL_BV_LIST
        for num_warps in BLACKWELL_NUM_WARPS
        for num_stages in BLACKWELL_NUM_STAGES
    ] + [
        # Additional configs for smaller sequences
        triton.Config(
            {'BK': 32, 'BV': 32}, 
            num_warps=2, 
            num_stages=2
        ),
        triton.Config(
            {'BK': 32, 'BV': 64}, 
            num_warps=4, 
            num_stages=3
        ),
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_blackwell(
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
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Blackwell-optimized kernel for computing hidden states h.
    
    Key optimizations:
    - Larger BK/BV (128) for better SM utilization on 156 SMs
    - Optimized warp count for sm_120 scheduler
    - Efficient shared memory usage with 128KB capacity
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    
    NTS = BS // BT

    # Compute decay for g_gamma mode
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # Initialize hidden state [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * K * V, (K, V), (V, 1), 
            (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    # Main loop over time chunks
    for i_t in range(NT):
        i_s = i_t // NTS
        
        # Load k [BK, BT] and v [BT, BV]
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (K, T), (1, H * K),
            (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(
            h + o_h, (K, V), (V, 1),
            (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )

        # Store intermediate state at split boundaries
        if i_t % NTS == 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        
        # Load tiles
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        # Apply scalar decay (g mode)
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_h *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # Apply scalar decay (g_gamma mode)
        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # Apply vector decay (gk mode) - h = Diag(gk) @ h
        if USE_GK:
            p_gk = tl.make_block_ptr(
                gk + (bos * H + i_h) * K, (K, T), (1, H * K),
                (i_k * BK, i_t * BT), (BK, BT), (0, 1)
            )
            p_gk_last = gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_h *= exp(b_gk_last)[:, None]
            
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # Apply vector decay (gv mode) - h = h @ Diag(gv)
        if USE_GV:
            p_gv = tl.make_block_ptr(
                gv + (bos * H + i_h) * V, (T, V), (H * V, 1),
                (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            p_gv_last = gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            
            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_h *= exp(b_gv_last)[None, :]
            
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        # Accumulate: h += k @ v
        b_h += tl.dot(b_k, b_v)

    # Store final state
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * K * V, (K, V), (V, 1),
            (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Kernel: chunk_fwd_o - Compute output o (Blackwell Optimized)
# ============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config(
            {'BK': BK, 'BV': BV}, 
            num_warps=num_warps, 
            num_stages=num_stages
        )
        for BK in BLACKWELL_BK_LIST
        for BV in BLACKWELL_BV_LIST
        for num_warps in BLACKWELL_NUM_WARPS
        for num_stages in BLACKWELL_NUM_STAGES
    ] + [
        # Optimized for smaller sequences
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 64}, num_warps=2, num_stages=2),
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_blackwell(
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
    Blackwell-optimized kernel for computing output o.
    
    Computes: o = (q @ h) * decay + (q @ k.T * mask * decay) @ v
    """
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # Offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    # Accumulators
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    # Loop over K dimension
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), 
            (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, H * K), 
            (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (K, V), (V, 1), 
            (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        
        # Load tiles
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # Accumulate contributions
        b_o += tl.dot(b_q, b_h)  # [BT, BV]
        b_A += tl.dot(b_q, b_k)  # [BT, BT]

    # Apply decay
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

    # Apply causal mask
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    # Load v and compute final output
    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), 
        (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), 
        (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )

    b_v = tl.load(p_v, boundary_check=(0, 1))
    # o = (q @ h + causal(q @ k.T) @ v) * scale
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def chunk_fwd_h_blackwell(
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    h0: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    split_size: Optional[int] = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Blackwell-optimized wrapper for computing hidden states h.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"split_size {BS} must be multiple of chunk_size {BT}"
    
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    
    def grid(meta): 
        return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    
    chunk_fwd_kernel_h_blackwell[grid](
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


def chunk_fwd_o_blackwell(
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
    """
    Blackwell-optimized wrapper for computing output o.
    """
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    
    def grid(meta): 
        return (triton.cdiv(V, meta['BV']), NT, B * H)
    
    chunk_fwd_kernel_o_blackwell[grid](
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


def chunk_simple_gla_fwd_blackwell(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Blackwell-optimized forward pass for chunk_simple_gla.
    
    This is the main entry point for inference (no backward needed).
    """
    # Compute hidden states h
    h, ht = chunk_fwd_h_blackwell(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
    )
    
    # Compute output o
    o = chunk_fwd_o_blackwell(
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
# Main Interface
# ============================================================================

@torch.compiler.disable
def chunk_simple_gla_blackwell(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Blackwell-optimized chunk_simple_gla for inference (forward only).
    
    Args:
        q: Queries [B, T, H, K]
        k: Keys [B, T, H, K]
        v: Values [B, T, H, V]
        g: Forget gates [B, T, H] (optional)
        g_gamma: Log decay [H] for head-wise decay (optional)
        scale: Attention scale, defaults to 1/sqrt(K)
        initial_state: Initial state [N, H, K, V] (optional)
        output_final_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths for varlen (optional)
        head_first: Deprecated, must be False
        
    Returns:
        o: Output [B, T, H, V]
        final_state: Final state [N, H, K, V] if output_final_state=True
        
    Note:
        This is a forward-only implementation optimized for Blackwell RTX 6000D.
        Backward pass is not supported.
    """
    if head_first:
        raise DeprecationWarning("head_first is deprecated. Use head_first=False.")
    
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "Ensure input is [B, T, H, ...] format."
        )
    
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(f"Batch size must be 1 with cu_seqlens, got {q.shape[0]}")
    
    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    # Use Blackwell-optimized chunk size
    if is_blackwell():
        T = q.shape[1]
        chunk_size = get_optimal_chunk_size_blackwell(T)
    else:
        chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))
    
    o, final_state = chunk_simple_gla_fwd_blackwell(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    
    return o, final_state
