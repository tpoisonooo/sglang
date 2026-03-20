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
"""
完全融合的 Chunk GLA + Output Processing

将以下三个步骤融合为单个 kernel，避免中间变量 h 和 o 的重复读写:
1. chunk_fwd_h: 计算 hidden states
2. chunk_fwd_o: 计算 attention output  
3. fused_output_final: RMSNorm + sigmoid gate

融合策略:
- Kernel 内部计算 h (保持在寄存器中)
- 直接使用 h 计算 o
- 对 o 就地应用 RMSNorm 和 gate，直接输出最终结果

相比原始的两/三 kernel 方案，这个完全融合版本:
- 消除 h 的显式读写 (saves K*V*4 bytes per position)
- 消除 o 的显式读写 (saves H*V*4 bytes per position)
- 预计额外 15-25% 带宽节省
"""

import torch
import triton
import triton.language as tl
from typing import Optional

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs, check_shared_mem


# Shared memory config for chunk_h
BKV_LIST = [32, 64] if check_shared_mem() else [16, 32]


# =============================================================================
# Inlined chunk_fwd_h from FLA (for self-containment)
# =============================================================================

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
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
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Inlined from FLA's chunk_h.py - computes hidden states.
    Grid: (K/BK, V/BV, N*H)
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
        b_gamma = tl.load(g_gamma + i_h).to(tl.float32)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV]
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

        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.).to(tl.float32)
            b_h *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        if USE_G_GAMMA:
            b_g_last = (b_gamma * min(BT, T - i_t * BT)).to(tl.float32)
            b_h *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.).to(tl.float32)
            b_h *= exp(b_gk_last)[:, None]

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.).to(tl.float32)
            b_h *= exp(b_gv_last)[None, :]

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inlined wrapper for chunk_fwd_kernel_h (from FLA's chunk_h.py).
    Computes hidden states h for chunk-wise GLA.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
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


# =============================================================================
# Fully Fused Kernel: h + o + RMSNorm + Gate in one pass
# =============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BK': 64, 'BV': 32}, num_warps=4, num_stages=2),
        triton.Config({'BK': 32, 'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=2),
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fused_all_kernel(
    q, k, v,
    g, g_gamma,
    z,              # gate input [B*T, H*V]
    norm_weight,    # RMSNorm weight [H*V]
    out,            # final output [B*T, H*V]
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EPS: tl.constexpr = 1e-6,
):
    """
    完全融合的 Chunk GLA Kernel。
    
    每个 block 处理一个 (chunk, head, V-tile) 组合。
    在 kernel 内部:
    1. 计算该 chunk 的 hidden states h (保持在 SRAM/寄存器)
    2. 计算 attention output o
    3. 应用 RMSNorm 和 sigmoid gate
    4. 直接写入最终输出
    
    Grid: (V/BV, T/BT, B*H)
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

    # Input offsets
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    # Output offsets (2D layout)
    out += bos * H * V + i_h * V
    z += bos * H * V + i_h * V
    norm_weight += i_h * V

    # =====================================================================
    # Step 1: Compute hidden states h for this chunk (on-the-fly)
    # =====================================================================
    # h accumulates k^T @ v for all previous positions
    # We compute it incrementally for efficiency
    
    # Allocate accumulator for h: [K, V] tile
    # Use smaller tiles to fit in shared memory
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    
    # First, accumulate hidden state from previous chunks
    # For position i_t * BT, we need sum of k[0:i_t*BT]^T @ v[0:i_t*BT]
    num_prev_chunks = i_t
    
    for i_prev in range(num_prev_chunks):
        prev_t_start = i_prev * BT
        
        for i_k in range(tl.cdiv(K, BK)):
            for i_v_acc in range(tl.cdiv(V, BV)):
                p_k_prev = tl.make_block_ptr(
                    k, (T, K), (H*K, 1),
                    (prev_t_start, i_k * BK), (BT, BK), (1, 0)
                )
                p_v_prev = tl.make_block_ptr(
                    v, (T, V), (H*V, 1),
                    (prev_t_start, i_v_acc * BV), (BT, BV), (1, 0)
                )
                
                b_k_prev = tl.load(p_k_prev, boundary_check=(0, 1))
                b_v_prev = tl.load(p_v_prev, boundary_check=(0, 1))
                
                # Accumulate: [BK, BT] @ [BT, BV] -> [BK, BV]
                if i_v_acc == i_v:  # Only for our V-tile
                    b_h_partial = tl.dot(tl.trans(b_k_prev), b_v_prev)
                    if i_k == tl.arange(0, 1):  # Only add once per k-tile
                        b_h += b_h_partial
    
    # Apply decay g to accumulated h if needed
    if USE_G:
        g += bos * H + i_h
        # Load g for the boundary
        if i_t > 0:
            boundary_pos = i_t * BT - 1
            b_g_boundary = tl.load(g + boundary_pos * H, mask=boundary_pos < T)
            b_h *= exp(b_g_boundary)
    
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        if i_t > 0:
            b_g_cum = b_gamma * (i_t * BT)
            b_h *= exp(b_g_cum)

    # =====================================================================
    # Step 2: Compute attention output for current chunk
    # =====================================================================
    # o = (q @ h + q @ k^T @ v) * scale
    
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        
        # Accumulate h for this k-tile and current v-tile
        p_v_chunk = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v_chunk = tl.load(p_v_chunk, boundary_check=(0, 1))
        
        # Update h with current chunk contribution
        # This is incremental: h_new = h_old + k_chunk^T @ v_chunk
        b_h_local = b_h + tl.dot(tl.trans(b_k), b_v_chunk)
        
        # o += q @ h
        b_o += tl.dot(b_q, b_h_local)
        
        # A += q @ k^T
        b_A += tl.dot(b_q, b_k)
    
    # Apply decay
    if USE_G:
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])
    
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g_local = b_gamma * (tl.arange(0, BT) + i_t * BT + 1)
        b_o = b_o * exp(b_g_local)[:, None]
        b_A = b_A * exp(b_g_local[:, None] - b_g_local[None, :])
    
    # Mask and compute final attention
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    
    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale

    # =====================================================================
    # Step 3: RMSNorm + Sigmoid Gate (in-place)
    # =====================================================================
    # Compute RMSNorm per row, then apply sigmoid gate
    
    # First pass: compute sum of squares
    sum_sq = tl.sum(b_o * b_o, axis=1)  # [BT]
    
    # Compute RMS per row
    mean_sq = sum_sq / (H * V)
    rms = tl.math.rsqrt(mean_sq + EPS)
    
    # Second pass: apply norm, weight, and gate
    # Load norm weight for this V-tile
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_v = offs_v < V
    
    for i_bt in range(BT):
        # Load output row
        b_o_row = b_o[i_bt, :]
        
        # Load norm weight
        b_norm_w = tl.load(norm_weight + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        
        # Apply RMSNorm
        b_o_row = b_o_row * rms[i_bt] * b_norm_w
        
        # Create output block pointer for dtype and store
        p_out_row = tl.make_block_ptr(
            out, (T, V), (H*V, 1), (i_t * BT + i_bt, i_v * BV), (1, BV), (1, 0)
        )
        
        # Match reference: bf16 round-trip
        b_o_row = b_o_row.to(p_out_row.dtype.element_ty).to(tl.float32)
        
        # Load z and apply sigmoid gate
        row_offset = (i_t * BT + i_bt) * H * V + i_v * BV
        b_z = tl.load(z + row_offset + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        gate = tl.sigmoid(b_z)
        b_o_row = b_o_row * gate
        
        # Store final output
        tl.store(out + row_offset + offs_v, b_o_row.to(p_out_row.dtype.element_ty), mask=mask_v)


# =============================================================================
# Optimized Version: Using pre-computed h (like original) but fusing o + output_final
# =============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BK': 64, 'BV': 32}, num_warps=4, num_stages=2),
        triton.Config({'BK': 32, 'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=2),
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_o_fused_all_kernel(
    q, k, v, h,
    g, g_gamma,
    z,
    norm_weight,
    out,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EPS: tl.constexpr = 1e-6,
):
    """
    融合 chunk_fwd_o + fused_output_final 的 kernel。
    
    Grid: (V/BV, T/BT, B*H)
    
    策略: 每个 block 遍历所有 V tiles 计算完整的 attention output，
    累加平方和得到正确的 RMS，然后只存储当前 block 负责的 tile。
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
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    # Output offsets (2D layout: [B*T, H*V])
    out += bos * H * V + i_h * V
    z += bos * H * V + i_h * V
    norm_weight += i_h * V

    # =====================================================================
    # Step 1: Precompute common matrices
    # =====================================================================
    # Compute A = q @ k^T (causal mask applied later)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_q, b_k)

    # Load g if needed
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h).to(tl.float32)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    # Apply causal mask
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    # =====================================================================
    # Step 2: Compute attention output for ALL V tiles + accumulate sum_sq
    # =====================================================================
    sum_sq_all = tl.zeros([BT], dtype=tl.float32)
    b_o_current = tl.zeros([BT, BV], dtype=tl.float32)
    
    num_v_tiles = tl.cdiv(V, BV)
    
    for i_v_acc in range(num_v_tiles):
        # Compute q @ h for this V tile
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v_acc * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q, b_h)

        # Apply g decay to b_o
        if USE_G:
            b_o = b_o * exp(b_g)[:, None]
        if USE_G_GAMMA:
            b_o = b_o * exp(b_g)[:, None]

        # Add attention term: b_A @ v
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v_acc * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        
        # Accumulate sum of squares for RMS
        sum_sq_all += tl.sum(b_o * b_o, axis=1)
        
        # Save if this is our target tile
        if i_v_acc == i_v:
            b_o_current = b_o

    # =====================================================================
    # Step 3: RMSNorm + Sigmoid Gate
    # =====================================================================
    mean_sq = sum_sq_all / (H * V)
    rms = tl.math.rsqrt(mean_sq + EPS)
    
    # Load norm weight and z for this V-tile
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_v = offs_v < V
    b_norm_w = tl.load(norm_weight + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    
    p_z = tl.make_block_ptr(z, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_z = tl.load(p_z, boundary_check=(0, 1)).to(tl.float32)
    b_gate = tl.sigmoid(b_z)
    
    # Create output block pointer first (needed for dtype)
    p_out = tl.make_block_ptr(
        out, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    
    # Apply: RMSNorm -> bf16 round-trip -> gate
    b_o_current = b_o_current * rms[:, None] * b_norm_w[None, :]
    b_o_current = b_o_current.to(p_out.dtype.element_ty).to(tl.float32)  # bf16 round-trip
    b_o_current = b_o_current * b_gate

    # Store final output
    tl.store(p_out, b_o_current.to(p_out.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Python Wrappers
# =============================================================================

def chunk_fwd_o_fused_all(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    融合 chunk_fwd_o + fused_output_final 的 wrapper（atomic 版本）。
    
    使用单 kernel + atomic_add 实现 RMSNorm 全局 reduction，
    避免显式的 o tensor 分配。
    """
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    if scale is None:
        scale = K ** -0.5

    # 分配最终输出
    out = v.new_empty(B * T, H * V)
    
    # 分配 sum_sq buffer: [B*H*NT, BT]
    sum_sq_buf = torch.zeros(B * H * NT, BT, device=v.device, dtype=torch.float32)
    
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    
    chunk_fwd_o_fused_atomic_kernel[grid](
        q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma,
        z=z, norm_weight=norm_weight, out=out,
        sum_sq_buf=sum_sq_buf,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, B=B, H=H, K=K, V=V, BT=BT,
        EPS=eps,
    )
    return out


# =============================================================================
# Main Interface: Fused Chunk GLA + Output Processing (Two-Kernel Version)
# =============================================================================

def chunk_gla_fused_all(
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
    融合的 Chunk GLA + Output Processing（完整融合版本）。
    
    相比原始的 chunk_gla_fused_output，这个版本:
    - 保持 chunk_fwd_h 计算 h (避免在 o kernel 中重复计算)
    - 将 chunk_fwd_o + fused_output_final 融合为一个 kernel
    - 消除显式的 o tensor 分配和读写
    
    内存带宽节省:
    - 原始: h (read) -> o (write) -> o (read) -> output (write) = 3N + 1N
    - 融合: h (read) -> output (write) = 1N + 1N
    - 节省: ~50% 的中间 tensor 带宽
    
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
    
    # Step 1: Compute hidden states (using inlined kernel)
    h, ht = _chunk_fwd_h(
        k=k, v=v, g=g, g_gamma=g_gamma,
        gk=None, gv=None,
        h0=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    )
    
    # Step 2: 融合 o + output_final (直接到最终输出)
    out = chunk_fwd_o_fused_all(
        q=q, k=k, v=v, h=h, z=z, norm_weight=norm_weight,
        g=g, g_gamma=g_gamma, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, eps=eps,
    )
    
    return out, ht


@torch.compiler.disable
def chunk_simple_gla_fused_all(
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
    """Simplified interface matching FLA's API style."""
    if head_first:
        raise DeprecationWarning("head_first is deprecated.")
    
    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    T = q.shape[1]
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))
    
    out, final_state = chunk_gla_fused_all(
        q=q, k=k, v=v, z=z, norm_weight=norm_weight,
        g=g, g_gamma=g_gamma, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, eps=eps,
    )
    
    return out, final_state
"""
Atomic-based fusion kernel for chunk_fwd_o + fused_output_final
"""
import torch
import triton
import triton.language as tl
from typing import Optional

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BK': 64, 'BV': 32}, num_warps=4, num_stages=2),
        triton.Config({'BK': 32, 'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=2),
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_o_fused_atomic_kernel(
    q, k, v, h,
    g, g_gamma,
    z,
    norm_weight,
    out,
    sum_sq_buf,     # [B*H*NT, BT] - global buffer for sum_sq
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EPS: tl.constexpr = 1e-6,
):
    """
    融合 chunk_fwd_o + fused_output_final 的 kernel（atomic 版本）。
    
    Grid: (V/BV, T/BT, B*H)
    
    策略:
    1. 每个 block 计算自己的 o tile
    2. 计算 sum_sq，atomic_add 到全局 buffer
    3. 所有 block 读取全局 sum_sq，计算 RMS，应用 RMSNorm + gate
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
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    # Output offsets (2D layout: [B*T, H*V])
    out += bos * H * V + i_h * V
    z += bos * H * V + i_h * V
    norm_weight += i_h * V
    
    # Sum_sq buffer offset: [B*H*NT, BT]
    sum_sq_buf += (i_bh * NT + i_t) * BT

    # =====================================================================
    # Step 1: Compute attention output for this V tile
    # =====================================================================
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        
        b_o += tl.dot(b_q, b_h)
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

    # =====================================================================
    # Step 2: Accumulate sum_sq using atomic_add
    # =====================================================================
    # Compute local sum_sq for this tile
    sum_sq_local = tl.sum(b_o * b_o, axis=1)  # [BT]
    
    # First block initializes the buffer
    if i_v == 0:
        for i in range(BT):
            tl.store(sum_sq_buf + i, 0.0)
    
    # Atomic add each element
    # sum_sq_local is block<BT>, we use pointer arithmetic
    offsets = tl.arange(0, BT)
    mask = offsets < BT
    # Store to a local temp, then atomic_add
    for i in range(BT):
        # Use tl.sum to extract scalar? No, that's inefficient
        # Instead, let's use a different approach
        pass
    
    # Alternative: just store local sum_sq to global and handle reduction later
    # For now, let's use a simple approach: each block stores its sum_sq
    # to a different offset, then we read all and sum in Python
    # But that's not fused...
    
    # Let me try a simpler atomic approach: just atomic_add the whole block
    # by iterating through BT at compile time
    # Note: BT is constexpr, so this loop is unrolled
    for i in tl.static_range(BT):
        if i < BT:  # Always true, but needed for mask
            val = tl.sum(b_o[:, i % BV] * b_o[:, i % BV]) if BV == 1 else sum_sq_local
            # This is getting complicated...
            # Let me just use a scalar approach
            pass
    
    # Simplest approach: use block-level atomic_add directly
    tl.atomic_add(sum_sq_buf + offsets, sum_sq_local, mask=mask)
    
    # =====================================================================
    # Step 3: Compute RMSNorm + Gate
    # =====================================================================
    # Load global sum_sq (accumulated by all tiles)
    sum_sq_global = tl.load(sum_sq_buf + tl.arange(0, BT), mask=m_t, other=0.0)
    
    mean_sq = sum_sq_global / (H * V)
    rms = tl.math.rsqrt(mean_sq + EPS)
    
    # Load norm weight and z for this V-tile
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_v = offs_v < V
    b_norm_w = tl.load(norm_weight + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    
    p_z = tl.make_block_ptr(z, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_z = tl.load(p_z, boundary_check=(0, 1)).to(tl.float32)
    b_gate = tl.sigmoid(b_z)
    
    # Create output block pointer first (needed for dtype)
    p_out = tl.make_block_ptr(
        out, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    
    # Apply: RMSNorm -> bf16 round-trip -> gate
    b_o = b_o * rms[:, None] * b_norm_w[None, :]
    b_o = b_o.to(p_out.dtype.element_ty).to(tl.float32)
    b_o = b_o * b_gate

    # Store final output
    tl.store(p_out, b_o.to(p_out.dtype.element_ty), boundary_check=(0, 1))


"""
Single-block-per-row fusion kernel
Grid: (T/BT, B*H) - each block handles all V tiles for a sequence position chunk
"""
import torch
import triton
import triton.language as tl
from typing import Optional

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


@triton.jit
def chunk_fused_single_block_kernel(
    q, k, v, h,
    g, g_gamma,
    z,
    norm_weight,
    out,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Fusion kernel with single block per row.
    Grid: (T/BT, B*H)
    Each block processes all V tiles for its sequence positions.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    # Input offsets
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    out += bos * H * V + i_h * V
    z += bos * H * V + i_h * V
    norm_weight += i_h * V

    # Compute A matrix (shared across all V)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g_decay = b_gamma * (tl.arange(0, BT) + 1)
        b_A = b_A * exp(b_g_decay[:, None] - b_g_decay[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    # Process all V tiles and accumulate sum_sq
    sum_sq = tl.zeros([BT], dtype=tl.float32)
    num_v_tiles = tl.cdiv(V, BV)
    
    # First pass: compute all o tiles and sum_sq
    for i_v in range(num_v_tiles):
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q, b_h)

        if USE_G:
            b_o = b_o * exp(b_g)[:, None]
        if USE_G_GAMMA:
            b_o = b_o * exp(b_g_decay)[:, None]

        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        
        sum_sq += tl.sum(b_o * b_o, axis=1)

    # Compute RMS with improved precision
    mean_sq = sum_sq / (H * V)
    rms = tl.math.rsqrt(mean_sq + EPS)

    # Second pass: apply RMSNorm + gate and store
    for i_v in range(num_v_tiles):
        # Recompute o (or we could store in shared memory if it fits)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q, b_h)

        if USE_G:
            b_o = b_o * exp(b_g)[:, None]
        if USE_G_GAMMA:
            b_o = b_o * exp(b_g_decay)[:, None]

        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        
        # Apply RMSNorm + gate
        offs_v = i_v * BV + tl.arange(0, BV)
        mask_v = offs_v < V
        b_norm_w = tl.load(norm_weight + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        
        p_z = tl.make_block_ptr(z, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1)).to(tl.float32)
        b_gate = tl.sigmoid(b_z)
        
        # Create output block pointer before using it for dtype
        p_out = tl.make_block_ptr(out, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        
        b_o = b_o * rms[:, None] * b_norm_w[None, :]
        b_o = b_o.to(p_out.dtype.element_ty).to(tl.float32)
        b_o = b_o * b_gate

        tl.store(p_out, b_o.to(p_out.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o_fused_all(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    融合 chunk_fwd_o + fused_output_final 的 wrapper（V3 - 改进数值精度）。
    
    使用单 kernel + 两遍计算，避免显式 o tensor 分配。
    """
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    if scale is None:
        scale = K ** -0.5

    out = v.new_empty(B * T, H * V)
    BV = 64 if V >= 64 else V
    
    def grid(meta): return (NT, B * H)
    
    chunk_fused_o_final_kernel_v2[grid](
        q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma,
        z=z, norm_weight=norm_weight, out=out,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, B=B, H=H, K=K, V=V, BT=BT,
        BK=64, BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        IS_VARLEN=cu_seqlens is not None,
        EPS=eps,
    )
    return out


@triton.jit
def chunk_fused_v128_kernel(
    q, k, v, h,
    g, g_gamma,
    z,
    norm_weight,
    out,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    # V is fixed at 128 for MiniCPM
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Single-pass fusion kernel optimized for V=128 (MiniCPM head_dim).
    
    Grid: (T/BT, B*H)
    
    Uses shared memory to store intermediate [BT, 128] output, enabling:
    - Single computation of attention output
    - Efficient RMSNorm computation
    - No redundant recomputation
    
    Preconditions:
    - V == 128 (asserted in Python wrapper)
    - BT * 128 * 4 bytes <= shared memory limit (128KB on Blackwell)
    """
    # V is compile-time constant 128
    V: tl.constexpr = 128
    
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    # Input offsets
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    out += bos * H * V + i_h * V
    z += bos * H * V + i_h * V
    norm_weight += i_h * V

    # Compute A matrix
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    # For V=128, process in single tile (BV=128) to avoid complex indexing
    BV: tl.constexpr = 128  # Single tile for V=128
    
    # Compute full attention output [BT, 128]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, 0), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)

    if USE_G:
        b_o = b_o * exp(b_g)[:, None]
    if USE_G_GAMMA:
        b_o = b_o * exp(b_g)[:, None]

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    
    # Compute RMS
    sum_sq = tl.sum(b_o * b_o, axis=1)
    mean_sq = sum_sq / (H * V)
    rms = tl.math.rsqrt(mean_sq + EPS)
    
    # Apply RMSNorm + gate and store
    b_norm_w = tl.load(norm_weight + tl.arange(0, V), mask=tl.arange(0, V) < V, other=0.0).to(tl.float32)
    
    p_z = tl.make_block_ptr(z, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_z = tl.load(p_z, boundary_check=(0, 1)).to(tl.float32)
    b_gate = tl.sigmoid(b_z)
    
    # Create output block pointer first (needed for dtype)
    p_out = tl.make_block_ptr(out, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    
    b_o = b_o * rms[:, None] * b_norm_w[None, :]
    b_o = b_o.to(p_out.dtype.element_ty).to(tl.float32)
    b_o = b_o * b_gate

    tl.store(p_out, b_o.to(p_out.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o_fused_all_v128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fusion wrapper optimized for V=128 (MiniCPM head_dim).
    
    Preconditions:
        - v.shape[-1] == 128, will raise AssertionError otherwise
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    # Assert V == 128 as this kernel is specialized for MiniCPM
    assert V == 128, f"chunk_fwd_o_fused_all_v128 requires V=128, got V={V}"
    
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    if scale is None:
        scale = K ** -0.5

    out = v.new_empty(B * T, H * V)
    
    def grid(meta): return (NT, B * H)
    
    chunk_fused_v128_kernel[grid](
        q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma,
        z=z, norm_weight=norm_weight, out=out,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, B=B, H=H, K=K, BT=BT,
        BK=64,  # Tile size for K
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        IS_VARLEN=cu_seqlens is not None,
        EPS=eps,
    )
    return out


# =============================================================================
# V2: Correct fused implementation (added 2025-01-XX)
# =============================================================================

@triton.jit
def chunk_fused_o_final_kernel_v2(
    q, k, v, h,
    g, g_gamma,
    z,
    norm_weight,
    out,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EPS: tl.constexpr,
):
    """Fused kernel: chunk_fwd_o + RMSNorm + gate (V2 - correct implementation)"""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    out += bos * H * V + i_h * V
    z += bos * H * V + i_h * V
    norm_weight += i_h * V

    # Compute A matrix
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h).to(tl.float32)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    # First pass: compute all o tiles and sum_sq
    num_v_tiles = tl.cdiv(V, BV)
    sum_sq = tl.zeros([BT], dtype=tl.float32)
    
    for i_v in range(num_v_tiles):
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q, b_h)

        if USE_G:
            b_o = b_o * exp(b_g)[:, None]
        if USE_G_GAMMA:
            b_o = b_o * exp(b_g)[:, None]

        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        sum_sq += tl.sum(b_o * b_o, axis=1)

    # Compute RMS
    mean_sq = sum_sq / (H * V)
    rms = tl.math.rsqrt(mean_sq + EPS)

    # Second pass: apply RMSNorm + gate and store
    for i_v in range(num_v_tiles):
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q, b_h)

        if USE_G:
            b_o = b_o * exp(b_g)[:, None]
        if USE_G_GAMMA:
            b_o = b_o * exp(b_g)[:, None]

        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        
        offs_v = i_v * BV + tl.arange(0, BV)
        mask_v = offs_v < V
        b_norm_w = tl.load(norm_weight + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        
        p_z = tl.make_block_ptr(z, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1)).to(tl.float32)
        b_gate = tl.sigmoid(b_z)
        
        # Create output block pointer first (needed for dtype)
        p_out = tl.make_block_ptr(out, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        
        b_o = b_o * rms[:, None] * b_norm_w[None, :]
        b_o = b_o.to(p_out.dtype.element_ty).to(tl.float32)
        b_o = b_o * b_gate

        tl.store(p_out, b_o.to(p_out.dtype.element_ty), boundary_check=(0, 1))



# =============================================================================
# Autotuned Wrapper for MiniCPM
# =============================================================================

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
):
    """
    Chunk GLA with RMSNorm and gate.
    
    Uses the proven-correct 3-kernel approach for all sequence lengths.
    (2-kernel approach has numerical issues and is disabled)
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


chunk_simple_gla = chunk_simple_gla_autotuned
chunk_gla = chunk_gla_fused_all
