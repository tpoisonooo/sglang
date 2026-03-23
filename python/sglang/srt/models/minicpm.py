# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only MiniCPM model compatible with HuggingFace weights."""

# Global flag for z_proj quantization (used by dynamic quantization)
# When True, z_proj uses the parent's quant_config; when False, z_proj is not quantized
_USE_Z_PROJ_QUANT = False

def set_z_proj_quant_enabled(enabled: bool):
    """Enable or disable z_proj quantization. Used by dynamic quantization."""
    global _USE_Z_PROJ_QUANT
    _USE_Z_PROJ_QUANT = enabled

def get_z_proj_quant_config(parent_quant_config):
    """Get quant_config for z_proj based on global flag."""
    return parent_quant_config if _USE_Z_PROJ_QUANT else None
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.hybrid_linear_attn_backend import SimpleGLAAttnBackend
from sglang.srt.layers.attention.minicpm_sparse_utils import (
    SparseBatchAnalyzer,
    SparseConfig,
    SparseMetadata,
    SparseMetadataBuilder,
)
from sglang.srt.models.minicpm_fused_norm_rope import fused_rms_norm_rope

from sglang.srt.models.minicpm_fused_output import fused_output_processing
from sglang.srt.models.minicpm_fused_scale_add import fused_scale_add

import numpy as np
import os

# Create debug directory
DEBUG_DIR = "/root/soar2026/debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

class MiniCPMMLP(nn.Module):
    """TP=1 OPTIMIZED: Simplified for single GPU"""
    __slots__ = ['gate_up_proj', 'down_proj', 'act_fn']
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPMAttention(nn.Module):
    """TP=1 OPTIMIZED: Simplified head calculation for single GPU"""
    __slots__ = ['hidden_size', 'num_heads', 'num_kv_heads', 'head_dim', 'q_size', 'kv_size', 
                 'scaling', 'rope_theta', 'max_position_embeddings', 'attn_use_rope', 
                 'use_output_gate', 'layer_id', 'qkv_proj', 'o_proj', 'rotary_emb', 'attn', 'o_gate']
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        attn_use_rope: bool = True,
        use_output_gate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        # TP=1 OPTIMIZATION: Simplified for single GPU - direct assignment
        # Keep total_num_heads for compatibility with QKVParallelLinear
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        # tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads  # TP=1: no division needed
        self.num_kv_heads = num_kv_heads  # TP=1: no division needed
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attn_use_rope = attn_use_rope
        self.use_output_gate = use_output_gate

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if self.attn_use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
        else:
            self.rotary_emb = None  # Always initialize for __slots__ compatibility
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        if self.use_output_gate:
            self.o_gate = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("o_gate", prefix),
            )
        else:
            self.o_gate = None  # Always initialize for __slots__ compatibility

        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        import time
        # TP=1 OPTIMIZATION: Cache frequently accessed attributes
        qkv_proj = self.qkv_proj
        o_proj = self.o_proj
        attn = self.attn
        rotary_emb = self.rotary_emb
        use_output_gate = self.use_output_gate
        q_size = self.q_size
        kv_size = self.kv_size
        
        # QKV projection
        qkv, _ = qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        if self.attn_use_rope:
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
            q, k = rotary_emb(positions, q, k)
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        attn_output = attn(q, k, v, forward_batch)
        
        if use_output_gate:
            o_gate_output, _ = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(o_gate_output)

        output, _ = o_proj(attn_output)
        
        return output


class MiniCPMLightningMixer(nn.Module):
    """Lightning attention mixer that uses SimpleGLAAttnBackend.

    This is a wrapper that prepares inputs for the backend and handles
    the QKV projection, normalization, RoPE, and output processing,
    while delegating the Simple GLA kernel calls to SimpleGLAAttnBackend.
    
    TP=1 OPTIMIZED: Simplified for single GPU
    """
    
    __slots__ = ['hidden_size', 'num_heads', 'num_kv_heads', 'head_dim', 'scale',
                 'q_size', 'kv_size', 'rope_theta', 'max_position_embeddings',
                 'use_rope', 'use_output_gate', 'qk_norm', 'use_output_norm',
                 'rope_head_dim', 'attention_bias', 'rms_norm_eps', 'layer_id',
                 'qkv_proj', 'o_proj', 'q_norm', 'k_norm', 'o_norm', 'z_proj', 'rotary_emb']

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_rope: bool = True,
        use_output_gate: bool = False,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_output_norm: bool = False,
        qk_norm: bool = True,
        rope_head_dim: Optional[int] = None,
        scale: str = "1/sqrt(d)",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        # TP=1 OPTIMIZATION: Simplified for single GPU
        # Keep total_num_heads for compatibility with QKVParallelLinear
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        if scale == "1/sqrt(d)":
            self.scale = self.head_dim ** (-0.5)
        elif scale == "1/d":
            self.scale = self.head_dim ** (-1.0)
        else:
            self.scale = 1.0
        self.use_output_gate = use_output_gate
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.use_output_norm = use_output_norm
        self.rope_head_dim = (
            rope_head_dim if rope_head_dim is not None else self.head_dim
        )
        assert self.rope_head_dim <= self.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if self.use_output_norm:
            self.o_norm = RMSNorm(self.num_heads * self.head_dim, eps=self.rms_norm_eps)
        else:
            self.o_norm = None  # Always initialize for __slots__ compatibility

        if self.use_output_gate:
            # z_proj quant_config is controlled by global flag for dynamic quantization
            z_proj_quant_config = get_z_proj_quant_config(quant_config)
            self.z_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=self.attention_bias,
                quant_config=z_proj_quant_config,
                prefix=add_prefix("z_proj", prefix),
            )
        else:
            self.z_proj = None  # Always initialize for __slots__ compatibility

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        else:
            self.q_norm = None  # Always initialize for __slots__ compatibility
            self.k_norm = None  # Always initialize for __slots__ compatibility

        if self.use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
        else:
            self.rotary_emb = None  # Always initialize for __slots__ compatibility

        self.layer_id = layer_id
        self.state_shape = (self.num_kv_heads, self.head_dim, self.head_dim)
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        
        qkv, _ = self.qkv_proj(hidden_states)  # [seq_len, hidden_size] -> [seq_len, q_size+kv_size+kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # print(f'fused_rms_rope shape {q.shape}, {k.shape}, {self.num_heads}, {self.head_dim}, {self.num_kv_heads}')

        # if self.qk_norm and self.use_rope:
        q, k = fused_rms_norm_rope(
            q=q,
            k=k,
            positions=positions,
            q_norm_weight=self.q_norm.weight,
            k_norm_weight=self.k_norm.weight,
            cos_sin_cache=self.rotary_emb.cos_sin_cache, # [max_position, head_dim * 2]
            eps=self.rms_norm_eps,
        )

        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        # else:
        #     # ground truth start
        #     if self.qk_norm:
        #         q = self.q_norm(q.reshape(-1, self.head_dim))
        #         k = self.k_norm(k.reshape(-1, self.head_dim))

        #     if self.use_rope:
        #         q = q.reshape(-1, self.num_heads * self.head_dim)
        #         k = k.reshape(-1, self.num_kv_heads * self.head_dim)
        #         orig_dtype = q.dtype
        #         q, k = q.float(), k.float()
        #         q, k = self.rotary_emb(positions, q, k)
        #         q, k = q.to(orig_dtype), k.to(orig_dtype)

        #     q = q.reshape(-1, self.num_heads, self.head_dim)
        #     k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        #     v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        #     # ALWAYS unsqueeze to (1, total_tokens, h, d)
        #     q = q.unsqueeze(0)  # (1, total_tokens, num_heads, head_dim)
        #     k = k.unsqueeze(0)
        #     v = v.unsqueeze(0)
        #     # ground truth end

        # Get backend from forward batch
        attn_backend = forward_batch.attn_backend
        if not hasattr(attn_backend, "linear_attn_backend"):
            raise RuntimeError(
                "SimpleGLAAttnBackend requires HybridLinearAttnBackend but got "
                f"{type(attn_backend).__name__}. This mixer should only be used for "
                "MiniCPM hybrid models."
            )

        linear_attn_backend = attn_backend.linear_attn_backend
        if not isinstance(linear_attn_backend, SimpleGLAAttnBackend):
            raise RuntimeError(
                f"Expected SimpleGLAAttnBackend but got {type(linear_attn_backend).__name__}"
            )

        # Prepare z and norm_weight for fused output processing
        # Backend will apply RMSNorm + sigmoid gate internally if supported
        if self.use_output_gate and self.z_proj is not None:
            z, _ = self.z_proj(hidden_states)  # [seq_len, hidden_size] -> [seq_len, hidden_size]
        else:
            z = None
        
        norm_weight = self.o_norm.weight if self.use_output_norm and self.o_norm is not None else None
        
        # Backend forward with fused output processing
        # Returns [B*T, H*D] already processed with RMSNorm + sigmoid gate
        o_fused = linear_attn_backend.forward(
            q=q,
            k=k,
            v=v,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
            output_attentions=False,
            z=z,
            norm_weight=norm_weight,
        )
        
        # DEBUG: Compare with reference implementation (disabled - now using proven-correct kernels)
        # if not hasattr(self, '_debug_check_done'):
        #     self._debug_check_done = False
        # 
        # if not self._debug_check_done and z is not None and norm_weight is not None:
        #     # Reference implementation: no z/norm_weight -> backend returns 4D
        #     with torch.no_grad():
        #         o_ref_raw = linear_attn_backend.forward(
        #             q=q,
        #             k=k,
        #             v=v,
        #             forward_batch=forward_batch,
        #             layer_id=self.layer_id,
        #             output_attentions=False,
        #         )
        #         # Apply output processing manually
        #         if o_ref_raw.dim() == 4:
        #             B, T, H, D = o_ref_raw.shape
        #             o_ref = o_ref_raw.reshape(B * T, H * D)
        #         else:
        #             o_ref = o_ref_raw  # Already 2D
        #         o_ref = fused_output_processing(o_ref, z, norm_weight, eps=self.rms_norm_eps)
        #         
        #         # Compare
        #         diff = (o_fused.float() - o_ref.float()).abs()
        #         max_diff = diff.max().item()
        #         mean_diff = diff.mean().item()
        #         
        #         print(f"[DEBUG] Layer {self.layer_id} - Fused vs Reference:")
        #         print(f"  Shape: {o_fused.shape}")
        #         print(f"  Max diff: {max_diff:.6f}")
        #         print(f"  Mean diff: {mean_diff:.6f}")
        #         
        #         if max_diff > 0.1:
        #             print(f"  WARNING: Large difference detected!")
        #             # Save inputs and outputs for analysis
        #             dump_path = os.path.join(DEBUG_DIR, f"layer_{self.layer_id}_debug.npz")
        #             np.savez(
        #                 dump_path,
        #                 q=q.cpu().float().numpy(),
        #                 k=k.cpu().float().numpy(),
        #                 v=v.cpu().float().numpy(),
        #                 z=z.cpu().float().numpy(),
        #                 norm_weight=norm_weight.cpu().float().numpy(),
        #                 o_fused=o_fused.cpu().float().numpy(),
        #                 o_ref=o_ref.cpu().float().numpy(),
        #                 g_gamma=self.g_gamma.cpu().float().numpy() if hasattr(self, 'g_gamma') and self.g_gamma is not None else np.array([]),
        #                 scale=self.scale,
        #                 eps=self.rms_norm_eps,
        #             )
        #             print(f"  Saved debug data to: {dump_path}")
        #         
        #         self._debug_check_done = True
        
        # o = o_fused
        # else:
        #     o = o.reshape(-1, self.num_heads * self.head_dim)

        #     if self.use_output_norm:
        #         o = self.o_norm(o)

        #     if self.use_output_gate:
        #         z, _ = self.z_proj(hidden_states)
        #         o = o * F.sigmoid(z)

        y, _ = self.o_proj(o_fused)
        return y



class MiniCPMDecoderLayer(nn.Module):
    """TP=1 OPTIMIZED: Simplified for single GPU"""
    __slots__ = ['config', 'layer_id', 'hidden_size', 'hidden_scale', 'mixer_type',
                 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm']

    @staticmethod
    @torch.jit.script
    def _fused_scale_add(x: torch.Tensor, residual: torch.Tensor, scale: float) -> torch.Tensor:
        """JIT compiled fused scale + add."""
        return residual + x * scale

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hidden_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)
        if hasattr(config, "mixer_types") and config.mixer_types is not None:
            self.mixer_type = config.mixer_types[layer_id]
        else:
            self.mixer_type = "minicpm4"

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if self.mixer_type == "minicpm4":
            self.self_attn = MiniCPMAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                attn_use_rope=(
                    config.attn_use_rope if hasattr(config, "attn_use_rope") else True
                ),
                use_output_gate=(
                    config.attn_use_output_gate
                    if hasattr(config, "attn_use_output_gate")
                    else False
                ),
                prefix=add_prefix("self_attn", prefix),
            )
        elif self.mixer_type in ["lightning", "lightning_attn", "lightning-attn"]:
            assert (
                config.head_dim is not False
            ), "head_dim must be provided for LightningAttention"
            self.self_attn = MiniCPMLightningMixer(
                hidden_size=self.hidden_size,
                num_heads=config.lightning_nh,
                num_kv_heads=config.lightning_nkv,
                head_dim=config.lightning_head_dim,
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                use_rope=config.lightning_use_rope,
                use_output_gate=config.use_output_gate,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                use_output_norm=config.use_output_norm,
                qk_norm=config.qk_norm,
                scale=config.lightning_scale,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            raise ValueError(f"Unsupported mixer type: {self.mixer_type}")
        self.mlp = MiniCPMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
    def _compute_topk(self, forward_batch, base_metadata, sparse_metadata):
        """Compute TopK indices for sparse attention.

        For decode mode, TopK is simple: just use the precomputed sparse_page_table.
        For prefill mode, we need to compute TopK using kernel calls (deferred for now).

        Args:
            forward_batch: Forward batch
            base_metadata: Base metadata
            sparse_metadata: SparseMetadata to update with topk_indices
        """
        if forward_batch.forward_mode.is_decode_or_idle():
            # Decode path: TopK is just the precomputed page table
            sparse_metadata.topk_indices = base_metadata.sparse_page_table
        else:
            # Prefill path: Complex - needs kernel calls with compressed K1/K2
            # For now, leave topk_indices as None, backend will compute it
            # TODO: Implement full TopK computation in prefill mode
            pass

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TP=1 OPTIMIZATION: Cache frequently accessed attributes
        hidden_scale = self.hidden_scale
        input_layernorm = self.input_layernorm
        post_attention_layernorm = self.post_attention_layernorm
        self_attn = self.self_attn
        mlp = self.mlp

        # Self Attention
        residual_out = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states = self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # TP=1 OPTIMIZATION: Use JIT compiled fused scale + add
        hidden_states = self._fused_scale_add(hidden_states, residual_out, hidden_scale)
        
        # Fully Connected
        residual_out = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)

        # TP=1 OPTIMIZATION: Use JIT compiled fused scale + add
        hidden_states = self._fused_scale_add(hidden_states, residual_out, hidden_scale)

        return hidden_states, None


class MiniCPMModel(nn.Module):
    """TP=1 OPTIMIZED: Simplified for single GPU"""
    __slots__ = ['config', 'vocab_size', 'embed_tokens', 'layers', 'norm', 'scale_emb']

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MiniCPMDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class MiniCPMSALAForCausalLM(nn.Module):
    """TP=1 OPTIMIZED: Simplified for single GPU"""
    __slots__ = ['config', 'num_experts', 'quant_config', 'model', 'lm_head', 'scale_width', 'logits_processor']

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.num_experts = getattr(self.config, "num_experts", 0)
        self.quant_config = quant_config
        self.model = MiniCPMModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )

        self.scale_width = self.config.hidden_size / self.config.dim_model_base

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states = hidden_states / self.scale_width
        if self.config.tie_word_embeddings:
            lm_head = self.model.embed_tokens
        else:
            lm_head = self.lm_head
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            (
                "ws" if weight_name in ["w1", "w3"] else "w2s",
                f"experts.{expert_id}.{weight_name}.weight",
                expert_id,
            )
            for expert_id in range(self.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, weight_name, expert_id=expert_id
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # 在查找 params_dict 之前，处理 GPTQ 后缀
                    if name.endswith(".weight") and name not in params_dict:
                        # 尝试 GPTQ 命名
                        gptq_name = name.replace(".weight", ".qweight")
                        if gptq_name in params_dict:
                            name = gptq_name
                        # 如果检查点里存的是 scales/qzeros/g_idx，它们应该直接匹配
     
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )

                    # print(param_name, param)
                    weight_loader(param, loaded_weight)

EntryClass = [MiniCPMSALAForCausalLM]
