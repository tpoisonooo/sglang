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
import pdb
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
from python.sglang.srt.models.minicpm_fused_scale_add import fused_scale_add
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
        import time
        t0 = time.perf_counter()
        print(f"[MiniCPMMLP] input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        gate_up, _ = self.gate_up_proj(x)
        t1 = time.perf_counter()
        print(f"[MiniCPMMLP] gate_up_proj: {x.shape} -> {gate_up.shape}, time: {(t1-t0)*1000:.3f}ms")
        x = self.act_fn(gate_up)
        t2 = time.perf_counter()
        print(f"[MiniCPMMLP] act_fn (SiluAndMul): {gate_up.shape} -> {x.shape}, time: {(t2-t1)*1000:.3f}ms")
        x, _ = self.down_proj(x)
        t3 = time.perf_counter()
        print(f"[MiniCPMMLP] down_proj: {gate_up.shape} -> {x.shape}, time: {(t3-t2)*1000:.3f}ms")
        print(f"[MiniCPMMLP] output shape: {x.shape}, total: {(t3-t0)*1000:.3f}ms")
        print(f"[MiniCPMMLP] compute: Linear(hidden->{gate_up.shape[-1]}) -> SiLU+Mul -> Linear({gate_up.shape[-1]}->hidden)")
        return x


class MiniCPMAttention(nn.Module):
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
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
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

        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        import time
        t0 = time.perf_counter()
        print(f"[MiniCPMAttention] input shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        print(f"[MiniCPMAttention] positions shape: {positions.shape}")
        
        qkv, _ = self.qkv_proj(hidden_states)
        t1 = time.perf_counter()
        print(f"[MiniCPMAttention] qkv_proj: {hidden_states.shape} -> {qkv.shape}, time: {(t1-t0)*1000:.3f}ms")
        
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        print(f"[MiniCPMAttention] split qkv: q={q.shape}, k={k.shape}, v={v.shape}")

        if self.attn_use_rope:
            t_rope0 = time.perf_counter()
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
            q, k = self.rotary_emb(positions, q, k)
            q, k = q.to(orig_dtype), k.to(orig_dtype)
            t_rope1 = time.perf_counter()
            print(f"[MiniCPMAttention] rotary_emb: time: {(t_rope1-t_rope0)*1000:.3f}ms")

        t_attn0 = time.perf_counter()
        attn_output = self.attn(q, k, v, forward_batch)
        t_attn1 = time.perf_counter()
        print(f"[MiniCPMAttention] RadixAttention: time: {(t_attn1-t_attn0)*1000:.3f}ms")

        if self.use_output_gate:
            t_gate0 = time.perf_counter()
            o_gate_output, _ = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(o_gate_output)
            t_gate1 = time.perf_counter()
            print(f"[MiniCPMAttention] output_gate: time: {(t_gate1-t_gate0)*1000:.3f}ms")

        output, _ = self.o_proj(attn_output)
        t2 = time.perf_counter()
        print(f"[MiniCPMAttention] o_proj: {attn_output.shape} -> {output.shape}, time: {(t2-t_attn1)*1000:.3f}ms")
        print(f"[MiniCPMAttention] output shape: {output.shape}, total: {(t2-t0)*1000:.3f}ms")
        print(f"[MiniCPMAttention] compute: Linear(hidden->qkv) -> RoPE -> Attention(num_heads={self.num_heads}, head_dim={self.head_dim}) -> Linear")
        return output



class MiniCPMLightningMixer(nn.Module):
    """Lightning attention mixer that uses SimpleGLAAttnBackend.

    This is a wrapper that prepares inputs for the backend and handles
    the QKV projection, normalization, RoPE, and output processing,
    while delegating the Simple GLA kernel calls to SimpleGLAAttnBackend.
    """

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
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
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

        if self.use_output_gate:
            self.z_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=self.attention_bias,
                quant_config=quant_config,
                prefix=add_prefix("z_proj", prefix),
            )

        if self.qk_norm:
            # Keep original norms for weight loading compatibility
            self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        if self.use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )

        self.layer_id = layer_id
        self.state_shape = (self.num_kv_heads, self.head_dim, self.head_dim)
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        import time
        t0 = time.perf_counter()
        print(f"[MiniCPMLightningMixer] input shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        print(f"[MiniCPMLightningMixer] positions shape: {positions.shape}")
        print(f"[MiniCPMLightningMixer] layer_id={self.layer_id}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
        qkv, _ = self.qkv_proj(hidden_states)  # [seq_len, hidden_size] -> [seq_len, q_size+kv_size+kv_size]
        t1 = time.perf_counter()
        print(f"[MiniCPMLightningMixer] qkv_proj: {hidden_states.shape} -> {qkv.shape}, time: {(t1-t0)*1000:.3f}ms")
        
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        print(f"[MiniCPMLightningMixer] split: q={q.shape}, k={k.shape}, v={v.shape}")

        if not self.qk_norm or not self.use_rope:
            pdb.set_trace()
            print('not qk_norm or not use_rope')
            pass

        # Use fused RMSNorm + RoPE kernel
        # Get cos_sin_cache from rotary_emb (already in FP32)
        t_fused0 = time.perf_counter()
        cos_sin_cache = self.rotary_emb.cos_sin_cache  # [max_position, head_dim * 2]
        q, k = fused_rms_norm_rope(
            q=q,
            k=k,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            q_norm_weight=self.q_norm.weight,
            k_norm_weight=self.k_norm.weight,
            eps=self.rms_norm_eps,
        )
        t_fused1 = time.perf_counter()
        print(f"[MiniCPMLightningMixer] fused_rms_norm_rope: time: {(t_fused1-t_fused0)*1000:.3f}ms")
        # else:
        # if True:
            # Fallback to original implementation
            # q = self.q_norm(q.contiguous().view(-1, self.head_dim))
            # k = self.k_norm(k.contiguous().view(-1, self.head_dim))
            # q = q.view(-1, self.num_heads * self.head_dim)
            # k = k.view(-1, self.num_kv_heads * self.head_dim)
        
            # orig_dtype = q.dtype
            # q, k = q.float(), k.float()
            # q, k = self.rotary_emb(positions, q, k)
            # q = q.to(orig_dtype).view(1, -1, self.num_heads, self.head_dim).contiguous()
            # k = k.to(orig_dtype).view(1, -1, self.num_kv_heads, self.head_dim).contiguous()

        v = v.reshape(1, -1, self.num_kv_heads, self.head_dim)
        print(f"[MiniCPMLightningMixer] reshape for backend: q={q.shape}, k={k.shape}, v={v.shape}")

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

        # Prepare backend inputs
        # Backend expects q, k, v, forward_batch, layer_id
        # It will handle state loading/saving internally
        t_backend0 = time.perf_counter()
        o = linear_attn_backend.forward(
            q=q,
            k=k,
            v=v,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
            output_attentions=False,
        )
        t_backend1 = time.perf_counter()
        print(f"[MiniCPMLightningMixer] SimpleGLAAttnBackend.forward: time: {(t_backend1-t_backend0)*1000:.3f}ms")

        if not self.use_output_gate:
            pdb.set_trace()
            print('no output gate')
            pass
    
        if not self.use_output_norm:
            pdb.set_trace()
            print('no output norm')
            pass

        z, _ = self.z_proj(hidden_states)   # [seq_len, hidden_size] -> [seq_len, 4096]
        t_z0 = time.perf_counter()
        print(f"[MiniCPMLightningMixer] z_proj: time: {(t_z0-t_backend1)*1000:.3f}ms")

        # o = o.reshape(-1, self.num_heads * self.head_dim)   # -> [142,4096]
        # fusing start - use fused kernel
        t_fused_out0 = time.perf_counter()
        o = fused_output_processing(
            o=o,
            z=z,
            norm_weight=self.o_norm.weight,
            eps=self.rms_norm_eps,
        )
        t_fused_out1 = time.perf_counter()
        print(f"[MiniCPMLightningMixer] fused_output_processing (o_norm + sigmoid gate): time: {(t_fused_out1-t_fused_out0)*1000:.3f}ms")
        # fusing end
        # else:
        # if True:
            # fusing start
            # o = self.o_norm(o)  # -> [142,4096]
            # o = o * F.sigmoid(z)
            # fusing end
        # diff = o2 - o1
        # pdb.set_trace()
        y, _ = self.o_proj(o)
        t2 = time.perf_counter()
        print(f"[MiniCPMLightningMixer] o_proj: {o.shape} -> {y.shape}, time: {(t2-t_fused_out1)*1000:.3f}ms")
        print(f"[MiniCPMLightningMixer] output shape: {y.shape}, total: {(t2-t0)*1000:.3f}ms")
        print(f"[MiniCPMLightningMixer] compute: Linear -> FusedRMSNorm+RoPE -> SimpleGLA -> FusedOutputNorm+Gate -> Linear")
        return y


class MiniCPMDecoderLayer(nn.Module):
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
        import time
        t_layer0 = time.perf_counter()
        print(f"\n[MiniCPMDecoderLayer {self.layer_id}] START input shape: {hidden_states.shape}, mixer_type={self.mixer_type}")
        
        # Build sparse metadata (model-specific logic!)

        # Self Attention
        t_attn0 = time.perf_counter()
        residual = hidden_states
        print(f"[MiniCPMDecoderLayer {self.layer_id}] residual saved")
        
        t_norm0 = time.perf_counter()
        hidden_states = self.input_layernorm(hidden_states)
        t_norm1 = time.perf_counter()
        print(f"[MiniCPMDecoderLayer {self.layer_id}] input_layernorm: time: {(t_norm1-t_norm0)*1000:.3f}ms")
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        t_attn1 = time.perf_counter()
        print(f"[MiniCPMDecoderLayer {self.layer_id}] self_attn total: {(t_attn1-t_attn0)*1000:.3f}ms")
        
        t_fused0 = time.perf_counter()
        hidden_states = fused_scale_add(hidden_states, residual, self.hidden_scale)
        t_fused1 = time.perf_counter()
        print(f"[MiniCPMDecoderLayer {self.layer_id}] fused_scale_add (residual + attn_out * {self.hidden_scale:.4f}): time: {(t_fused1-t_fused0)*1000:.3f}ms")
        
        # Fully Connected
        t_mlp0 = time.perf_counter()
        residual = hidden_states
        print(f"[MiniCPMDecoderLayer {self.layer_id}] residual saved for MLP")
        
        t_norm2 = time.perf_counter()
        hidden_states = self.post_attention_layernorm(hidden_states)
        t_norm3 = time.perf_counter()
        print(f"[MiniCPMDecoderLayer {self.layer_id}] post_attention_layernorm: time: {(t_norm3-t_norm2)*1000:.3f}ms")
        
        hidden_states = self.mlp(hidden_states)
        t_mlp1 = time.perf_counter()
        print(f"[MiniCPMDecoderLayer {self.layer_id}] mlp total: {(t_mlp1-t_norm3)*1000:.3f}ms")
        
        t_fused2 = time.perf_counter()
        hidden_states = fused_scale_add(hidden_states, residual, self.hidden_scale)
        t_fused3 = time.perf_counter()
        print(f"[MiniCPMDecoderLayer {self.layer_id}] fused_scale_add (residual + mlp_out * {self.hidden_scale:.4f}): time: {(t_fused3-t_fused2)*1000:.3f}ms")
        
        t_layer1 = time.perf_counter()
        print(f"[MiniCPMDecoderLayer {self.layer_id}] END output shape: {hidden_states.shape}, layer total: {(t_layer1-t_layer0)*1000:.3f}ms")

        return hidden_states, None


class MiniCPMModel(nn.Module):
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
        import time
        t0 = time.perf_counter()
        print(f"\n{'='*80}")
        print(f"[MiniCPMModel] START forward")
        print(f"[MiniCPMModel] input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"[MiniCPMModel] positions shape: {positions.shape}")
        
        t_embed0 = time.perf_counter()
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
            print(f"[MiniCPMModel] embed_tokens + scale({self.config.scale_emb})")
        else:
            hidden_states = input_embeds
            print(f"[MiniCPMModel] using input_embeds")
        t_embed1 = time.perf_counter()
        print(f"[MiniCPMModel] embedding: {input_ids.shape} -> {hidden_states.shape}, time: {(t_embed1-t_embed0)*1000:.3f}ms")
        
        residual = None
        num_layers = len(self.layers)
        print(f"[MiniCPMModel] num_layers: {num_layers}")

        for i in range(num_layers):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        
        t_norm0 = time.perf_counter()
        hidden_states = self.norm(hidden_states)
        t_norm1 = time.perf_counter()
        print(f"[MiniCPMModel] final norm: time: {(t_norm1-t_norm0)*1000:.3f}ms")
        
        t1 = time.perf_counter()
        print(f"[MiniCPMModel] END output shape: {hidden_states.shape}, total: {(t1-t0)*1000:.3f}ms")
        print(f"{'='*80}\n")
        return hidden_states


class MiniCPMSALAForCausalLM(nn.Module):
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
        import time
        t0 = time.perf_counter()
        print(f"\n{'#'*80}")
        print(f"[MiniCPMSALAForCausalLM] START forward")
        print(f"[MiniCPMSALAForCausalLM] input_ids: {input_ids.shape}, positions: {positions.shape}")
        
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
            print(f"[MiniCPMSALAForCausalLM] scaled input_embeds")

        t_model0 = time.perf_counter()
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        t_model1 = time.perf_counter()
        print(f"[MiniCPMSALAForCausalLM] model forward: {(t_model1-t_model0)*1000:.3f}ms")
        
        t_scale0 = time.perf_counter()
        hidden_states = hidden_states / self.scale_width
        t_scale1 = time.perf_counter()
        print(f"[MiniCPMSALAForCausalLM] scale_width ({self.scale_width:.4f}): time: {(t_scale1-t_scale0)*1000:.3f}ms")
        
        if self.config.tie_word_embeddings:
            lm_head = self.model.embed_tokens
            print(f"[MiniCPMSALAForCausalLM] using embed_tokens as lm_head")
        else:
            lm_head = self.lm_head
            print(f"[MiniCPMSALAForCausalLM] using lm_head")

        t_logits0 = time.perf_counter()
        result = self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)
        t_logits1 = time.perf_counter()
        print(f"[MiniCPMSALAForCausalLM] logits_processor: time: {(t_logits1-t_logits0)*1000:.3f}ms")
        
        t1 = time.perf_counter()
        print(f"[MiniCPMSALAForCausalLM] END total: {(t1-t0)*1000:.3f}ms")
        print(f"{'#'*80}\n")
        return result

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
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

# EntryClass = [MiniCPMSALAForCausalLM]
