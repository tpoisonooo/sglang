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
"""A tensor parallel worker."""
from __future__ import annotations

import copy
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    SendWeightsToRemoteInstanceReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class BaseTpWorker(ABC):
    @abstractmethod
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        pass

    @property
    @abstractmethod
    def model_runner(self) -> "ModelRunner":
        pass

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        return self.model_runner.is_hybrid_swa

    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return getattr(self.model_runner.attention_tp_group, "cpu_group", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        success, message = (
            self.model_runner.init_weights_send_group_for_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_rank,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.group_name,
            recv_req.load_format,
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):

        monkey_patch_torch_reductions()
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def can_run_lora_batch(self, lora_ids: list[str]) -> bool:
        lora_ids_set = set(lora_ids) if isinstance(lora_ids, list) else lora_ids
        return self.model_runner.lora_manager.validate_lora_batch(lora_ids_set)

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch).logits_output
        embeddings = logits_output.embeddings
        return embeddings


class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        is_multi_layer_eagle: bool = False,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.is_draft_worker = is_draft_worker
        self.is_multi_layer_eagle = is_multi_layer_eagle
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        # MTP model runners
        self.model_runner_list = []

        # Dynamic quantization support (auto-detect dual model structure)
        self.enable_dynamic_quant = getattr(server_args, 'enable_dynamic_quant', False)
        self._fp4_model_runner = None
        self._int4_model_runner = None
        self._active_runner_type = "fp4"  # default
        
        # Auto-detect dual model structure using _dual_model_base_path if available
        if not is_draft_worker:
            # Use stored base path if available (set by server_args), otherwise use model_path
            dual_base = getattr(self.server_args, '_dual_model_base_path', None)
            check_path = dual_base or self.server_args.model_path
            
            fp4_path = os.path.join(check_path, "fp4")
            int4_path = os.path.join(check_path, "int4")
            if os.path.isdir(fp4_path) and os.path.isdir(int4_path):
                if dual_base is None:
                    logger.info(f"Detected dual model structure: {check_path}")
                    self.enable_dynamic_quant = True
                # Store paths for later use
                self._dual_fp4_path = fp4_path
                self._dual_int4_path = int4_path
            else:
                self._dual_fp4_path = None
                self._dual_int4_path = None

        self._init_model_config()
        
        # Allow disabling dual runner via environment variable for testing
        if os.environ.get('SGLANG_DISABLE_DUAL_RUNNER') == '1':
            logger.info("Dual runner disabled via SGLANG_DISABLE_DUAL_RUNNER=1")
            self.enable_dynamic_quant = False
        
        if self.enable_dynamic_quant and not is_draft_worker:
            self._init_dual_model_runners()
        else:
            self._init_model_runner()

        if is_multi_layer_eagle:
            self._init_multi_layer_eagle_model_runners()

        self._init_dllm_algorithm()

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = self.model_runner.max_running_requests
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = server_args.speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None

    def _init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=(
                self.server_args.model_path
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_path
            ),
            model_revision=(
                self.server_args.revision
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_revision
            ),
            is_draft_model=self.is_draft_worker,
        )

    def _init_model_runner(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )

    def _init_dual_model_runners(self):
        """Initialize both FP4 and INT4 model runners for dynamic quantization."""
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.configs.model_config import ModelConfig
        
        logger.info("Initializing dual model runners for dynamic quantization")
        
        # Import z_proj quant config controller
        from sglang.srt.models.minicpm import set_z_proj_quant_enabled
        
        # Split memory fraction between two runners
        mem_fraction_per_runner = self.server_args.mem_fraction_static / 2
        
        # Determine paths: use stored dual base path or auto-detect
        import os
        dual_base = getattr(self.server_args, '_dual_model_base_path', None)
        if dual_base:
            fp4_path = os.path.join(dual_base, "fp4")
            int4_path = os.path.join(dual_base, "int4")
        else:
            # Fallback to explicit paths or model_path
            fp4_path = getattr(
                self.server_args, 'dynamic_quant_fp4_path', self.server_args.model_path
            )
            int4_path = getattr(
                self.server_args, 'dynamic_quant_int4_path', self.server_args.model_path
            )
        
        logger.info(f"Dual runner paths - FP4: {fp4_path}, INT4: {int4_path}")
        
        # 1. Initialize INT4 runner first (default for low concurrency)
        # For INT4/GPTQ, z_proj is NOT quantized
        set_z_proj_quant_enabled(False)
        
        int4_server_args = copy.deepcopy(self.server_args)
        int4_server_args.model_path = int4_path
        int4_server_args.quantization = "gptq"  # INT4 uses GPTQ/Marlin
        int4_server_args.dtype = "half"  # GPTQ requires float16
        # Keep tokenizer from FP4 model (shared tokenizer)
        int4_server_args.tokenizer_path = self.server_args.tokenizer_path
        
        # Create INT4 model config
        int4_model_config = ModelConfig.from_server_args(
            int4_server_args,
            model_path=int4_server_args.model_path,
            model_revision=self.server_args.revision,
            is_draft_model=False,
        )
        
        self._int4_model_runner = ModelRunner(
            model_config=int4_model_config,
            mem_fraction_static=mem_fraction_per_runner,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=int4_server_args,
            is_draft_worker=False,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            draft_model_idx=None,
        )
        
        logger.info(f"INT4 runner initialized: {int4_path}")
        
        # Debug: Check INT4 z_proj status immediately after init
        try:
            # Find first layer with z_proj (mixed architecture)
            for i, layer in enumerate(self._int4_model_runner.model.model.layers):
                if hasattr(layer.self_attn, 'z_proj'):
                    int4_z_proj_quant = getattr(layer.self_attn.z_proj, 'quant_config', None) is not None
                    logger.info(f"[DualRunner] INT4 layer {i} z_proj quantized: {int4_z_proj_quant}")
                    break
        except Exception as e:
            logger.warning(f"[DualRunner] Failed to check INT4 z_proj: {e}")
        
        # Reset distributed state for second runner (TP=1 case)
        # This is safe because TP=1 doesn't actually use distributed
        from sglang.srt.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        
        # 2. Initialize FP4 runner (for high concurrency)
        # For FP4, z_proj IS quantized
        set_z_proj_quant_enabled(True)
        
        fp4_server_args = copy.deepcopy(self.server_args)
        fp4_server_args.model_path = fp4_path
        fp4_server_args.quantization = "modelopt_fp4"  # FP4 uses modelopt_fp4 quantization
        fp4_server_args.dtype = "half"  # Use float16 to match INT4 runner for consistency
        
        # Create FP4 model config with correct quantization
        fp4_model_config = ModelConfig.from_server_args(
            fp4_server_args,
            model_path=fp4_path,
            model_revision=self.server_args.revision,
            is_draft_model=False,
        )
        
        # Create FP4 runner sharing INT4 runner's memory pools
        # This ensures KV cache is shared between prefill (FP4) and decode (INT4)
        logger.info("Creating FP4 runner with shared memory pools from INT4 runner")
        self._fp4_model_runner = ModelRunner(
            model_config=fp4_model_config,
            mem_fraction_static=mem_fraction_per_runner,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=fp4_server_args,
            is_draft_worker=False,
            req_to_token_pool=self._int4_model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self._int4_model_runner.token_to_kv_pool_allocator,
            draft_model_idx=None,
        )
        
        # Also share token_to_kv_pool, attn_backend and sampler directly
        self._fp4_model_runner.token_to_kv_pool = self._int4_model_runner.token_to_kv_pool
        #self._fp4_model_runner.attn_backend = self._int4_model_runner.attn_backend
        self._fp4_model_runner.sampler = self._int4_model_runner.sampler
        #if hasattr(self._int4_model_runner, 'decode_attn_backend'):
        #    self._fp4_model_runner.decode_attn_backend = self._int4_model_runner.decode_attn_backend
        
        logger.info(f"FP4 runner initialized: {fp4_path}")
        
        # Debug: Check FP4 z_proj status immediately after init
        try:
            # Find first layer with z_proj (mixed architecture)
            for i, layer in enumerate(self._fp4_model_runner.model.model.layers):
                if hasattr(layer.self_attn, 'z_proj'):
                    fp4_z_proj_quant = getattr(layer.self_attn.z_proj, 'quant_config', None) is not None
                    logger.info(f"[DualRunner] FP4 layer {i} z_proj quantized: {fp4_z_proj_quant}")
                    break
        except Exception as e:
            logger.warning(f"[DualRunner] Failed to check FP4 z_proj: {e}")
        
        # Debug: Check z_proj quantization status
        try:
            # Find first layer with z_proj in both runners
            int4_z_proj_quant = None
            fp4_z_proj_quant = None
            for i, layer in enumerate(self._int4_model_runner.model.model.layers):
                if hasattr(layer.self_attn, 'z_proj'):
                    int4_z_proj_quant = getattr(layer.self_attn.z_proj, 'quant_config', None) is not None
                    break
            for i, layer in enumerate(self._fp4_model_runner.model.model.layers):
                if hasattr(layer.self_attn, 'z_proj'):
                    fp4_z_proj_quant = getattr(layer.self_attn.z_proj, 'quant_config', None) is not None
                    break
            logger.info(f"[DualRunner] z_proj quantization - INT4: {int4_z_proj_quant}, FP4: {fp4_z_proj_quant}")
        except Exception as e:
            logger.warning(f"[DualRunner] Failed to check z_proj status: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        
        # Default to INT4 runner (for memory pool reference)
        # Scheduler uses tp_worker's memory pool references
        self._model_runner = self._int4_model_runner
        self._active_runner_type = "int4"
        self.req_to_token_pool = self._int4_model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = self._int4_model_runner.token_to_kv_pool_allocator
        logger.info("Dynamic quantization ready: Prefill-Decode separation (default INT4, scheduler uses INT4 memory pools)")

    def switch_model_runner(self, runner_type: str):
        """Switch between FP4 and INT4 model runners.
        
        Args:
            runner_type: "fp4" or "int4"
        """
        if not self.enable_dynamic_quant:
            return
            
        if runner_type == self._active_runner_type:
            return
        
        if runner_type == "fp4":
            if self._fp4_model_runner is None:
                logger.warning("FP4 runner not available")
                return
            self._model_runner = self._fp4_model_runner
            self._active_runner_type = "fp4"
            # Verify memory pool sharing
            int4_pool_id = id(self._int4_model_runner.token_to_kv_pool)
            fp4_pool_id = id(self._fp4_model_runner.token_to_kv_pool)
            logger.info(f"[DualRunner] Switched to FP4 | KV pool INT4: {int4_pool_id}, FP4: {fp4_pool_id}, shared: {int4_pool_id == fp4_pool_id}")
        elif runner_type == "int4":
            if self._int4_model_runner is None:
                logger.warning("INT4 runner not available")
                return
            self._model_runner = self._int4_model_runner
            self._active_runner_type = "int4"
            logger.info(f"[DualRunner] Switched to INT4 runner (batch size hint: low)")
        else:
            logger.warning(f"Unknown runner type: {runner_type}")

    @property
    def active_runner_type(self) -> str:
        """Get current active runner type."""
        return self._active_runner_type if self.enable_dynamic_quant else "fp4"

    def _init_multi_layer_eagle_model_runners(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self.model_runner_list.append(self.model_runner)
        for i in range(1, self.server_args.speculative_num_steps):
            self.model_runner_list.append(
                ModelRunner(
                    model_config=self.model_config,
                    mem_fraction_static=self.server_args.mem_fraction_static,
                    gpu_id=self.gpu_id,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    moe_ep_rank=self.moe_ep_rank,
                    moe_ep_size=self.ep_size,
                    pp_rank=self.pp_rank,
                    pp_size=self.pp_size,
                    nccl_port=self.nccl_port,
                    dp_rank=self.dp_rank,
                    server_args=self.server_args,
                    is_draft_worker=self.is_draft_worker,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    draft_model_idx=i,
                )
            )

    def _init_dllm_algorithm(self):
        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        if self.server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)
        else:
            self.dllm_algorithm = None

    @property
    def model_runner(self) -> "ModelRunner":
        return self._model_runner

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def is_dllm(self):
        return self.dllm_algorithm is not None

    def _forward_batch_generation_dllm(
        self, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        logits_output, next_token_ids, can_run_cuda_graph = self.dllm_algorithm.run(
            self.model_runner, forward_batch
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def get_remote_instance_transfer_engine_info(self):
        return (
            self.model_runner.remote_instance_transfer_engine_session_id,
            self.model_runner.remote_instance_transfer_engine_weight_info,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        # FIXME(lsyin): maybe remove skip_attn_backend_init in forward_batch_generation,
        #               which requires preparing replay to always be in this function

        # Get forward batch from model worker batch
        if model_worker_batch is not None:
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)

            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            assert forward_batch is not None

        if self.is_dllm():
            return self._forward_batch_generation_dllm(forward_batch)

        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

            if is_verify:
                # Skip sampling and return logits for target forward
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and model_worker_batch.sampling_info.grammars is not None
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not model_worker_batch.is_prefill_only:
                # For normal requests, sample the next token ids.
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
                if (
                    model_worker_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    self.model_runner.compute_logprobs_only(
                        logits_output, model_worker_batch
                    )

            return batch_result
        else:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            batch.split_forward_batch = forward_batch
            batch.seq_lens_cpu_cache = model_worker_batch.seq_lens_cpu
        else:
            model_worker_batch = batch.get_model_worker_batch(batch.seq_lens_cpu_cache)

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
