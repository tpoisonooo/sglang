#!/usr/bin/env python3
"""
Example: ModelOpt Quantization and Export with SGLang

This example demonstrates the streamlined workflow for quantizing a model with
ModelOpt and automatically exporting it for deployment with SGLang.
"""

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn

import sglang as sgl
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.model_loader.loader import get_model_loader


# =============================================================================
# Custom Quantization Module for LightningAttention
# =============================================================================
# LightningAttention uses custom GLA kernels (chunk_simple_gla) instead of 
# standard matmul/bmm/sdpa, so ModelOpt cannot auto-detect the attention pattern.
# We register a custom QuantModule to ensure its Linear layers are properly quantized.

def register_lightning_attention_quantization():
    """Register custom quantization module for LightningAttention."""
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
        from modelopt.torch.quantization.conversion import register
        
        class QuantLightningAttention(QuantModule):
            """
            Quantized LightningAttention module.
            
            This module quantizes:
            1. Linear layers (q_proj, k_proj, v_proj, o_proj) - weights and inputs
            2. Recurrent state for GLA attention (similar to KV Cache quantization)
            
            The custom GLA attention kernel itself is not quantized as it uses 
            custom CUDA kernels.
            """
            
            def _setup(self):
                """Setup quantizers for LightningAttention's Linear layers and recurrent state."""
                # Input quantizers for each projection (weight quantization)
                self.q_proj_input_quantizer = TensorQuantizer()
                self.k_proj_input_quantizer = TensorQuantizer()
                self.v_proj_input_quantizer = TensorQuantizer()
                self.o_proj_input_quantizer = TensorQuantizer()
                
                # Recurrent state quantizers (KV Cache equivalent for GLA)
                self.recurrent_state_quantizer = TensorQuantizer()
                self.q_attn_quantizer = TensorQuantizer()
                self.k_attn_quantizer = TensorQuantizer()
                self.v_attn_quantizer = TensorQuantizer()
                
            def forward(
                self,
                hidden_states: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[tuple] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[object] = None,
                **kwargs,
            ):
                """Forward with quantized Linear layers."""
                bsz, seqlen, _ = hidden_states.shape
                
                # Quantize inputs before Linear projections
                q = self.q_proj(self.q_proj_input_quantizer(hidden_states))
                k = self.k_proj(self.k_proj_input_quantizer(hidden_states))
                v = self.v_proj(self.v_proj_input_quantizer(hidden_states))
                
                # ... (rest of original forward logic)
                # Reuse the original forward implementation
                from einops import rearrange, repeat
                
                q = rearrange(q, "b t (h d) -> b h t d", d=self.head_dim)
                k = rearrange(k, "b t (h d) -> b h t d", d=self.head_dim)
                v = rearrange(v, "b t (h d) -> b h t d", d=self.head_dim)

                if self.qk_norm:
                    q = self.q_norm(q)
                    k = self.k_norm(k)

                if self.use_rope and position_embeddings is not None:
                    kv_seq_len = position_ids.max().item() + 1 if position_ids is not None else seqlen
                    cos, sin = self.rotary_emb(v.to(torch.float32), seq_len=kv_seq_len)
                    q, k = self._apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                k = repeat(k, "b h t d -> b (h g) t d", g=self.num_key_value_groups)
                v = repeat(v, "b h t d -> b (h g) t d", g=self.num_key_value_groups)

                s = self._get_slope_tensor().to(k.device, dtype=torch.float32) * (-1.0)

                initial_state = None
                if past_key_value is not None:
                    layer_state = past_key_value.layers[self.layer_idx].state
                    initial_state = layer_state.get("recurrent_state", None)

                q = rearrange(q, "b h t d -> b t h d").to(torch.float32)
                k = rearrange(k, "b h t d -> b t h d").to(torch.float32)
                v = rearrange(v, "b h t d -> b t h d").to(torch.float32)
                s = s.to(torch.float32)
                
                # Apply attention input quantizers (KV Cache quantization equivalent)
                q = self.q_attn_quantizer(q)
                k = self.k_attn_quantizer(k)
                v = self.v_attn_quantizer(v)
                
                # Quantize initial recurrent state if present
                if initial_state is not None:
                    initial_state = self.recurrent_state_quantizer(initial_state)

                # Call the custom attention function
                o, final_state = self.attn_fn(
                    q=q,
                    k=k,
                    v=v,
                    decay=s,
                    initial_state=initial_state,
                    scale=self.scale,
                    attention_mask=attention_mask,
                )
                
                # Quantize final recurrent state before returning
                if final_state is not None:
                    final_state = self.recurrent_state_quantizer(final_state)

                if past_key_value is not None:
                    past_key_value.layers[self.layer_idx].update(
                        recurrent_state=final_state,
                        layer_idx=self.layer_idx,
                        offset=seqlen,
                    )

                o = rearrange(o, "b t h d -> b t (h d)").contiguous().to(hidden_states.dtype)

                if self.use_output_norm:
                    o = self.o_norm(o)

                if self.use_output_gate:
                    z = torch.sigmoid(self.z_proj(hidden_states))
                    o = o * z

                # Quantize input before o_proj
                y = self.o_proj(self.o_proj_input_quantizer(o))
                return y, None, past_key_value
            
            def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
                """Apply rotary position embedding."""
                from transformers.models.llama.modeling_llama import rotate_half
                
                # Expand cos/sin for broadcasting
                cos = cos.unsqueeze(1)  # [batch, 1, seq_len, dim]
                sin = sin.unsqueeze(1)
                
                # Apply RoPE
                q_embed = (q * cos) + (rotate_half(q) * sin)
                k_embed = (k * cos) + (rotate_half(k) * sin)
                return q_embed, k_embed
            
            def _get_slope_tensor(self):
                """Get slope tensor for GLA attention."""
                import math
                h = self.num_attention_heads
                # Create slope tensor similar to _build_slope_tensor in original code
                def get_slopes(n):
                    def get_slopes_power_of_2(n):
                        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                        ratio = start
                        return [start * ratio ** i for i in range(n)]
                    
                    if math.log2(n).is_integer():
                        return get_slopes_power_of_2(n)
                    else:
                        closest_power_of_2 = 2 ** math.floor(math.log2(n))
                        return (
                            get_slopes_power_of_2(closest_power_of_2)
                            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                        )
                
                slopes = get_slopes(h)
                return torch.tensor(slopes, dtype=torch.float32)
        
        # Try to import LightningAttention and register the quantized version
        try:
            # LightningAttention will be available after model is loaded with trust_remote_code
            # We'll register it dynamically in the quantize function
            print("✅ Custom QuantLightningAttention defined successfully")
            return QuantLightningAttention
        except Exception as e:
            print(f"⚠️  Could not register LightningAttention quantization: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠️  ModelOpt not available for custom quantization: {e}")
        return None


# Global variable to hold the custom quant class
_QuantLightningAttention = register_lightning_attention_quantization()


def _validate_export(export_dir: str) -> bool:
    """Validate that an exported model directory contains the expected files."""
    import glob

    required_files = ["config.json", "tokenizer_config.json"]

    if not os.path.exists(export_dir):
        return False

    # Check required files
    for file in required_files:
        if not os.path.exists(os.path.join(export_dir, file)):
            return False

    # Check for model files using pattern matching to handle sharded models
    model_patterns = [
        "model*.safetensors",
        "pytorch_model*.bin",
    ]

    has_model_file = False
    for pattern in model_patterns:
        matching_files = glob.glob(os.path.join(export_dir, pattern))
        if matching_files:
            has_model_file = True
            break

    return has_model_file


def _get_export_info(export_dir: str) -> Optional[dict]:
    """Get information about an exported model."""
    import json

    if not _validate_export(export_dir):
        return None

    try:
        config_path = os.path.join(export_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        return {
            "model_type": config.get("model_type", "unknown"),
            "architectures": config.get("architectures", []),
            "quantization_config": config.get("quantization_config", {}),
            "export_dir": export_dir,
        }
    except Exception:
        return None


def quantize_and_export_model(
    model_path: str,
    export_dir: str,
    quantization_method: str = "modelopt_fp8",
    checkpoint_save_path: Optional[str] = None,
    device: str = "cuda",
    skip_first_n_layers: int = 0,
    skip_last_n_layers: int = 0,
    layer_prefix: str = "model.layers",
) -> None:
    """
    Quantize a model with ModelOpt and export it for SGLang deployment.

    Args:
        model_path: Path to the original model
        export_dir: Directory to export the quantized model
        quantization_method: Quantization method ("modelopt_fp8" or "modelopt_fp4")
        checkpoint_save_path: Optional path to save ModelOpt checkpoint
        device: Device to use for quantization
    """
    print("🚀 Starting ModelOpt quantization and export workflow")
    print(f"📥 Input model: {model_path}")
    print(f"📤 Export directory: {export_dir}")
    print(f"⚙️  Quantization method: {quantization_method}")

    # Initialize minimal distributed environment for single GPU quantization
    if not torch.distributed.is_initialized():
        print("🔧 Initializing distributed environment...")
        # Set up environment variables for single-process distributed
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"  # Use a different port than tests
        os.environ["LOCAL_RANK"] = "0"

        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl" if device == "cuda" else "gloo",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    # Pre-load model code to register custom quantization for LightningAttention
    print("🔧 Pre-loading model code for custom quantization registration...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Import the modeling module to get LightningAttention class
        import importlib.util
        import sys
        
        model_file = os.path.join(model_path, "modeling_minicpm_sala.py")
        if os.path.exists(model_file):
            spec = importlib.util.spec_from_file_location(
                "modeling_minicpm_sala", model_file
            )
            modeling_module = importlib.util.module_from_spec(spec)
            sys.modules["modeling_minicpm_sala"] = modeling_module
            spec.loader.exec_module(modeling_module)
            
            # Get LightningAttention class and register custom quantization
            if hasattr(modeling_module, "LightningAttention") and _QuantLightningAttention is not None:
                import modelopt.torch.quantization as mtq
                from modelopt.torch.quantization.conversion import register
                LightningAttention = modeling_module.LightningAttention
                mtq.register(LightningAttention, _QuantLightningAttention)
                print(f"✅ Registered custom weight quantization for {LightningAttention}")
                
                # Also register to QuantModuleRegistry to prevent AST-based KV Cache quantization attempt
                # Since LightningAttention uses GLA (not standard attention), we skip KV Cache quantization
                try:
                    from modelopt.torch.quantization.nn import QuantModuleRegistry
                    # Check if already registered
                    if QuantModuleRegistry.get(LightningAttention) is None:
                        QuantModuleRegistry.register({LightningAttention: "LightningAttention"})(_QuantLightningAttention)
                        print(f"✅ Registered to QuantModuleRegistry (skipping AST-based KV Cache quantization)")
                except Exception as e:
                    print(f"⚠️  Could not register to QuantModuleRegistry: {e}")
            else:
                print("ℹ️  LightningAttention will use default quantization (no custom module)")
            
            # Note about KV Cache quantization
            print("""
📌 Note about KV Cache Quantization:
   - LightningAttention uses GLA (Gated Linear Attention) with recurrent state
   - It does NOT use traditional KV Cache, so KV Cache quantization is not applicable
   - MiniCPMSdpaAttention layers (if any) will have KV Cache quantized normally
            """)
    except Exception as e:
        print(f"⚠️  Could not pre-register LightningAttention quantization: {e}")
        print("   Will proceed with default quantization...")

    # Configure model loading with ModelOpt quantization and export
    print(f"📋 Layer quantization settings:")
    print(f"   - Skip first {skip_first_n_layers} layers")
    print(f"   - Skip last {skip_last_n_layers} layers")
    print(f"   - Layer prefix: '{layer_prefix}'")
    
    model_config = ModelConfig(
        model_path=model_path,
        quantization=quantization_method,  # Use unified quantization flag
        trust_remote_code=True,
        modelopt_skip_first_n_layers=skip_first_n_layers,
        modelopt_skip_last_n_layers=skip_last_n_layers,
        modelopt_layer_prefix=layer_prefix,
    )

    load_config = LoadConfig(
        modelopt_checkpoint_save_path=checkpoint_save_path,
        modelopt_export_path=export_dir,
    )
    device_config = DeviceConfig(device=device)

    # Load and quantize the model (export happens automatically)
    print("🔄 Loading and quantizing model...")
    model_loader = get_model_loader(load_config, model_config)

    try:
        model_loader.load_model(
            model_config=model_config,
            device_config=device_config,
        )
        print("✅ Model quantized successfully!")

        # Validate the export
        if _validate_export(export_dir):
            print("✅ Export validation passed!")

            info = _get_export_info(export_dir)
            if info:
                print("📋 Model info:")
                print(f"   - Type: {info['model_type']}")
                print(f"   - Architecture: {info['architectures']}")
                print(f"   - Quantization: {info['quantization_config']}")
        else:
            print("❌ Export validation failed!")
            return

    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return

    print("\n🎉 Workflow completed successfully!")
    print(f"📁 Quantized model exported to: {export_dir}")
    print("\n🚀 To use the exported model:")
    print(
        f"   python -m sglang.launch_server --model-path {export_dir} --quantization modelopt"
    )
    print("\n   # Or in Python:")
    print("   import sglang as sgl")
    print(f"   llm = sgl.Engine(model_path='{export_dir}', quantization='modelopt')")
    print("   # Note: 'modelopt' auto-detects FP4/FP8 from model config")


def deploy_exported_model(
    export_dir: str,
    host: str = "127.0.0.1",
    port: int = 30000,
) -> None:
    """
    Deploy an exported ModelOpt quantized model with SGLang.

    Args:
        export_dir: Directory containing the exported model
        host: Host to bind the server to
        port: Port to bind the server to
    """
    print(f"🚀 Deploying exported model from: {export_dir}")

    # Validate export first
    if not _validate_export(export_dir):
        print("❌ Invalid export directory!")
        return

    try:
        # Launch SGLang engine with the exported model
        # Using generic "modelopt" for auto-detection of FP4/FP8
        llm = sgl.Engine(
            model_path=export_dir,
            quantization="modelopt",
            host=host,
            port=port,
        )

        print("✅ Model deployed successfully!")
        print(f"🌐 Server running at http://{host}:{port}")

        # Example inference
        prompts = ["Hello, how are you?", "What is the capital of France?"]
        sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 100}

        print("\n🧪 Running example inference...")
        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Output: {output['text']}")
            print()

    except Exception as e:
        print(f"❌ Deployment failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ModelOpt Quantization and Export with SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize and export a model (recommended workflow)
  python modelopt_quantize_and_export.py quantize \\
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --export-dir ./quantized_model \\
    --quantization-method modelopt_fp8

  # Deploy a pre-exported model
  python modelopt_quantize_and_export.py deploy \\
    --export-dir ./quantized_model
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quantize command
    quantize_parser = subparsers.add_parser(
        "quantize", help="Quantize and export a model"
    )
    quantize_parser.add_argument(
        "--model-path", required=True, help="Path to the model to quantize"
    )
    quantize_parser.add_argument(
        "--export-dir", required=True, help="Directory to export the quantized model"
    )
    quantize_parser.add_argument(
        "--quantization-method",
        choices=["modelopt_fp8", "modelopt_fp4"],
        default="modelopt_fp8",
        help="Quantization method to use. Note: FP4 only supports group_size=16 (hardware limitation)",
    )
    quantize_parser.add_argument(
        "--checkpoint-save-path", help="Optional path to save ModelOpt checkpoint"
    )
    quantize_parser.add_argument(
        "--device", default="cuda", help="Device to use for quantization"
    )
    quantize_parser.add_argument(
        "--skip-first-n-layers",
        type=int,
        default=0,
        help="Skip quantization for first N layers (0 = quantize all). "
             "Skipping first/last layers can improve accuracy. Default: 0",
    )
    quantize_parser.add_argument(
        "--skip-last-n-layers",
        type=int,
        default=0,
        help="Skip quantization for last N layers (0 = quantize all). "
             "Skipping first/last layers can improve accuracy. Default: 0",
    )
    quantize_parser.add_argument(
        "--layer-prefix",
        default="model.layers",
        help="Prefix for layer names in the model. Default: 'model.layers'. "
             "For MiniCPM-SALA: 'model.layers'",
    )

    # TODO: Quantize-and-serve command removed due to compatibility issues
    # Use the separate quantize-then-deploy workflow instead

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy an exported model")
    deploy_parser.add_argument(
        "--export-dir", required=True, help="Directory containing the exported model"
    )
    deploy_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind the server to"
    )
    deploy_parser.add_argument(
        "--port", type=int, default=30000, help="Port to bind the server to"
    )

    args = parser.parse_args()

    if args.command == "quantize":
        quantize_and_export_model(
            model_path=args.model_path,
            export_dir=args.export_dir,
            quantization_method=args.quantization_method,
            checkpoint_save_path=args.checkpoint_save_path,
            device=args.device,
            skip_first_n_layers=args.skip_first_n_layers,
            skip_last_n_layers=args.skip_last_n_layers,
            layer_prefix=args.layer_prefix,
        )
    elif args.command == "deploy":
        deploy_exported_model(
            export_dir=args.export_dir,
            host=args.host,
            port=args.port,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
