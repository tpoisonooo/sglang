import os
from anyio import Path
import pandas as pd
import argparse

from gptqmodel import QuantizeConfig, GPTQModel
from gptqmodel.models.base import BaseQModel
from gptqmodel.models import MODEL_MAP
# import pdb; pdb.set_trace()

class MiniCPMSALAQModel(BaseQModel):
    """
    MiniCPM SALA (Sparse Attention with Linear Attention) model support.
    
    This model supports two mixer types:
    1. minicpm4: Standard attention with optional output gate (o_gate)
    2. lightning: Lightning attention with output gate (z_proj)
    
    Note: z_proj is not quantized as per the original model design.
    """
    
    pre_lm_head_norm_module = "model.norm"
    layer_modules_strict = False  # 允许动态模块

    # Module tree for MiniCPM SALA model
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "o_gate:1", "z_proj:2"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

MODEL_MAP['minicpm_sala'] = MiniCPMSALAQModel


def load_q_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    files = ['common.jsonl', 'math.jsonl', 'extra.jsonl']
    # files = ['extra.jsonl']
    calibration_dataset = []
    for filename in files:
        # 获取脚本所在目录，拼接 jsonl 路径
        jsonl_path = os.path.join(script_dir, filename)
        # 使用 pandas 读取 jsonl（处理混合类型），只保留 question 列
        df = pd.read_json(jsonl_path, lines=True)
        calibration_dataset += df["question"].tolist()
    return calibration_dataset

def main():
    parser = argparse.ArgumentParser(description="RTN W4A16 quantization")
    parser.add_argument("--input", required=True, default='/data/share/MiniCPM-SALA', help="Original model directory")
    parser.add_argument("--output", required=True, default='/data/share/MiniCPM-SALA-int4', help="Quantized model output directory")
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    model_path = args.input
    quant_path = args.output

    quant_config = QuantizeConfig(bits=4, group_size=128) # quantization config
    model = GPTQModel.load(model_path, quant_config, trust_remote_code=True, attn_implementation="flash_attention_2") # load model

    model.layer_modules_strict = False
    calibration_dataset = load_q_dataset()
    model.quantize(calibration_dataset, batch_size=2) # quantize
    model.save(quant_path) # save model
