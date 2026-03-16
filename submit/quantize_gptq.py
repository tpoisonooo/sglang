import os
from anyio import Path
import pandas as pd
import argparse

# 现在导入 GPTQModel
from gptqmodel import QuantizeConfig, GPTQModel
# 先导入 MODEL_MAP 并注册 MiniCPMSALAQModel，然后再导入 GPTQModel
# from gptqmodel.models import MODEL_MAP
# from gptqmodel.models.base import BaseQModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import torch
# torch.set_float32_matmul_precision('medium')  # 允许 TF32，略快略省内存


# 每次调整 量化的 layer 后，需要改动：
# 1. minicpm.py  里的 opr 的quant参数
# 2. submit 里 minicpm.py 对应修改
# 3. prepare_model 里的 md5

# 20260310 中午提交
# "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "o_gate:1", "z_proj:2") 对应的精度：
"""
{
  "acc": 95.5, # 需要到 97
  "acc_ori": 76.4,
  "final_score": 0.0,
  "benchmark_duration": {
    "S1": 413.33,
    "S8": 575.32,
    "Smax": 1055.36
  }
}
"""
# 20260311 早晨提交
# "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "o_gate:2", "z_proj:!"),
"""
{
  "acc": 96.75, # 需要到 97
  "acc_ori": 77.4,
  "final_score": 0.0,
  "benchmark_duration": {
    "S1": 429.59
    "S8": 591.43,
    "Smax": 1066.22
  }
}
"""
# 20260313 中午提交
"""
{
  "acc": 98.81,
  "acc_ori": 79.04,
  "final_score": 74.92,
  "benchmark_duration": {
    "S1": 434.16,
    "S8": 594.42,
    "Smax": 1067.92
  }
}
"""

# o_gate 后面有 F.sigmoid 还好； down_proj 属于 MLP 最后一层，影响比较大。
# GT 是  82.24%
# GT + fp8_kvcache 80.31%
# GT + math + fp8_kvcache 76.09%
# GT + common + math + fp8_kvcache 模型在 /workspace-moredata (加 common data 到底是否有效？)  76.67%
# GT + common + math + fp8_kvcache + 首尾 g128 模型在 /data/share/minicpm   78.64%
# GT + common + math + fp8_kvcache + 首尾 g128 + 放弃 downgate   76.18%

def load_q_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # files = ['common.jsonl', 'math.jsonl', 'perf_public_set.jsonl']
    # files = ['math.jsonl', 'extra.jsonl', 'common.jsonl']
    files = ['extra.jsonl']
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
    parser.add_argument("--input", required=False, default='/data/share/MiniCPM-SALA', help="Original model directory")
    parser.add_argument("--output", required=True, default='/data/share/MiniCPM-SALA-int4', help="Quantized model output directory")
    args = parser.parse_args()

    model_path = args.input
    quant_path = args.output

    quant_config = QuantizeConfig(bits=4, group_size=128, dynamic={
            # 跳过第一层和最后一层 
            '+:model\\.model\\.layers\\.0\\..*': {'bits': 8},
            '+:model\\.model\\.layers\\.31\\..*': {'bits': 8},

            # o_gate 使用 8-bit（所有层）
            # r'+:model\.model\.layers\.\d+\.self_attn\.o_gate': {'bits': 8, 'group_size': 64},
        }
    ) # quantization config
    # model = GPTQModel.load(model_path, quant_config, trust_remote_code=True, attn_implementation="flash_attention_2") # load model
    model = GPTQModel.load(model_path, quant_config, trust_remote_code=True) # load model

    model.layer_modules_strict = False
    calibration_dataset = load_q_dataset()
    model.quantize(calibration_dataset, batch_size=1) # quantize
    model.save(quant_path) # save model

main()
