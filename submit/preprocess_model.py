#!/usr/bin/env python3
"""
================================================================================
模型预处理脚本 - GPTQ W4A16 量化
被 prepare_model.sh 调用
================================================================================

Usage:
    python preprocess_model.py --input /path/to/model --output /path/to/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 导入量化脚本中的函数
from quantize_model import quantize_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SALA model with GPTQ quantization"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input model path (original FP16 model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output model path (quantized model)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    # 验证输入路径
    if not input_path.exists():
        print(f"[preprocess] Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    print(f"[preprocess] Input: {input_path}")
    print(f"[preprocess] Output: {output_path}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 执行量化
    # 使用推荐的配置：W4A16, group_size=128, desc_act=False
    try:
        quantize_model(
            model_path=str(input_path),
            output_path=str(output_path),
            bits=4,
            group_size=128,
            desc_act=False,
            num_calibration_samples=128,
        )
        print("[preprocess] Quantization completed successfully")
    except Exception as e:
        print(f"[preprocess] Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
