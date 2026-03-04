#!/usr/bin/env bash
# =============================================================================
# 快速测试脚本 - 验证量化模型
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-/models/MiniCPM-SALA}"
OUTPUT_PATH="${2:-/models/MiniCPM-SALA-quant}"

echo "======================================================================"
echo "量化测试脚本"
echo "======================================================================"
echo ""
echo "Input:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo ""

# 1. 运行量化
echo "[test] Step 1: Quantizing model..."
rm -rf "${OUTPUT_PATH}"
python3 "${SCRIPT_DIR}/quantize_gptq.py" \
    --input "${MODEL_PATH}" \
    --output "${OUTPUT_PATH}" \
    --group-size 128 \
    --bits 4

# 2. 检查输出文件
echo ""
echo "[test] Step 2: Checking output files..."

required_files=(
    "config.json"
    "quantize_config.json"
    "model.safetensors.index.json"
)

for file in "${required_files[@]}"; do
    if [ -f "${OUTPUT_PATH}/${file}" ]; then
        echo "  ✓ ${file}"
    else
        echo "  ✗ ${file} MISSING"
        exit 1
    fi
done

# 3. 显示配置
echo ""
echo "[test] Step 3: Quantization config:"
cat "${OUTPUT_PATH}/quantize_config.json"

# 4. 显示大小
echo ""
echo "[test] Step 4: Model size:"
du -sh "${OUTPUT_PATH}"
echo ""
ls -lh "${OUTPUT_PATH}"/*.safetensors | awk '{print "  " $5, $9}'

# 5. 对比原始模型
echo ""
echo "[test] Step 5: Size comparison:"
original_size=$(du -sb "${MODEL_PATH}" | cut -f1)
quantized_size=$(du -sb "${OUTPUT_PATH}" | cut -f1)
compression_ratio=$(echo "scale=2; ${original_size} / ${quantized_size}" | bc)

echo "  Original:  $(du -sh "${MODEL_PATH}" | cut -f1)"
echo "  Quantized: $(du -sh "${OUTPUT_PATH}" | cut -f1)"
echo "  Ratio:     ${compression_ratio}x"

echo ""
echo "======================================================================"
echo "测试完成！"
echo "======================================================================"
echo ""
echo "量化模型已保存到: ${OUTPUT_PATH}"
echo ""
echo "启动命令:"
echo "  python -m sglang.launch_server \\"
echo "    --model ${OUTPUT_PATH} \\"
echo "    --quantization gptq_marlin \\"
echo "    --kv-cache-dtype fp8_e5m2 \\"
echo "    --dtype float16"
