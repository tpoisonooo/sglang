#!/usr/bin/env bash
# =============================================================================
# 打包脚本 - 将提交物打包为 .tar.gz 文件
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="sala-gptq-marlin"
VERSION=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${PROJECT_NAME}_${VERSION}.tar.gz"

echo "[package] Packaging submission files..."
echo "[package] Working directory: ${SCRIPT_DIR}"
echo "[package] Output file: ${OUTPUT_FILE}"

# 进入 submit 目录
cd "${SCRIPT_DIR}"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

# 复制必要文件
echo "[package] Copying files..."
mkdir -p "${TEMP_DIR}/${PROJECT_NAME}"

cp prepare_env.sh "${TEMP_DIR}/${PROJECT_NAME}/"
cp prepare_model.sh "${TEMP_DIR}/${PROJECT_NAME}/"
cp preprocess_model.py "${TEMP_DIR}/${PROJECT_NAME}/"
cp quantize_model.py "${TEMP_DIR}/${PROJECT_NAME}/"
cp verify_accuracy.py "${TEMP_DIR}/${PROJECT_NAME}/"
cp README.md "${TEMP_DIR}/${PROJECT_NAME}/"

# 复制 sglang 目录（如果有自定义代码）
if [ -d "sglang" ]; then
    echo "[package] Copying sglang directory..."
    cp -r sglang "${TEMP_DIR}/${PROJECT_NAME}/"
fi

# 创建打包
echo "[package] Creating tar.gz archive..."
tar -czf "${OUTPUT_FILE}" -C "${TEMP_DIR}" "${PROJECT_NAME}"

# 显示结果
echo "[package] Done!"
echo "[package] Output: ${SCRIPT_DIR}/${OUTPUT_FILE}"
echo "[package] Size: $(du -h "${OUTPUT_FILE}" | cut -f1)"
echo ""
echo "[package] Package contents:"
tar -tzf "${OUTPUT_FILE}" | head -20

# 验证
echo ""
echo "[package] Verifying package..."
tar -tzf "${OUTPUT_FILE}" | grep -q "prepare_env.sh" && echo "  ✓ prepare_env.sh found"
tar -tzf "${OUTPUT_FILE}" | grep -q "prepare_model.sh" && echo "  ✓ prepare_model.sh found"
tar -tzf "${OUTPUT_FILE}" | grep -q "quantize_model.py" && echo "  ✓ quantize_model.py found"

echo ""
echo "[package] Package ready for submission: ${OUTPUT_FILE}"
