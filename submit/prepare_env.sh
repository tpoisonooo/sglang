#!/usr/bin/env bash
set -euo pipefail

echo "[prepare_env] start $(date '+%F %T')"

# ---- patch minicpm_backend hardcoded bfloat16 -> float16 for GPTQ compat ----
BACKEND_DIR=/opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python/sglang/srt/layers/attention

for pyfile in "$BACKEND_DIR"/minicpm_backend.py "$BACKEND_DIR"/minicpm_sparse_utils.py; do
    if [ -f "$pyfile" ]; then
        sed -i 's/torch\.bfloat16/torch.float16/g'  "$pyfile"
        sed -i 's/"bfloat16"/"float16"/g'            "$pyfile"
        echo "[prepare_env] patched $(basename "$pyfile")"
    fi
done
find "$BACKEND_DIR/__pycache__" -name "minicpm_backend*" -o -name "minicpm_sparse_utils*" 2>/dev/null | xargs rm -f

# ---- override SGLang launch args for GPTQ Marlin + FP8 KV Cache ----
# 关键优化：
# 1. --quantization gptq_marlin : 使用 Marlin Kernel 加速
# 2. --kv-cache-dtype fp8_e5m2 : FP8 KV Cache 减少显存带宽
# 3. --dtype float16 : 保持 FP16 激活
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --kv-cache-dtype fp8_e5m2 --dtype float16 --disable-cuda-graph"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
