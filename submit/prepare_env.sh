#!/usr/bin/env bash

uv pip install --no-deps -e ./sglang/python

cd flash-attention/
pip install -e "flash_attn/cute[dev]"
cd -

# Patch FA4 interface to support SM12.x (Blackwell GPUs like RTX 6000D)
# This patch adds support for compute capability 12.x and fixes page_size handling
SGL_KERNEL_PATH=$(python -c "import sgl_kernel; print(sgl_kernel.__file__[:-12])" 2>/dev/null)
if [ -n "$SGL_KERNEL_PATH" ]; then
    if [ -f "./_fa4_interface.py" ]; then
        echo "Patching sgl_kernel/_fa4_interface.py for SM12.x support..."
        cp ./_fa4_interface.py "$SGL_KERNEL_PATH/_fa4_interface.py"
        echo "Patch applied successfully."
    fi
    if [ -f "./flash_attn.py" ]; then
        echo "Patching sgl_kernel/flash_attn.py for FA4 API fix..."
        cp ./flash_attn.py "$SGL_KERNEL_PATH/flash_attn.py"
        echo "Patch applied successfully."
    fi
else
    echo "Warning: Could not find sgl_kernel path. FA4 patches not applied."
fi

export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --attention-backend flashinfer --kv-cache-dtype fp8_e4m3"
export HF_ENDPOINT=https://hf-mirror.com

