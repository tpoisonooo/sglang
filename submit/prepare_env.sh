#!/usr/bin/env bash

uv pip install --no-deps -e ./sglang/python
uv pip install nvidia-modelopt
uv pip install flashinfer_python==0.6.6
uv pip install flashinfer_cubin==0.6.6

cd flash-attention/
pip install -e "flash_attn/cute[dev]"
cd -

# Copy SM120 kernel files to flash_attn_origin/cute/ for SM120 (Blackwell) support
FLASH_ATTN_ORIGIN_CUTE="./sglang_minicpm_sala_env/lib/python3.12/site-packages/flash_attn_origin/cute"
if [ -d "./flash-attention/flash_attn/cute" ]; then
    mkdir -p "$FLASH_ATTN_ORIGIN_CUTE"
    echo "Copying SM120 kernel files to $FLASH_ATTN_ORIGIN_CUTE/..."
    cp ./flash-attention/flash_attn/cute/flash_fwd_sm120.py "$FLASH_ATTN_ORIGIN_CUTE/" && echo "  - flash_fwd_sm120.py copied" || echo "  - flash_fwd_sm120.py copy failed"
    cp ./flash-attention/flash_attn/cute/flash_bwd_sm120.py "$FLASH_ATTN_ORIGIN_CUTE/" && echo "  - flash_bwd_sm120.py copied" || echo "  - flash_bwd_sm120.py copy failed"
fi

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

# Patch infllm_v2 for FP8 support
INFLLM_PATH=$(python -c "import infllm_v2; print(infllm_v2.__file__[:-12])" 2>/dev/null)
if [ -n "$INFLLM_PATH" ]; then
    if [ -f "./infllmv2_sparse_attention.py" ]; then
        echo "Patching infllm_v2/infllmv2_sparse_attention.py for FP8 support..."
        cp ./infllmv2_sparse_attention.py "$INFLLM_PATH/infllmv2_sparse_attention.py"
        echo "Patch applied successfully."
    fi
else
    echo "Warning: Could not find infllm_v2 path. infllm_v2 patches not applied."
fi

export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --attention-backend flashinfer"
export HF_ENDPOINT=https://hf-mirror.com


