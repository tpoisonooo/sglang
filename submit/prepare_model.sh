#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
INPUT_PATH=""
OUTPUT_PATH=""
USE_LOCAL=false  # 默认从 HuggingFace 下载
LOCAL_SOURCE_DIR="/root/models/openbmb/dual"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_PATH="$2"  # 接受但不使用
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --local)
            USE_LOCAL=true
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --output <path> [--input <path>] [--local]"
            exit 1
            ;;
    esac
done

if [[ -z "${OUTPUT_PATH}" ]]; then
    echo "Error: --output is required"
    exit 1
fi

echo "[prepare_model] Output: ${OUTPUT_PATH}"
echo "[prepare_model] Use local: ${USE_LOCAL}"

export HF_ENDPOINT=https://hf-mirror.com

# HuggingFace 仓库配置
# TODO: 上传模型到 HuggingFace 后修改以下配置
REMOTE_REPO="tpoisonooo/dual0324"  # 需要上传到的仓库名

# 期望的 MD5 校验值
# FP4 模型
EXPECTED_FP4_MD5_1="683439315237988fcb2378816a240ff5"
EXPECTED_FP4_MD5_2="d18353502728cca3c31143c2cebc2493"
# INT4 模型  
EXPECTED_INT4_MD5_1="c873d39fc585131c02e42b4b9f3048f7"
EXPECTED_INT4_MD5_2="e73d7784bc5f8f37223a18a791575942"

# 下载函数，带重试机制
download_with_retry() {
    local max_retries=10
    local retry_count=0
    local local_dir="$1"
    local repo="$2"
    
    while [[ $retry_count -lt $max_retries ]]; do
        echo "[prepare_model] Download attempt $((retry_count + 1))/$max_retries from ${repo}..."
        
        # 清理之前的下载（如果存在）
        if [[ -d "$local_dir" ]]; then
            rm -rf "$local_dir"
        fi
        
        # 下载整个仓库（包含 fp4 和 int4 子目录）
        if huggingface-cli download "${repo}" --local-dir "$local_dir" --local-dir-use-symlinks False; then
            echo "[prepare_model] Download completed successfully."
            return 0
        else
            echo "[prepare_model] Download failed."
            retry_count=$((retry_count + 1))
            if [[ $retry_count -lt $max_retries ]]; then
                echo "[prepare_model] Retrying in 30 seconds..."
                sleep 30
            fi
        fi
    done
    
    echo "[prepare_model] Download failed after $max_retries attempts."
    return 1
}

# 本地复制函数
copy_model_files() {
    local local_dir="$1"
    local source_dir="$2"
    
    echo "[prepare_model] Copying model files from ${source_dir}..."
    
    # 检查源目录是否存在
    if [[ ! -d "$source_dir" ]]; then
        echo "[prepare_model] Error: Source directory not found: $source_dir"
        return 1
    fi
    
    # 检查是否包含 int4 子目录 (fp4 文件直接在 source_dir 中)
    if [[ ! -d "$source_dir/int4" ]]; then
        echo "[prepare_model] Error: Source directory must contain 'int4' subdirectory (fp4 is in root)"
        return 1
    fi
    
    # 清理目标目录（如果存在）
    if [[ -d "$local_dir" ]]; then
        rm -rf "$local_dir"
    fi
    
    # 创建目标目录并复制
    mkdir -p "$local_dir"
    
    # 复制 fp4 文件 (直接在 source_dir 中) 到目标目录根
    local copy_ok=true
    echo "[prepare_model] Copying fp4 files from ${source_dir}..."
    if ! cp -r "${source_dir}"/* "$local_dir/" 2>/dev/null; then
        echo "[prepare_model] Error: Failed to copy fp4 files"
        copy_ok=false
    fi
    
    # 复制 int4 子目录
    if ! cp -r "${source_dir}/int4" "$local_dir/"; then
        echo "[prepare_model] Error: Failed to copy int4 directory"
        copy_ok=false
    fi
    
    if [[ "$copy_ok" == true ]]; then
        echo "[prepare_model] Copy completed successfully."
        return 0
    else
        return 1
    fi
}

# MD5 校验函数
verify_md5() {
    local file="$1"
    local expected_md5="$2"
    
    if [[ ! -f "$file" ]]; then
        echo "[prepare_model] Error: File not found: $file"
        return 1
    fi
    
    local actual_md5
    actual_md5=$(md5sum "$file" | awk '{print $1}')
    
    echo "[prepare_model] MD5 check for $file:"
    echo "  Expected: $expected_md5"
    echo "  Actual:   $actual_md5"
    
    if [[ "$actual_md5" == "$expected_md5" ]]; then
        echo "  Result: PASS"
        return 0
    else
        echo "  Result: FAIL"
        return 1
    fi
}

# 检查模型文件完整性
check_model_files() {
    local base_dir="$1"
    local model_type="$2"  # fp4 或 int4
    local expected_md5_1="$3"
    local expected_md5_2="$4"
    
    # fp4 模型文件直接在 base_dir 中，int4 在子目录中
    if [[ "$model_type" == "fp4" ]]; then
        local model_dir="${base_dir}"
    else
        local model_dir="${base_dir}/${model_type}"
    fi
    local model_file_1="${model_dir}/model-00001-of-00002.safetensors"
    local model_file_2="${model_dir}/model-00002-of-00002.safetensors"
    
    # 检查文件是否存在
    if [[ ! -f "$model_file_1" ]] || [[ ! -f "$model_file_2" ]]; then
        echo "[prepare_model] ${model_type} model files not found."
        return 1
    fi
    
    echo "[prepare_model] Verifying ${model_type} model..."
    
    # MD5 校验
    local md5_ok=true
    if ! verify_md5 "$model_file_1" "$expected_md5_1"; then
        md5_ok=false
    fi
    
    if ! verify_md5 "$model_file_2" "$expected_md5_2"; then
        md5_ok=false
    fi
    
    if [[ "$md5_ok" == true ]]; then
        echo "[prepare_model] ${model_type} model is valid."
        return 0
    else
        echo "[prepare_model] ${model_type} model is corrupted."
        return 1
    fi
}

# 检查目标目录是否已存在且文件完整
check_existing_model() {
    local output_dir="$1"
    
    if [[ ! -d "$output_dir" ]]; then
        return 1
    fi
    
    echo "[prepare_model] Found existing model directory, verifying..."
    
    # 检查 int4 子目录 (fp4 文件直接在 output_dir 中)
    if [[ ! -d "$output_dir/int4" ]]; then
        echo "[prepare_model] Missing int4 subdirectory."
        return 1
    fi
    
    # 校验两个模型
    local fp4_ok=false
    local int4_ok=false
    
    if check_model_files "$output_dir" "fp4" "$EXPECTED_FP4_MD5_1" "$EXPECTED_FP4_MD5_2"; then
        fp4_ok=true
    fi
    
    if check_model_files "$output_dir" "int4" "$EXPECTED_INT4_MD5_1" "$EXPECTED_INT4_MD5_2"; then
        int4_ok=true
    fi
    
    if [[ "$fp4_ok" == true ]] && [[ "$int4_ok" == true ]]; then
        echo "[prepare_model] Existing model is valid, skipping copy/download."
        return 0
    else
        echo "[prepare_model] Existing model is corrupted or incomplete."
        return 1
    fi
}

# 主流程
# 1. 检查目标目录是否已存在且有效
if check_existing_model "${OUTPUT_PATH}"; then
    echo "[prepare_model] Model already exists at ${OUTPUT_PATH}"
    echo "[prepare_model] Done!"
    exit 0
fi

# 2. 获取模型文件
if [[ "${USE_LOCAL}" == true ]]; then
    # 使用本地模型
    if ! copy_model_files "$OUTPUT_PATH" "$LOCAL_SOURCE_DIR"; then
        echo "[prepare_model] Error: Failed to copy model files from local directory."
        exit 1
    fi
else
    # 从 HuggingFace 下载
    if ! download_with_retry "$OUTPUT_PATH" "$REMOTE_REPO"; then
        echo "[prepare_model] Error: Failed to download model from HuggingFace."
        echo "[prepare_model] Please ensure the model is uploaded to: ${REMOTE_REPO}"
        exit 1
    fi
fi

# 3. 校验两个模型
echo "[prepare_model] Verifying model files..."

ALL_OK=true

if ! check_model_files "$OUTPUT_PATH" "fp4" "$EXPECTED_FP4_MD5_1" "$EXPECTED_FP4_MD5_2"; then
    ALL_OK=false
fi

if ! check_model_files "$OUTPUT_PATH" "int4" "$EXPECTED_INT4_MD5_1" "$EXPECTED_INT4_MD5_2"; then
    ALL_OK=false
fi

if [[ "$ALL_OK" != true ]]; then
    echo "[prepare_model] Error: MD5 verification failed. Files may be corrupted."
    exit 1
fi

echo "[prepare_model] All MD5 checks passed."
echo "[prepare_model] Model structure:"
echo "  ${OUTPUT_PATH}/fp4/"
echo "  ${OUTPUT_PATH}/int4/"
echo "[prepare_model] Done!"
