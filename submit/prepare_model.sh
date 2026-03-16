#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
INPUT_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${INPUT_PATH}" ]] || [[ -z "${OUTPUT_PATH}" ]]; then
    echo "Error: --input and --output are required"
    exit 1
fi

echo "[prepare_model] Input: ${INPUT_PATH}"
echo "[prepare_model] Output: ${OUTPUT_PATH}"

export HF_ENDPOINT=https://hf-mirror.com

# 期望的 MD5 校验值
EXPECTED_MD5_1="5820dd4be58042e10697defe42719d4c"
EXPECTED_MD5_2="66265791d93952a94cbef09c8928e194"

# 下载函数，带重试机制
download_with_retry() {
    local max_retries=10
    local retry_count=0
    local local_dir="$1"
    
    while [[ $retry_count -lt $max_retries ]]; do
        echo "[prepare_model] Download attempt $((retry_count + 1))/$max_retries..."
        
        # 清理之前的下载（如果存在）
        if [[ -d "$local_dir" ]]; then
            rm -rf "$local_dir"
        fi
        
        if huggingface-cli download tpoisonooo/test03152 --local-dir "$local_dir"; then
            echo "[prepare_model] Download completed successfully."
            return 0
        else
            echo "[prepare_model] Download failed."
            retry_count=$((retry_count + 1))
            if [[ $retry_count -lt $max_retries ]]; then
                echo "[prepare_model] Retrying in 5 seconds..."
                sleep 30
            fi
        fi
    done
    
    echo "[prepare_model] Download failed after $max_retries attempts."
    return 1
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

# 检查目标目录是否已存在且文件完整
check_existing_model() {
    local output_dir="$1"
    
    if [[ ! -d "$output_dir" ]]; then
        return 1
    fi
    
    local model_file_1="$output_dir/model-00001-of-00002.safetensors"
    local model_file_2="$output_dir/model-00002-of-00002.safetensors"
    
    # 检查文件是否存在
    if [[ ! -f "$model_file_1" ]] || [[ ! -f "$model_file_2" ]]; then
        echo "[prepare_model] Model files not found in existing directory."
        return 1
    fi
    
    echo "[prepare_model] Found existing model directory, verifying MD5..."
    
    # MD5 校验
    local md5_ok=true
    if ! verify_md5 "$model_file_1" "$EXPECTED_MD5_1"; then
        md5_ok=false
    fi
    
    if ! verify_md5 "$model_file_2" "$EXPECTED_MD5_2"; then
        md5_ok=false
    fi
    
    if [[ "$md5_ok" == true ]]; then
        echo "[prepare_model] Existing model is valid, skipping download."
        return 0
    else
        echo "[prepare_model] Existing model is corrupted or incomplete, will re-download."
        return 1
    fi
}


# 1. 检查目标目录是否已存在且有效
if check_existing_model "${OUTPUT_PATH}"; then
    echo "[prepare_model] Model already exists at ${OUTPUT_PATH}"
    echo "[prepare_model] Done!"
    exit 0
fi

# 2. 下载模型（带重试）
if ! download_with_retry "$OUTPUT_PATH"; then
    echo "[prepare_model] Error: Failed to download model."
    exit 1
fi

# 3. MD5 校验下载的文件
MODEL_FILE_1="$OUTPUT_PATH/model-00001-of-00002.safetensors"
MODEL_FILE_2="$OUTPUT_PATH/model-00002-of-00002.safetensors"

echo "[prepare_model] Verifying downloaded files..."

MD5_OK=true
if ! verify_md5 "$MODEL_FILE_1" "$EXPECTED_MD5_1"; then
    MD5_OK=false
fi

if ! verify_md5 "$MODEL_FILE_2" "$EXPECTED_MD5_2"; then
    MD5_OK=false
fi

if [[ "$MD5_OK" != true ]]; then
    echo "[prepare_model] Error: MD5 verification failed. Downloaded files may be corrupted."
    exit 1
fi

echo "[prepare_model] All MD5 checks passed."
echo "[prepare_model] Done!"
