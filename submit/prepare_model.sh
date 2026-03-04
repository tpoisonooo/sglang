#!/usr/bin/env bash
set -euo pipefail

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
echo "[prepare_model] Using optimized config: group_size=128, bits=4, sym=True"

# Run quantization with optimized settings
python3 "${SCRIPT_DIR}/quantize_gptq.py" \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_PATH}" \
    --group-size 128 \
    --bits 4

echo "[prepare_model] Done!"
