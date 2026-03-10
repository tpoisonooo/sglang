#!/usr/bin/env bash

uv pip install --no-deps -e ./sglang/python
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --kv-cache-dtype fp8_e4m3"
export HF_ENDPOINT=https://hf-mirror.com
