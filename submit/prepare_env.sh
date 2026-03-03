#!/usr/bin/env bash

echo "[prepare_env] start $(date '+%F %T')"

uv pip install --no-deps -e ./sglang/python

export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"

echo "[prepare_env] done"
