python3 -m sglang.launch_server     --model /data/share/MiniCPM-SALA  --host "0.0.0.0"    --trust-remote-code     --disable-radix-cache     --attention-backend minicpm_flashinfer     --chunked-prefill-size 8192     --max-running-requests 32     --skip-server-warmup     --port 31111     --dense-as-sparse


python -m sglang.bench_one_batch --model-path /data/share/MiniCPM-SALA  --batch 32 --input-len 256 --output-len 32