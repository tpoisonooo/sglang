python3 -m sglang.launch_server --model /data/share/MiniCPM-SALA  --host "0.0.0.0"    --trust-remote-code     --disable-radix-cache     --attention-backend minicpm_flashinfer     --chunked-prefill-size 8192     --max-running-requests 32    --port 31111     --dense-as-sparse

python -m sglang.launch_server --model /data/share/MiniCPM-SALA  --port 31111

python -m sglang.bench_one_batch --model-path /data/share/MiniCPM-SALA  --batch 32 --input-len 256 --output-len 32

# 先进 docker 再启动
docker run --rm -it   --gpus "device=4"  \
    --privileged --runtime=nvidia -p 30000:30000 \
    -v /data/share/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
    --entrypoint=/bin/bash \
    modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest

python3 -m sglang.launch_server \
    --model /models/MiniCPM-SALA \
    --host "0.0.0.0"  \
    --trust-remote-code \
    --port 30000 \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 --skip-server-warmup --dense-as-sparse \
    --max-running-requests 32 \
    --tp-size 1

# 参考docker启动命令
docker stop sglang-server
docker rm sglang-server
docker run --name sglang-server \
  --runtime=nvidia \
  --privileged \
  --gpus 'device=4' \
  -p 30000:30000 \
  -v /data/share/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
  modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest 


# 测速度
export SPEED_DATA_S1=/home/khj/workspace/sglang/speech.jsonl
export SPEED_DATA_S8=/home/khj/workspace/sglang/speech.jsonl
export SPEED_DATA_SMAX=/home/khj/workspace/sglang/speech.jsonl
bash ../SOAR-Toolkit/bench_serving.sh  http://127.0.0.1:30000

# 测精度
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_name /models/MiniCPM-SALA \
  --data_path /data/khj/workspace/sglang/perf_public_set.jsonl \
  --concurrency 32
