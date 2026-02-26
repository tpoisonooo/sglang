安装 
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

python3 -m sglang.launch_server --model /data/share/MiniCPM-SALA  --host "0.0.0.0"    --trust-remote-code     --disable-radix-cache     --attention-backend minicpm_flashinfer     --chunked-prefill-size 8192     --max-running-requests 32    --port 31111     --dense-as-sparse

python -m sglang.launch_server --model /data/share/MiniCPM-SALA  --port 31111

python -m sglang.bench_one_batch --model-path /data/share/MiniCPM-SALA  \
    --batch 8 --input-len 256 --output-len 32 \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse \
    --tp-size 1

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

python3 -m sglang.launch_server \
    --model /data/share/MiniCPM-SALA \
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


# 测试代码
python -m sglang.bench_one_batch --model-path /data/share/MiniCPM-SALA   \
   --batch 4 --input-len 1024 --output-len 512  \
   --trust-remote-code     --disable-radix-cache   \
   --attention-backend minicpm_flashinfer  \
   --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse  \
   --tp-size 1

[2026-02-26 10:25:20 TP0] Reset HybridReqToTokenPool
Prefill. latency: 0.46342 s, throughput:   9316.59 token/s
Decode 0. Batch size: 4, latency: 0.29446 s, throughput:     13.58 token/s
Decode 1. Batch size: 4, latency: 0.01225 s, throughput:    326.46 token/s
Decode 2. Batch size: 4, latency: 0.01189 s, throughput:    336.47 token/s
Decode 3. Batch size: 4, latency: 0.01190 s, throughput:    336.16 token/s
Decode 4. Batch size: 4, latency: 0.01187 s, throughput:    337.04 token/s
Decode.  median latency: 0.01178 s, median throughput:    339.57 token/s
Total. latency:  1.112 s, throughput:   3981.76 token/s
Benchmark ...
[2026-02-26 10:25:21 TP0] Reset HybridReqToTokenPool
Prefill. latency: 0.37183 s, throughput:  11015.86 token/s
Decode 0. Batch size: 4, latency: 0.01211 s, throughput:    330.38 token/s
Decode 1. Batch size: 4, latency: 0.01191 s, throughput:    335.75 token/s
Decode 2. Batch size: 4, latency: 0.01187 s, throughput:    337.05 token/s
Decode 3. Batch size: 4, latency: 0.01187 s, throughput:    337.09 token/s
Decode 4. Batch size: 4, latency: 0.01184 s, throughput:    337.83 token/s
Decode.  median latency: 0.01175 s, median throughput:    340.47 token/s
Total. latency:  6.395 s, throughput:    960.75 token/s
