docker stop sglang-server
docker rm sglang-server
docker run --rm -it --gpus "device=0"  \
    --privileged --runtime=nvidia -p 30000:30000 \
    -v /data/share/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
    -v /home/khj/workspace/sglang:/source/sglang \
    -v /data/share/MiniCPM-SALA-int4:/models/MiniCPM-SALA-int4:ro \
    --entrypoint=/bin/bash \
    modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest

python3 -m sglang.launch_server \
    --model /models/MiniCPM-SALA-int4 \
    --host "0.0.0.0"  \
    --trust-remote-code \
    --port 30000 \
    --disable-radix-cache \
    --attention-backend flashinfer \
    --chunked-prefill-size 32768 --skip-server-warmup --dense-as-sparse \
    --max-running-requests 32 \
    --tp-size 1 \
    --kv-cache-dtype fp8_e4m3