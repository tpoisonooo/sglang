import time
import requests
import concurrent.futures
from sglang.utils import print_highlight

port = 31111
url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}

# 配置
N = 16  # 并发数
TOTAL_REQUESTS = 400  # 总请求数


"""
开始测试: 并发数=16, 总请求数=400

==================================================
测试结果 (基于 Server 返回数据)
==================================================
并发数: 16
总请求数: 400
成功请求数: 400
失败请求数: 0
客户端总耗时: 48.57s
吞吐量: 8.24 req/s
--------------------------------------------------
Server E2E Latency (服务端端到端延迟):
  平均: 1.910s
  最小: 0.058s
  最大: 2.322s
--------------------------------------------------
Token 统计:
  总 Prompt Tokens: 3200
  总 Completion Tokens: 46510
  总 Tokens: 49710
--------------------------------------------------
生成速度:
  平均 Tokens/s (per request): 60.48
  整体 Throughput (tokens/s): 957.57
==================================================
"""

def send_request(i):
    """发送单个请求"""
    try:
        response = requests.post(url, json=data, timeout=60)
        resp_json = response.json()
        meta_info = resp_json.get("meta_info", {})
        return {
            "idx": i,
            "success": True,
            "e2e_latency": meta_info.get("e2e_latency"),
            "response_sent_to_client_ts": meta_info.get("response_sent_to_client_ts"),
            "completion_tokens": meta_info.get("completion_tokens"),
            "prompt_tokens": meta_info.get("prompt_tokens"),
        }
    except Exception as e:
        return {"idx": i, "success": False, "error": str(e)}


def main():
    print_highlight(f"开始测试: 并发数={N}, 总请求数={TOTAL_REQUESTS}")
    
    start_time = time.time()
    
    # 使用线程池并发发送请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
        futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    fail_count = TOTAL_REQUESTS - success_count
    success_results = [r for r in results if r["success"]]
    
    # 提取 server 返回的指标
    e2e_latencies = [r["e2e_latency"] for r in success_results if r["e2e_latency"] is not None]
    completion_tokens_list = [r["completion_tokens"] for r in success_results if r["completion_tokens"] is not None]
    prompt_tokens_list = [r["prompt_tokens"] for r in success_results if r["prompt_tokens"] is not None]
    
    # 计算吞吐和延迟（基于 server 返回的 e2e_latency）
    throughput = success_count / total_time if total_time > 0 else 0
    
    # Server 端统计的 e2e_latency
    avg_e2e_latency = sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0
    min_e2e_latency = min(e2e_latencies) if e2e_latencies else 0
    max_e2e_latency = max(e2e_latencies) if e2e_latencies else 0
    
    # Token 统计
    total_completion_tokens = sum(completion_tokens_list) if completion_tokens_list else 0
    total_prompt_tokens = sum(prompt_tokens_list) if prompt_tokens_list else 0
    total_tokens = total_completion_tokens + total_prompt_tokens
    
    # Tokens per second（基于 server e2e_latency）
    tokens_per_sec_list = []
    for r in success_results:
        comp_tokens = r.get("completion_tokens")
        e2e_lat = r.get("e2e_latency")
        if comp_tokens is not None and e2e_lat is not None and e2e_lat > 0:
            tokens_per_sec_list.append(comp_tokens / e2e_lat)
    
    avg_tokens_per_sec = sum(tokens_per_sec_list) / len(tokens_per_sec_list) if tokens_per_sec_list else 0
    
    print_highlight("\n" + "=" * 50)
    print_highlight("测试结果 (基于 Server 返回数据)")
    print_highlight("=" * 50)
    print_highlight(f"并发数: {N}")
    print_highlight(f"总请求数: {TOTAL_REQUESTS}")
    print_highlight(f"成功请求数: {success_count}")
    print_highlight(f"失败请求数: {fail_count}")
    print_highlight(f"客户端总耗时: {total_time:.2f}s")
    print_highlight(f"吞吐量: {throughput:.2f} req/s")
    print_highlight("-" * 50)
    print_highlight("Server E2E Latency (服务端端到端延迟):")
    print_highlight(f"  平均: {avg_e2e_latency:.3f}s")
    print_highlight(f"  最小: {min_e2e_latency:.3f}s")
    print_highlight(f"  最大: {max_e2e_latency:.3f}s")
    print_highlight("-" * 50)
    print_highlight("Token 统计:")
    print_highlight(f"  总 Prompt Tokens: {total_prompt_tokens}")
    print_highlight(f"  总 Completion Tokens: {total_completion_tokens}")
    print_highlight(f"  总 Tokens: {total_tokens}")
    print_highlight("-" * 50)
    print_highlight("生成速度:")
    print_highlight(f"  平均 Tokens/s (per request): {avg_tokens_per_sec:.2f}")
    if total_time > 0:
        overall_tokens_per_sec = total_completion_tokens / total_time
        print_highlight(f"  整体 Throughput (tokens/s): {overall_tokens_per_sec:.2f}")
    print_highlight("=" * 50)


if __name__ == "__main__":
    main()
