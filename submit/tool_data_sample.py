#!/usr/bin/env python3
"""
从 JSONL 文件中按 source 采样数据。
输入文件可能非常大，且没有换行符分隔。
输出每个 source 100 条，只保留 index, question, source 字段。
"""

import json
import sys
from collections import defaultdict
import random


def extract_question(messages):
    """从 messages 数组中提取用户的问题"""
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')
    return ''


def stream_json_objects(filepath):
    """
    流式读取文件，逐个解析 JSON 对象。
    处理没有换行符的大文件。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    i = 0
    obj_index = 0
    while i < len(content):
        if content[i] == '{':
            brace_count = 0
            in_string = False
            escape = False
            start = i
            
            for j in range(i, len(content)):
                c = content[j]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                obj = json.loads(content[start:j+1])
                                if 'source' in obj:
                                    yield obj_index, obj
                                    obj_index += 1
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
            else:
                break
        i += 1


def sample_by_source(input_path, output_path, samples_per_source=100):
    """
    按 source 采样数据。
    使用 reservoir sampling 来处理大数据集。
    """
    # 每个 source 的 reservoir
    reservoirs = defaultdict(list)
    # 记录每个 source 看到的总数量
    source_counts = defaultdict(int)
    
    print(f"Processing {input_path}...")
    
    for idx, obj in stream_json_objects(input_path):
        source = obj.get('source')
        if not source:
            continue
        
        source_counts[source] += 1
        count = source_counts[source]
        
        # Reservoir sampling
        if len(reservoirs[source]) < samples_per_source:
            reservoirs[source].append(obj)
        else:
            # 以 samples_per_source/count 的概率替换
            j = random.randint(0, count - 1)
            if j < samples_per_source:
                reservoirs[source][j] = obj
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} objects, sources: {dict(source_counts)}")
    
    print(f"\nTotal objects processed: {sum(source_counts.values())}")
    print(f"Source distribution: {dict(source_counts)}")
    
    # 写入输出文件
    print(f"\nWriting samples to {output_path}...")
    
    total_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for source in sorted(reservoirs.keys()):
            samples = reservoirs[source]
            print(f"Source '{source}': {len(samples)} samples (from {source_counts[source]} total)")
            
            for obj in samples:
                question = extract_question(obj.get('messages', []))
                output_obj = {
                    'index': total_written,
                    'question': question,
                    'source': source
                }
                f.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                total_written += 1
    
    print(f"\nTotal samples written: {total_written}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python sample.py <input_jsonl> <output_jsonl> [samples_per_source]")
        print("Example: python sample.py ep00.jsonl sampled.jsonl 100")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    samples_per_source = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    sample_by_source(input_path, output_path, samples_per_source)
