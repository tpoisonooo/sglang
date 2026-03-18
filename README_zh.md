## 当前 WIP 优化

1. 原始 fa + scale_add fused 
Average Score: 77.89%
Total Duration: 2705.45 s
Total Tokens: In=8644166, Out=996649
Average Tokens/Sample: In=57627.8, Out=6644.3
Overall TPS (Output): 368.39 tokens/s
Detailed results saved to outputs/20260317_034349/predictions.jsonl

2. fa4 prefill （确认生效）+ fa decode

Average Score: 76.82%
Total Duration: 3051.68 s
Total Tokens: In=8644166, Out=1317022
Average Tokens/Sample: In=57627.8, Out=8780.1
Overall TPS (Output): 431.57 tokens/s
