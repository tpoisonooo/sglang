# Test Chunk GLA - Blackwell 6000D Optimization

针对 NVIDIA Blackwell RTX 6000D 优化的 `chunk_simple_gla` 实现。

## 硬件目标

- **GPU**: NVIDIA RTX 6000D (Blackwell)
- **Compute Capability**: sm_120 (12.0)
- **SMs**: 156
- **Shared Memory**: 128 KB/SM
- **Memory**: GDDR7, ~1568 GB/s
- **L2 Cache**: 128 MB

## 文件结构

```
test_chunk_gla/
├── __init__.py                      # 包初始化，自动选择最优实现
├── chunk_simple_gla.py              # 原版 FLA 实现（参考）
├── chunk_simple_gla_blackwell.py    # Blackwell 优化版（仅 forward）
├── chunk_simple_gla_fused.py        # 融合 kernel 实验版
├── test_chunk_gla.py                # 测试和 benchmark
├── OPTIMIZATION_NOTES.md            # 详细优化分析
└── README.md                        # 本文件
```

## 已完成的优化

### 1. 移除 Backward（✓）
- 删除了所有反向传播 kernel（`chunk_bwd_*`）
- 减少代码体积，专注于推理优化

### 2. 增大 Tile Size（✓）
```python
# 原版
BK/BV = [32, 64]

# Blackwell 版
BK/BV = [64, 128]  # 利用 128KB shared memory
```

### 3. 优化 Chunk Size（✓）
```python
def get_optimal_chunk_size_blackwell(seq_len):
    if seq_len <= 64:
        return seq_len
    else:
        return 128  # 利用 GDDR7 高带宽
```

### 4. Warp/Stage 配置（✓）
```python
# 针对 sm_120 优化
num_warps = [4, 8]  # 原版: [1, 2, 4, 8]
num_stages = [2, 3, 4]
```

## 使用方法

### 自动选择（推荐）
```python
from test_chunk_gla import chunk_simple_gla

# 自动检测 Blackwell 并使用优化版本
o, ht = chunk_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)
```

### 强制使用 Blackwell 版
```python
from test_chunk_gla import chunk_simple_gla_blackwell

o, ht = chunk_simple_gla_blackwell(q, k, v, g_gamma=g_gamma, scale=scale)
```

### 原始 FLA 版本
```python
from test_chunk_gla import chunk_simple_gla_original

o, ht = chunk_simple_gla_original(q, k, v, g_gamma=g_gamma, scale=scale)
```

## 测试

```bash
cd /root/soar2026
python test_chunk_gla/test_chunk_gla.py
```

## 进一步优化机会

### 高优先级
1. **Kernel Fusion**
   - 将 `chunk_fwd_h` 和 `chunk_fwd_o` 融合为单个 kernel
   - 消除中间 `h` tensor 的 HBM 读写
   - 预计收益: 20-40% 内存带宽节省

2. **更激进的 Tile Size**
   - 尝试 BK=256 或 BV=256 的非对称配置
   - 利用 Blackwell 的大 shared memory

3. **FP4 量化**
   - Blackwell 支持 FP4 tensor core 操作
   - 可减少 50% 内存带宽

### 中优先级
4. **Grid 优化**
   - 针对 156 SMs 优化 grid 大小
   - 避免 tail effect

5. **异步执行**
   - 使用 CUDA graph 减少 launch overhead
   - 对小序列（T≤128）特别有效

6. **TMA (Tensor Memory Accelerator)**
   - Blackwell 支持 TMA
   - 可以更高效地加载 tile 数据

## 性能基准

初步测试结果（RTX 6000D）：

| Seq Len | Blackwell (ms) | Original (ms) | Speedup |
|---------|---------------|---------------|---------|
| 16      | 0.048         | 0.052         | 1.09x   |
| 64      | 0.046         | 0.049         | 1.08x   |
| 256     | 0.046         | 0.050         | 1.10x   |
| 1024    | 0.051         | 0.050         | 0.98x   |
| 4096    | 0.193         | 0.225         | 1.16x   |

**分析**: 
- 小序列有轻微提升（~10%）
- 中等序列性能相当
- 大序列有 16% 提升
- 主要瓶颈可能是 Triton autotune 开销和 kernel launch

## 下一步工作

1. [ ] Profile 分析（Nsight Compute）
2. [ ] 实现真正的 Kernel Fusion
3. [ ] 针对 MiniCPM 特定形状 (H=32, K=V=128) 专门优化
4. [ ] FP4 量化可行性研究

## 参考

- FLA: https://github.com/fla-org/flash-linear-attention
- Triton Docs: https://triton-lang.org/
