# Blackwell RTX 6000D chunk_simple_gla 优化分析

## 硬件规格

| 特性 | RTX 6000D |
|------|-----------|
| Compute Capability | sm_120 (12.0) |
| SMs | 156 |
| Shared Memory/SM | 128 KB |
| Memory | GDDR7, ~1568 GB/s |
| L2 Cache | 128 MB |
| Tensor Cores | 5th Gen (FP4 support) |

## 当前实现分析

### FLA 原版的限制

1. **保守的 Tile Size**
   - BK/BV 最大只有 64（为了兼容旧 GPU）
   - Blackwell 有 128KB shared memory，可以支持 BK=Bv=128

2. **通用 Autotune 配置**
   - num_warps: 1-8（没有针对 sm_120 优化）
   - num_stages: 2-4（可能需要调整）

3. **Chunk Size 选择**
   - 原版: `min(64, max(16, next_power_of_2(T)))`
   - 对于 GDDR7 高带宽，可以使用更大的 chunk（128）

4. **包含 Backward 代码**
   - 原版包含完整的反向传播 kernel
   - 推理时不需要，增加了代码复杂度和内存占用

## 已应用的优化（当前版本）

### 1. 移除 Backward（✓ 完成）
- 删除了 `chunk_bwd_kernel_dh`、`chunk_bwd_kernel_dqkwg`、`chunk_bwd_kernel_dv`
- 减少了代码体积和编译时间

### 2. 增大 Tile Size（✓ 完成）
```python
BLACKWELL_BK_LIST = [64, 128]  # 原版: [32, 64]
BLACKWELL_BV_LIST = [64, 128]
```

### 3. 优化 Chunk Size 选择（✓ 完成）
```python
def get_optimal_chunk_size_blackwell(seq_len):
    if seq_len <= 64:
        return seq_len  # 小序列保持原样
    else:
        return 128  # 大序列使用 128 提高计算效率
```

### 4. Warp/Stage 配置（✓ 完成）
```python
BLACKWELL_NUM_WARPS = [4, 8]  # 聚焦高 occupancy
BLACKWELL_NUM_STAGES = [2, 3, 4]
```

## 潜在的进一步优化

### 1. FP4 量化支持（待研究）
Blackwell 的 5th Gen Tensor Cores 支持 FP4，可以显著减少内存带宽：
```python
# 未来可能的优化
if use_fp4:
    q_fp4 = quantize_to_fp4(q)  # 2-bit per element
    k_fp4 = quantize_to_fp4(k)
    # 使用 FP4 tensor core MMA
```

### 2. 更激进的 Tile Size
尝试 BK=256, BV=64 或 BK=64, BV=256 的非对称配置：
```python
# 利用 128KB shared memory 的极限
# K=V=128 时: 128 * 128 * 4 bytes = 64KB per tile
# 可以同时加载更多 tile 或增加 register 使用
```

### 3. 软件流水线优化
```python
@triton.jit
def kernel_with_software_pipeline(...):
    # 使用 tl.range 的 pipeline 参数
    for i_t in tl.range(0, NT, pipeline=num_stages):
        # 自动插入 cp.async 指令
```

### 4. 针对 156 SMs 的 Grid 优化
```python
# 计算最优 grid 大小以充分利用 156 SMs
def optimal_grid(B, H, K, V, BK, BV):
    total_tiles = (K // BK) * (V // BV) * B * H
    # 确保 grid 大小是 156 的倍数或因子，避免 tail effect
    return grid
```

### 5. L2 Cache 优化
Blackwell 有 128MB L2 cache，可以缓存更多数据：
```python
# 对于 B=1, H=32, K=V=128, T=4096
# h tensor: [1, 32, 128, 128, 32] = ~67MB (fp32)
# 可以完全放入 L2 cache，减少 HBM 访问
```

## 性能分析

### 当前瓶颈

从初步 benchmark 来看：
1. **短序列 (T≤512)**: 计算瓶颈，优化效果不明显
2. **中等序列 (T=1024)**: 与原版性能相当
3. **长序列 (T≥2048)**: 可能有内存带宽瓶颈

### 进一步优化建议

1. **减少 Autotune 开销**
   - 当前 autotune configs 太多，首次运行慢
   - 可以针对 Blackwell 预选出最优 2-3 个配置

2. **Kernel Fusion**
   - 将 `chunk_fwd_h` 和 `chunk_fwd_o` 融合为单个 kernel
   - 减少中间 tensor `h` 的 HBM 读写

3. **异步执行**
   - 利用 CUDA graph 减少 launch overhead
   - 对于小序列（T≤128）尤其重要

## 实现文件说明

| 文件 | 说明 |
|------|------|
| `chunk_simple_gla.py` | 原版 FLA 实现（完整参考） |
| `chunk_simple_gla_blackwell.py` | Blackwell 优化版（仅 forward） |
| `test_chunk_gla.py` | 测试和 benchmark |
| `__init__.py` | 自动选择最优实现 |

## 使用方法

```python
from test_chunk_gla import chunk_simple_gla

# 自动检测硬件并选择最优实现
o, ht = chunk_simple_gla(q, k, v, g_gamma=g_gamma, scale=scale)

# 强制使用 Blackwell 优化版
from test_chunk_gla import chunk_simple_gla_blackwell
o, ht = chunk_simple_gla_blackwell(q, k, v, g_gamma=g_gamma, scale=scale)
```

## 下一步工作

1. [ ] Profile 详细分析瓶颈（使用 Nsight Compute）
2. [ ] 尝试 Kernel Fusion（h + o 合并）
3. [ ] 研究 FP4 量化的可行性
4. [ ] 针对 MiniCPM 特定形状 (H=32, K=V=128) 专门优化
