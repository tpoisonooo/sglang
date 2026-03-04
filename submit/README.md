# MiniCPM-SALA GPTQ W4A16 量化提交方案

本目录包含完整的 GPTQ W4A16 + Marlin Kernel + FP8 KV Cache 量化加速方案提交物。

## 方案概述

| 组件 | 配置 |
|------|------|
| 权重量化 | GPTQ W4A16 (4-bit权重, 16-bit激活) |
| Linear加速 | Marlin Kernel (高性能反量化GEMM) |
| KV Cache | FP8 E5M2 (减少Decode阶段显存带宽) |
| 量化组大小 | 128 |
| 激活顺序 | 关闭 (desc_act=False，性能更优) |

## 目录结构

```
submit/
├── prepare_env.sh          # 必须 - 环境构建脚本
├── prepare_model.sh        # 可选 - 模型预处理入口
├── preprocess_model.py     # prepare_model.sh 调用的 Python 脚本
├── quantize_model.py       # 独立的量化脚本（核心实现）
├── verify_accuracy.py      # 精度验证脚本
├── README.md               # 本文件
└── sglang/python/          # 自定义 sglang 源码（editable install）
```

## 各文件说明

### 1. prepare_env.sh（必须）

平台在基础环境启动后自动执行此脚本，完成以下任务：

1. **安装依赖**：
   - PyTorch (CUDA 12.1)
   - GPTQModel (量化工具)
   - transformers, accelerate, safetensors

2. **安装自定义 SGLang**：
   ```bash
   uv pip install --no-deps -e ./sglang/python
   ```

3. **配置启动参数**：
   ```bash
   export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} \
       --quantization gptq_marlin \
       --kv-cache-dtype fp8_e5m2 \
       --log-level info"
   ```

### 2. prepare_model.sh（可选）

平台在环境就绪后调用此脚本，接口固定为：

```bash
bash prepare_model.sh --input <原始模型路径> --output <处理后模型路径>
```

脚本会调用 `preprocess_model.py` 执行实际的量化操作。

### 3. quantize_model.py（核心）

使用 GPTQModel 对 SALA FP16 模型进行 W4A16 量化，主要流程：

1. **配置量化参数**：
   - bits=4 (W4A16)
   - group_size=128
   - desc_act=False
   - checkpoint_format="marlin"

2. **准备校准数据**：
   - 128 条多样化文本样本
   - 涵盖中文、英文、代码、逻辑推理等

3. **执行量化**：
   - 加载 FP16 模型
   - 使用校准数据进行 GPTQ 量化
   - 直接输出 Marlin 格式

4. **保存模型**：
   - 量化权重 (safetensors)
   - 配置文件 (config.json)
   - 量化配置 (quantize_config.json)
   - 分词器文件

**独立使用方式**：
```bash
python quantize_model.py \
    --model-path /path/to/sala-fp16 \
    --output-path /path/to/output \
    --bits 4 \
    --group-size 128 \
    --desc-act False
```

### 4. verify_accuracy.py（验证工具）

验证量化后模型的精度是否满足要求（> 97%）。

**在线验证**（模型已加载到 SGLang 服务）：
```bash
python verify_accuracy.py --server-url http://localhost:30000
```

**离线验证**（直接加载模型）：
```bash
python verify_accuracy.py \
    --model-path /path/to/quantized/model \
    --offline
```

**测试用例覆盖**：
- 中文常识问答（3题）
- 英文常识问答（3题）
- 逻辑推理（2题）
- 代码相关（2题）
- 长文本理解（2题，测试 Lightning Attention）

### 5. sglang/python/

自定义的 sglang 源码目录。通过 editable install，平台会使用此目录下的代码替代镜像内置 sglang。

如需修改推理引擎实现（如调整 Marlin Kernel 参数），可在此目录下修改：
- `sglang/srt/layers/quantization/gptq.py` - GPTQ 量化逻辑
- `sglang/srt/layers/quantization/marlin_utils.py` - Marlin 工具函数

## 执行流程

### 平台执行流程

```
1. 启动基础环境
   ↓
2. 执行 prepare_env.sh
   - 安装 GPTQModel 等依赖
   - 安装自定义 SGLang
   - 配置启动参数
   ↓
3. 执行 prepare_model.sh --input <模型路径> --output <输出路径>
   - 调用 preprocess_model.py
   - 执行 GPTQ 量化
   - 输出 Marlin 格式模型
   ↓
4. 启动 SGLang 服务
   - 自动加载量化后的模型
   - 使用 gptq_marlin 量化方式
   - 启用 FP8 KV Cache
```

### 本地测试流程

```bash
# 1. 准备环境
bash prepare_env.sh

# 2. 量化模型（如尚未量化）
python quantize_model.py \
    --model-path /path/to/sala-fp16 \
    --output-path ./sala-gptq-marlin

# 3. 启动服务
python -m sglang.launch_server \
    --model-path ./sala-gptq-marlin \
    --quantization gptq_marlin \
    --kv-cache-dtype fp8_e5m2 \
    --tp-size 1

# 4. 验证精度
python verify_accuracy.py --server-url http://localhost:30000
```

## 关键参数说明

### 量化参数

| 参数 | 值 | 说明 |
|------|-----|------|
| bits | 4 | 权重量化位数，4-bit 提供最佳压缩比 |
| group_size | 128 | 量化组大小，128 是 Marlin 推荐值 |
| desc_act | False | 禁用激活顺序量化，提升推理性能 |
| sym | True | 对称量化，Marlin 要求 |
| checkpoint_format | marlin | 直接输出 Marlin 格式 |

### SGLang 启动参数

| 参数 | 值 | 说明 |
|------|-----|------|
| --quantization | gptq_marlin | 使用 GPTQ + Marlin 量化 |
| --kv-cache-dtype | fp8_e5m2 | KV Cache 使用 FP8 E5M2 格式 |

## 精度与性能

### 预期精度

- **目标精度**: > 97%（相对于 FP16 基线）
- **校准样本**: 128 条
- **测试覆盖**: 中文、英文、逻辑、代码、长文本

### 性能收益

- **显存节省**: 约 75%（4-bit vs 16-bit）
- **推理加速**: Marlin Kernel 提供高效 GEMM 计算
- **Decode加速**: FP8 KV Cache 减少显存带宽瓶颈

## 故障排查

### 量化阶段

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 显存不足 | 模型太大 | 使用更小的 batch size 或启用 CPU offload |
| 量化精度低 | 校准数据不足 | 增加 --num-calibration-samples |
| 转换失败 | 不支持的层 | 检查模型结构，可能需要跳过某些层 |

### 推理阶段

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 模型加载失败 | 量化配置不匹配 | 检查 quantize_config.json 格式 |
| 精度下降严重 | 量化损失大 | 尝试 W8A8 量化或增加校准样本 |
| 性能不如预期 | Marlin 配置不当 | 检查 GPU 是否支持 Marlin (SM80+) |

## 进阶调优

### 调整 Marlin Kernel 参数（针对 6000D）

如需调整 tile/warp 配置，修改 `sglang/python/sglang/srt/layers/quantization/marlin_utils.py`：

```python
# 当前默认值
GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16
```

### 动态量化配置

如需对不同层使用不同配置，可在 `quantize_model.py` 中使用 GPTQModel 的 `dynamic` 参数：

```python
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    dynamic={
        # 对特定层使用 8-bit 量化
        r"+:.*\.layers\.(20|21|22|23)\..*": {"bits": 8},
    }
)
```

## 参考链接

- [GPTQModel 文档](https://github.com/ModelCloud/GPTQModel)
- [Marlin Paper](https://arxiv.org/abs/2401.14151)
- [SGLang 量化文档](https://docs.sglang.ai/quantization/)

## 提交检查清单

- [ ] `prepare_env.sh` 可执行且包含所有依赖安装
- [ ] `prepare_model.sh` 可执行且接口正确
- [ ] `quantize_model.py` 可正常运行
- [ ] `sglang/python/` 包含必要的自定义代码
- [ ] 本地测试通过（精度 > 97%）
- [ ] 打包为 .tar.gz 格式
