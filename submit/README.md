# MiniCPM-SALA 提交 Demo

本目录是一个最小可运行的提交示例，演示如何按照平台要求组织 `prepare_env.sh` + `prepare_model.sh` 提交包。

## 目录结构

```
.
├── prepare_env.sh          # 必须 — 环境构建脚本
├── prepare_model.sh        # 可选 — 模型预处理入口
├── preprocess_model.py     # prepare_model.sh 调用的 Python 脚本
└── sglang/python/          # 自定义 sglang 源码（editable install）
```

## 各文件说明

### `prepare_env.sh`（必须）

平台在基础环境启动后自动执行此脚本。本 demo 中做了两件事：

1. 用 `uv pip install --no-deps -e ./sglang/python` 将自定义 sglang 以 editable 模式安装，替换镜像内置版本
2. 通过 `export SGLANG_SERVER_ARGS` 追加推理启动参数（示例中添加了 `--log-level info`）

```bash
uv pip install --no-deps -e ./sglang/python
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"
```

> **注意**：`prepare_env.sh` 会被 `source` 进入平台主脚本，因此 `export` 的环境变量可以直接生效。

### `prepare_model.sh`（可选）

平台在环境就绪后调用此脚本，接口固定为：

```bash
bash prepare_model.sh --input <原始模型路径> --output <处理后模型路径>
```

两个路径均由平台提供，选手无需关心容器内的具体挂载位置。本 demo 中仅做简单的模型文件复制，不做任何量化或转换。

实际参赛时，可以在 `preprocess_model.py` 中实现量化（GPTQ、AWQ 等）、剪枝、权重融合等预处理逻辑。

### `sglang/python/`

自定义的 sglang 源码目录。通过 editable install，平台会使用此目录下的代码替代镜像内置 sglang，选手可以在此修改推理引擎的实现。

## 扩展示例

| 场景 | 修改点 |
|---|---|
| 安装额外 pip 包 | `prepare_env.sh` 中添加 `uv pip install xxx` |
| 自定义推理参数 | `prepare_env.sh` 中修改 `SGLANG_SERVER_ARGS` |
| GPTQ 量化 | `preprocess_model.py` 中实现 GPTQ 打包，`prepare_env.sh` 中追加 `--quantization gptq` |
| 模型剪枝/蒸馏 | `preprocess_model.py` 中实现，输出到 `--output` 目录 |
