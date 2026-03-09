---
name: cuda-or-triton-kernel-dev-paradigm
description: Guidelines for developing and optimizing GPU kernels (CUDA or Triton). Use when: (1) Writing new CUDA/Triton kernels, (2) Optimizing existing operators, (3) Benchmarking kernel performance, (4) Comparing kernel implementations for accuracy and speed
---

## Kernel dev workflow

1. 在一切开始前，要确保曾经源码分析过 kernel 实现
- 这是个 compute bound 还是个 memory bound？
- sm occupancy 能否打满？
- 计算中间的 data type 是否有进一步调整的空间？
- 是否有 ping-pong 或 fuse 的可能？
- 根据当前硬件（指令集或硬件架构）是否有优化点？
- 实现复杂度如何？
- 考虑用户的输入（哪些是 fixed shape 和 fixed dtype），有没有多余的代码可以删除？

2. 具体优化算子时，检查是否存在对应的 benchmark 代码
- benchmark 在代码的根目录，以 test 或 bench 作为目录的开头
- benchmark 一定存在 baseline 做精度对比。baseline 可能是调用标准库，或者从源码里扣一个在本地文件。kernel 输入的 dtype 多数是 bf16。现在 fp32 并不常用（除了 softmax 这类精度敏感的算子）

3. 算子优化期间，往往考虑 input 或 weight 的特定 shape、硬件特性做针对性优化。因为通用写法的效率已经不差

4. 除非函数名或类名明确指出只对某种 shape 有效，assistant 生成的 kernel 都需要对通用 shape 兜底

5. benchmark 运行结束后，需要 print 或 plot 出不同版本的性能差异

6. 优化结束后，清理为了 bugfix 调试写的小样例（例如 test 机器是否正常）；不要清理新开发/新调整的完整 benchmark 项目。
