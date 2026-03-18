# AGENTS.md - test_chunk_gla

## Project Overview

This project implements **fused GPU kernels for Chunk GLA (Gated Linear Attention)** optimized for NVIDIA Blackwell architecture (RTX 6000D, sm_120). It is part of the [SGLang](https://github.com/sgl-project/sglang) project and specifically targets the MiniCPM model's Lightning Attention mechanism.

The main goal is to reduce memory bandwidth usage by fusing multiple operations into fewer Triton kernels, achieving ~10-15x speedup compared to PyTorch baseline for output processing.

### Key Optimization: Two-Kernel Fusion Scheme

The implementation uses a two-kernel approach (Scheme B):

1. **chunk_fwd_h** (from FLA): Computes hidden states using proven-correct FLA kernel
2. **chunk_fwd_o_fused** (custom): Computes attention output directly in 2D layout `[B*T, H*V]`
3. **fused_output_final** (custom): Fused RMSNorm + sigmoid gate per sequence position

This eliminates intermediate tensor reshaping and reduces HBM traffic.

---

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | >=3.10 | Runtime language |
| PyTorch | 2.9.1+cu128 | Deep learning framework |
| Triton | 3.5.1 | GPU kernel development |
| flash-linear-attention (FLA) | 0.4.1 | Baseline GLA implementation |
| CUDA | 12.x | GPU compute platform |

### Hardware Target

- **GPU**: NVIDIA RTX 6000D (Blackwell)
- **Compute Capability**: sm_120 (12.0)
- **SMs**: 156
- **Shared Memory**: 128 KB/SM
- **Memory**: GDDR7, ~1568 GB/s
- **L2 Cache**: 128 MB
- **Tensor Cores**: 5th Gen (FP4 support)

---

## Project Structure

```
test_chunk_gla/
├── __init__.py                      # Package initialization, exports main interfaces
├── chunk_gla_fused_output.py        # Core fused kernels (Triton) + Python wrappers
├── test_chunk_gla_with_output.py    # End-to-end test and benchmark suite
├── README.md                        # Project documentation (Chinese)
├── OPTIMIZATION_NOTES.md            # Detailed optimization analysis
└── AGENTS.md                        # This file
```

### File Details

#### `chunk_gla_fused_output.py`
Contains the core implementation:

- `chunk_fwd_kernel_o_fused`: Modified chunk_fwd_o kernel with 2D output layout
  - Grid: `(V/BV, T/BT, B*H)`
  - Autotune configs: `(BK=128,BV=128,warps=8)`, `(BK=64,BV=64,warps=4)`, `(BK=32,BV=32,warps=2)`
  
- `kernel_fused_output_final`: RMSNorm + sigmoid gate kernel
  - Grid: `(B*T,)` - one block per sequence position
  - BLOCK_SIZE: 1024 or 512 based on hidden_size

- `chunk_gla_fused_output()`: Main interface combining all three steps
- `chunk_simple_gla_fused_output()`: Simplified API matching FLA style

#### `test_chunk_gla_with_output.py`
Comprehensive test and benchmark suite:

- **Correctness Tests**: Compares against PyTorch reference implementation
- **Performance Benchmarks**: Memory bandwidth analysis with roofline model
- **Test Configs**: Sequence lengths [16, 64, 128, 256, 1024, 4096]

Three implementations compared:
1. `real_baseline_fla_4d`: FLA chunk_gla (4D) → reshape → fused_output_processing
2. `fla_fused_2d`: FLA chunk_fwd_h + our 2D output + fused_output_final
3. `torch_reference`: Pure PyTorch reference for correctness checking

#### `__init__.py`
Package exports:
- `chunk_simple_gla_fused_output` - Main simplified interface
- `chunk_gla_fused_output` - Full interface with all parameters
- `chunk_fwd_o_fused` - Low-level output kernel wrapper
- `fused_output_final` - Low-level RMSNorm+gate wrapper

---

## Build and Test Commands

### Prerequisites

The project depends on the parent SGLang project structure:

```bash
# Required packages (from parent project's pyproject.toml)
pip install torch==2.9.1 triton==3.5.1
pip install flash-linear-attention==0.4.1

# For output processing kernel (dependency)
# Ensure /root/soar2026/python is in PYTHONPATH
export PYTHONPATH=/root/soar2026/python:$PYTHONPATH
```

### Running Tests

```bash
cd /root/soar2026

# Run end-to-end test with benchmark
python test_chunk_gla/test_chunk_gla_with_output.py
```

### Expected Output

The test outputs:
1. **Correctness Table**: MAE (Mean Absolute Error) and Max Error for both baseline and fused implementations
2. **Performance Table**: 
   - Time (ms) for baseline vs fused
   - HBM traffic comparison (baseline | fused | saved%)
   - Achieved bandwidth (GB/s)
   - Utilization % of practical peak (1344 GB/s)
   - Speedup vs FLA baseline

---

## Code Style Guidelines

### General Conventions

1. **Language**: Documentation and comments are primarily in Chinese
2. **Docstrings**: Use triple quotes with descriptive text for all public functions
3. **Type Hints**: Use Python 3.10+ type hints (`tuple[torch.Tensor, torch.Tensor]`)
4. **Constants**: Use UPPER_CASE for kernel configuration constants

### Triton Kernel Conventions

```python
# 1. Use triton.heuristics for optional arguments
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})

# 2. Use triton.autotune for performance tuning
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        # ... more configs
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)

# 3. Use tl.constexpr for compile-time constants
def kernel(..., B: tl.constexpr, H: tl.constexpr, ...):
```

### Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Kernels | `kernel_*` or `*_kernel_*` | `kernel_fused_output_final` |
| Wrappers | Same as kernel without decorator | `chunk_fwd_o_fused` |
| Main interfaces | `chunk_*_fused_output` | `chunk_gla_fused_output` |
| Internal tensors | `b_*` prefix | `b_o`, `b_q`, `b_h` |
| Pointers | `p_*` prefix | `p_q`, `p_o` |

---

## Testing Instructions

### Adding New Test Cases

Edit `test_chunk_gla_with_output.py`:

```python
# Add to test_configs list in main()
test_configs = [
    {"seq_len": 16, "num_heads": 32, "head_dim": 128},
    # Add your config here
    {"seq_len": 8192, "num_heads": 32, "head_dim": 128},
]
```

### Accuracy Validation

Tests pass if:
- `output_mae < 0.1` (Mean Absolute Error < 0.1)
- `output_max < threshold` (no outliers)
- No NaN values in output

### Performance Targets (RTX 6000D)

| Sequence Length | Target Utilization | Expected Speedup |
|-----------------|-------------------|------------------|
| 16 - 128 | >40% (⚠️ Latency bound) | 1.0-1.2x |
| 256 - 1024 | >70% (✅ Good) | 1.1-1.3x |
| 4096+ | >70% (✅ Good) | 1.2-1.5x |

---

## Integration with SGLang

This module integrates with the broader SGLang project:

```
SGLang Project (/root/soar2026/)
├── python/sglang/srt/models/
│   ├── minicpm.py                   # Main MiniCPM model
│   ├── minicpm_fused_output.py      # Fused output processing kernel
│   └── ...
└── test_chunk_gla/                  # This project
    ├── chunk_gla_fused_output.py    # Fused GLA + output processing
    └── test_chunk_gla_with_output.py
```

### Usage in Model Code

```python
from test_chunk_gla import chunk_simple_gla_fused_output

# Inside MiniCPM Lightning Attention layer
out, ht = chunk_simple_gla_fused_output(
    q, k, v,           # [B, T, H, D] attention inputs
    z=z,               # [B*T, H*V] gate input
    norm_weight=norm_w,  # [H*V] RMSNorm weight
    g_gamma=g_gamma,   # decay parameter
    scale=scale,
    output_final_state=True,
)
# Returns: out [B*T, H*V], ht [B, H, K, V]
```

---

## Optimization Notes

### Blackwell-Specific Optimizations Applied

1. **Increased Tile Sizes**: BK/BV up to 128 (vs 64 in FLA) leveraging 128KB shared memory
2. **Optimized Warp/Stage Configs**: Focus on num_warps=[4,8], num_stages=[2,3,4]
3. **Chunk Size**: Use 128 for sequences >64 (vs 64 in FLA)
4. **2D Output Layout**: Direct `[B*T, H*V]` output eliminates reshape overhead

### Further Optimization Opportunities

Documented in `OPTIMIZATION_NOTES.md`:
- FP4 quantization support (Blackwell 5th Gen Tensor Cores)
- Kernel fusion (h + o into single kernel)
- TMA (Tensor Memory Accelerator) usage
- CUDA Graphs for small sequences

### Memory Bandwidth Savings

The fused approach saves HBM traffic by:
- Eliminating `h` tensor write+read (stays in registers between kernels)
- Direct 2D output (no 4D→2D reshape copy)
- Fused RMSNorm+gate (single kernel instead of multiple ops)

Estimated savings: 10-30% depending on sequence length.

---

## Dependencies

### Required
- `torch>=2.0`
- `triton>=3.0`
- `flash-linear-attention>=0.4.1`
- CUDA-capable GPU (sm_120 for full optimization)

### Optional
- `sglang` (for integration with full serving stack)

---

## Security Considerations

1. **Kernel Safety**: All kernels include boundary checks (`boundary_check=(0,1)`)
2. **Numerical Stability**: Uses float32 accumulation, bf16 for storage
3. **Memory Alignment**: Assumes properly aligned tensors (standard PyTorch guarantees)
4. **No User Input**: This is a compute library; no direct user input handling

---

## References

- [FLA: Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [Triton Documentation](https://triton-lang.org/)
- [SGLang Project](https://github.com/sgl-project/sglang)
- [MiniCPM Paper](https://arxiv.org/abs/2404.06395)
