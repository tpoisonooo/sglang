# Fused Recurrent Simple GLA Test Environment

This directory contains a standalone implementation of `fused_recurrent_simple_gla` extracted from the FLA (Flash Linear Attention) library for testing and optimization.

## Files

- `recurrent_simple_gla.py` - Complete standalone implementation including:
  - Triton kernels (`fused_recurrent_fwd_kernel`, `fused_recurrent_bwd_kernel`)
  - Python wrappers (`fused_recurrent_fwd`, `fused_recurrent_bwd`)
  - PyTorch autograd function (`FusedRecurrentFunction`)
  - Main interface (`fused_recurrent_simple_gla`)

- `test_recurrent_gla.py` - Test and benchmark script including:
  - Correctness tests against PyTorch reference implementation
  - Correctness tests against FLA library implementation
  - Tests with initial state (for prefix caching scenarios)
  - Performance benchmarks
  - Memory usage analysis

## Usage

### Run Tests

```bash
cd /root/soar2026
python test_recurrent_gla/test_recurrent_gla.py
```

### Use the Kernel in Your Code

```python
from test_recurrent_gla.recurrent_simple_gla import fused_recurrent_simple_gla
import torch

# Create inputs
B, T, H, K, V = 1, 128, 32, 128, 128
q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g_gamma = torch.randn(H, device='cuda', dtype=torch.float32) * -0.01
scale = K ** -0.5

# Forward pass
o, final_state = fused_recurrent_simple_gla(
    q, k, v, 
    g_gamma=g_gamma, 
    scale=scale,
    output_final_state=True
)
```

## Implementation Details

### Kernel Algorithm

The fused recurrent kernel implements the following recurrence:

```
h_t = decay * h_{t-1} + k_t^T @ v_t
o_t = q_t @ h_t * scale
```

Where:
- `h_t` is the hidden state of shape `[B, H, K, V]`
- `decay = exp(g_gamma)` is the head-wise decay factor
- `k_t, v_t, q_t` are key, value, query at timestep t
- `scale` is the attention scale factor (typically `1/sqrt(K)`)

### Key Features

1. **Fused Single Kernel**: The forward pass is computed in a single Triton kernel without intermediate materialization

2. **Variable Length Support**: Supports `cu_seqlens` for variable-length sequences (batch size must be 1)

3. **Initial State**: Supports `initial_state` for prefix caching / continuation

4. **Autograd Support**: Full backward pass support for training

5. **No Chunking Overhead**: Unlike chunk-based methods, this is purely recurrent - suitable for short sequences (< 128)

### Comparison with Chunk-based Methods

| Aspect | Fused Recurrent | Chunk-based (e.g., chunk_simple_gla) |
|--------|----------------|--------------------------------------|
| Memory | O(B*H*K*V) state | O(B*H*K*V) state + O(B*NS*H*K*V) intermediate h |
| Best for | Short sequences (< 128) | Long sequences (> 128) |
| Parallelism | Limited (sequential) | High (chunk-level parallel) |
| HBM traffic | Low | Higher (h tensor traffic) |

## Test Results

The test script verifies:
1. **Numerical correctness** against PyTorch reference (MAE < 1e-5)
2. **Bit-exact match** with FLA implementation
3. **Correctness with initial state** for prefix caching
4. **Performance** compared to FLA

## Notes for Optimization

This implementation is extracted from FLA for further optimization. Potential areas:

1. **Warp-level optimization**: Current kernel uses simple loop structure
2. **Shared memory usage**: Could optimize data layout for better SM utilization
3. **Instruction-level optimization**: Fuse operations, reduce memory accesses
4. **Pipeline optimization**: Overlap computation and memory for consecutive elements
