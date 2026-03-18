"""
test_chunk_gla: Fused Chunk GLA with Output Processing

This package contains:
- chunk_gla_fused_output.py: FLA chunk_fwd_h + 2D output + RMSNorm+gate fusion (2 kernels)
- chunk_gla_fused_all.py: Fully fused version - chunk_fwd_h + o + RMSNorm+gate (1 kernel for o+final)
- test_chunk_gla_with_output.py: End-to-end test comparing FLA baseline vs fused

Optimization approaches:

Scheme B (chunk_gla_fused_output):
1. Uses FLA's proven-correct chunk_fwd_h
2. Custom chunk_fwd_o_fused with 2D output [B*T, H*V]
3. Fused RMSNorm + sigmoid gate kernel
Total: 3 kernel launches (h, o, final)

Scheme C - Fully Fused (chunk_gla_fused_all):
1. Uses FLA's proven-correct chunk_fwd_h
2. Fused chunk_fwd_o + RMSNorm + sigmoid gate in single kernel
Total: 2 kernel launches (h, o+final)
- Eliminates explicit o tensor storage
- Saves ~50% intermediate tensor memory bandwidth

Scheme D - Autotuned (chunk_gla_autotuned):
- Automatically selects best kernel based on sequence length
- T <= 512: 2-kernel (~1.4-1.7x speedup)
- T == 1024: 3-kernel (avoids occupancy dip)
- T >= 2048: 2-kernel (~1.07-1.16x speedup)
"""

from .chunk_gla_fused_output import (
    chunk_simple_gla_fused_output,
    chunk_gla_fused_output,
    chunk_fwd_o_fused,
    fused_output_final,
)

from .chunk_gla_fused_all import (
    chunk_simple_gla_fused_all,
    chunk_gla_fused_all,
    chunk_fwd_o_fused_all,
)

from .chunk_gla_autotuned import (
    chunk_simple_gla_autotuned,
    chunk_gla_autotuned,
    chunk_simple_gla,
    chunk_gla,
)

__all__ = [
    # Original 3-kernel version
    "chunk_simple_gla_fused_output",
    "chunk_gla_fused_output",
    "chunk_fwd_o_fused",
    "fused_output_final",
    # Fully fused 2-kernel version
    "chunk_simple_gla_fused_all",
    "chunk_gla_fused_all",
    "chunk_fwd_o_fused_all",
    # Autotuned version (recommended)
    "chunk_simple_gla_autotuned",
    "chunk_gla_autotuned",
    "chunk_simple_gla",
    "chunk_gla",
]
