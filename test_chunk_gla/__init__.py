"""
test_chunk_gla: FLA chunk_simple_gla implementation for hardware optimization.

This package contains:
- chunk_simple_gla.py: Original FLA implementation (reference)
- chunk_simple_gla_blackwell.py: Blackwell 6000D optimized forward-only version
- test_chunk_gla.py: Test inputs
"""

# Import Blackwell-optimized version by default if on Blackwell
from .chunk_simple_gla_blackwell import chunk_simple_gla_blackwell, is_blackwell

# Keep reference to original
from .chunk_simple_gla import chunk_simple_gla as chunk_simple_gla_original

# Auto-select based on hardware
def chunk_simple_gla(q, k, v, **kwargs):
    """
    Auto-select optimal implementation based on hardware.
    
    On Blackwell (sm_120): Uses optimized forward-only kernel
    On other GPUs: Falls back to original FLA implementation
    """
    if is_blackwell():
        return chunk_simple_gla_blackwell(q, k, v, **kwargs)
    else:
        return chunk_simple_gla_original(q, k, v, **kwargs)

__all__ = [
    "chunk_simple_gla",
    "chunk_simple_gla_blackwell",
    "chunk_simple_gla_original",
    "is_blackwell",
]
