/*
 * Optimized implementation using __nv_bfloat162 for vectorized computation
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace optimized {

__global__ void __launch_bounds__(1024) silu_and_mul_kernel_bf162(
    __nv_bfloat16* __restrict__ out, 
    const __nv_bfloat16* __restrict__ input, 
    const int d) {
  
  // 128-bit = 8 x bf16 = 4 x __nv_bfloat162
  constexpr int vec_size = 4;
  
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;
  
  const auto* input_bf162 = reinterpret_cast<const __nv_bfloat162*>(input + offset);
  const auto* gate_bf162 = reinterpret_cast<const __nv_bfloat162*>(input + offset + d);
  auto* out_bf162 = reinterpret_cast<__nv_bfloat162*>(out + token_idx * d);
  
  // Each thread processes 4 x __nv_bfloat162 = 8 bf16 elements
  const int64_t num_vec_per_row = d / (vec_size * 2);  // d / 8
  
  #pragma unroll 1
  for (int64_t idx = thread_idx; idx < num_vec_per_row; idx += stride) {
    // Load 128 bits (4 x __nv_bfloat162)
    uint4 x_u4 = reinterpret_cast<const uint4*>(input_bf162)[idx];
    uint4 y_u4 = reinterpret_cast<const uint4*>(gate_bf162)[idx];
    
    // Process 4 __nv_bfloat162 vectors
    __nv_bfloat162 out_vec[vec_size];
    
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      __nv_bfloat162 x_val = reinterpret_cast<const __nv_bfloat162*>(&x_u4)[i];
      __nv_bfloat162 y_val = reinterpret_cast<const __nv_bfloat162*>(&y_u4)[i];
      
      // Convert to float2 for computation
      float2 x_f2 = __bfloat1622float2(x_val);
      float2 y_f2 = __bfloat1622float2(y_val);
      
      // silu(x) = x / (1 + exp(-x))
      float2 silu_f2;
      silu_f2.x = x_f2.x / (1.0f + __expf(-x_f2.x));
      silu_f2.y = x_f2.y / (1.0f + __expf(-x_f2.y));
      
      // Multiply: silu(x) * y
      float2 out_f2;
      out_f2.x = silu_f2.x * y_f2.x;
      out_f2.y = silu_f2.y * y_f2.y;
      
      // Convert back to __nv_bfloat162
      out_vec[i] = __float22bfloat162_rn(out_f2);
    }
    
    // Store 128 bits
    uint4 out_u4;
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      reinterpret_cast<__nv_bfloat162*>(&out_u4)[i] = out_vec[i];
    }
    reinterpret_cast<uint4*>(out_bf162)[idx] = out_u4;
  }
  
  // Handle remaining elements (tail processing)
  const int64_t remaining_start = num_vec_per_row * vec_size * 2;
  for (int64_t idx = remaining_start + thread_idx; idx < d; idx += stride) {
    float x = __bfloat162float(input[offset + idx]);
    float y = __bfloat162float(input[offset + d + idx]);
    float silu_val = x / (1.0f + __expf(-x));
    out[token_idx * d + idx] = __float2bfloat16(silu_val * y);
  }
}

void launch_silu_and_mul_bf162(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int d,
    int num_tokens,
    cudaStream_t stream) {
  
  dim3 grid(num_tokens);
  
  // Compute optimal block size
  constexpr int vec_size = 4;  // 4 x __nv_bfloat162 = 8 bf16 elements
  uint32_t block_size = (d / (vec_size * 2) + 31) & ~31U;
  block_size = (block_size > 1024) ? 1024 : block_size;
  block_size = (block_size < 128) ? 128 : block_size;
  
  dim3 block(block_size);
  
  silu_and_mul_kernel_bf162<<<grid, block, 0, stream>>>(out, input, d);
}

} // namespace optimized
