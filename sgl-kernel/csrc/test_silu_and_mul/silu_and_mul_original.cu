/*
 * Original implementation - scalar processing with 128-bit vectorized memory access
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace original {

template <typename T>
__device__ __forceinline__ float to_f32(const T& x) {
  return static_cast<float>(x);
}

template <typename T>
__device__ __forceinline__ T from_f32(float f32) {
  return static_cast<T>(f32);
}

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = to_f32(x);
  return from_f32<T>(f32_val / (1.0f + __expf(-f32_val)));
}

template <typename T, T (*Activation)(const T&)>
__global__ void __launch_bounds__(1024) act_and_mul_kernel(
    T* __restrict__ out, 
    const T* __restrict__ input, 
    const int d) {
  constexpr int vec_size = 16 / sizeof(T);
  
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;
  
  #pragma unroll 1
  for (int64_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    uint4 x_u4 = reinterpret_cast<const uint4*>(input + offset)[idx];
    uint4 y_u4 = reinterpret_cast<const uint4*>(input + offset + d)[idx];
    
    float x_f[vec_size], y_f[vec_size];
    
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      x_f[i] = static_cast<float>(reinterpret_cast<const T*>(&x_u4)[i]);
      y_f[i] = static_cast<float>(reinterpret_cast<const T*>(&y_u4)[i]);
    }
    
    float out_f[vec_size];
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out_f[i] = static_cast<float>(Activation(static_cast<T>(x_f[i]))) * y_f[i];
    }
    
    uint4 out_u4;
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      reinterpret_cast<T*>(&out_u4)[i] = static_cast<T>(out_f[i]);
    }
    
    reinterpret_cast<uint4*>(out + token_idx * d)[idx] = out_u4;
  }
  
  const int64_t remaining_start = (d / vec_size) * vec_size;
  for (int64_t idx = remaining_start + thread_idx; idx < d; idx += stride) {
    float x = static_cast<float>(input[offset + idx]);
    float y = static_cast<float>(input[offset + d + idx]);
    out[token_idx * d + idx] = static_cast<T>(static_cast<float>(Activation(static_cast<T>(x))) * y);
  }
}

// Explicit instantiation for __nv_bfloat16
template __global__ void act_and_mul_kernel<__nv_bfloat16, silu<__nv_bfloat16>>(
    __nv_bfloat16* __restrict__ out, 
    const __nv_bfloat16* __restrict__ input, 
    const int d);

void launch_silu_and_mul(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int d,
    int num_tokens,
    cudaStream_t stream) {
  
  dim3 grid(num_tokens);
  
  // Compute optimal block size
  constexpr int vec_size = 16 / sizeof(__nv_bfloat16);  // 8 for bf16
  uint32_t block_size = (d / vec_size + 31) & ~31U;
  block_size = (block_size > 1024) ? 1024 : block_size;
  block_size = (block_size < 128) ? 128 : block_size;
  
  dim3 block(block_size);
  
  act_and_mul_kernel<__nv_bfloat16, silu<__nv_bfloat16>>
      <<<grid, block, 0, stream>>>(out, input, d);
}

} // namespace original
