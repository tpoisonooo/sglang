/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#ifndef USE_ROCM

#include "utils.h"

#else
#include "hip/hip_act_and_mul.cuh"
#endif

// Fully independent implementation of activation kernels
// Optimized based on benchmark results:
// - 128-bit vectorized memory access
// - __expf for faster computation
// - Optimal block size calculation

namespace detail {

template <typename T>
__device__ __forceinline__ float to_f32(const T& x) {
#if USE_ROCM
  return castToFloat(x);
#else
  return static_cast<float>(x);
#endif
}

template <typename T>
__device__ __forceinline__ T from_f32(float f32) {
#if USE_ROCM
  return castFromFloat<T>(f32);
#else
  return static_cast<T>(f32);
#endif
}

}  // namespace detail

// Activation functions
template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  // Use __expf for faster computation (intrinsic function)
  return detail::from_f32<T>(f32_val / (1.0f + __expf(-f32_val)));
}

template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
}

// gelu_quick(x) = x * torch.sigmoid(1.702 * x)
template <typename T>
__device__ __forceinline__ T gelu_quick_act(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + __expf(-f32_val * 1.702f)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  float f32_val = detail::to_f32(x);
  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));
  return detail::from_f32<T>(f32_val * cdf);
}

// Fully independent optimized kernel for activation_and_mul
// This is a complete implementation optimized for maximum performance
template <typename T, T (*Activation)(const T&)>
__global__ void __launch_bounds__(1024) act_and_mul_kernel(
    T* __restrict__ out, 
    const T* __restrict__ input, 
    const int d) {
  // Use 128-bit vector size for optimal memory throughput
  constexpr int vec_size = 16 / sizeof(T);  // 8 for fp16/bf16, 4 for fp32
  
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;
  
  // Main vectorized loop
  // Each thread processes vec_size elements at a time
  #pragma unroll 1
  for (int64_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    // Load 128 bits directly from global memory using uint4
    uint4 x_u4 = reinterpret_cast<const uint4*>(input + offset)[idx];
    uint4 y_u4 = reinterpret_cast<const uint4*>(input + offset + d)[idx];
    
    // Process all elements with unrolled loop
    float x_f[vec_size], y_f[vec_size];
    
    // Cast to float
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      x_f[i] = static_cast<float>(reinterpret_cast<const T*>(&x_u4)[i]);
      y_f[i] = static_cast<float>(reinterpret_cast<const T*>(&y_u4)[i]);
    }
    
    // Compute activation and multiply
    float out_f[vec_size];
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out_f[i] = Activation(static_cast<T>(x_f[i])) * y_f[i];
    }
    
    // Cast back and store
    uint4 out_u4;
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      reinterpret_cast<T*>(&out_u4)[i] = static_cast<T>(out_f[i]);
    }
    
    // Store 128 bits to global memory
    reinterpret_cast<uint4*>(out + token_idx * d)[idx] = out_u4;
  }
  
  // Handle remaining elements (tail processing)
  const int64_t remaining_start = (d / vec_size) * vec_size;
  for (int64_t idx = remaining_start + thread_idx; idx < d; idx += stride) {
    float x = static_cast<float>(input[offset + idx]);
    float y = static_cast<float>(input[offset + d + idx]);
    out[token_idx * d + idx] = static_cast<T>(Activation(static_cast<T>(x)) * y);
  }
}

// Kernel for activation only (no multiply)
template <typename T, T (*Activation)(const T&)>
__global__ void __launch_bounds__(1024) act_only_kernel(
    T* __restrict__ out, 
    const T* __restrict__ input, 
    const int d) {
  constexpr int vec_size = 16 / sizeof(T);
  
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * d;
  
  #pragma unroll 1
  for (int64_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    uint4 x_u4 = reinterpret_cast<const uint4*>(input + offset)[idx];
    
    float x_f[vec_size];
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      x_f[i] = static_cast<float>(reinterpret_cast<const T*>(&x_u4)[i]);
    }
    
    float out_f[vec_size];
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out_f[i] = Activation(static_cast<T>(x_f[i]));
    }
    
    uint4 out_u4;
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      reinterpret_cast<T*>(&out_u4)[i] = static_cast<T>(out_f[i]);
    }
    
    reinterpret_cast<uint4*>(out + offset)[idx] = out_u4;
  }
  
  const int64_t remaining_start = (d / vec_size) * vec_size;
  for (int64_t idx = remaining_start + thread_idx; idx < d; idx += stride) {
    float x = static_cast<float>(input[offset + idx]);
    out[offset + idx] = static_cast<T>(Activation(static_cast<T>(x)));
  }
}

// Helper to compute optimal block size
static inline dim3 get_optimal_block_size(int d, int vec_size) {
  uint32_t block_size = std::min(static_cast<uint32_t>(d / vec_size), 1024U);
  // Round up to nearest multiple of 32 (warp size) for coalesced access
  block_size = (block_size + 31) & ~31U;
  // Ensure minimum block size for good occupancy
  block_size = std::max(block_size, 128U);
  return dim3(block_size);
}

void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    constexpr uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block = get_optimal_block_size(d, vec_size);
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    constexpr uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block = get_optimal_block_size(d, vec_size);
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    constexpr uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block = get_optimal_block_size(d, vec_size);
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}

#if USE_ROCM
void gelu_quick(at::Tensor& out, const at::Tensor& input) {
  int d = input.size(-1);
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    constexpr uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block = get_optimal_block_size(d, vec_size);
    sgl_hip::activation::act_only_kernel<c_type, gelu_quick_act>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
    return true;
  });
}
#endif
