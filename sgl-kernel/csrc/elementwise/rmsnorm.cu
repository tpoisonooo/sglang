/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "utils.h"

// PDL (Programmatic Dependent Launch) support for Hopper (SM90+)
// Automatically disabled for older architectures
#ifndef ENABLE_HOPPER_PDL
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define ENABLE_HOPPER_PDL 1
#else
#define ENABLE_HOPPER_PDL 0
#endif
#endif

namespace sgl_kernel {

// Type conversion utilities
template <typename T>
__device__ __forceinline__ float to_float_t(T x);

template <>
__device__ __forceinline__ float to_float_t<half>(half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float_t<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <>
__device__ __forceinline__ float to_float_t<float>(float x) {
    return x;
}

template <typename T>
__device__ __forceinline__ T from_float_t(float x);

template <>
__device__ __forceinline__ half from_float_t<half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float_t<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

template <>
__device__ __forceinline__ float from_float_t<float>(float x) {
    return x;
}

// Vector type helpers
template <typename T, int VecSize>
struct VecType {
    using Type = T;
};

template <> struct VecType<half, 2> { using Type = half2; };
template <> struct VecType<__nv_bfloat16, 2> { using Type = __nv_bfloat162; };
template <> struct VecType<float, 2> { using Type = float2; };
template <> struct VecType<half, 4> { using Type = uint2; };
template <> struct VecType<__nv_bfloat16, 4> { using Type = uint2; };
template <> struct VecType<float, 4> { using Type = float4; };
template <> struct VecType<half, 8> { using Type = uint4; };
template <> struct VecType<__nv_bfloat16, 8> { using Type = uint4; };

// Warp shuffle reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction using shared memory
template <int MaxWarps = 32>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane_id < blockDim.x / 32) ? shared_mem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// Kernel for large hidden_size (> 512)
template <typename T, int VecSize>
__global__ void __launch_bounds__(1024) rmsnorm_kernel_large(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const int hidden_size,
    const int batch_size,
    const float eps) {
    
    using VecT = typename VecType<T, VecSize>::Type;
    
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // PDL: wait for previous kernel (Hopper only)
    #if ENABLE_HOPPER_PDL
    asm volatile("griddepcontrol.wait;");
    #endif
    
    extern __shared__ float shared_mem[];
    
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;
    
    float sum_sq = 0.0f;
    
    const int vec_hidden_size = hidden_size / VecSize;
    const VecT* vec_input = reinterpret_cast<const VecT*>(row_input);
    
    #pragma unroll 4
    for (int i = tid; i < vec_hidden_size; i += num_threads) {
        VecT vec = vec_input[i];
        T* ptr = reinterpret_cast<T*>(&vec);
        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float val = to_float_t<T>(ptr[j]);
            sum_sq += val * val;
        }
    }
    
    const int remainder_start = vec_hidden_size * VecSize;
    for (int i = remainder_start + tid; i < hidden_size; i += num_threads) {
        float val = to_float_t<T>(row_input[i]);
        sum_sq += val * val;
    }
    
    sum_sq = block_reduce_sum(sum_sq, shared_mem);
    
    if (threadIdx.x == 0) {
        shared_mem[0] = rsqrtf(sum_sq / float(hidden_size) + eps);
    }
    __syncthreads();
    float rms_rcp = shared_mem[0];
    
    VecT* vec_output = reinterpret_cast<VecT*>(row_output);
    const VecT* vec_weight = reinterpret_cast<const VecT*>(weight);
    
    #pragma unroll 4
    for (int i = tid; i < vec_hidden_size; i += num_threads) {
        VecT in_vec = vec_input[i];
        VecT w_vec = vec_weight[i];
        VecT out_vec;
        
        T* in_ptr = reinterpret_cast<T*>(&in_vec);
        T* w_ptr = reinterpret_cast<T*>(&w_vec);
        T* out_ptr = reinterpret_cast<T*>(&out_vec);
        
        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float in_val = to_float_t<T>(in_ptr[j]);
            float w_val = to_float_t<T>(w_ptr[j]);
            out_ptr[j] = from_float_t<T>(in_val * rms_rcp * w_val);
        }
        
        vec_output[i] = out_vec;
    }
    
    for (int i = remainder_start + tid; i < hidden_size; i += num_threads) {
        float in_val = to_float_t<T>(row_input[i]);
        float w_val = to_float_t<T>(weight[i]);
        row_output[i] = from_float_t<T>(in_val * rms_rcp * w_val);
    }
    
    // PDL: signal next kernel (Hopper only)
    #if ENABLE_HOPPER_PDL
    asm volatile("griddepcontrol.launch_dependents;");
    #endif
}

// Kernel for small hidden_size (<= 512) - warp-specialized
template <typename T, int VecSize>
__global__ void __launch_bounds__(512) rmsnorm_kernel_small(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const int hidden_size,
    const int batch_size,
    const float eps) {
    
    using VecT = typename VecType<T, VecSize>::Type;
    
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;
    
    if (row >= batch_size) return;
    
    // PDL: wait for previous kernel (Hopper only)
    #if ENABLE_HOPPER_PDL
    if (threadIdx.x == 0) asm volatile("griddepcontrol.wait;");
    #endif
    
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;
    
    float sum_sq = 0.0f;
    
    const int vec_hidden_size = hidden_size / VecSize;
    const VecT* vec_input = reinterpret_cast<const VecT*>(row_input);
    
    #pragma unroll
    for (int i = lane_id; i < vec_hidden_size; i += 32) {
        VecT vec = vec_input[i];
        T* ptr = reinterpret_cast<T*>(&vec);
        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float val = to_float_t<T>(ptr[j]);
            sum_sq += val * val;
        }
    }
    
    const int remainder_start = vec_hidden_size * VecSize;
    for (int i = remainder_start + lane_id; i < hidden_size; i += 32) {
        float val = to_float_t<T>(row_input[i]);
        sum_sq += val * val;
    }
    
    sum_sq = warp_reduce_sum(sum_sq);
    float rms_rcp = rsqrtf(sum_sq / float(hidden_size) + eps);
    
    VecT* vec_output = reinterpret_cast<VecT*>(row_output);
    const VecT* vec_weight = reinterpret_cast<const VecT*>(weight);
    
    #pragma unroll
    for (int i = lane_id; i < vec_hidden_size; i += 32) {
        VecT in_vec = vec_input[i];
        VecT w_vec = vec_weight[i];
        VecT out_vec;
        
        T* in_ptr = reinterpret_cast<T*>(&in_vec);
        T* w_ptr = reinterpret_cast<T*>(&w_vec);
        T* out_ptr = reinterpret_cast<T*>(&out_vec);
        
        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float in_val = to_float_t<T>(in_ptr[j]);
            float w_val = to_float_t<T>(w_ptr[j]);
            out_ptr[j] = from_float_t<T>(in_val * rms_rcp * w_val);
        }
        
        vec_output[i] = out_vec;
    }
    
    for (int i = remainder_start + lane_id; i < hidden_size; i += 32) {
        float in_val = to_float_t<T>(row_input[i]);
        float w_val = to_float_t<T>(weight[i]);
        row_output[i] = from_float_t<T>(in_val * rms_rcp * w_val);
    }
    
    // PDL: signal next kernel (Hopper only)
    #if ENABLE_HOPPER_PDL
    if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
    #endif
}

// Launcher function
template <typename T>
cudaError_t launch_rmsnorm(
    T* output,
    const T* input,
    const T* weight,
    int batch_size,
    int hidden_size,
    float eps,
    bool enable_pdl,
    cudaStream_t stream) {
    
    // Determine vector size based on alignment
    const int vec_size = (hidden_size % 8 == 0 && sizeof(T) <= 2) ? 8 :
                         (hidden_size % 4 == 0 && sizeof(T) <= 2) ? 4 :
                         (hidden_size % 2 == 0) ? 2 : 1;
    
    if (hidden_size <= 512) {
        // Use warp-specialized kernel for small hidden_size
        const int warps_per_block = 8;
        const int threads_per_block = warps_per_block * 32;
        const int blocks = (batch_size + warps_per_block - 1) / warps_per_block;
        
        if (vec_size == 8) {
            rmsnorm_kernel_small<T, 8><<<blocks, threads_per_block, 0, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        } else if (vec_size == 4) {
            rmsnorm_kernel_small<T, 4><<<blocks, threads_per_block, 0, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        } else if (vec_size == 2) {
            rmsnorm_kernel_small<T, 2><<<blocks, threads_per_block, 0, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        } else {
            rmsnorm_kernel_small<T, 1><<<blocks, threads_per_block, 0, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        }
    } else {
        // Use standard kernel for large hidden_size
        const int threads_per_block = std::min(1024, ((hidden_size / vec_size + 31) / 32) * 32);
        const int smem_size = (threads_per_block / 32) * sizeof(float);
        
        if (vec_size == 8) {
            rmsnorm_kernel_large<T, 8><<<batch_size, threads_per_block, smem_size, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        } else if (vec_size == 4) {
            rmsnorm_kernel_large<T, 4><<<batch_size, threads_per_block, smem_size, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        } else if (vec_size == 2) {
            rmsnorm_kernel_large<T, 2><<<batch_size, threads_per_block, smem_size, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        } else {
            rmsnorm_kernel_large<T, 1><<<batch_size, threads_per_block, smem_size, stream>>>(
                output, input, weight, hidden_size, batch_size, eps);
        }
    }
    
    return cudaGetLastError();
}

} // namespace sgl_kernel

// PyTorch binding - optimized implementation
void sgl_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl) {
    CHECK_INPUT(output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    auto device = input.device();
    CHECK_EQ(output.device(), device);
    CHECK_EQ(weight.device(), device);
    CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
    CHECK_DIM(2, output);  // output: (batch_size, hidden_size)
    CHECK_DIM(1, weight);  // weight: (hidden_size)
    CHECK_EQ(input.size(0), output.size(0));
    CHECK_EQ(input.size(1), output.size(1));
    CHECK_EQ(input.size(1), weight.size(0));
    
    unsigned int batch_size = input.size(0);
    unsigned int hidden_size = input.size(1);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
        cudaError_t status = sgl_kernel::launch_rmsnorm(
            static_cast<c_type*>(output.data_ptr()),
            static_cast<c_type*>(input.data_ptr()),
            static_cast<c_type*>(weight.data_ptr()),
            batch_size,
            hidden_size,
            static_cast<float>(eps),
            enable_pdl,
            stream);
        TORCH_CHECK(status == cudaSuccess, 
                    "rmsnorm failed with error code " + std::string(cudaGetErrorString(status)));
        return true;
    });
}
