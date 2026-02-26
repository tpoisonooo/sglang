/*
 * Copyright (c) 2023 by FlashInfer team.
 * Copyright (c) 2025 SGLang Team.
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
#ifndef SGL_POS_ENC_CUH_
#define SGL_POS_ENC_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <flashinfer/pos_enc.cuh>  // upstream

namespace flashinfer {

namespace kv_buffer_saver {

template <typename DType, typename IdType, uint32_t vec_size>
__device__ __forceinline__ void prepare(
    vec_t<float, vec_size>& v_vec,
    IdType& kv_cache_offset,
    DType* v,
    IdType* kv_cache_loc,
    uint32_t idx,
    uint32_t tx,
    uint32_t kv_head_idx,
    size_t v_stride_n,
    size_t v_stride_h) {
  kv_cache_offset = kv_cache_loc[idx];

  DType* v_ptr = v + get_elem_offset_impl(idx, kv_head_idx, 0, v_stride_n, v_stride_h);
  v_vec.cast_load(v_ptr + tx * vec_size);
}

template <typename DType, typename IdType, uint32_t vec_size>
__device__ __forceinline__ void save(
    IdType& kv_cache_offset,
    vec_t<float, vec_size>& k_vec,
    vec_t<float, vec_size>& v_vec,
    DType* k_buffer,
    DType* v_buffer,
    uint32_t idx,
    uint32_t tx,
    uint32_t kv_head_idx,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h) {
  DType* k_buffer_ptr =
      k_buffer + get_elem_offset_impl(kv_cache_offset, kv_head_idx, 0, k_buffer_stride_n, k_buffer_stride_h);
  DType* v_buffer_ptr =
      v_buffer + get_elem_offset_impl(kv_cache_offset, kv_head_idx, 0, v_buffer_stride_n, v_buffer_stride_h);
  k_vec.cast_store(k_buffer_ptr + tx * vec_size);
  v_vec.cast_store(v_buffer_ptr + tx * vec_size);
}

}  // namespace kv_buffer_saver

// ============================================================================
// BF16/FP16 Native Operations (avoid FP32 conversion)
// ============================================================================

namespace optimized {

/**
 * @brief Apply RoPE using native BF16 operations without FP32 conversion
 * Non-interleave mode (neox style)
 */
template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ vec_t<nv_bfloat16, vec_size> vec_apply_llama_rope_cos_sin_bf16_native(
    const nv_bfloat16* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<nv_bfloat16, vec_size> vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    // Load permuted elements (x[i + rotary_dim/2] or x[i - rotary_dim/2])
    const uint32_t permute_offset = (threadIdx.x * vec_size < rotary_dim / 2)
        ? threadIdx.x * vec_size + rotary_dim / 2
        : threadIdx.x * vec_size - rotary_dim / 2;

    vec_t<nv_bfloat16, vec_size> permuted_vec;
    permuted_vec.cast_load(x + permute_offset);

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      nv_bfloat16 cos_bf16 = __float2bfloat16(cos[i]);
      nv_bfloat16 sin_bf16 = __float2bfloat16(sin[i]);

      // x * cos
      nv_bfloat16 result = __hmul(vec[i], cos_bf16);

      // + permuted * sin (with sign flip for first half)
      nv_bfloat16 permuted_val = permuted_vec[i];
      if (threadIdx.x * vec_size < rotary_dim / 2) {
        permuted_val = __hneg(permuted_val);
      }
      result = __hfma(permuted_val, sin_bf16, result);

      vec[i] = result;
    }
  }
  return vec;
}

/**
 * @brief Optimized interleave mode with BF16 native operations
 */
template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ vec_t<nv_bfloat16, vec_size>
vec_apply_llama_rope_cos_sin_interleave_bf16_native(
    const nv_bfloat16* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<nv_bfloat16, vec_size> vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    // Save original values before modification
    vec_t<nv_bfloat16, vec_size> vec_before = vec;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      // Use cos[i/2] and sin[i/2] like the original implementation
      nv_bfloat16 cos_bf16 = __float2bfloat16(cos[i / 2]);
      nv_bfloat16 sin_bf16 = __float2bfloat16(sin[i / 2]);

      nv_bfloat16 paired_val = vec_before[i ^ 1];  // Get paired element
      if (i % 2 == 0) {
        // Even index: x * cos - paired * sin
        paired_val = __hneg(paired_val);
      }
      // Odd index: x * cos + paired * sin
      vec[i] = __hfma(paired_val, sin_bf16, __hmul(vec[i], cos_bf16));
    }
  }
  return vec;
}

/**
 * @brief Apply RoPE using native FP16 operations without FP32 conversion
 * Non-interleave mode (neox style)
 */
template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ vec_t<half, vec_size> vec_apply_llama_rope_cos_sin_fp16_native(
    const half* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<half, vec_size> vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    const uint32_t permute_offset = (threadIdx.x * vec_size < rotary_dim / 2)
        ? threadIdx.x * vec_size + rotary_dim / 2
        : threadIdx.x * vec_size - rotary_dim / 2;

    vec_t<half, vec_size> permuted_vec;
    permuted_vec.cast_load(x + permute_offset);

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      half cos_fp16 = __float2half(cos[i]);
      half sin_fp16 = __float2half(sin[i]);

      half result = __hmul(vec[i], cos_fp16);
      half permuted_val = permuted_vec[i];
      if (threadIdx.x * vec_size < rotary_dim / 2) {
        permuted_val = __hneg(permuted_val);
      }
      result = __hfma(permuted_val, sin_fp16, result);

      vec[i] = result;
    }
  }
  return vec;
}

/**
 * @brief Optimized interleave mode with FP16 native operations
 */
template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ vec_t<half, vec_size>
vec_apply_llama_rope_cos_sin_interleave_fp16_native(
    const half* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<half, vec_size> vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_t<half, vec_size> vec_before = vec;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      half cos_fp16 = __float2half(cos[i / 2]);
      half sin_fp16 = __float2half(sin[i / 2]);

      half paired_val = vec_before[i ^ 1];
      if (i % 2 == 0) {
        paired_val = __hneg(paired_val);
      }
      vec[i] = __hfma(paired_val, sin_fp16, __hmul(vec[i], cos_fp16));
    }
  }
  return vec;
}

/**
 * @brief Optimized kernel with native BF16/FP16 support and __ldg for cos_sin_cache
 */
template <
    bool save_kv_cache,
    bool interleave,
    uint32_t head_dim,
    uint32_t vec_size,
    uint32_t bdx,
    typename DType,
    typename IdType>
__global__ void __launch_bounds__(256) BatchQKApplyRotaryPosIdsCosSinCacheOptimizedKernel(
    DType* __restrict__ q,
    DType* __restrict__ k,
    DType* __restrict__ v,
    DType* __restrict__ q_rope,
    DType* __restrict__ k_rope,
    DType* __restrict__ k_buffer,
    DType* __restrict__ v_buffer,
    const float* __restrict__ cos_sin_cache,
    const IdType* __restrict__ pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    const IdType* __restrict__ kv_cache_loc) {
  uint32_t bx = blockIdx.x;
  uint32_t tx = threadIdx.x;
  uint32_t ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    // Load cos/sin with __ldg for read-only cache
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;
      }

      // Use __ldg for read-only cos_sin_cache
      const float* cos_ptr = cos_sin_cache + (pos * rotary_dim) + vec_idx;
      const float* sin_ptr = cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx);

#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        cos[i] = __ldg(cos_ptr + i);
        sin[i] = __ldg(sin_ptr + i);
      }
    }

    // Process Q heads
#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);

      if constexpr (std::is_same_v<DType, nv_bfloat16>) {
        // BF16 native path
        vec_t<nv_bfloat16, vec_size> q_vec;
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_cos_sin_interleave_bf16_native<vec_size, bdx>(
              reinterpret_cast<nv_bfloat16*>(q_ptr), cos, sin, rotary_dim);
        } else {
          q_vec = vec_apply_llama_rope_cos_sin_bf16_native<vec_size, bdx>(
              reinterpret_cast<nv_bfloat16*>(q_ptr), cos, sin, rotary_dim);
        }
        q_vec.cast_store(reinterpret_cast<nv_bfloat16*>(q_rope_ptr) + tx * vec_size);
      } else if constexpr (std::is_same_v<DType, half>) {
        // FP16 native path
        vec_t<half, vec_size> q_vec;
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_cos_sin_interleave_fp16_native<vec_size, bdx>(
              reinterpret_cast<half*>(q_ptr), cos, sin, rotary_dim);
        } else {
          q_vec = vec_apply_llama_rope_cos_sin_fp16_native<vec_size, bdx>(
              reinterpret_cast<half*>(q_ptr), cos, sin, rotary_dim);
        }
        q_vec.cast_store(reinterpret_cast<half*>(q_rope_ptr) + tx * vec_size);
      } else {
        // Fallback to FP32 path
        vec_t<float, vec_size> q_vec;
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
        } else {
          q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
        }
        q_vec.cast_store(q_rope_ptr + tx * vec_size);
      }
    }

    // Process K heads
#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);

      // Load V for KV cache if needed
      vec_t<float, vec_size> v_vec_float;
      IdType kv_cache_offset;
      if constexpr (save_kv_cache) {
        kv_cache_offset = kv_cache_loc[idx];
        DType* v_ptr = v + get_elem_offset_impl(idx, kv_head_idx, 0, v_stride_n, v_stride_h);
        v_vec_float.cast_load(v_ptr + tx * vec_size);
      }

      if constexpr (std::is_same_v<DType, nv_bfloat16>) {
        // BF16 native path
        vec_t<nv_bfloat16, vec_size> k_vec;
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_cos_sin_interleave_bf16_native<vec_size, bdx>(
              reinterpret_cast<nv_bfloat16*>(k_ptr), cos, sin, rotary_dim);
        } else {
          k_vec = vec_apply_llama_rope_cos_sin_bf16_native<vec_size, bdx>(
              reinterpret_cast<nv_bfloat16*>(k_ptr), cos, sin, rotary_dim);
        }
        k_vec.cast_store(reinterpret_cast<nv_bfloat16*>(k_rope_ptr) + tx * vec_size);

        if constexpr (save_kv_cache) {
          // Store K and V to buffer
          DType* k_buffer_ptr = k_buffer + get_elem_offset_impl(
              kv_cache_offset, kv_head_idx, 0, k_buffer_stride_n, k_buffer_stride_h);
          DType* v_buffer_ptr = v_buffer + get_elem_offset_impl(
              kv_cache_offset, kv_head_idx, 0, v_buffer_stride_n, v_buffer_stride_h);

          k_vec.cast_store(reinterpret_cast<nv_bfloat16*>(k_buffer_ptr) + tx * vec_size);

          vec_t<nv_bfloat16, vec_size> v_vec_bf16;
          v_vec_bf16.cast_from(v_vec_float);
          v_vec_bf16.cast_store(reinterpret_cast<nv_bfloat16*>(v_buffer_ptr) + tx * vec_size);
        }
      } else if constexpr (std::is_same_v<DType, half>) {
        // FP16 native path
        vec_t<half, vec_size> k_vec;
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_cos_sin_interleave_fp16_native<vec_size, bdx>(
              reinterpret_cast<half*>(k_ptr), cos, sin, rotary_dim);
        } else {
          k_vec = vec_apply_llama_rope_cos_sin_fp16_native<vec_size, bdx>(
              reinterpret_cast<half*>(k_ptr), cos, sin, rotary_dim);
        }
        k_vec.cast_store(reinterpret_cast<half*>(k_rope_ptr) + tx * vec_size);

        if constexpr (save_kv_cache) {
          DType* k_buffer_ptr = k_buffer + get_elem_offset_impl(
              kv_cache_offset, kv_head_idx, 0, k_buffer_stride_n, k_buffer_stride_h);
          DType* v_buffer_ptr = v_buffer + get_elem_offset_impl(
              kv_cache_offset, kv_head_idx, 0, v_buffer_stride_n, v_buffer_stride_h);

          k_vec.cast_store(reinterpret_cast<half*>(k_buffer_ptr) + tx * vec_size);

          vec_t<half, vec_size> v_vec_fp16;
          v_vec_fp16.cast_from(v_vec_float);
          v_vec_fp16.cast_store(reinterpret_cast<half*>(v_buffer_ptr) + tx * vec_size);
        }
      } else {
        // Fallback to FP32 path
        vec_t<float, vec_size> k_vec;
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
        } else {
          k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
        }
        k_vec.cast_store(k_rope_ptr + tx * vec_size);

        if constexpr (save_kv_cache) {
          DType* k_buffer_ptr = k_buffer + get_elem_offset_impl(
              kv_cache_offset, kv_head_idx, 0, k_buffer_stride_n, k_buffer_stride_h);
          DType* v_buffer_ptr = v_buffer + get_elem_offset_impl(
              kv_cache_offset, kv_head_idx, 0, v_buffer_stride_n, v_buffer_stride_h);
          k_vec.cast_store(k_buffer_ptr + tx * vec_size);
          v_vec_float.cast_store(v_buffer_ptr + tx * vec_size);
        }
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace optimized

// ============================================================================
// Original Kernels (kept for backward compatibility)
// ============================================================================

template <
    bool save_kv_cache,
    bool interleave,
    uint32_t head_dim,
    uint32_t vec_size,
    uint32_t bdx,
    typename DType,
    typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheEnhancedHeadParallelismKernel(
    DType* q,
    DType* k,
    DType* v,
    DType* q_rope,
    DType* k_rope,
    DType* k_buffer,
    DType* v_buffer,
    float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    IdType* __restrict__ kv_cache_loc) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);

      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);

      vec_t<float, vec_size> v_vec;
      IdType kv_cache_offset;
      if constexpr (save_kv_cache) {
        kv_buffer_saver::prepare<DType, IdType, vec_size>(
            v_vec, kv_cache_offset, v, kv_cache_loc, idx, tx, kv_head_idx, v_stride_n, v_stride_h);
      }

      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);

      if constexpr (save_kv_cache) {
        kv_buffer_saver::save<DType, IdType, vec_size>(
            kv_cache_offset,
            k_vec,
            v_vec,
            k_buffer,
            v_buffer,
            idx,
            tx,
            kv_head_idx,
            k_buffer_stride_n,
            k_buffer_stride_h,
            v_buffer_stride_n,
            v_buffer_stride_h);
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <
    bool save_kv_cache,
    bool interleave,
    uint32_t head_dim,
    uint32_t vec_size,
    uint32_t bdx,
    typename DType,
    typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel(
    DType* q,
    DType* k,
    DType* v,
    DType* q_rope,
    DType* k_rope,
    DType* k_buffer,
    DType* v_buffer,
    float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    IdType* __restrict__ kv_cache_loc) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    // not to unroll the loop, because num head might be large and might lead to worse performance
#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);

      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);

      vec_t<float, vec_size> v_vec;
      IdType kv_cache_offset;
      if constexpr (save_kv_cache) {
        kv_buffer_saver::prepare<DType, IdType, vec_size>(
            v_vec, kv_cache_offset, v, kv_cache_loc, idx, tx, kv_head_idx, v_stride_n, v_stride_h);
      }

      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);

      if constexpr (save_kv_cache) {
        kv_buffer_saver::save<DType, IdType, vec_size>(
            kv_cache_offset,
            k_vec,
            v_vec,
            k_buffer,
            v_buffer,
            idx,
            tx,
            kv_head_idx,
            k_buffer_stride_n,
            k_buffer_stride_h,
            v_buffer_stride_n,
            v_buffer_stride_h);
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

#define DISPATCH_SAVE_KV_CACHE(save_kv_cache, SAVE_KV_CACHE, ...) \
  if (save_kv_cache) {                                            \
    const bool SAVE_KV_CACHE = true;                              \
    __VA_ARGS__                                                   \
  } else {                                                        \
    const bool SAVE_KV_CACHE = false;                             \
    __VA_ARGS__                                                   \
  }

/**
 * @brief Enhanced RoPE kernel with optional BF16/FP16 native support
 * 
 * For BF16 and FP16 data types, uses optimized native operations.
 * For FP32, uses the original implementation.
 */
template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIdsCosSinCacheEnhanced(
    DType* q,
    DType* k,
    DType* v,
    DType* q_rope,
    DType* k_rope,
    DType* k_buffer,
    DType* v_buffer,
    float* cos_sin_cache,
    IdType* pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    uint32_t head_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    IdType* kv_cache_loc,
    bool interleave,
    bool save_kv_cache,
    bool enable_pdl,
    cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

#define LAUNCH_KERNEL_RAW(kernel_name)                                \
  do {                                                                \
    cudaLaunchConfig_t config = {};                                   \
    config.gridDim = nblks;                                           \
    config.blockDim = nthrs;                                          \
    config.dynamicSmemBytes = 0;                                      \
    config.stream = stream;                                           \
    cudaLaunchAttribute attrs[1] = {};                                \
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl; \
    config.numAttrs = 1;                                              \
    config.attrs = attrs;                                             \
                                                                      \
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                          \
        &config,                                                      \
        kernel_name,                                                  \
        q,                                                            \
        k,                                                            \
        v,                                                            \
        q_rope,                                                       \
        k_rope,                                                       \
        k_buffer,                                                     \
        v_buffer,                                                     \
        cos_sin_cache,                                                \
        pos_ids,                                                      \
        nnz,                                                          \
        num_qo_heads,                                                 \
        num_kv_heads,                                                 \
        rotary_dim,                                                   \
        q_stride_n,                                                   \
        q_stride_h,                                                   \
        k_stride_n,                                                   \
        k_stride_h,                                                   \
        v_stride_n,                                                   \
        v_stride_h,                                                   \
        q_rope_stride_n,                                              \
        q_rope_stride_h,                                              \
        k_rope_stride_n,                                              \
        k_rope_stride_h,                                              \
        k_buffer_stride_n,                                            \
        k_buffer_stride_h,                                            \
        v_buffer_stride_n,                                            \
        v_buffer_stride_h,                                            \
        kv_cache_loc));                                               \
  } while (0)

  DISPATCH_SAVE_KV_CACHE(save_kv_cache, SAVE_KV_CACHE, {
    DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        // operate on 16 Bytes at a time
        constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
        // how many threads needed per head_dim
        constexpr uint32_t bdx = HEAD_DIM / vec_size;
        // how many threads needed per block
        uint32_t num_threads = std::max(128U, bdx);
        // how many tokens can we process in a block
        uint32_t bdy = num_threads / bdx;
        // how many blocks needed to process all tokens
        uint32_t nblks_x = (nnz + bdy - 1) / bdy;

        // Use optimized kernel for BF16 and FP16
        if constexpr (std::is_same_v<DType, nv_bfloat16> || std::is_same_v<DType, half>) {
          // Use fixed 256 threads for better occupancy with low-precision types
          constexpr uint32_t opt_num_threads = 256;
          constexpr uint32_t opt_bdy = opt_num_threads / bdx;
          uint32_t opt_nblks_x = (nnz + opt_bdy - 1) / opt_bdy;

          auto kernel_0 = optimized::BatchQKApplyRotaryPosIdsCosSinCacheOptimizedKernel<
              SAVE_KV_CACHE,
              INTERLEAVE,
              HEAD_DIM,
              vec_size,
              bdx,
              DType,
              IdType>;

          int num_blocks_per_sm_0 = 0;
          FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &num_blocks_per_sm_0, kernel_0, opt_num_threads, /*smem_size=*/0));
          uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

          if ((nnz + opt_bdy - 1) / opt_bdy >= num_ctas_0) {
            dim3 nblks(opt_nblks_x);
            dim3 nthrs(bdx, opt_bdy);
            LAUNCH_KERNEL_RAW(kernel_0);
          } else {
            // For small batch, use head parallelism
            dim3 nblks(opt_nblks_x, num_qo_heads + num_kv_heads);
            dim3 nthrs(bdx, opt_bdy);
            LAUNCH_KERNEL_RAW(kernel_0);
          }
        } else {
          // Use original kernel for FP32
          auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel<
              SAVE_KV_CACHE,
              INTERLEAVE,
              HEAD_DIM,
              vec_size,
              bdx,
              DType,
              IdType>;

          int num_blocks_per_sm_0 = 0;
          FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
          uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

          if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
            dim3 nblks(nblks_x);
            dim3 nthrs(bdx, bdy);
            LAUNCH_KERNEL_RAW(kernel_0);
          } else {
            dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
            dim3 nthrs(bdx, bdy);
            auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheEnhancedHeadParallelismKernel<
                SAVE_KV_CACHE,
                INTERLEAVE,
                HEAD_DIM,
                vec_size,
                bdx,
                DType,
                IdType>;
            LAUNCH_KERNEL_RAW(kernel_1);
          }
        }
      });
    });
  });
#undef LAUNCH_KERNEL_RAW

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // SGL_POS_ENC_CUH_
