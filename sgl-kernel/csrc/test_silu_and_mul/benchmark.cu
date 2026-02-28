/*
 * Benchmark: Compare original vs optimized silu_and_mul implementations
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

// Function declarations from other files
namespace original {
    void launch_silu_and_mul(
        __nv_bfloat16* out,
        const __nv_bfloat16* input,
        int d,
        int num_tokens,
        cudaStream_t stream);
}

namespace optimized {
    void launch_silu_and_mul_bf162(
        __nv_bfloat16* out,
        const __nv_bfloat16* input,
        int d,
        int num_tokens,
        cudaStream_t stream);
}

// Utility: Check CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Utility: Initialize data
void init_data(__nv_bfloat16* data, size_t n) {
    std::vector<float> tmp(n);
    for (size_t i = 0; i < n; ++i) {
        // Random values in range [-2, 2]
        tmp[i] = (float(rand()) / RAND_MAX) * 4.0f - 2.0f;
    }
    for (size_t i = 0; i < n; ++i) {
        data[i] = __float2bfloat16(tmp[i]);
    }
}

// Utility: Check results match
bool check_results(const __nv_bfloat16* out1, const __nv_bfloat16* out2, size_t n, float tol = 1e-2f) {
    std::vector<__nv_bfloat16> h_out1(n);
    std::vector<__nv_bfloat16> h_out2(n);
    CHECK_CUDA(cudaMemcpy(h_out1.data(), out1, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out2.data(), out2, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    
    int mismatch_count = 0;
    for (size_t i = 0; i < n; ++i) {
        float f1 = __bfloat162float(h_out1[i]);
        float f2 = __bfloat162float(h_out2[i]);
        float rel_err = fabsf(f1 - f2) / (fabsf(f1) + 1e-6f);
        if (rel_err > tol && fabsf(f1 - f2) > tol) {
            if (mismatch_count < 5) {
                printf("  Mismatch at %zu: %f vs %f (rel_err=%f)\n", i, f1, f2, rel_err);
            }
            mismatch_count++;
        }
    }
    if (mismatch_count > 0) {
        printf("  Total mismatches: %d / %zu\n", mismatch_count, n);
    }
    return mismatch_count == 0;
}

// Benchmark function
template<typename Func>
float benchmark(Func&& func, int warmup_iters, int iters, cudaStream_t stream) {
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        func();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        func();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return elapsed_ms / iters;  // Average time per iteration
}

void run_benchmark(int num_tokens, int d, int warmup_iters = 10, int iters = 100) {
    printf("\n========================================\n");
    printf("Benchmark: num_tokens=%d, d=%d\n", num_tokens, d);
    printf("Input shape: [%d, %d] (up_proj: [%d, %d])\n", num_tokens, 2*d, num_tokens, d);
    printf("========================================\n");
    
    size_t input_size = (size_t)num_tokens * 2 * d;
    size_t output_size = (size_t)num_tokens * d;
    
    // Allocate device memory
    __nv_bfloat16 *d_input, *d_out_orig, *d_out_opt;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out_orig, output_size * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out_opt, output_size * sizeof(__nv_bfloat16)));
    
    // Initialize input data
    std::vector<__nv_bfloat16> h_input(input_size);
    init_data(h_input.data(), input_size);
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Verify correctness first
    printf("\nVerifying correctness...\n");
    original::launch_silu_and_mul(d_out_orig, d_input, d, num_tokens, stream);
    optimized::launch_silu_and_mul_bf162(d_out_opt, d_input, d, num_tokens, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    bool correct = check_results(d_out_orig, d_out_opt, output_size);
    printf("Correctness check: %s\n", correct ? "PASSED" : "FAILED");
    
    if (!correct) {
        printf("WARNING: Results don't match! Benchmark may not be meaningful.\n");
    }
    
    // Benchmark original implementation
    printf("\nBenchmarking original implementation...\n");
    float time_orig = benchmark([&]() {
        original::launch_silu_and_mul(d_out_orig, d_input, d, num_tokens, stream);
    }, warmup_iters, iters, stream);
    
    // Benchmark optimized implementation
    printf("Benchmarking optimized (__nv_bfloat162) implementation...\n");
    float time_opt = benchmark([&]() {
        optimized::launch_silu_and_mul_bf162(d_out_opt, d_input, d, num_tokens, stream);
    }, warmup_iters, iters, stream);
    
    // Calculate metrics
    // Memory traffic: read input (2*d per token) + write output (d per token) = 3*d elements
    // Each element is 2 bytes (bf16)
    double mem_bytes = (double)num_tokens * d * 3 * sizeof(__nv_bfloat16);
    double bandwidth_orig = (mem_bytes / (time_orig * 1e-3)) / 1e9;  // GB/s
    double bandwidth_opt = (mem_bytes / (time_opt * 1e-3)) / 1e9;    // GB/s
    double speedup = time_orig / time_opt;
    
    printf("\nResults:\n");
    printf("  Original:  %.3f ms (%.2f GB/s)\n", time_orig, bandwidth_orig);
    printf("  Optimized: %.3f ms (%.2f GB/s)\n", time_opt, bandwidth_opt);
    printf("  Speedup:   %.2fx\n", speedup);
    
    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_out_orig));
    CHECK_CUDA(cudaFree(d_out_opt));
}

int main() {
    // Check GPU properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    double theoretical_bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", theoretical_bw);
    
    // Test cases
    std::vector<std::pair<int, int>> test_cases = {
        // {num_tokens, d}
        {1, 32768},
        {4, 32768},
        {8, 32768},
        {16, 32768},
        {32, 32768},
        {64, 32768},
        {128, 32768},
        {256, 32768},
        {512, 32768},
        {1024, 32768},
        {2048, 32768},
        {4096, 32768},
    };
    
    for (const auto& [num_tokens, d] : test_cases) {
        run_benchmark(num_tokens, d, 20, 100);
    }
    
    printf("\n========================================\n");
    printf("All benchmarks completed!\n");
    printf("========================================\n");
    
    return 0;
}
