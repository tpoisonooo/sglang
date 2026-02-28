# Makefile for silu_and_mul benchmark

# CUDA settings
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS += -nvvm-path=/usr/local/cuda/nvvm
CUDA_ARCH = -gencode arch=compute_90,code=sm_90  # H800 is Hopper (sm_90)
NVCC_FLAGS = -O3 -std=c++17 $(CUDA_ARCH) \
    -lineinfo \
    --use_fast_math \
    -Xcompiler -fPIC

# Include paths
INCLUDES = -I/usr/local/cuda/include

# Library paths
LIBRARIES = -L/usr/local/cuda/lib64 -lcudart

# Source files
SRCS = silu_and_mul_original.cu silu_and_mul_bf162.cu benchmark.cu
OBJS = $(SRCS:.cu=.o)

# Target executable
TARGET = benchmark_silu_and_mul

# Default target
all: $(TARGET)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Link executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBRARIES)

# Run benchmark
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Debug build
debug: NVCC_FLAGS = -g -G -O0 -std=c++17 $(CUDA_ARCH) -Xcompiler -fPIC
debug: clean all

# Profile with nsys
profile: $(TARGET)
	nsys profile -o silu_and_mul_profile ./$(TARGET)

# Print CUDA info
info:
	@echo "NVCC: $(shell which $(NVCC))"
	@echo "NVCC version: $(shell $(NVCC) --version | grep release)"
	@echo "Target architecture: sm_90 (Hopper/H800)"

.PHONY: all clean run debug profile info
