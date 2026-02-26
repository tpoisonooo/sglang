#!/bin/bash
set -e

# Get the directory where this script resides (i.e., the sglang repo root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/sglang_minicpm_sala_env"
DEPS_DIR="${REPO_ROOT}/3rdparty"

# PyPI mirror: prefer CLI argument, then env var, default to official source
if [ -n "$1" ]; then
    export UV_INDEX_URL="$1"
elif [ -z "${UV_INDEX_URL}" ]; then
    export UV_INDEX_URL="https://pypi.org/simple"
fi

echo "============================================"
echo " MiniCPM-SALA Installation (uv)"
echo "============================================"
echo "Root Directory: ${REPO_ROOT}"
echo "PyPI mirror:    ${UV_INDEX_URL}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first (e.g., pip install uv)."
    exit 1
fi

# ---- Prepare Dependencies (via git submodule) ----
echo "[0/4] Initializing submodules (infllmv2_cuda_impl, sparse_kernel)..."
git submodule update --init --recursive

# ---- Ensure uv-managed Python ----
REQUIRED_PY="3.12"
echo "[1/4] Ensuring Python ${REQUIRED_PY} is managed by uv..."
uv python install "${REQUIRED_PY}"
UV_PYTHON=$(uv python find --python-preference only-managed "${REQUIRED_PY}")
echo "  uv-managed Python: ${UV_PYTHON} ($(${UV_PYTHON} --version))"

# ---- Create venv ----
if [ -d "${VENV_DIR}" ]; then
    VENV_PY_VER=$("${VENV_DIR}/bin/python" --version 2>&1 | awk '{print $2}')
    if [[ "${VENV_PY_VER}" != ${REQUIRED_PY}.* ]]; then
        echo "  venv exists but Python version is ${VENV_PY_VER} (expected ${REQUIRED_PY}.x), recreating..."
        rm -rf "${VENV_DIR}"
        uv venv --python "${UV_PYTHON}" "${VENV_DIR}"
    else
        echo "  venv already exists (Python ${VENV_PY_VER}), skipping"
    fi
else
    echo "  Creating virtual environment..."
    uv venv --python "${UV_PYTHON}" "${VENV_DIR}"
fi

# Activate environment variables for the script execution
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:$PATH"
echo "Python: $(python --version)"

# ---- Prepare build environment ----
# Fix compiler (python-build-standalone sets CXX="clang++ -pthread", incompatible with CMake)
if command -v g++ &> /dev/null; then
    export CC=gcc CXX=g++
fi
# Ensure nvcc is in PATH
if [ -z "${CUDA_HOME}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME="/usr/local/cuda"
    fi
fi
if [ -n "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

# ---- Install Packages ----

# Install sglang from the repo root
echo "[2/4] Installing sglang (current directory)..."
uv pip install "cmake>=3.26"
uv pip install --upgrade pip setuptools wheel

# Build and install sgl-kernel from source (local changes)
echo "  - Building sgl-kernel from source..."
cd "${REPO_ROOT}/sgl-kernel"
# Force rebuild by cleaning first
rm -rf build dist *.egg-info _skbuild
# Install build dependencies first
uv pip install scikit-build-core torch wheel
# Set MAX_JOBS for faster compilation (use all CPU cores)
export MAX_JOBS=8
echo "    Using MAX_JOBS=${MAX_JOBS} for compilation"
uv pip install -e . --no-build-isolation

# Install sglang
cd "${REPO_ROOT}"
uv pip install -e "${REPO_ROOT}/python[all]"

# Build and install CUDA kernel dependencies
echo "[3/4] Building CUDA kernels..."

# infllm_v2 (has its own submodules, e.g. cutlass)
echo "  - Installing infllm_v2..."
cd "${DEPS_DIR}/infllmv2_cuda_impl"
git submodule update --init --recursive
python setup.py install

# sparse_kernel
echo "  - Installing sparse_kernel..."
cd "${DEPS_DIR}/sparse_kernel"
python setup.py install

# Install additional libraries
echo "[4/4] Installing additional libraries..."
uv pip install tilelang flash-linear-attention

# ---- Done ----
echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo "To activate the environment, run:"
echo "source ${VENV_DIR}/bin/activate"
