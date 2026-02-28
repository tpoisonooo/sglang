#!/bin/bash
set -ex

# HTTP Proxy configuration for build
HTTP_PROXY="http://10.15.1.122:7890"
HTTPS_PROXY="http://10.15.1.122:7890"
NO_PROXY="localhost,127.0.0.1,.local"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"          # e.g. 3.10
CUDA_VERSION="$2"            # e.g. 12.9
ARCH="${3:-$(uname -i)}"     # optional override

if [ "${ARCH}" = "aarch64" ]; then
  BASE_IMG="pytorch/manylinuxaarch64-builder"
else
  BASE_IMG="pytorch/manylinux2_28-builder"
fi

# Create cache directories for persistent build artifacts in home directory
# Using home directory to persist across workspace cleanups/checkouts
CACHE_DIR="${HOME}/.cache/sgl-kernel"
BUILDX_CACHE_DIR="${CACHE_DIR}/buildx"
mkdir -p "${BUILDX_CACHE_DIR}"

# Ensure a buildx builder
# Use docker driver if USE_LOCAL_DOCKER_IMAGES is set (allows using locally built images)
# Use docker-container driver for better cache export support
BUILDER_NAME="sgl-kernel-builder"
DRIVER_TYPE="${USE_LOCAL_DOCKER_IMAGES:+docker}"
DRIVER_TYPE="${DRIVER_TYPE:-docker-container}"

if [ "${DRIVER_TYPE}" = "docker" ]; then
  # docker driver is the default, use default builder
  echo "Using default docker driver (local images supported)"
  BUILDER_NAME="default"
  docker buildx use default
elif ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --driver "${DRIVER_TYPE}" --use --bootstrap
else
  docker buildx use "${BUILDER_NAME}"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"

# Clean up local build directory to avoid CMake cache conflicts with Docker build
# Docker builds in /sgl-kernel but host may have build/ with different paths
if [ -d "build" ]; then
  echo "Removing local build/ directory to avoid CMake cache conflicts..."
  rm -rf build
fi

echo "----------------------------------------"
echo "Build configuration"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "PYTHON_TAG:     ${PY_TAG}"
echo "Output:         ${DIST_DIR}/"
echo "Buildx cache:   ${BUILDX_CACHE_DIR}"
echo "Builder:        ${BUILDER_NAME}"
echo "----------------------------------------"

BUILD_ARGS=()
# Proxy settings for Dockerfile
BUILD_ARGS+=(--build-arg http_proxy="${HTTP_PROXY}")
BUILD_ARGS+=(--build-arg https_proxy="${HTTPS_PROXY}")
BUILD_ARGS+=(--build-arg no_proxy="${NO_PROXY}")
# Optional profiling build-args (empty string disables)
[ -n "${ENABLE_CMAKE_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_CMAKE_PROFILE="${ENABLE_CMAKE_PROFILE}")
[ -n "${ENABLE_BUILD_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_BUILD_PROFILE="${ENABLE_BUILD_PROFILE}")
# Optional extra cmake build-args (empty string disables)
[ -n "${CMAKE_EXTRA_ARGS:-}" ] && BUILD_ARGS+=(--build-arg CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS}")

# Build cache args (docker driver doesn't support --cache-to)
CACHE_ARGS=(--cache-from "type=local,src=${BUILDX_CACHE_DIR}")
if [ "${DRIVER_TYPE}" = "docker-container" ]; then
  CACHE_ARGS+=(--cache-to "type=local,dest=${BUILDX_CACHE_DIR},mode=max")
fi

docker buildx build \
  --builder "${BUILDER_NAME}" \
  -f Dockerfile . \
  --build-arg BASE_IMG="${BASE_IMG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ARCH="${ARCH}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg PYTHON_TAG="${PY_TAG}" \
  "${BUILD_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  --target artifact \
  --output "type=local,dest=${DIST_DIR}" \
  --network=host

echo "Done. Wheels are in ${DIST_DIR}/"
