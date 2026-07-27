#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${GPGPUSIM_BUILD_DIR:-"${repo_dir}/cmake-build-debug"}
cuda_samples=${GPGPUSIM_ENABLE_CUDA_SAMPLES:-ON}

cmake -S "${repo_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGPGPUSIM_ENABLE_CUDA_SAMPLES="${cuda_samples}"
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
