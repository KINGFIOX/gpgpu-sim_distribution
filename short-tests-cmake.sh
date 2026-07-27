#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${GPGPUSIM_BUILD_DIR:-"${repo_dir}/build"}
cuda_samples=${GPGPUSIM_ENABLE_CUDA_SAMPLES:-OFF}

cmake -S "${repo_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGPGPUSIM_ENABLE_CUDA_SAMPLES="${cuda_samples}"
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
