# GPGPU-Sim CUDA 11.8 Runtime

This tree provides a focused GPGPU-Sim runtime and CMake/CTest regression
suite. It intentionally targets one host/toolkit combination instead of
maintaining compatibility layers for multiple CUDA releases and integrations.

## Supported Environment

- Recommended distribution: Ubuntu 22.04 LTS
- Recommended host compiler: GCC/G++ 11.4.0
- Linux on x86-64 or AArch64
- CUDA Toolkit 11.8 (`nvcc` 11.8 and its CUDA headers/tools)
- CMake 3.18 or newer
- CUDA C test targets linked to the simulator runtime

CMake rejects non-Linux hosts, non-GCC host compilers, and CUDA Toolkit
versions other than 11.8.

## Install CUDA Toolkit 11.8

On Ubuntu 22.04 for AArch64/SBSA, install CUDA Toolkit 11.8 from NVIDIA's
package repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

The repository path in these commands is for SBSA systems. On x86-64, replace
`sbsa` in the download URL with `x86_64`.

## Build And Test

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda-11.8 \
  -DGPGPUSIM_GPU_MODEL=SM7_QV100
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The simulator `libcudart` is a build-tree test dependency. It is not installed,
and this project does not guarantee running an arbitrary CUDA executable by
exporting `LD_LIBRARY_PATH`. Tests link the build-tree target directly and use
CTest to provide their simulator configuration.

The build also writes a build-tree CMake package. An external CMake project can
opt in explicitly with:

```cmake
find_package(GPGPUSim CONFIG REQUIRED)
set_target_properties(my_cuda_test PROPERTIES
  CUDA_ARCHITECTURES "${GPGPUSIM_CUDA_ARCHITECTURE}-virtual"
  CUDA_RUNTIME_LIBRARY None)
target_link_libraries(my_cuda_test PRIVATE GPGPUSim::cudart)
```

Configure that project with `-DGPGPUSim_DIR=/path/to/gpgpu-sim/build` (or add
the same directory to `CMAKE_PREFIX_PATH`). The package refers to the existing
build-tree library and exposes `GPGPUSIM_GPU_MODEL` and
`GPGPUSIM_CUDA_ARCHITECTURE`; there is deliberately no install package for
`libcudart`.

## GPU Model

Each build embeds one GPU model directly into `libcudart`. The supported values
of `GPGPUSIM_GPU_MODEL` are `SM3_KEPLER_TITAN`, `SM6_TITANX`, `SM7_GV100`,
`SM7_QV100`, `SM7_TITANV`, `SM75_RTX2060`, `SM75_RTX2060_S`, and
`SM86_RTX3070`. Changing the model requires reconfiguring and rebuilding.

The runtime does not read `gpgpusim.config` or an interconnect configuration
from the process working directory. CMake embeds the selected files from
`configs/tested-cfgs`, and test PTX is compiled for the corresponding virtual
architecture.

`scripts/short-tests-cmake.sh` runs the configure, build, and test sequence in one
command. Set `GPGPUSIM_BUILD_DIR` to select a different build directory.

## CUDA Samples Tests

Twenty-six focused CUDA Samples 11.8 tests and their upstream `Common` helpers are
kept directly under `tests/cuda_samples`. They are built whenever
`BUILD_TESTING` is enabled:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda-11.8
cmake --build build --parallel
ctest --test-dir build -L cuda-samples-gate --output-on-failure
```

The sample targets are `asyncAPI`, `bandwidthTest`, `clock`, `cppIntegration`,
`cppOverload`, `cudaOpenMP`, `deviceQuery`, `dwtHaar1D`,
`fp16ScalarProduct`, `FDTD3d`, `inlinePTX`, `interval`, `matrixMul`, `newdelete`,
`reduction`, `scalarProd`, `simpleAtomicIntrinsics`, `simpleOccupancy`, `simplePrintf`,
`simpleTemplates`, `simpleVoteIntrinsics`, `simpleZeroCopy`, `template`,
`threadFenceReduction`, `transpose`, and `vectorAdd`. Each target is built once.
Short-running targets are registered in both simulator modes, while larger
workloads use functional mode only. Tests use the `<sample>.functional` and
`<sample>.performance` names and can be selected by mode label:

```bash
ctest --test-dir build -L cuda-samples-functional --output-on-failure
ctest --test-dir build -L cuda-samples-performance --output-on-failure
```

The `gpgpusim_add_cuda_sample()` helper accepts `FUNCTIONAL`, `PERFORMANCE`,
`OPENMP`, `SOURCES`, `INCLUDE_DIRS`, `ARGS`, `DATA`, `PASS_REGEX`, and
`TIMEOUT`; for example:

```cmake
gpgpusim_add_cuda_sample(vectorAdd
  FUNCTIONAL PERFORMANCE
  SOURCES vectorAdd/vectorAdd.cu
  TIMEOUT 60)
```

Selecting both modes registers both CTest tests while keeping one executable
target. Selecting neither mode defaults to functional simulation. `TIMEOUT`
applies independently to every registered mode. `DATA` files are copied into
each mode's isolated working directory, and `PASS_REGEX` can require an
upstream success marker in addition to a zero exit status. The `asyncAPI` and
`matrixMul` test copies intentionally omit unsupported CUDA profiler calls and
add workload controls while preserving the upstream defaults. The `interval`
test copy keeps the double-precision interval Newton algorithm and CPU/GPU
validation intact, but reduces its compile-time workload to one equation and
one run. CTest uses these reduced workloads for practical simulation times.

All device-code test binaries use the selected model's virtual architecture, so
their CUDA fatbins contain host ELF plus PTX and no device ELF/cubin. Separate
linkage and fatbin audits verify the simulator `libcudart` dependency and
PTX-only output. Apart from the documented `asyncAPI`, `matrixMul`, and `interval`
adaptations, the upstream sample sources are not modified, and their license is
retained alongside them.

## Libraries

The build produces one unversioned implementation library for the in-tree
tests and explicit CMake consumers:

- `libcudart.so`

The Runtime API covers device discovery, allocation, copies, kernel launch,
streams, events, synchronization, and the CUDA registration hooks required by
`nvcc`-generated host code. The test suite runs CUDA Samples and a vector-add
smoke program against this library.

## Deliberately Unsupported

- PyTorch and other framework-specific compatibility APIs
- device SASS/cubin execution or loading
- cuBLAS, cuDNN, and other CUDA library emulation
- CUDA Driver API
- OpenCL
- CUDA/OpenGL interoperability
- SST/Balar integration
- AccelWattch, GPUWattch, and interconnect power/energy modeling
- PTXPlus and the cuobjdump-to-PTXPlus converter
- legacy CUDA texture-reference registration
- non-Linux platforms and CUDA Toolkit versions other than 11.8
- the former Make-based build

Unsupported symbols are not exported as fake-success or abort-only public API
stubs. Add support by implementing the simulator behavior and a focused test,
not by returning success without doing the operation.
