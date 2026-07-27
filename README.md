# GPGPU-Sim CUDA 11.8 Runtime

This tree provides a focused GPGPU-Sim runtime for standalone CUDA C
applications. It intentionally targets one host/toolkit combination instead of
maintaining compatibility layers for multiple CUDA releases and integrations.

## Supported Environment

- Linux on x86-64 or AArch64
- CUDA Toolkit 11.8 (`nvcc` 11.8 and its CUDA headers/tools)
- GCC/G++
- CMake 3.18 or newer
- Standalone CUDA C applications using the supported Runtime API

CMake rejects non-Linux hosts, non-GCC host compilers, and CUDA Toolkit
versions other than 11.8.

## Build And Test

```bash
git submodule update --init third_party/cuda-samples
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda-11.8
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build
source build/setup
```

The install step writes the simulator libraries under:

```text
lib/gcc-<version>/cuda-11080/<build-mode>/
```

The generated `build/setup` script adds that directory to
`LD_LIBRARY_PATH`, adds CUDA 11.8 to `PATH`, and exports the simulator paths.

`short-tests-cmake.sh` runs the configure, build, and test sequence in one
command. Set `GPGPUSIM_BUILD_DIR` to select a different build directory.

## CUDA Samples Tests

The CUDA Samples 11.8 dataset is pinned as a Git submodule. The regular build
does not compile it. Enable the focused regression suite explicitly:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda-11.8 \
  -DGPGPUSIM_ENABLE_CUDA_SAMPLES=ON
cmake --build build --parallel
ctest --test-dir build -L cuda-samples-gate --output-on-failure
```

The six detailed-simulation gates are `deviceQuery`, `vectorAdd`,
`simpleTemplates`, `simpleVoteIntrinsics`, `simpleAtomicIntrinsics`, and
`clock`. All device-code test binaries use `70-virtual`, so their CUDA fatbins
contain host ELF plus PTX and no device ELF/cubin; the host-only `deviceQuery`
sample has no device fatbin by design. Separate linkage and fatbin audits verify
the shared CUDA 11.8 Runtime library and PTX-only output. The upstream sample
sources are not modified.

For broader compatibility data, run the report-only functional survey:

```bash
cmake --build build --target cuda-samples-survey
```

It classifies every sample, builds standalone CUDA Runtime candidates in an
isolated copy, and writes `manifest.tsv`, `results.tsv`, per-sample logs, and
`summary.md` under `build/tests/cuda_samples/survey`. Unsupported categories
such as Driver/JIT, graphics, multi-GPU, and CUDA library samples are recorded
as exclusions. Build or runtime failures are recorded in the report and do not
make the target fail. Set `CUDA_SAMPLES_SURVEY_FILTER` to an extended regular
expression when surveying a subset.

To include the focused gates in `short-tests-cmake.sh`, set
`GPGPUSIM_ENABLE_CUDA_SAMPLES=ON`.

## Libraries

The build produces one implementation library with CUDA 11.8 ABI names:

- `libcudart.so.11.0`
- `libcuda.so.1` -> `libcudart.so.11.0`
- `libcuda.so` -> `libcuda.so.1`

The Runtime API covers device discovery, allocation, copies, kernel launch,
streams, events, synchronization, and the CUDA registration hooks required by
`nvcc`-generated host code. The test suite runs a CUDA C vector-add program
against this library.

The Driver API is deliberately limited to the following delegated subset:

- initialization, version, error, and device queries
- context synchronization
- device allocation and host/device/device copies
- stream creation, destruction, and synchronization
- event creation, record, synchronization, and destruction

Applications that require other Driver API entry points are outside the
supported surface.

## Deliberately Unsupported

- PyTorch and other framework-specific compatibility APIs
- device SASS/cubin execution or loading
- cuBLAS, cuDNN, and other CUDA library emulation
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
