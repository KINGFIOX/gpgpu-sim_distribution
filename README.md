# GPGPU-Sim CUDA 11.8 Runtime

This tree provides a focused GPGPU-Sim runtime for standalone CUDA C
applications. It intentionally targets one host/toolkit combination instead of
maintaining compatibility layers for multiple CUDA releases and integrations.

## Supported Environment

- Linux on x86-64 or AArch64
- CUDA Toolkit 11.8 (`nvcc` 11.8 and its CUDA headers/tools)
- GCC/G++
- CMake 3.17 or newer
- Standalone CUDA C applications using the supported Runtime API

CMake rejects non-Linux hosts, non-GCC host compilers, and CUDA Toolkit
versions other than 11.8.

## Build And Test

```bash
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
- cuBLAS, cuDNN, and other CUDA library emulation
- OpenCL
- CUDA/OpenGL interoperability
- SST/Balar integration
- PTXPlus and the cuobjdump-to-PTXPlus converter
- legacy CUDA texture-reference registration
- non-Linux platforms and CUDA Toolkit versions other than 11.8
- the former Make-based build

Unsupported symbols are not exported as fake-success or abort-only public API
stubs. Add support by implementing the simulator behavior and a focused test,
not by returning success without doing the operation.
