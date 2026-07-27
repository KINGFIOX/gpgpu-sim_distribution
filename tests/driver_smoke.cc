#include <cuda.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

template <typename Function>
Function load(void *library, const char *name) {
  dlerror();
  Function function = reinterpret_cast<Function>(dlsym(library, name));
  const char *error = dlerror();
  if (error != nullptr) {
    std::fprintf(stderr, "dlsym(%s) failed: %s\n", name, error);
    std::exit(EXIT_FAILURE);
  }
  return function;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) return EXIT_FAILURE;
  const std::string library_path = std::string(argv[1]) + "/libcuda.so";
  void *library = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return EXIT_FAILURE;
  }

  const auto init = load<CUresult (*)(unsigned int)>(library, "cuInit");
  const auto driver_get_version =
      load<CUresult (*)(int *)>(library, "cuDriverGetVersion");
  const auto device_get_count =
      load<CUresult (*)(int *)>(library, "cuDeviceGetCount");
  const auto mem_alloc =
      load<CUresult (*)(CUdeviceptr *, size_t)>(library, "cuMemAlloc_v2");
  const auto mem_free =
      load<CUresult (*)(CUdeviceptr)>(library, "cuMemFree_v2");
  const auto memcpy_htod = load<CUresult (*)(CUdeviceptr, const void *, size_t)>(
      library, "cuMemcpyHtoD_v2");
  const auto memcpy_dtoh = load<CUresult (*)(void *, CUdeviceptr, size_t)>(
      library, "cuMemcpyDtoH_v2");
  const auto synchronize =
      load<CUresult (*)(void)>(library, "cuCtxSynchronize");
  const auto stream_create =
      load<CUresult (*)(CUstream *, unsigned int)>(library, "cuStreamCreate");
  const auto stream_synchronize = load<CUresult (*)(CUstream)>(
      library, "cuStreamSynchronize");
  const auto stream_destroy =
      load<CUresult (*)(CUstream)>(library, "cuStreamDestroy_v2");
  const auto event_create =
      load<CUresult (*)(CUevent *, unsigned int)>(library, "cuEventCreate");
  const auto event_record =
      load<CUresult (*)(CUevent, CUstream)>(library, "cuEventRecord");
  const auto event_synchronize =
      load<CUresult (*)(CUevent)>(library, "cuEventSynchronize");
  const auto event_destroy =
      load<CUresult (*)(CUevent)>(library, "cuEventDestroy_v2");

  int version = 0;
  int count = 0;
  if (init(0) != CUDA_SUCCESS ||
      driver_get_version(&version) != CUDA_SUCCESS || version != 11080 ||
      device_get_count(&count) != CUDA_SUCCESS || count < 1) {
    std::fprintf(stderr, "driver subset returned version=%d count=%d\n",
                 version, count);
    return EXIT_FAILURE;
  }

  const int input[] = {3, 1, 4, 1};
  int output[] = {0, 0, 0, 0};
  CUdeviceptr device_pointer = 0;
  if (mem_alloc(&device_pointer, sizeof(input)) != CUDA_SUCCESS ||
      memcpy_htod(device_pointer, input, sizeof(input)) != CUDA_SUCCESS ||
      memcpy_dtoh(output, device_pointer, sizeof(output)) != CUDA_SUCCESS ||
      synchronize() != CUDA_SUCCESS ||
      mem_free(device_pointer) != CUDA_SUCCESS) {
    std::fprintf(stderr, "driver memory subset failed\n");
    return EXIT_FAILURE;
  }
  for (size_t index = 0; index < sizeof(input) / sizeof(input[0]); ++index) {
    if (output[index] != input[index]) {
      std::fprintf(stderr, "driver copy mismatch at %zu\n", index);
      return EXIT_FAILURE;
    }
  }

  CUstream stream = nullptr;
  CUevent event = nullptr;
  if (stream_create(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS ||
      event_create(&event, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
      event_record(event, stream) != CUDA_SUCCESS ||
      event_synchronize(event) != CUDA_SUCCESS ||
      stream_synchronize(stream) != CUDA_SUCCESS ||
      event_destroy(event) != CUDA_SUCCESS ||
      stream_destroy(stream) != CUDA_SUCCESS) {
    std::fprintf(stderr, "driver stream/event subset failed\n");
    return EXIT_FAILURE;
  }

  // GPGPU-Sim owns process-lifetime state and exit handlers after cuInit.
  return EXIT_SUCCESS;
}
