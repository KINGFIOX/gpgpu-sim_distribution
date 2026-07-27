#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>

namespace {

CUresult to_driver_error(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return CUDA_SUCCESS;
    case cudaErrorInvalidValue:
      return CUDA_ERROR_INVALID_VALUE;
    case cudaErrorMemoryAllocation:
      return CUDA_ERROR_OUT_OF_MEMORY;
    case cudaErrorInvalidDevice:
      return CUDA_ERROR_INVALID_DEVICE;
    case cudaErrorNotReady:
      return CUDA_ERROR_NOT_READY;
    default:
      return CUDA_ERROR_UNKNOWN;
  }
}

const char *driver_error_name(CUresult error) {
  switch (error) {
    case CUDA_SUCCESS:
      return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
      return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
      return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
      return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:
      return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_INVALID_DEVICE:
      return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_NOT_READY:
      return "CUDA_ERROR_NOT_READY";
    default:
      return "CUDA_ERROR_UNKNOWN";
  }
}

}  // namespace

extern "C" {

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **name) {
  if (name == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *name = driver_error_name(error);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **message) {
  if (message == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *message = driver_error_name(error);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuInit(unsigned int flags) {
  if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
  int count = 0;
  return to_driver_error(cudaGetDeviceCount(&count));
}

CUresult CUDAAPI cuDriverGetVersion(int *driver_version) {
  return to_driver_error(cudaDriverGetVersion(driver_version));
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
  return to_driver_error(cudaGetDeviceCount(count));
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  if (device == nullptr) return CUDA_ERROR_INVALID_VALUE;
  int count = 0;
  cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) return to_driver_error(error);
  if (ordinal < 0 || ordinal >= count) return CUDA_ERROR_INVALID_DEVICE;
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int length, CUdevice device) {
  if (name == nullptr || length <= 0) return CUDA_ERROR_INVALID_VALUE;
  cudaDeviceProp properties{};
  cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return to_driver_error(error);
  snprintf(name, static_cast<size_t>(length), "%s", properties.name);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice device) {
  if (bytes == nullptr) return CUDA_ERROR_INVALID_VALUE;
  cudaDeviceProp properties{};
  cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error == cudaSuccess) *bytes = properties.totalGlobalMem;
  return to_driver_error(error);
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor,
                                           CUdevice device) {
  if (major == nullptr || minor == nullptr) return CUDA_ERROR_INVALID_VALUE;
  cudaDeviceProp properties{};
  cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error == cudaSuccess) {
    *major = properties.major;
    *minor = properties.minor;
  }
  return to_driver_error(error);
}

CUresult CUDAAPI cuDeviceGetAttribute(int *value,
                                      CUdevice_attribute attribute,
                                      CUdevice device) {
  return to_driver_error(cudaDeviceGetAttribute(
      value, static_cast<cudaDeviceAttr>(attribute), device));
}

CUresult CUDAAPI cuCtxSynchronize(void) {
  return to_driver_error(cudaDeviceSynchronize());
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *device_pointer, size_t bytes) {
  if (device_pointer == nullptr) return CUDA_ERROR_INVALID_VALUE;
  void *pointer = nullptr;
  cudaError_t error = cudaMalloc(&pointer, bytes);
  if (error == cudaSuccess) {
    *device_pointer = reinterpret_cast<uintptr_t>(pointer);
  }
  return to_driver_error(error);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr device_pointer) {
  return to_driver_error(cudaFree(reinterpret_cast<void *>(
      static_cast<uintptr_t>(device_pointer))));
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr destination, const void *source,
                              size_t bytes) {
  return to_driver_error(cudaMemcpy(
      reinterpret_cast<void *>(static_cast<uintptr_t>(destination)), source,
      bytes, cudaMemcpyHostToDevice));
}

CUresult CUDAAPI cuMemcpyDtoH(void *destination, CUdeviceptr source,
                              size_t bytes) {
  return to_driver_error(cudaMemcpy(
      destination, reinterpret_cast<const void *>(static_cast<uintptr_t>(source)),
      bytes, cudaMemcpyDeviceToHost));
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr destination, CUdeviceptr source,
                              size_t bytes) {
  return to_driver_error(cudaMemcpy(
      reinterpret_cast<void *>(static_cast<uintptr_t>(destination)),
      reinterpret_cast<const void *>(static_cast<uintptr_t>(source)), bytes,
      cudaMemcpyDeviceToDevice));
}

CUresult CUDAAPI cuStreamCreate(CUstream *stream, unsigned int flags) {
  return to_driver_error(cudaStreamCreateWithFlags(
      reinterpret_cast<cudaStream_t *>(stream), flags));
}

CUresult CUDAAPI cuStreamDestroy(CUstream stream) {
  return to_driver_error(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
  return to_driver_error(
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

CUresult CUDAAPI cuEventCreate(CUevent *event, unsigned int flags) {
  return to_driver_error(
      cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(event), flags));
}

CUresult CUDAAPI cuEventDestroy(CUevent event) {
  return to_driver_error(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
}

CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream) {
  return to_driver_error(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                         reinterpret_cast<cudaStream_t>(stream)));
}

CUresult CUDAAPI cuEventSynchronize(CUevent event) {
  return to_driver_error(
      cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
}

}  // extern "C"
