#include <cuda_runtime_api.h>

int main() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess ? 0 : 1;
}
