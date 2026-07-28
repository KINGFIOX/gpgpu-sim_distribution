#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kElementCount = 32;

__global__ void vector_add(const int *left, const int *right, int *result) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kElementCount) result[index] = left[index] + right[index];
}

void check(cudaError_t error, const char *operation) {
  if (error == cudaSuccess) return;
  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
  std::exit(EXIT_FAILURE);
}

}  // namespace

int main() {
  int compute_mode = -1;
  check(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, 0),
        "cudaDeviceGetAttribute(compute mode)");
  if (compute_mode != cudaComputeModeDefault) return EXIT_FAILURE;

  int can_access_peer = -1;
  check(cudaDeviceCanAccessPeer(&can_access_peer, 0, 0),
        "cudaDeviceCanAccessPeer");
  if (can_access_peer != 0) return EXIT_FAILURE;

  int left[kElementCount];
  int right[kElementCount];
  int result[kElementCount] = {};
  for (int i = 0; i < kElementCount; ++i) {
    left[i] = i;
    right[i] = kElementCount - i;
  }
  int host_copy[kElementCount] = {};
  check(cudaMemcpy(host_copy, left, sizeof(left), cudaMemcpyHostToHost),
        "cudaMemcpy(host-to-host)");
  for (int i = 0; i < kElementCount; ++i) {
    if (host_copy[i] != left[i]) return EXIT_FAILURE;
  }

  int *device_left = nullptr;
  int *device_right = nullptr;
  int *device_result = nullptr;
  const size_t bytes = sizeof(left);
  check(cudaMalloc(reinterpret_cast<void **>(&device_left), bytes),
        "cudaMalloc(left)");
  check(cudaMalloc(reinterpret_cast<void **>(&device_right), bytes),
        "cudaMalloc(right)");
  check(cudaMalloc(reinterpret_cast<void **>(&device_result), bytes),
        "cudaMalloc(result)");
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");
  check(
      cudaMemcpyAsync(device_left, left, bytes, cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync(left)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaMemcpy(device_right, right, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(right)");

  vector_add<<<1, kElementCount>>>(device_left, device_right, device_result);
  check(cudaGetLastError(), "vector_add launch");
  check(cudaMemcpy(result, device_result, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(result)");

  for (int value : result) {
    if (value != kElementCount) {
      std::fprintf(stderr, "unexpected result: %d\n", value);
      return EXIT_FAILURE;
    }
  }

  check(cudaFree(device_result), "cudaFree(result)");
  check(cudaFree(device_right), "cudaFree(right)");
  check(cudaFree(device_left), "cudaFree(left)");
  return EXIT_SUCCESS;
}
