if(NOT IS_SYMLINK "${LIBRARY_DIR}/libcuda.so")
  message(FATAL_ERROR "libcuda.so is not a symlink")
endif()
if(EXISTS "${LIBRARY_DIR}/libcuda.so.1")
  message(FATAL_ERROR "versioned libcuda alias is present")
endif()

execute_process(
  COMMAND "${READELF}" -d "${LIBRARY}"
  OUTPUT_VARIABLE dynamic_section
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT dynamic_section MATCHES "SONAME.*\\[libcudart\\.so\\]")
  message(FATAL_ERROR "unexpected libcudart SONAME")
endif()
if(dynamic_section MATCHES "lib(OpenCL|cublas|cudnn)")
  message(FATAL_ERROR "unsupported accelerator library dependency found")
endif()

execute_process(
  COMMAND "${READELF}" --wide --dyn-syms "${LIBRARY}"
  OUTPUT_VARIABLE symbols
  COMMAND_ERROR_IS_FATAL ANY)
foreach(required
    cudaMalloc cudaMemcpy cudaLaunchKernel cudaDeviceSynchronize
    cudaDeviceGetAttribute cudaDeviceCanAccessPeer
    cuInit cuDriverGetVersion cuDeviceGetCount cuMemAlloc_v2
    cuMemcpyHtoD_v2 cuMemcpyDtoH_v2 cuStreamCreate cuEventCreate)
  if(NOT symbols MATCHES " ${required}(\n|$)")
    message(FATAL_ERROR "required symbol ${required} is not exported")
  endif()
endforeach()
foreach(forbidden
    __cudaRegisterTexture cudaGetExportTable
    cudaGLRegisterBufferObject cudaWGLGetDevice
    cuModuleLoad cuLaunchKernel)
  if(symbols MATCHES " ${forbidden}(\n|$)")
    message(FATAL_ERROR "unsupported symbol ${forbidden} is exported")
  endif()
endforeach()
