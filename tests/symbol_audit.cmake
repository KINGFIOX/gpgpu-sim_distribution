foreach(alias libcuda.so libcuda.so.1)
  if(NOT IS_SYMLINK "${LIBRARY_DIR}/${alias}")
    message(FATAL_ERROR "${alias} is not a symlink")
  endif()
endforeach()

execute_process(
  COMMAND "${READELF}" -d "${LIBRARY}"
  OUTPUT_VARIABLE dynamic_section
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT dynamic_section MATCHES "SONAME.*libcudart\\.so\\.11\\.0")
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
  if(NOT symbols MATCHES " ${required}@@")
    message(FATAL_ERROR "required symbol ${required} is not exported")
  endif()
endforeach()
foreach(forbidden
    __cudaRegisterTexture cudaGetExportTable
    cudaGLRegisterBufferObject cudaWGLGetDevice
    cuModuleLoad cuLaunchKernel)
  if(symbols MATCHES " ${forbidden}@@")
    message(FATAL_ERROR "unsupported symbol ${forbidden} is exported")
  endif()
endforeach()
