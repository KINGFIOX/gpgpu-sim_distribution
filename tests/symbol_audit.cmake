execute_process(
        COMMAND "${READELF}" -d "${LIBRARY}"
        OUTPUT_VARIABLE dynamic_section
        COMMAND_ERROR_IS_FATAL ANY)
if (NOT dynamic_section MATCHES "SONAME.*\\[libcudart\\.so\\]")
    message(FATAL_ERROR "unexpected libcudart SONAME")
endif ()
if (dynamic_section MATCHES "lib(OpenCL|cublas|cudnn)")
    message(FATAL_ERROR "unsupported accelerator library dependency found")
endif ()

execute_process(
        COMMAND "${READELF}" --wide --dyn-syms "${LIBRARY}"
        OUTPUT_VARIABLE symbols
        COMMAND_ERROR_IS_FATAL ANY)
foreach (required
        cudaMalloc cudaMemcpy cudaLaunchKernel cudaDeviceSynchronize
        cudaDeviceGetAttribute cudaDeviceCanAccessPeer cudaDeviceSetLimit
        cudaHostRegister
        cudaHostUnregister cudaSetDeviceFlags)
    if (NOT symbols MATCHES " ${required}(\n|$)")
        message(FATAL_ERROR "required symbol ${required} is not exported")
    endif ()
endforeach ()
foreach (forbidden
        __cudaRegisterTexture cudaGetExportTable
        cudaGLRegisterBufferObject cudaWGLGetDevice)
    if (symbols MATCHES " ${forbidden}(\n|$)")
        message(FATAL_ERROR "unsupported symbol ${forbidden} is exported")
    endif ()
endforeach ()
if (symbols MATCHES " cu[A-Z][A-Za-z0-9_]*(\n|$)")
    message(FATAL_ERROR "CUDA Driver API symbol is exported")
endif ()
