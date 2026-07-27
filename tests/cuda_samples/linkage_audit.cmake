string(REPLACE "|" ";" executable_list "${EXECUTABLES}")
foreach(executable IN LISTS executable_list)
  if(NOT EXISTS "${executable}")
    message(FATAL_ERROR "CUDA sample executable not found: ${executable}")
  endif()

  execute_process(
    COMMAND "${READELF}" -d "${executable}"
    OUTPUT_VARIABLE dynamic_section
    COMMAND_ERROR_IS_FATAL ANY)
  if(NOT dynamic_section MATCHES
     "NEEDED.*Shared library: \\[libcudart\\.so\\.11\\.0\\]")
    message(FATAL_ERROR
      "${executable} is not dynamically linked to libcudart.so.11.0")
  endif()
  if(dynamic_section MATCHES
     "Shared library: \\[lib(cuda\\.so|cublas|cufft|curand|cusolver|cusparse|npp|nvrtc)")
    message(FATAL_ERROR
      "${executable} depends on an unsupported CUDA library")
  endif()
endforeach()
