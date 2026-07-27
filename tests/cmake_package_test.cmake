foreach(required SOURCE_DIR BINARY_DIR GPGPUSIM_DIR GENERATOR CUDA_COMPILER)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required argument: ${required}")
  endif()
endforeach()

set(configure_command
  "${CMAKE_COMMAND}"
  -S "${SOURCE_DIR}"
  -B "${BINARY_DIR}"
  -G "${GENERATOR}"
  "-DGPGPUSim_DIR=${GPGPUSIM_DIR}"
  "-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER}")
if(DEFINED MAKE_PROGRAM AND NOT "${MAKE_PROGRAM}" STREQUAL "")
  list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(DEFINED BUILD_TYPE AND NOT "${BUILD_TYPE}" STREQUAL "")
  list(APPEND configure_command "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
endif()

execute_process(
  COMMAND ${configure_command}
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_stdout
  ERROR_VARIABLE configure_stderr)
if(configure_result)
  message(FATAL_ERROR
    "GPGPU-Sim package consumer configure failed:\n"
    "${configure_stdout}${configure_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${BINARY_DIR}"
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_stdout
  ERROR_VARIABLE build_stderr)
if(build_result)
  message(FATAL_ERROR
    "GPGPU-Sim package consumer build failed:\n"
    "${build_stdout}${build_stderr}")
endif()
