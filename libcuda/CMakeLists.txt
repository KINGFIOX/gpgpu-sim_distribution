add_library(cuda STATIC
  cuda_runtime_api.cc
  cuda_driver_api.cc)

target_include_directories(cuda
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(cuda PRIVATE
  GPGPUSIM_CUOBJDUMP="${CUDAToolkit_BIN_DIR}/cuobjdump")
