add_library(cudart_impl STATIC
  cuda_runtime_api.cc)

target_include_directories(cudart_impl
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(cudart_impl PRIVATE
  GPGPUSIM_CUOBJDUMP="${CUDAToolkit_BIN_DIR}/cuobjdump")
