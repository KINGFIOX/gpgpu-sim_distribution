set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
enable_language(CUDA)

add_executable(runtime_smoke runtime_smoke.cu)
set_target_properties(runtime_smoke PROPERTIES
  CUDA_ARCHITECTURES 70-virtual
  CUDA_RUNTIME_LIBRARY None)
target_link_libraries(runtime_smoke PRIVATE cudart)

add_executable(driver_smoke driver_smoke.cc)
target_include_directories(driver_smoke PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(driver_smoke PRIVATE dl)

set(TEST_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/config")
file(MAKE_DIRECTORY "${TEST_CONFIG_DIR}")
configure_file(
  "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/SM7_QV100/gpgpusim.config"
  "${TEST_CONFIG_DIR}/gpgpusim.config" COPYONLY)
configure_file(
  "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/SM7_QV100/config_volta_islip.icnt"
  "${TEST_CONFIG_DIR}/config_volta_islip.icnt" COPYONLY)

add_test(NAME runtime_vector_add COMMAND runtime_smoke)
set_tests_properties(runtime_vector_add PROPERTIES
  WORKING_DIRECTORY "${TEST_CONFIG_DIR}"
  ENVIRONMENT
    "GPGPUSIM_ROOT=${PROJECT_SOURCE_DIR};CUDA_INSTALL_PATH=${CUDAToolkit_TARGET_DIR};PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH}")

add_test(NAME driver_api_subset COMMAND driver_smoke "$<TARGET_FILE_DIR:cudart>")
set_tests_properties(driver_api_subset PROPERTIES
  WORKING_DIRECTORY "${TEST_CONFIG_DIR}"
  ENVIRONMENT
    "GPGPUSIM_ROOT=${PROJECT_SOURCE_DIR};CUDA_INSTALL_PATH=${CUDAToolkit_TARGET_DIR};PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH}")

add_test(
  NAME gpgpusim_cmake_package
  COMMAND ${CMAKE_COMMAND}
    "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cmake_package"
    "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake_package_build"
    "-DGPGPUSIM_DIR=${CMAKE_BINARY_DIR}"
    "-DGENERATOR=${CMAKE_GENERATOR}"
    "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    "-DBUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCUDA_COMPILER=${CUDAToolkit_NVCC_EXECUTABLE}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake_package_test.cmake)

add_test(
  NAME exported_symbol_audit
  COMMAND ${CMAKE_COMMAND}
    -DLIBRARY=$<TARGET_FILE:cudart>
    -DLIBRARY_DIR=$<TARGET_FILE_DIR:cudart>
    -DREADELF=${CMAKE_READELF}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/symbol_audit.cmake)

if(GPGPUSIM_ENABLE_CUDA_SAMPLES)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cuda_samples/CudaSamples.cmake)
endif()
