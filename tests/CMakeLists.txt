set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
enable_language(CUDA)

add_executable(runtime_smoke runtime_smoke.cu)
set_target_properties(runtime_smoke PROPERTIES
        CUDA_ARCHITECTURES "${GPGPUSIM_CUDA_ARCHITECTURE}-virtual"
        CUDA_RUNTIME_LIBRARY None)
target_link_libraries(runtime_smoke PRIVATE cudart)

set(TEST_RUN_DIR "${CMAKE_CURRENT_BINARY_DIR}/run/runtime_smoke")
file(MAKE_DIRECTORY "${TEST_RUN_DIR}")

add_test(NAME runtime_vector_add COMMAND runtime_smoke)
set_tests_properties(runtime_vector_add PROPERTIES
        WORKING_DIRECTORY "${TEST_RUN_DIR}"
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
        -DREADELF=${CMAKE_READELF}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/symbol_audit.cmake)

add_subdirectory(cuda_samples)
