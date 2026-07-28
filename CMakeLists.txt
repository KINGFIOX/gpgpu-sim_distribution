cmake_minimum_required(VERSION 3.18)

set(CMAKE_SKIP_INSTALL_RULES TRUE)

project(GPGPU-Sim
        DESCRIPTION "cycle-level simulator modeling contemporary graphics processing units (GPUs)"
        HOMEPAGE_URL https://github.com/accel-sim/gpgpu-sim_distribution
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GPGPUSIM_ENABLE_TRACE "Enable GPGPU-Sim debug tracing" ON)
option(GPGPUSIM_ENABLE_CUDA_SAMPLES "Build the pinned CUDA Samples 11.8 regression tests" ON)

include(gpgpusim_check.cmake)
include(CTest)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG=1)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-unused-function;-Wno-sign-compare;-g;-fPIC>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wno-unused-function;-Wno-sign-compare;-ggdb;-fPIC>")
else ()
    add_compile_definitions(DEBUG=0)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3;-g;-Wall;-Wno-unused-function;-Wno-sign-compare;-fPIC>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wno-unused-function;-Wno-sign-compare;-fPIC>")
endif ()

if (GPGPUSIM_ENABLE_TRACE)
    add_compile_definitions(TRACING_ON=1)
endif ()

add_subdirectory(src)
add_subdirectory(libcuda)

add_library(cudart SHARED $<TARGET_OBJECTS:gpgpusim_entrypoint>)
add_library(entrypoint STATIC $<TARGET_OBJECTS:gpgpusim_entrypoint>)

target_link_libraries(cudart PRIVATE
        -Wl,--whole-archive cudart_impl -Wl,--no-whole-archive
        ptxsim gpgpusim intersim m z pthread)
target_link_libraries(entrypoint PUBLIC cudart_impl ptxsim gpgpusim intersim)

# Export the simulator runtime from the build tree.  The library is
# intentionally not installed: consumers must opt into the simulator by
# pointing find_package at this build directory.
export(TARGETS cudart
        FILE "${CMAKE_BINARY_DIR}/GPGPUSimTargets.cmake"
        NAMESPACE GPGPUSim::)
configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/GPGPUSimConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/GPGPUSimConfig.cmake"
        @ONLY)
message(STATUS
        "GPGPU-Sim build-tree package: ${CMAKE_BINARY_DIR}/GPGPUSimConfig.cmake")

if (BUILD_TESTING)
    add_subdirectory(tests)
endif ()
