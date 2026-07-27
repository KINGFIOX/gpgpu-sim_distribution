cmake_minimum_required(VERSION 3.18)

project(GPGPU-Sim
        VERSION 4.2.0
        DESCRIPTION "cycle-level simulator modeling contemporary graphics processing units (GPUs)"
        HOMEPAGE_URL https://github.com/accel-sim/gpgpu-sim_distribution
        LANGUAGES CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}" CACHE PATH
      "GPGPU-Sim installation root" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GPGPUSIM_ENABLE_TRACE "Enable GPGPU-Sim debug tracing" ON)
option(GPGPUSIM_ENABLE_CUDA_SAMPLES
       "Build the pinned CUDA Samples 11.8 regression tests" OFF)

include(gpgpusim_check.cmake)
include(CTest)

add_custom_target(gen_build_string ALL
  COMMAND ${CMAKE_COMMAND}
          -D INPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -D OUTPUT_DIR=${CMAKE_BINARY_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/gpgpusim_gen_build_string.cmake
  COMMENT "Generating build string file in ${CMAKE_BINARY_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DEBUG=1)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-unused-function;-Wno-sign-compare;-g;-fPIC>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wno-unused-function;-Wno-sign-compare;-ggdb;-fPIC>")
else()
  add_compile_definitions(DEBUG=0)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3;-g;-Wall;-Wno-unused-function;-Wno-sign-compare;-fPIC>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wno-unused-function;-Wno-sign-compare;-fPIC>")
endif()

if(GPGPUSIM_ENABLE_TRACE)
  add_compile_definitions(TRACING_ON=1)
endif()

add_subdirectory(src)
add_subdirectory(libcuda)

add_library(cudart SHARED $<TARGET_OBJECTS:gpgpusim_entrypoint>)
add_library(entrypoint STATIC $<TARGET_OBJECTS:gpgpusim_entrypoint>)

set_target_properties(cudart PROPERTIES
  OUTPUT_NAME cudart
  VERSION 11.8.0
  SOVERSION 11.0)

target_link_options(cudart PRIVATE
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/linux-so-version.txt")
target_link_libraries(cudart PRIVATE
  -Wl,--whole-archive cuda -Wl,--no-whole-archive
  ptxsim gpgpusim intersim m z pthread)
target_link_libraries(entrypoint PUBLIC cuda ptxsim gpgpusim intersim)

# The Driver API subset shares the same process-wide simulator state as cudart.
add_custom_command(TARGET cudart POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "$<TARGET_SONAME_FILE_NAME:cudart>"
          "$<TARGET_FILE_DIR:cudart>/libcuda.so.1"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "libcuda.so.1"
          "$<TARGET_FILE_DIR:cudart>/libcuda.so")

set(GPGPUSIM_CONFIG
    "gcc-${CMAKE_CXX_COMPILER_VERSION}/cuda-11080/${GPGPUSIM_BUILD_MODE}")
set(GPGPUSIM_INSTALL_DIR "lib/${GPGPUSIM_CONFIG}")
set(GPGPUSIM_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${GPGPUSIM_INSTALL_DIR}")

include(gpgpusim_gen_setup_environment.cmake)

install(TARGETS cudart LIBRARY DESTINATION ${GPGPUSIM_INSTALL_DIR})
install(CODE "
  set(alias_dir \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${GPGPUSIM_INSTALL_DIR}\")
  execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
    \"libcudart.so.11.0\" \"\${alias_dir}/libcuda.so.1\")
  execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
    \"libcuda.so.1\" \"\${alias_dir}/libcuda.so\")")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
