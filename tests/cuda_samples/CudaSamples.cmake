set(CUDA_SAMPLES_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/third_party/cuda-samples")
set(CUDA_SAMPLES_PINNED_COMMIT
    "81992093d2b8c33cab22dbf6852c070c330f1715")

if(NOT EXISTS "${CUDA_SAMPLES_SOURCE_DIR}/README.md")
  message(FATAL_ERROR
    "CUDA Samples is not initialized. Run: git submodule update --init "
    "third_party/cuda-samples")
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${CUDA_SAMPLES_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE cuda_samples_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE cuda_samples_git_result)
if(cuda_samples_git_result OR
   NOT cuda_samples_commit STREQUAL CUDA_SAMPLES_PINNED_COMMIT)
  message(FATAL_ERROR
    "CUDA Samples must be at ${CUDA_SAMPLES_PINNED_COMMIT}; found "
    "${cuda_samples_commit}")
endif()

set(CUDA_SAMPLES_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_samples/bin")
set(CUDA_SAMPLES_RUN_ROOT "${CMAKE_CURRENT_BINARY_DIR}/cuda_samples/run")
set(CUDA_SAMPLES_GATE_TARGETS "")

function(gpgpusim_add_cuda_sample name sample_path)
  cmake_parse_arguments(SAMPLE "" "" "SOURCES;ARGS" ${ARGN})
  if(NOT SAMPLE_SOURCES)
    message(FATAL_ERROR "CUDA sample ${name} has no sources")
  endif()

  set(target "cuda_sample_${name}")
  set(sources "")
  foreach(source IN LISTS SAMPLE_SOURCES)
    list(APPEND sources
      "${CUDA_SAMPLES_SOURCE_DIR}/${sample_path}/${source}")
  endforeach()

  add_executable(${target} ${sources})
  target_include_directories(${target} PRIVATE
    "${CUDA_SAMPLES_SOURCE_DIR}/Common")
  target_link_libraries(${target} PRIVATE CUDA::cudart)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 70-virtual
    CUDA_RUNTIME_LIBRARY Shared
    RUNTIME_OUTPUT_DIRECTORY "${CUDA_SAMPLES_BINARY_DIR}")

  set(run_dir "${CUDA_SAMPLES_RUN_ROOT}/${name}")
  file(MAKE_DIRECTORY "${run_dir}")
  configure_file(
    "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/SM7_QV100/gpgpusim.config"
    "${run_dir}/gpgpusim.config" COPYONLY)
  configure_file(
    "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/SM7_QV100/config_volta_islip.icnt"
    "${run_dir}/config_volta_islip.icnt" COPYONLY)

  add_test(NAME ${target} COMMAND "$<TARGET_FILE:${target}>" ${SAMPLE_ARGS})
  set_tests_properties(${target} PROPERTIES
    LABELS "cuda-samples;cuda-samples-gate"
    RUN_SERIAL TRUE
    TIMEOUT 180
    WORKING_DIRECTORY "${run_dir}"
    ENVIRONMENT
      "GPGPUSIM_ROOT=${PROJECT_SOURCE_DIR};CUDA_INSTALL_PATH=${CUDAToolkit_TARGET_DIR};PTX_SIM_MODE_FUNC=0;PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:cudart>")

  set(CUDA_SAMPLES_GATE_TARGETS
      ${CUDA_SAMPLES_GATE_TARGETS} ${target} PARENT_SCOPE)
endfunction()

gpgpusim_add_cuda_sample(deviceQuery Samples/1_Utilities/deviceQuery
  SOURCES deviceQuery.cpp)
gpgpusim_add_cuda_sample(vectorAdd Samples/0_Introduction/vectorAdd
  SOURCES vectorAdd.cu)
gpgpusim_add_cuda_sample(simpleTemplates Samples/0_Introduction/simpleTemplates
  SOURCES simpleTemplates.cu)
gpgpusim_add_cuda_sample(simpleVoteIntrinsics
  Samples/0_Introduction/simpleVoteIntrinsics
  SOURCES simpleVoteIntrinsics.cu)
gpgpusim_add_cuda_sample(simpleAtomicIntrinsics
  Samples/0_Introduction/simpleAtomicIntrinsics
  SOURCES simpleAtomicIntrinsics.cu simpleAtomicIntrinsics_cpu.cpp)
gpgpusim_add_cuda_sample(clock Samples/0_Introduction/clock
  SOURCES clock.cu)

set(cuda_samples_executables "")
foreach(target IN LISTS CUDA_SAMPLES_GATE_TARGETS)
  list(APPEND cuda_samples_executables "$<TARGET_FILE:${target}>")
endforeach()
string(JOIN "|" cuda_samples_executable_arg ${cuda_samples_executables})
add_test(
  NAME cuda_samples_linkage_audit
  COMMAND "${CMAKE_COMMAND}"
    "-DEXECUTABLES=${cuda_samples_executable_arg}"
    "-DREADELF=${CMAKE_READELF}"
    -P "${CMAKE_CURRENT_LIST_DIR}/linkage_audit.cmake")
set_tests_properties(cuda_samples_linkage_audit PROPERTIES
  LABELS "cuda-samples;cuda-samples-linkage")

add_test(
  NAME cuda_samples_fatbin_audit
  COMMAND "${CMAKE_COMMAND}"
    "-DEXECUTABLES=${cuda_samples_executable_arg}"
    "-DCUOBJDUMP=${CUDAToolkit_BIN_DIR}/cuobjdump"
    -P "${CMAKE_CURRENT_LIST_DIR}/fatbin_audit.cmake")
set_tests_properties(cuda_samples_fatbin_audit PROPERTIES
  LABELS "cuda-samples;cuda-samples-fatbin")

find_program(BASH_EXECUTABLE bash)
if(NOT BASH_EXECUTABLE)
  message(FATAL_ERROR "bash is required for the CUDA Samples survey")
endif()
add_custom_target(cuda-samples-survey
  COMMAND "${BASH_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/run_survey.sh"
    "${CUDA_SAMPLES_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_samples/survey"
    "$<TARGET_FILE_DIR:cudart>"
    "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/SM7_QV100"
    "${PROJECT_SOURCE_DIR}"
    "${CUDAToolkit_TARGET_DIR}"
    "${CMAKE_COMMAND}"
    "${CMAKE_READELF}"
    "${CUDAToolkit_BIN_DIR}/cuobjdump"
  DEPENDS cudart
  USES_TERMINAL
  COMMENT "Running the CUDA Samples 11.8 compatibility survey")
