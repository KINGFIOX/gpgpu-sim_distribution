set(CUDA_SAMPLES_SOURCE_DIR
        "${PROJECT_SOURCE_DIR}/third_party/cuda-samples")
set(CUDA_SAMPLES_PINNED_COMMIT
        "81992093d2b8c33cab22dbf6852c070c330f1715")

if (NOT EXISTS "${CUDA_SAMPLES_SOURCE_DIR}/README.md")
    message(FATAL_ERROR
            "CUDA Samples is not initialized. Run: git submodule update --init "
            "third_party/cuda-samples")
endif ()

execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${CUDA_SAMPLES_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE cuda_samples_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE cuda_samples_git_result)
if (cuda_samples_git_result OR
        NOT cuda_samples_commit STREQUAL CUDA_SAMPLES_PINNED_COMMIT)
    message(FATAL_ERROR
            "CUDA Samples must be at ${CUDA_SAMPLES_PINNED_COMMIT}; found "
            "${cuda_samples_commit}")
endif ()

set(CUDA_SAMPLES_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_samples/bin")
set(CUDA_SAMPLES_RUN_ROOT "${CMAKE_CURRENT_BINARY_DIR}/cuda_samples/run")
set(CUDA_SAMPLES_GATE_TARGETS "")

function(gpgpusim_add_cuda_sample name sample_path)
    cmake_parse_arguments(SAMPLE "FUNCTIONAL;PERFORMANCE" "TIMEOUT"
            "SOURCES;ARGS" ${ARGN})
    if (SAMPLE_UNPARSED_ARGUMENTS)
        string(JOIN " " unparsed_arguments ${SAMPLE_UNPARSED_ARGUMENTS})
        message(FATAL_ERROR
                "CUDA sample ${name} has unknown arguments: ${unparsed_arguments}")
    endif ()
    if (SAMPLE_KEYWORDS_MISSING_VALUES)
        string(JOIN ", " missing_values ${SAMPLE_KEYWORDS_MISSING_VALUES})
        message(FATAL_ERROR
                "CUDA sample ${name} has arguments without values: ${missing_values}")
    endif ()
    if (NOT SAMPLE_SOURCES)
        message(FATAL_ERROR "CUDA sample ${name} has no sources")
    endif ()
    if (TARGET ${name})
        message(FATAL_ERROR "CUDA sample target already exists: ${name}")
    endif ()
    if (NOT SAMPLE_TIMEOUT)
        set(SAMPLE_TIMEOUT 180)
    elseif (NOT SAMPLE_TIMEOUT MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "CUDA sample ${name} TIMEOUT must be a positive integer")
    endif ()
    if (NOT SAMPLE_FUNCTIONAL AND NOT SAMPLE_PERFORMANCE)
        set(SAMPLE_FUNCTIONAL TRUE)
    endif ()

    set(sources "")
    foreach (source IN LISTS SAMPLE_SOURCES)
        set(source_path "${CUDA_SAMPLES_SOURCE_DIR}/${sample_path}/${source}")
        if (NOT EXISTS "${source_path}")
            message(FATAL_ERROR "CUDA sample source not found: ${source_path}")
        endif ()
        list(APPEND sources "${source_path}")
    endforeach ()

    add_executable(${name} ${sources})
    target_include_directories(${name} PRIVATE
            "${CUDA_SAMPLES_SOURCE_DIR}/Common"
            ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE cudart)
    set_target_properties(${name} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES 70-virtual
            CUDA_RUNTIME_LIBRARY None
            BUILD_RPATH "$<TARGET_FILE_DIR:cudart>"
            RUNTIME_OUTPUT_DIRECTORY "${CUDA_SAMPLES_BINARY_DIR}")

    set(sample_modes "")
    if (SAMPLE_FUNCTIONAL)
        list(APPEND sample_modes functional)
    endif ()
    if (SAMPLE_PERFORMANCE)
        list(APPEND sample_modes performance)
    endif ()

    foreach (mode IN LISTS sample_modes)
        if (mode STREQUAL "functional")
            set(ptx_sim_mode_func 1)
        else ()
            set(ptx_sim_mode_func 0)
        endif ()

        set(test_name "${name}.${mode}")
        set(run_dir "${CUDA_SAMPLES_RUN_ROOT}/${name}/${mode}")
        file(MAKE_DIRECTORY "${run_dir}")
        configure_file(
                "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/SM7_QV100/gpgpusim.config"
                "${run_dir}/gpgpusim.config" COPYONLY)
        configure_file(
                "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/SM7_QV100/config_volta_islip.icnt"
                "${run_dir}/config_volta_islip.icnt" COPYONLY)

        add_test(NAME "${test_name}"
                COMMAND "$<TARGET_FILE:${name}>" ${SAMPLE_ARGS})
        set_tests_properties("${test_name}" PROPERTIES
                LABELS "cuda-samples;cuda-samples-gate;cuda-samples-${mode}"
                RUN_SERIAL TRUE
                TIMEOUT "${SAMPLE_TIMEOUT}"
                WORKING_DIRECTORY "${run_dir}"
                ENVIRONMENT
                "GPGPUSIM_ROOT=${PROJECT_SOURCE_DIR};CUDA_INSTALL_PATH=${CUDAToolkit_TARGET_DIR};PTX_SIM_MODE_FUNC=${ptx_sim_mode_func};PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH}")
    endforeach ()

    set(CUDA_SAMPLES_GATE_TARGETS
            ${CUDA_SAMPLES_GATE_TARGETS} ${name} PARENT_SCOPE)
endfunction()

gpgpusim_add_cuda_sample(deviceQuery Samples/1_Utilities/deviceQuery
        FUNCTIONAL PERFORMANCE
        SOURCES deviceQuery.cpp)
gpgpusim_add_cuda_sample(vectorAdd Samples/0_Introduction/vectorAdd
        FUNCTIONAL PERFORMANCE
        SOURCES vectorAdd.cu)
gpgpusim_add_cuda_sample(simpleTemplates Samples/0_Introduction/simpleTemplates
        FUNCTIONAL PERFORMANCE
        SOURCES simpleTemplates.cu)
gpgpusim_add_cuda_sample(simpleVoteIntrinsics
        Samples/0_Introduction/simpleVoteIntrinsics
        FUNCTIONAL PERFORMANCE
        SOURCES simpleVoteIntrinsics.cu)
# The detailed 64 x 256 atomic workload takes about 330 seconds on this model.
gpgpusim_add_cuda_sample(simpleAtomicIntrinsics
        Samples/0_Introduction/simpleAtomicIntrinsics
        FUNCTIONAL PERFORMANCE
        SOURCES simpleAtomicIntrinsics.cu simpleAtomicIntrinsics_cpu.cpp
        TIMEOUT 600)
gpgpusim_add_cuda_sample(clock Samples/0_Introduction/clock
        FUNCTIONAL PERFORMANCE
        SOURCES clock.cu)

set(cuda_samples_executables "")
foreach (target IN LISTS CUDA_SAMPLES_GATE_TARGETS)
    list(APPEND cuda_samples_executables "$<TARGET_FILE:${target}>")
endforeach ()
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
