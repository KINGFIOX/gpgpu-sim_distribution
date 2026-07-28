set(CUDA_SAMPLES_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(CUDA_SAMPLES_RUN_ROOT "${CMAKE_CURRENT_BINARY_DIR}/run")
set(CUDA_SAMPLES_GATE_TARGETS "")

function(gpgpusim_add_cuda_sample name)
    cmake_parse_arguments(SAMPLE "FUNCTIONAL;PERFORMANCE;OPENMP"
            "TIMEOUT;PASS_REGEX" "SOURCES;ARGS;DATA;INCLUDE_DIRS" ${ARGN})
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

    add_executable(${name} ${SAMPLE_SOURCES})
    target_include_directories(${name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/Common"
            ${SAMPLE_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE cudart)
    if (SAMPLE_OPENMP)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_compile_options(${name} PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif ()
    set_target_properties(${name} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${GPGPUSIM_CUDA_ARCHITECTURE}-virtual"
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
        foreach (data_file IN LISTS SAMPLE_DATA)
            set(data_source "${CMAKE_CURRENT_SOURCE_DIR}/${name}/${data_file}")
            if (NOT EXISTS "${data_source}")
                message(FATAL_ERROR
                        "CUDA sample ${name} data file does not exist: ${data_file}")
            endif ()
            get_filename_component(data_directory "${data_file}" DIRECTORY)
            file(MAKE_DIRECTORY "${run_dir}/${data_directory}")
            configure_file("${data_source}" "${run_dir}/${data_file}" COPYONLY)
        endforeach ()

        add_test(NAME "${test_name}"
                COMMAND "$<TARGET_FILE:${name}>" ${SAMPLE_ARGS})
        set_tests_properties("${test_name}" PROPERTIES
                LABELS "cuda-samples;cuda-samples-gate;cuda-samples-${mode}"
                RUN_SERIAL TRUE
                TIMEOUT "${SAMPLE_TIMEOUT}"
                WORKING_DIRECTORY "${run_dir}"
                ENVIRONMENT
                "GPGPUSIM_ROOT=${PROJECT_SOURCE_DIR};CUDA_INSTALL_PATH=${CUDAToolkit_TARGET_DIR};PTX_SIM_MODE_FUNC=${ptx_sim_mode_func};PATH=${CUDAToolkit_BIN_DIR}:$ENV{PATH}")
        if (SAMPLE_PASS_REGEX)
            set_tests_properties("${test_name}" PROPERTIES
                    PASS_REGULAR_EXPRESSION "${SAMPLE_PASS_REGEX}")
        endif ()
    endforeach ()

    set(CUDA_SAMPLES_GATE_TARGETS
            ${CUDA_SAMPLES_GATE_TARGETS} ${name} PARENT_SCOPE)
endfunction()

gpgpusim_add_cuda_sample(deviceQuery
        FUNCTIONAL PERFORMANCE
        SOURCES deviceQuery/deviceQuery.cpp)
gpgpusim_add_cuda_sample(vectorAdd
        FUNCTIONAL PERFORMANCE
        SOURCES vectorAdd/vectorAdd.cu)
gpgpusim_add_cuda_sample(simpleTemplates
        FUNCTIONAL PERFORMANCE
        SOURCES simpleTemplates/simpleTemplates.cu)
gpgpusim_add_cuda_sample(simpleVoteIntrinsics
        FUNCTIONAL PERFORMANCE
        SOURCES simpleVoteIntrinsics/simpleVoteIntrinsics.cu)
# The detailed 64 x 256 atomic workload takes about 330 seconds on this model.
gpgpusim_add_cuda_sample(simpleAtomicIntrinsics
        FUNCTIONAL PERFORMANCE
        SOURCES simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu
        simpleAtomicIntrinsics/simpleAtomicIntrinsics_cpu.cpp
        TIMEOUT 600)
gpgpusim_add_cuda_sample(clock
        FUNCTIONAL PERFORMANCE
        SOURCES clock/clock.cu)

gpgpusim_add_cuda_sample(asyncAPI
        FUNCTIONAL PERFORMANCE
        SOURCES asyncAPI/asyncAPI.cu
        ARGS --size=4096)
gpgpusim_add_cuda_sample(bandwidthTest
        FUNCTIONAL PERFORMANCE
        SOURCES bandwidthTest/bandwidthTest.cu
        ARGS --mode=range --start=1024 --end=4096 --increment=1024
        --htod --dtoh --dtod
        PASS_REGEX "Result = PASS")
gpgpusim_add_cuda_sample(cppIntegration
        FUNCTIONAL PERFORMANCE
        SOURCES cppIntegration/main.cpp cppIntegration/cppIntegration.cu
        cppIntegration/cppIntegration_gold.cpp
        PASS_REGEX "Hello World")
gpgpusim_add_cuda_sample(cppOverload
        FUNCTIONAL PERFORMANCE
        SOURCES cppOverload/cppOverload.cu
        PASS_REGEX "PASSED")
gpgpusim_add_cuda_sample(cudaOpenMP
        FUNCTIONAL OPENMP
        SOURCES cudaOpenMP/cudaOpenMP.cu)
gpgpusim_add_cuda_sample(dwtHaar1D
        FUNCTIONAL
        SOURCES dwtHaar1D/dwtHaar1D.cu
        DATA data/regression.gold.dat data/regression_2_14.gold.dat
        data/regression_2_18.gold.dat data/signal.dat data/signal_2_14.dat
        data/signal_2_18.dat
        PASS_REGEX "Test success")
gpgpusim_add_cuda_sample(fp16ScalarProduct
        FUNCTIONAL
        SOURCES fp16ScalarProduct/fp16ScalarProduct.cu
        PASS_REGEX "fp16ScalarProduct PASSED")
gpgpusim_add_cuda_sample(FDTD3d
        FUNCTIONAL
        SOURCES FDTD3d/src/FDTD3d.cpp FDTD3d/src/FDTD3dGPU.cu
        FDTD3d/src/FDTD3dReference.cpp
        INCLUDE_DIRS FDTD3d/inc
        ARGS --dimx=96 --dimy=96 --dimz=96 --timesteps=1 --block-size=128
        --noprompt
        TIMEOUT 180)
gpgpusim_add_cuda_sample(inlinePTX
        FUNCTIONAL PERFORMANCE
        SOURCES inlinePTX/inlinePTX.cu
        PASS_REGEX "Test Successful")
gpgpusim_add_cuda_sample(matrixMul
        FUNCTIONAL PERFORMANCE
        SOURCES matrixMul/matrixMul.cu
        ARGS --wA=32 --hA=32 --wB=32 --hB=32 --iterations=1
        PASS_REGEX "Result = PASS")
gpgpusim_add_cuda_sample(newdelete
        FUNCTIONAL PERFORMANCE
        SOURCES newdelete/newdelete.cu
        TIMEOUT 600
        PASS_REGEX "Test Summary: 3/3 succesfully run")
gpgpusim_add_cuda_sample(interval
        FUNCTIONAL PERFORMANCE
        SOURCES interval/interval.cu
        INCLUDE_DIRS interval
        TIMEOUT 600
        PASS_REGEX "Found 2 intervals")
gpgpusim_add_cuda_sample(reduction
        FUNCTIONAL
        SOURCES reduction/reduction.cpp reduction/reduction_kernel.cu
        ARGS --n=4096 --threads=128 --kernel=0 --maxblocks=64
        PASS_REGEX "Test passed")
gpgpusim_add_cuda_sample(scalarProd
        FUNCTIONAL
        SOURCES scalarProd/scalarProd.cu scalarProd/scalarProd_cpu.cpp
        PASS_REGEX "Test passed")
gpgpusim_add_cuda_sample(simpleOccupancy
        FUNCTIONAL
        SOURCES simpleOccupancy/simpleOccupancy.cu
        PASS_REGEX "Test PASSED")
gpgpusim_add_cuda_sample(simplePrintf
        FUNCTIONAL PERFORMANCE
        SOURCES simplePrintf/simplePrintf.cu)
gpgpusim_add_cuda_sample(simpleZeroCopy
        FUNCTIONAL
        SOURCES simpleZeroCopy/simpleZeroCopy.cu
        ARGS --use_generic_memory)
gpgpusim_add_cuda_sample(template
        FUNCTIONAL PERFORMANCE
        SOURCES template/template.cu template/template_cpu.cpp)
gpgpusim_add_cuda_sample(threadFenceReduction
        FUNCTIONAL
        SOURCES threadFenceReduction/threadFenceReduction.cu
        ARGS --n=4096 --threads=128 --maxblocks=64 --multipass)
gpgpusim_add_cuda_sample(transpose
        FUNCTIONAL
        SOURCES transpose/transpose.cu
        ARGS --dimX=32 --dimY=32
        TIMEOUT 240
        PASS_REGEX "Test passed")

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
        -P "${CMAKE_CURRENT_SOURCE_DIR}/linkage_audit.cmake")
set_tests_properties(cuda_samples_linkage_audit PROPERTIES
        LABELS "cuda-samples;cuda-samples-linkage")

add_test(
        NAME cuda_samples_fatbin_audit
        COMMAND "${CMAKE_COMMAND}"
        "-DEXECUTABLES=${cuda_samples_executable_arg}"
        "-DCUOBJDUMP=${CUDAToolkit_BIN_DIR}/cuobjdump"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/fatbin_audit.cmake")
set_tests_properties(cuda_samples_fatbin_audit PROPERTIES
        LABELS "cuda-samples;cuda-samples-fatbin")
