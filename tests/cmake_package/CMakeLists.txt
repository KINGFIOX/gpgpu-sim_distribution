cmake_minimum_required(VERSION 3.18)
project(GPGPUSimPackageConsumer LANGUAGES CUDA)

find_package(GPGPUSim CONFIG REQUIRED)
if (NOT TARGET GPGPUSim::cudart)
    message(FATAL_ERROR "GPGPUSim::cudart target was not exported")
endif ()
if (NOT GPGPUSIM_GPU_MODEL OR NOT GPGPUSIM_CUDA_ARCHITECTURE)
    message(FATAL_ERROR "GPGPU-Sim package does not describe its GPU model")
endif ()

get_target_property(cudart_configs GPGPUSim::cudart IMPORTED_CONFIGURATIONS)
foreach (config IN LISTS cudart_configs)
    get_target_property(cudart_location GPGPUSim::cudart
            "IMPORTED_LOCATION_${config}")
    if (cudart_location)
        break()
    endif ()
endforeach ()
if (NOT cudart_location)
    get_target_property(cudart_location GPGPUSim::cudart IMPORTED_LOCATION)
endif ()
if (NOT cudart_location OR NOT EXISTS "${cudart_location}")
    message(FATAL_ERROR
            "GPGPUSim::cudart does not point to a built library: ${cudart_location}")
endif ()

add_executable(gpgpusim_package_smoke package_smoke.cu)
set_target_properties(gpgpusim_package_smoke PROPERTIES
        CUDA_ARCHITECTURES "${GPGPUSIM_CUDA_ARCHITECTURE}-virtual"
        CUDA_RUNTIME_LIBRARY None)
target_link_libraries(gpgpusim_package_smoke PRIVATE GPGPUSim::cudart)
