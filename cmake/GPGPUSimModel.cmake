set(GPGPUSIM_GPU_MODEL "SM7_QV100" CACHE STRING
        "GPU model compiled into the GPGPU-Sim runtime")
set(GPGPUSIM_GPU_MODELS
        SM3_KEPLER_TITAN
        SM6_TITANX
        SM7_GV100
        SM7_QV100
        SM7_TITANV
        SM75_RTX2060
        SM75_RTX2060_S
        SM86_RTX3070)
set_property(CACHE GPGPUSIM_GPU_MODEL PROPERTY STRINGS ${GPGPUSIM_GPU_MODELS})

list(FIND GPGPUSIM_GPU_MODELS "${GPGPUSIM_GPU_MODEL}" model_index)
if (model_index EQUAL -1)
    string(JOIN ", " supported_models ${GPGPUSIM_GPU_MODELS})
    message(FATAL_ERROR
            "Unsupported GPGPUSIM_GPU_MODEL '${GPGPUSIM_GPU_MODEL}'. "
            "Supported models: ${supported_models}")
endif ()

if (GPGPUSIM_GPU_MODEL STREQUAL "SM3_KEPLER_TITAN")
    set(GPGPUSIM_CUDA_ARCHITECTURE 35)
    set(interconnect_config_name config_kepler_islip.icnt)
elseif (GPGPUSIM_GPU_MODEL STREQUAL "SM6_TITANX")
    set(GPGPUSIM_CUDA_ARCHITECTURE 61)
    set(interconnect_config_name config_pascal_islip.icnt)
elseif (GPGPUSIM_GPU_MODEL MATCHES "^SM7_(GV100|QV100|TITANV)$")
    set(GPGPUSIM_CUDA_ARCHITECTURE 70)
    set(interconnect_config_name config_volta_islip.icnt)
elseif (GPGPUSIM_GPU_MODEL MATCHES "^SM75_RTX2060(_S)?$")
    set(GPGPUSIM_CUDA_ARCHITECTURE 75)
    set(interconnect_config_name config_turing_islip.icnt)
elseif (GPGPUSIM_GPU_MODEL STREQUAL "SM86_RTX3070")
    set(GPGPUSIM_CUDA_ARCHITECTURE 86)
    set(interconnect_config_name config_ampere_islip.icnt)
endif ()

set(model_config_dir
        "${PROJECT_SOURCE_DIR}/configs/tested-cfgs/${GPGPUSIM_GPU_MODEL}")
set(gpgpusim_config_file "${model_config_dir}/gpgpusim.config")
set(interconnect_config_file
        "${model_config_dir}/${interconnect_config_name}")
foreach (config_file IN ITEMS
        "${gpgpusim_config_file}" "${interconnect_config_file}")
    if (NOT EXISTS "${config_file}")
        message(FATAL_ERROR "GPU model configuration not found: ${config_file}")
    endif ()
endforeach ()

file(READ "${gpgpusim_config_file}" GPGPUSIM_CONFIG_CONTENT)
file(READ "${interconnect_config_file}" GPGPUSIM_INTERCONNECT_CONFIG_CONTENT)
string(FIND "${GPGPUSIM_CONFIG_CONTENT}"
        ")GPGPUSIM_CONFIG\"" gpgpusim_delimiter_pos)
string(FIND "${GPGPUSIM_INTERCONNECT_CONFIG_CONTENT}"
        ")GPGPUSIM_ICNT\"" interconnect_delimiter_pos)
if (NOT gpgpusim_delimiter_pos EQUAL -1 OR
        NOT interconnect_delimiter_pos EQUAL -1)
    message(FATAL_ERROR
            "Selected GPU model configuration contains a reserved raw-string delimiter")
endif ()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${gpgpusim_config_file}" "${interconnect_config_file}")
set(generated_dir "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated_dir}")
set(GPGPUSIM_EMBEDDED_CONFIG_SOURCE
        "${generated_dir}/gpgpusim_embedded_config.cc")
configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/gpgpusim_embedded_config.cc.in"
        "${GPGPUSIM_EMBEDDED_CONFIG_SOURCE}" @ONLY)

message(STATUS
        "GPGPU-Sim GPU model: ${GPGPUSIM_GPU_MODEL} "
        "(compute_${GPGPUSIM_CUDA_ARCHITECTURE})")
