#!/usr/bin/env bash
set -euo pipefail

if (( $# != 9 )); then
  echo "usage: $0 SAMPLES_ROOT WORKSPACE LIB_DIR CONFIG_DIR GPGPUSIM_ROOT CUDA_ROOT CMAKE READELF CUOBJDUMP" >&2
  exit 2
fi

samples_root=$1
workspace=$2
simulator_lib_dir=$3
config_source_dir=$4
gpgpusim_root=$5
cuda_root=$6
cmake_command=$7
readelf_command=$8
cuobjdump_command=$9
survey_filter=${CUDA_SAMPLES_SURVEY_FILTER:-}
build_timeout=${CUDA_SAMPLES_SURVEY_BUILD_TIMEOUT:-300}
run_timeout=${CUDA_SAMPLES_SURVEY_RUN_TIMEOUT:-300}

for required in \
  "${samples_root}/Common" \
  "${samples_root}/Samples" \
  "${simulator_lib_dir}/libcudart.so.11.0" \
  "${config_source_dir}/gpgpusim.config" \
  "${config_source_dir}/config_volta_islip.icnt" \
  "${cuda_root}/bin/nvcc"; do
  if [[ ! -e ${required} ]]; then
    echo "required survey input is missing: ${required}" >&2
    exit 2
  fi
done
if [[ ! -x ${cuobjdump_command} ]]; then
  echo "required survey input is missing: ${cuobjdump_command}" >&2
  exit 2
fi

case ${workspace} in
  ""|/|"${samples_root}"|"${gpgpusim_root}")
    echo "refusing unsafe survey workspace: ${workspace}" >&2
    exit 2
    ;;
esac

shadow_root=${workspace}/source
run_root=${workspace}/run
log_root=${workspace}/logs
results_file=${workspace}/results.tsv
manifest_file=${workspace}/manifest.tsv
summary_file=${workspace}/summary.md

"${cmake_command}" -E remove_directory "${workspace}"
"${cmake_command}" -E make_directory \
  "${shadow_root}" "${run_root}" "${log_root}"
"${cmake_command}" -E copy_directory \
  "${samples_root}/Common" "${shadow_root}/Common"

printf 'sample\tclassification\treason\n' >"${manifest_file}"
printf 'sample\tstatus\texit_code\tlog\n' >"${results_file}"

declare -A counts=()

record_result() {
  local sample=$1
  local status=$2
  local exit_code=$3
  local log=$4
  printf '%s\t%s\t%s\t%s\n' \
    "${sample}" "${status}" "${exit_code}" "${log}" >>"${results_file}"
  counts["${status}"]=$(( ${counts["${status}"]:-0} + 1 ))
}

classify_sample() {
  local relative_path=$1
  local source_dir=$2
  local sample_name=${relative_path##*/}

  case ${relative_path} in
    Samples/4_CUDA_Libraries/*)
      printf 'excluded\tcuda_library'
      return
      ;;
  esac

  if [[ ${sample_name} =~ (Drv|drv|DynlinkJIT|cuHook|ptxjit) ]]; then
    printf 'excluded\tdriver_or_jit_api'
    return
  fi
  if [[ ${sample_name} =~ (nvrtc|NVRTC) ]]; then
    printf 'excluded\tnvrtc'
    return
  fi
  if [[ ${sample_name} =~ (CUDA2GL|GL|EGL|Vulkan|D3D|DirectX) ]]; then
    printf 'excluded\tgraphics_interop'
    return
  fi
  if [[ ${sample_name} =~ (MPI|OpenMP|IPC|P2P|MultiGPU|CrossGPU|NvSci|MMAP) ]]; then
    printf 'excluded\tprocess_or_multi_gpu_integration'
    return
  fi
  if grep -REiq \
      '(cublas|cufft|curand|cusparse|cusolver|npp\.h|nvjpeg|thrust/|nvrtc)' \
      --include='Makefile' --include='*.cu' --include='*.cpp' --include='*.h' \
      "${source_dir}"; then
    printf 'excluded\tauxiliary_cuda_library'
    return
  fi
  if grep -REq \
      '\<cu[A-Z][A-Za-z0-9_]*[[:space:]]*\(' \
      --include='*.cu' --include='*.cpp' --include='*.cc' --include='*.h' \
      "${source_dir}"; then
    printf 'excluded\tdriver_api'
    return
  fi

  case ${relative_path} in
    Samples/0_Introduction/*|Samples/1_Utilities/*|\
    Samples/2_Concepts_and_Techniques/*|Samples/3_CUDA_Features/*|\
    Samples/5_Domain_Specific/*|Samples/6_Performance/*)
      printf 'included\tstandalone_cuda_runtime'
      ;;
    *)
      printf 'excluded\tout_of_scope_category'
      ;;
  esac
}

run_sample() {
  local source_dir=$1
  local relative_path=$2
  local sample_name=${relative_path##*/}
  local result_key=${relative_path//\//__}
  local destination=${shadow_root}/${relative_path}
  local build_log=${log_root}/${result_key}.build.log
  local run_log=${log_root}/${result_key}.run.log
  local executable=${destination}/${sample_name}
  local sample_run_dir=${run_root}/${result_key}
  local rc=0

  "${cmake_command}" -E make_directory "$(dirname "${destination}")"
  "${cmake_command}" -E copy_directory "${source_dir}" "${destination}"

  if timeout --signal=TERM --kill-after=10s "${build_timeout}s" \
      make -j1 -C "${destination}" \
        CUDA_PATH="${cuda_root}" SMS=70 \
        GENCODE_FLAGS='-gencode arch=compute_70,code=compute_70' \
        EXTRA_NVCCFLAGS=--cudart=shared >"${build_log}" 2>&1; then
    rc=0
  else
    rc=$?
    if (( rc == 124 )); then
      record_result "${relative_path}" build_timeout "${rc}" "${build_log}"
    else
      record_result "${relative_path}" build_failed "${rc}" "${build_log}"
    fi
    return
  fi

  if [[ ! -x ${executable} ]]; then
    if grep -qi 'waiv' "${build_log}"; then
      record_result "${relative_path}" waived 2 "${build_log}"
    else
      record_result "${relative_path}" no_executable 1 "${build_log}"
    fi
    return
  fi

  if ! "${readelf_command}" -d "${executable}" | \
      grep -q 'Shared library: \[libcudart\.so\.11\.0\]'; then
    record_result "${relative_path}" invalid_linkage 1 "${build_log}"
    return
  fi
  if "${readelf_command}" -d "${executable}" | \
      grep -Eq 'Shared library: \[lib(cuda\.so|cublas|cufft|curand|cusolver|cusparse|npp|nvrtc)'; then
    record_result "${relative_path}" invalid_linkage 1 "${build_log}"
    return
  fi

  local elf_listing
  local ptx_listing
  if ! elf_listing=$("${cuobjdump_command}" --list-elf "${executable}" 2>&1); then
    if [[ ${elf_listing} == *"does not contain device code"* ]]; then
      :
    else
      printf '\nPTX-only audit (ELF listing):\n%s\n' \
        "${elf_listing}" >>"${build_log}"
      record_result "${relative_path}" invalid_fatbin 1 "${build_log}"
      return
    fi
  elif grep -Eq '(^|$)[[:space:]]*ELF file[[:space:]]' <<<"${elf_listing}"; then
    printf '\nPTX-only audit (ELF listing):\n%s\n' \
      "${elf_listing}" >>"${build_log}"
    record_result "${relative_path}" invalid_fatbin 1 "${build_log}"
    return
  else
    if ! ptx_listing=$("${cuobjdump_command}" --list-ptx "${executable}" 2>&1) ||
       [[ ${ptx_listing} != *"PTX file"* ]]; then
      printf '\nPTX-only audit (PTX listing):\n%s\n' \
        "${ptx_listing}" >>"${build_log}"
      record_result "${relative_path}" invalid_fatbin 1 "${build_log}"
      return
    fi
  fi

  "${cmake_command}" -E make_directory "${sample_run_dir}"
  "${cmake_command}" -E copy_if_different \
    "${config_source_dir}/gpgpusim.config" \
    "${sample_run_dir}/gpgpusim.config"
  "${cmake_command}" -E copy_if_different \
    "${config_source_dir}/config_volta_islip.icnt" \
    "${sample_run_dir}/config_volta_islip.icnt"

  if timeout --signal=TERM --kill-after=10s "${run_timeout}s" \
      env GPGPUSIM_ROOT="${gpgpusim_root}" \
        CUDA_INSTALL_PATH="${cuda_root}" \
        PTX_SIM_MODE_FUNC=1 \
        PATH="${cuda_root}/bin:${PATH}" \
        LD_LIBRARY_PATH="${simulator_lib_dir}" \
        "${cmake_command}" -E chdir "${sample_run_dir}" \
        "${executable}" >"${run_log}" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  case ${rc} in
    0) record_result "${relative_path}" passed 0 "${run_log}" ;;
    2) record_result "${relative_path}" waived 2 "${run_log}" ;;
    124) record_result "${relative_path}" timeout 124 "${run_log}" ;;
    *) record_result "${relative_path}" runtime_failed "${rc}" "${run_log}" ;;
  esac
}

while IFS= read -r makefile; do
  source_dir=$(dirname "${makefile}")
  relative_path=${source_dir#"${samples_root}/"}
  classification=$(classify_sample "${relative_path}" "${source_dir}")
  class=${classification%%$'\t'*}
  reason=${classification#*$'\t'}
  printf '%s\t%s\t%s\n' \
    "${relative_path}" "${class}" "${reason}" >>"${manifest_file}"

  if [[ ${class} == excluded ]]; then
    record_result "${relative_path}" excluded 0 "${reason}"
    continue
  fi
  if [[ -n ${survey_filter} && ! ${relative_path} =~ ${survey_filter} ]]; then
    record_result "${relative_path}" filtered 0 "filter"
    continue
  fi

  echo "[cuda-samples] ${relative_path}"
  run_sample "${source_dir}" "${relative_path}"
done < <(find "${samples_root}/Samples" -name Makefile -type f | sort)

{
  echo '# CUDA Samples 11.8 compatibility survey'
  echo
  echo '| Status | Count |'
  echo '|---|---:|'
  while IFS= read -r status; do
    printf '| %s | %d |\n' "${status}" "${counts[${status}]}"
  done < <(printf '%s\n' "${!counts[@]}" | sort)
  echo
  printf 'Detailed results: `%s`\n' "${results_file}"
} >"${summary_file}"

cat "${summary_file}"
