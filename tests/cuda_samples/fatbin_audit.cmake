if(NOT DEFINED EXECUTABLES OR NOT DEFINED CUOBJDUMP)
  message(FATAL_ERROR "EXECUTABLES and CUOBJDUMP are required")
endif()

string(REPLACE "|" ";" executable_list "${EXECUTABLES}")
foreach(executable IN LISTS executable_list)
  if(NOT EXISTS "${executable}")
    message(FATAL_ERROR "CUDA sample executable not found: ${executable}")
  endif()

  execute_process(
    COMMAND "${CUOBJDUMP}" --list-elf "${executable}"
    OUTPUT_VARIABLE elf_listing
    ERROR_VARIABLE elf_errors
    RESULT_VARIABLE elf_result)
  if(NOT elf_result EQUAL 0)
    if(elf_errors MATCHES "does not contain device code")
      continue()
    endif()
    message(FATAL_ERROR
      "cuobjdump --list-elf failed for ${executable}: ${elf_errors}")
  endif()
  if(elf_listing MATCHES "(^|\\n)[ \t]*ELF file[ \t]")
    message(FATAL_ERROR
      "${executable} contains device ELF/cubin data:\n${elf_listing}")
  endif()

  execute_process(
    COMMAND "${CUOBJDUMP}" --list-ptx "${executable}"
    OUTPUT_VARIABLE ptx_listing
    ERROR_VARIABLE ptx_errors
    RESULT_VARIABLE ptx_result)
  if(NOT ptx_result EQUAL 0 OR NOT ptx_listing MATCHES "PTX file")
    message(FATAL_ERROR
      "${executable} does not contain extractable PTX:\n"
      "${ptx_listing}${ptx_errors}")
  endif()
endforeach()
