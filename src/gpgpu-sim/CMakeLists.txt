list(APPEND gpgpusim_SRC addrdec.cc
    ${GPGPUSIM_EMBEDDED_CONFIG_SOURCE}
    dram.cc
    dram_sched.cc
    gpu-cache.cc
    gpu-misc.cc
    gpu-sim.cc
    hashing.cc
    histogram.cc
    icnt_wrapper.cc
    l2cache.cc
    local_interconnect.cc
    mem_fetch.cc
    mem_latency_stat.cc
    scoreboard.cc
    shader.cc
    stack.cc
    stat-tool.cc
    traffic_breakdown.cc
    visualizer.cc)

# Create libgpgpusim.a
add_library(gpgpusim STATIC ${gpgpusim_SRC})
target_include_directories(gpgpusim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src)
target_include_directories(gpgpusim PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
