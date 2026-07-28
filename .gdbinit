# Provides useful debugging commands for a running GPGPU-Sim instance.  Load
# it with "source /path/to/.gdbinit" after starting GDB with debug symbols.

printf "\n  ** loading GPGPU-Sim debugging macros... ** \n\n"

set print pretty
set print array-indexes
set unwindonsignal on

define dp
        set $gpusim = GPGPU_Context()->the_gpgpusim
        if ($gpusim == 0 || $gpusim->g_the_gpu == 0)
                printf "GPGPU-Sim is not initialized.\n"
        else
                call $gpusim->g_the_gpu->dump_pipeline((0x40|0x4|0x1),$arg0,0)
        end
end

document dp
Usage: dp <index>
Display pipeline state.
<index>: index of shader core you would like to see the pipeline state of

This function displays the state of the pipeline on a single shader core
(setting different values for the first argument of the call to
dump_pipeline will cause different information to be displayed--
see the source code for more details)
end

define dpc
        set $gpusim = GPGPU_Context()->the_gpgpusim
        if ($gpusim == 0 || $gpusim->g_the_gpu == 0)
                printf "GPGPU-Sim is not initialized.\n"
        else
                call $gpusim->g_the_gpu->dump_pipeline((0x40|0x4|0x1),$arg0,0)
                continue
        end
end

document dpc
Usage: dpc <index>
Display pipeline state, then continue to next breakpoint.
<index>: index of shader core you would like to see the pipeline state of

This version is useful if you set a breakpoint where gpu_sim_cycle is
incremented in gpu_sim_loop() in src/gpgpu-sim/gpu-sim.c
repeatedly hitting enter will advance to show the pipeline contents on
the next cycle.
end

define dm
        set $gpusim = GPGPU_Context()->the_gpgpusim
        if ($gpusim == 0 || $gpusim->g_the_gpu == 0)
                printf "GPGPU-Sim is not initialized.\n"
        else
                call $gpusim->g_the_gpu->dump_pipeline(0x10000|0x10000000,0,$arg0)
        end
end

define ptxdis
	set $addr = $arg0
	set $end = $arg1
	printf "disassemble instructions from 0x%x to 0x%x\n", $addr, $end
	while ($addr <= $end)
		set $insn = GPGPU_Context()->pc_to_instruction($addr)
		if ($insn == 0)
			set $addr = $addr + 1
		else
			printf "0x%04x (%4u)  : ", $addr, $addr
			call GPGPU_Context()->func_sim->ptx_print_insn($addr, stdout)
			call fflush(stdout)
			printf "\n"
			set $size = $insn->inst_size()
			if ($size == 0)
				set $size = 1
			end
			set $addr = $addr + $size
		end
	end
end

document ptxdis
Usage: ptxdis <start> <end>
Disassemble PTX instructions between <start> and <end> (PCs).
end

define ptxdis_func
	set $sid = $arg0
	set $gpusim = GPGPU_Context()->the_gpgpusim
	if ($gpusim == 0 || $gpusim->g_the_gpu == 0)
		printf "GPGPU-Sim is not initialized.\n"
	else
		set $shader_config = $gpusim->g_the_gpu->getShaderCoreConfig()
		set $cluster = $shader_config->sid_to_cluster($sid)
		set $cid = $shader_config->sid_to_cid($sid)
		set $core = $gpusim->g_the_gpu->m_cluster[$cluster]->m_core[$cid]
		set $threads = $core->get_thread_info()
		set $ptx_tinfo = $threads[$arg1]
		if ($ptx_tinfo == 0)
			printf "No thread %u on shader core %u.\n", $arg1, $sid
		else
			set $finfo = $ptx_tinfo->get_finfo()
			if ($finfo == 0)
				printf "Thread %u on shader core %u has no function.\n", $arg1, $sid
			else
				set $minpc = $finfo->get_start_PC()
				set $maxpc = $minpc + $finfo->m_instr_mem_size - 1
				printf "disassembly of function %s (min pc = %u, max pc = %u):\n", $finfo->get_name().c_str(), $minpc, $maxpc
				ptxdis $minpc $maxpc
			end
		end
	end
end

document ptxdis_func
Usage: ptxdis_func <shd_idx> <tid> (requires debug build)
<shd_idx>: shader core number
<tid>: thread ID
end

define ptx_tids2pcs
	set $sid = $arg2
	set $gpusim = GPGPU_Context()->the_gpgpusim
	if ($gpusim == 0 || $gpusim->g_the_gpu == 0)
		printf "GPGPU-Sim is not initialized.\n"
	else
		set $shader_config = $gpusim->g_the_gpu->getShaderCoreConfig()
		set $cluster = $shader_config->sid_to_cluster($sid)
		set $cid = $shader_config->sid_to_cid($sid)
		set $core = $gpusim->g_the_gpu->m_cluster[$cluster]->m_core[$cid]
		set $threads = $core->get_thread_info()
		set $i = 0
		while ($i < $arg1)
			set $tid = $arg0[$i]
			set $ptx_tinfo = $threads[$tid]
			if ($ptx_tinfo == 0)
				printf "%2u : tid = %3u  => <no thread>\n", $i, $tid
			else
				printf "%2u : tid = %3u  => pc = %u\n", $i, $tid, $ptx_tinfo->get_pc()
			end
			set $i = $i + 1
		end
	end
end

document ptx_tids2pcs
Usage: ptx_tids2pcs <tids> <tidslen> <shd_idx>
<tids>: array of tids
<tidslen>: length of <tids> array
<shd_idx>: shader core number
end
