// This file created from cuda_runtime_api.h distributed with CUDA 1.1
// Changes Copyright 2009,  Tor M. Aamodt, Ali Bakhoda and George L. Yuan
// University of British Columbia

/*
 * cuda_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
 * George L. Yuan and the University of British Columbia, Vancouver,
 * BC V6T 1Z4, All Rights Reserved.
 *
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and
 * benchmarks/template/ are derived from the CUDA SDK available from
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from
 * src/intersim/ are derived from Booksim (a simulator provided with the
 * textbook "Principles and Practices of Interconnection Networks" available
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by
 * the corresponding legal terms and conditions set forth separately (original
 * copyright notices are left in files from these sources and where we have
 * modified a file our copyright notice appears before the original copyright
 * notice).
 *
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed separately by NVIDIA under separate terms and
 * conditions.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use
 * only.
 *
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia,
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/gpgpusim_entrypoint.h"
#include "../src/stream_manager.h"
#include "cuda_api_object.h"
#include "gpgpu_context.h"

#include <pthread.h>
#include <semaphore.h>

static_assert(CUDART_VERSION == 11080,
              "libcudart must be built with CUDA Toolkit 11.8 headers");

/*DEVICE_BUILTIN*/
struct cudaArray {
  void *devPtr;
  int devPtr32;
  struct cudaChannelFormatDesc desc;
  int width;
  int height;
  int size;  // in bytes
  unsigned dimensions;
};

cudaError_t g_last_cudaError = cudaSuccess;

void register_ptx_function(const char *name, function_info *impl) {
  // no longer need this
}

#define __my_func__ __PRETTY_FUNCTION__

struct _cuda_device_id *gpgpu_context::GPGPUSim_Init() {
  _cuda_device_id *the_device = the_gpgpusim->the_cude_device;
  if (!the_device) {
    gpgpu_sim *the_gpu = gpgpu_ptx_sim_init_perf();

    cudaDeviceProp *prop = (cudaDeviceProp *)calloc(sizeof(cudaDeviceProp), 1);
    snprintf(prop->name, 256, "GPGPU-Sim");
    prop->major = the_gpu->compute_capability_major();
    prop->minor = the_gpu->compute_capability_minor();
    prop->totalGlobalMem = 0x80000000 /* 2 GB */;
    prop->memPitch = 0;
    if (prop->major >= 2) {
      prop->maxThreadsPerBlock = 1024;
      prop->maxThreadsDim[0] = 1024;
      prop->maxThreadsDim[1] = 1024;
    } else {
      prop->maxThreadsPerBlock = 512;
      prop->maxThreadsDim[0] = 512;
      prop->maxThreadsDim[1] = 512;
    }

    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 0x40000000;
    prop->maxGridSize[1] = 0x40000000;
    prop->maxGridSize[2] = 0x40000000;
    prop->totalConstMem = 0x40000000;
    prop->textureAlignment = 0;
    //        * TODO: Update the .config and xml files of all GPU config files
    //        with new value of sharedMemPerBlock and regsPerBlock
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
    prop->regsPerMultiprocessor = the_gpu->num_registers_per_core();
    prop->sharedMemPerMultiprocessor = the_gpu->shared_mem_size();
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
    prop->regsPerBlock = the_gpu->num_registers_per_block();
    prop->warpSize = the_gpu->wrp_size();
    prop->clockRate = the_gpu->shader_clock();
    prop->multiProcessorCount = the_gpu->get_config().num_shader();
    prop->maxThreadsPerMultiProcessor = the_gpu->threads_per_core();
    prop->computeMode = cudaComputeModeDefault;
    prop->canMapHostMemory = 1;
    the_gpu->set_prop(prop);
    the_gpgpusim->the_cude_device = new _cuda_device_id(the_gpu);
    the_device = the_gpgpusim->the_cude_device;
  }
  start_sim_thread(1);
  return the_device;
}

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx) {
  // static CUctx_st *the_context = NULL;
  CUctx_st *the_context = ctx->the_gpgpusim->the_context;
  if (the_context == NULL) {
    _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();
    ctx->the_gpgpusim->the_context = new CUctx_st(the_gpu);
    the_context = ctx->the_gpgpusim->the_context;
  }
  return the_context;
}

/// singleton pattern
gpgpu_context *GPGPU_Context() {
  static gpgpu_context *gpgpu_ctx = NULL;
  if (gpgpu_ctx == NULL) {
    gpgpu_ctx = new gpgpu_context();
  }
  return gpgpu_ctx;
}

void ptxinfo_data::ptxinfo_addinfo() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  if (!get_ptxinfo_kname()) {
    /* This info is not per kernel (since CUDA 5.0 some info (e.g. gmem, and
     * cmem) is added at the beginning for the whole binary ) */
    print_ptxinfo();
    context->add_ptxinfo(get_ptxinfo());
    clear_ptxinfo();
    return;
  }
  if (!strcmp("__cuda_dummy_entry__", get_ptxinfo_kname())) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    clear_ptxinfo();
    return;
  }
  print_ptxinfo();
  context->add_ptxinfo(get_ptxinfo_kname(), get_ptxinfo());
  clear_ptxinfo();
}

void announce_call(const char *func) {
  printf("\n\nGPGPU-Sim PTX: CUDA API function \"%s\" has been called.\n",
         func);
  fflush(stdout);
}

typedef std::map<unsigned, CUevent_st *> event_tracker_t;

int CUevent_st::m_next_event_uid;
event_tracker_t g_timer_events;

//! Return the executable file of the process containing the PTX code
//!
//! This Function returns the executable file ran by the process.  This
//! executable is supposed to contain the PTX code.  It provides a workaround
//! for processes running on valgrind by dereferencing /proc/<pid>/exe within
//! the GPGPU-Sim process before calling cuobjdump to extract PTX.  This is
//! needed because valgrind uses x86 emulation to detect memory leaks.  Other
//! processes (e.g. cuobjdump) reading /proc/<pid>/exe will see the emulator
//! executable instead of the application binary.
//!
std::string get_app_binary() {
  char self_exe_path[1025];
  ssize_t path_length = readlink("/proc/self/exe", self_exe_path, 1024);
  assert(path_length != -1);
  self_exe_path[path_length] = '\0';

  printf("self exe links to: %s\n", self_exe_path);
  return self_exe_path;
}

/*******************************************************************************
 * Add internal cuda runtime API call to accept gpgpu_context *
 *******************************************************************************/
cudaError_t cudaSetDeviceInternal(int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // set the active device to run cuda
  if (device >= 0 && device < ctx->GPGPUSim_Init()->num_devices()) {
    ctx->api->g_active_device = device;
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

cudaError_t cudaGetDeviceInternal(int *device,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *device = ctx->api->g_active_device;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetLimitInternal(
    size_t *pValue, cudaLimit limit, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (!pValue) return g_last_cudaError = cudaErrorInvalidValue;
  std::map<int, size_t>::const_iterator configured_limit =
      ctx->api->device_limits.find(static_cast<int>(limit));
  if (configured_limit != ctx->api->device_limits.end()) {
    *pValue = configured_limit->second;
    return g_last_cudaError = cudaSuccess;
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  const struct cudaDeviceProp *prop = dev->get_prop();
  const gpgpu_sim_config &config = dev->get_gpgpu()->get_config();
  switch (limit) {
    case 0:  // cudaLimitStackSize
      *pValue = config.stack_limit();
      break;
    case 2:  // cudaLimitMallocHeapSize
      *pValue = config.heap_limit();
      break;
    case 3:  // cudaLimitDevRuntimeSyncDepth
      if (prop->major > 2) {
        *pValue = config.sync_depth_limit();
        break;
      }
      return g_last_cudaError = cudaErrorUnsupportedLimit;
    case 4:  // cudaLimitDevRuntimePendingLaunchCount
      if (prop->major > 2) {
        *pValue = config.pending_launch_count_limit();
        break;
      }
      return g_last_cudaError = cudaErrorUnsupportedLimit;
    default:
      return g_last_cudaError = cudaErrorUnsupportedLimit;
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceSetLimitInternal(
    cudaLimit limit, size_t value, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  const struct cudaDeviceProp *prop = dev->get_prop();
  switch (limit) {
    case cudaLimitStackSize:
    case cudaLimitMallocHeapSize:
      break;
    case cudaLimitDevRuntimeSyncDepth:
    case cudaLimitDevRuntimePendingLaunchCount:
      if (prop->major <= 2) return g_last_cudaError = cudaErrorUnsupportedLimit;
      break;
    default:
      return g_last_cudaError = cudaErrorUnsupportedLimit;
  }
  ctx->api->device_limits[static_cast<int>(limit)] = value;
  return g_last_cudaError = cudaSuccess;
}

static void validate_fatbin_handle(cuda_runtime_api *api,
                                   void **fatbin_handle) {
  if (!api->fatbin_registered || fatbin_handle != api->fatbin_handle()) {
    fprintf(stderr, "GPGPU-Sim PTX: invalid fat binary handle\n");
    exit(EXIT_FAILURE);
  }
}

void **cudaRegisterFatBinaryInternal(void *fatCubin,
                                     gpgpu_context *gpgpu_ctx = NULL) {
  (void)fatCubin;
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (ctx->api->fatbin_registered) {
    fprintf(stderr, "GPGPU-Sim PTX: multiple fat binaries are unsupported\n");
    exit(EXIT_FAILURE);
  }

  ctx->api->fatbin_registered = true;
  void **fatbin_handle = ctx->api->fatbin_handle();
  printf("GPGPU-Sim PTX: __cudaRegisterFatBinary, fatbin_handle = %p\n",
         (void *)fatbin_handle);
  ctx->api->cuobjdumpInit();
  return fatbin_handle;
}

void cudaRegisterFunctionInternal(void **fatCubinHandle, const char *hostFun,
                                  char *deviceFun, const char *deviceName,
                                  int thread_limit, uint3 *tid, uint3 *bid,
                                  dim3 *bDim, dim3 *gDim,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  validate_fatbin_handle(ctx->api, fatCubinHandle);
  CUctx_st *context = GPGPUSim_Context(ctx);
  printf(
      "GPGPU-Sim PTX: __cudaRegisterFunction %s : hostFun 0x%p, "
      "fatbin_handle = %p\n",
      deviceFun, hostFun, (void *)fatCubinHandle);
  ctx->load_fatbin_ptx();
  context->register_function(hostFun, deviceFun);
}

void cudaRegisterVarInternal(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  validate_fatbin_handle(ctx->api, fatCubinHandle);
  printf(
      "GPGPU-Sim PTX: __cudaRegisterVar: hostVar = %p; deviceAddress = %s; "
      "deviceName = %s\n",
      hostVar, deviceAddress, deviceName);
  printf(
      "GPGPU-Sim PTX: __cudaRegisterVar: Registering const memory space of %d "
      "bytes\n",
      size);
  ctx->load_fatbin_ptx();
  fflush(stdout);
  if (constant && !global && !ext) {
    ctx->func_sim->gpgpu_ptx_sim_register_const_variable(hostVar, deviceName,
                                                         size);
  } else if (!constant && !global && !ext) {
    ctx->func_sim->gpgpu_ptx_sim_register_global_variable(hostVar, deviceName,
                                                          size);
  } else {
    fprintf(stderr,
            "GPGPU-Sim PTX: unsupported external/global variable registration "
            "for %s\n",
            deviceName);
  }
}

cudaError_t cudaConfigureCallInternal(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, cudaStream_t stream,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  ctx->api->g_cuda_launch_stack.push_back(
      kernel_config(gridDim, blockDim, sharedMem, s));
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceCountInternal(int *count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *count = dev->num_devices();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetDevicePropertiesInternal(
    struct cudaDeviceProp *prop, int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  if (device >= 0 && device < dev->num_devices()) {
    *prop = *dev->get_prop();
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

__host__ cudaError_t CUDARTAPI
cudaChooseDeviceInternal(int *device, const struct cudaDeviceProp *prop,
                         gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *device = dev->get_id();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaSetupArgumentInternal(const void *arg, size_t size,
                                      size_t offset,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (ctx->api->g_cuda_launch_stack.empty())
    return g_last_cudaError = cudaErrorInvalidConfiguration;
  kernel_config &config = ctx->api->g_cuda_launch_stack.back();
  config.set_arg(arg, size, offset);
  printf(
      "GPGPU-Sim PTX: Setting up arguments for %zu bytes starting at "
      "0x%llx..\n",
      size, (unsigned long long)arg);

  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaLaunchInternal(const char *hostFun,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if (mode) sscanf(mode, "%u", &(ctx->func_sim->g_ptx_sim_mode));
  if (ctx->api->g_cuda_launch_stack.empty())
    return g_last_cudaError = cudaErrorInvalidConfiguration;
  kernel_config config = ctx->api->g_cuda_launch_stack.back();
  {
    dim3 gridDim = config.grid_dim();
    dim3 blockDim = config.block_dim();
    if (gridDim.x * gridDim.y * gridDim.z == 0 ||
        blockDim.x * blockDim.y * blockDim.z == 0) {
      // can't launch
      printf("can't launch a empty kernel\n");
      ctx->api->g_cuda_launch_stack.pop_back();
      return g_last_cudaError = cudaErrorInvalidConfiguration;
    }
  }
  struct CUstream_st *stream = config.get_stream();
  if (context->get_kernel(hostFun) == NULL) {
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaErrorInvalidDeviceFunction;
  }

  printf("\nGPGPU-Sim PTX: cudaLaunch for 0x%p (mode=%s) on stream %u\n",
         hostFun,
         (ctx->func_sim->g_ptx_sim_mode) ? "functional simulation"
                                         : "performance simulation",
         stream ? stream->get_uid() : 0);
  kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      hostFun, config.get_args(), config.grid_dim(), config.block_dim(),
      context);
  // do dynamic PDOM analysis for performance simulation scenario
  std::string kname = grid->name();
  function_info *kernel_func_info = grid->entry();
  if (kernel_func_info->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kname.c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kname.c_str());
    kernel_func_info->do_pdom();
    kernel_func_info->set_pdom();
  }
  dim3 gridDim = config.grid_dim();
  dim3 blockDim = config.block_dim();

  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();
  class memory_space *global_mem;
  global_mem = gpu->get_global_memory();

  if (gpu->resume_option == 1 && (grid->get_uid() == gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    for (int i = 0; i < gpu->resume_CTA; i++) grid->increment_cta_id();
  }
  if (gpu->resume_option == 1 && (grid->get_uid() < gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    printf("Skipping kernel %d as resuming from kernel %d\n", grid->get_uid(),
           gpu->resume_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  if (gpu->checkpoint_option == 1 &&
      (grid->get_uid() > gpu->checkpoint_kernel)) {
    printf("Skipping kernel %d as checkpoint from kernel %d\n", grid->get_uid(),
           gpu->checkpoint_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  printf(
      "GPGPU-Sim PTX: pushing kernel \'%s\' to stream %u, gridDim= (%u,%u,%u) "
      "blockDim = (%u,%u,%u) \n",
      kname.c_str(), stream ? stream->get_uid() : 0, gridDim.x, gridDim.y,
      gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
  ctx->the_gpgpusim->g_stream_manager->push(op);
  ctx->api->g_cuda_launch_stack.pop_back();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaMallocInternal(void **devPtr, size_t size,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*devPtr);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*devPtr] = size;
  }
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

cudaError_t cudaMallocHostInternal(void **ptr, size_t size,
                                   gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *ptr = malloc(size);
  if (*ptr) {
    // track pinned memory size allocated in the host so that same amount of
    // memory is also allocated in GPU.
    ctx->api->pinned_memory_size[*ptr] = size;
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI
cudaMallocPitchInternal(void **devPtr, size_t *pitch, size_t width,
                        size_t height, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  unsigned malloc_width_inbytes = width;
  printf("GPGPU-Sim PTX: cudaMallocPitch (width = %d)\n", malloc_width_inbytes);
  CUctx_st *context = GPGPUSim_Context(ctx);
  *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(
      malloc_width_inbytes * height);
  pitch[0] = malloc_width_inbytes;
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

cudaError_t cudaHostGetDevicePointerInternal(void **pDevice, void *pHost,
                                             unsigned int flags,
                                             gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (!pDevice || !pHost || flags != 0)
    return g_last_cudaError = cudaErrorInvalidValue;

  // Host mappings use a simulator allocation and are synchronized back to the
  // registered host range by the blocking synchronization APIs.
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  std::map<void *, size_t>::const_iterator i =
      ctx->api->pinned_memory_size.find(pHost);
  if (i == ctx->api->pinned_memory_size.end())
    return g_last_cudaError = cudaErrorInvalidValue;

  std::map<void *, void *>::const_iterator mapped =
      ctx->api->pinned_memory.find(pHost);
  if (mapped != ctx->api->pinned_memory.end()) {
    *pDevice = mapped->second;
    return g_last_cudaError = cudaSuccess;
  }

  size_t size = i->second;
  *pDevice = gpu->gpu_malloc(size);
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*pDevice);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*pDevice] = size;
  }
  if (*pDevice) {
    ctx->api->pinned_memory[pHost] = *pDevice;
    gpu->memcpy_to_gpu((size_t)*pDevice, pHost, size);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

static void copy_mapped_memory_to_host(gpgpu_context *ctx) {
  if (ctx->api->pinned_memory.empty()) return;

  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  for (const std::pair<void *const, void *> &mapping :
       ctx->api->pinned_memory) {
    std::map<void *, size_t>::const_iterator size =
        ctx->api->pinned_memory_size.find(mapping.first);
    if (size != ctx->api->pinned_memory_size.end())
      gpu->memcpy_from_gpu(mapping.first, (size_t)mapping.second, size->second);
  }
}

__host__ cudaError_t CUDARTAPI cudaMallocArrayInternal(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  unsigned size =
      width * height * ((desc->x + desc->y + desc->z + desc->w) / 8);
  CUctx_st *context = GPGPUSim_Context(ctx);
  (*array) = (struct cudaArray *)malloc(sizeof(struct cudaArray));
  (*array)->desc = *desc;
  (*array)->width = width;
  (*array)->height = height;
  (*array)->size = size;
  (*array)->dimensions = 2;
  ((*array)->devPtr32) =
      (int)(long long)context->get_device()->get_gpgpu()->gpu_mallocarray(size);
  printf("GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n",
         ((*array)->devPtr32));
  ((*array)->devPtr) = (void *)(long long)((*array)->devPtr32);
  if (((*array)->devPtr)) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI
cudaMemcpyInternal(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  // gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);
  if (count == 0) return g_last_cudaError = cudaSuccess;
  if (dst == NULL || src == NULL)
    return g_last_cudaError = cudaErrorInvalidValue;

  if (kind == cudaMemcpyHostToHost)
    memcpy(dst, src, count);
  else if (kind == cudaMemcpyHostToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation(src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDeviceToHost)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, dst, count, 0));
  else if (kind == cudaMemcpyDeviceToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDefault) {
    if ((size_t)src >= GLOBAL_HEAP_START) {
      if ((size_t)dst >= GLOBAL_HEAP_START)
        ctx->the_gpgpusim->g_stream_manager->push(stream_operation(
            (size_t)src, (size_t)dst, count, 0));  // device to device
      else
        ctx->the_gpgpusim->g_stream_manager->push(
            stream_operation((size_t)src, dst, count, 0));  // device to host
    } else {
      if ((size_t)dst >= GLOBAL_HEAP_START)
        ctx->the_gpgpusim->g_stream_manager->push(
            stream_operation(src, (size_t)dst, count, 0));
      else
        memcpy(dst, src, count);
    }
  } else {
    return g_last_cudaError = cudaErrorInvalidMemcpyDirection;
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t count, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if (count == 0) return g_last_cudaError = cudaSuccess;
  if (dst == NULL || src == NULL)
    return g_last_cudaError = cudaErrorInvalidValue;
  size_t channel_size = dst->desc.w + dst->desc.x + dst->desc.y + dst->desc.z;
  if (channel_size == 0 || channel_size % 8 != 0)
    return g_last_cudaError = cudaErrorInvalidValue;
  size_t row_size = dst->width * (channel_size / 8);
  size_t offset = hOffset * row_size + wOffset;
  if (wOffset > row_size || hOffset >= static_cast<size_t>(dst->height) ||
      count > static_cast<size_t>(dst->size) - offset)
    return g_last_cudaError = cudaErrorInvalidValue;
  size_t destination = (size_t)dst->devPtr + offset;
  printf("GPGPU-Sim PTX: cudaMemcpyToArray\n");
  if (kind == cudaMemcpyDefault)
    kind = (size_t)src >= GLOBAL_HEAP_START ? cudaMemcpyDeviceToDevice
                                            : cudaMemcpyHostToDevice;
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu(destination, src, count);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu(destination, (size_t)src, count);
  else
    return g_last_cudaError = cudaErrorInvalidMemcpyDirection;
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DInternal(
    void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
    size_t height, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if (width == 0 || height == 0) return g_last_cudaError = cudaSuccess;
  if (dst == NULL || src == NULL || width > dpitch || width > spitch)
    return g_last_cudaError = cudaErrorInvalidValue;
  if (kind == cudaMemcpyDefault) {
    const bool src_device = (size_t)src >= GLOBAL_HEAP_START;
    const bool dst_device = (size_t)dst >= GLOBAL_HEAP_START;
    if (src_device)
      kind = dst_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    else
      kind = dst_device ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
  }
  for (size_t row = 0; row < height; ++row) {
    void *row_dst = static_cast<char *>(dst) + row * dpitch;
    const void *row_src = static_cast<const char *>(src) + row * spitch;
    if (kind == cudaMemcpyHostToHost)
      memcpy(row_dst, row_src, width);
    else if (kind == cudaMemcpyHostToDevice)
      gpu->memcpy_to_gpu((size_t)row_dst, row_src, width);
    else if (kind == cudaMemcpyDeviceToHost)
      gpu->memcpy_from_gpu(row_dst, (size_t)row_src, width);
    else if (kind == cudaMemcpyDeviceToDevice)
      gpu->memcpy_gpu_to_gpu((size_t)row_dst, (size_t)row_src, width);
    else
      return g_last_cudaError = cudaErrorInvalidMemcpyDirection;
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if (width == 0 || height == 0) return g_last_cudaError = cudaSuccess;
  if (dst == NULL || src == NULL || spitch < width)
    return g_last_cudaError = cudaErrorInvalidValue;
  size_t channel_size = dst->desc.w + dst->desc.x + dst->desc.y + dst->desc.z;
  if (channel_size == 0 || channel_size % 8 != 0 || dst->dimensions != 2)
    return g_last_cudaError = cudaErrorInvalidValue;
  unsigned elem_size = channel_size / 8;
  size_t destination_pitch = elem_size * dst->width;
  if (wOffset > destination_pitch || width > destination_pitch - wOffset ||
      hOffset > static_cast<size_t>(dst->height) ||
      height > static_cast<size_t>(dst->height) - hOffset)
    return g_last_cudaError = cudaErrorInvalidValue;
  if (kind == cudaMemcpyDefault)
    kind = (size_t)src >= GLOBAL_HEAP_START ? cudaMemcpyDeviceToDevice
                                            : cudaMemcpyHostToDevice;
  for (size_t row = 0; row < height; ++row) {
    size_t row_dst =
        (size_t)dst->devPtr + (hOffset + row) * destination_pitch + wOffset;
    const void *row_src = static_cast<const char *>(src) + row * spitch;
    if (kind == cudaMemcpyHostToDevice)
      gpu->memcpy_to_gpu(row_dst, row_src, width);
    else if (kind == cudaMemcpyDeviceToDevice)
      gpu->memcpy_gpu_to_gpu(row_dst, (size_t)row_src, width);
    else
      return g_last_cudaError = cudaErrorInvalidMemcpyDirection;
  }
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolInternal(
    const char *symbol, const void *src, size_t count, size_t offset,
    enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDefault)
    return g_last_cudaError = cudaErrorInvalidMemcpyDirection;
  printf("GPGPU-Sim PTX: cudaMemcpyToSymbol: symbol = %p\n", symbol);
  // stream_operation( const char *symbol, const void *src, size_t count, size_t
  // offset )
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(src, symbol, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,src,count,offset,1,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolInternal(
    void *dst, const char *symbol, size_t count, size_t offset,
    enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDefault)
    return g_last_cudaError = cudaErrorInvalidMemcpyDirection;
  printf("GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(symbol, dst, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,dst,count,offset,0,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyAsyncInternal(
    void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  if (count == 0) return g_last_cudaError = cudaSuccess;
  if (dst == NULL || src == NULL)
    return g_last_cudaError = cudaErrorInvalidValue;
  if (kind == cudaMemcpyDefault) {
    const bool src_device = (size_t)src >= GLOBAL_HEAP_START;
    const bool dst_device = (size_t)dst >= GLOBAL_HEAP_START;
    if (src_device)
      kind = dst_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    else
      kind = dst_device ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
  }
  switch (kind) {
    case cudaMemcpyHostToHost:
      memcpy(dst, src, count);
      break;
    case cudaMemcpyHostToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation(src, (size_t)dst, count, s));
      break;
    case cudaMemcpyDeviceToHost:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, dst, count, s));
      break;
    case cudaMemcpyDeviceToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, (size_t)dst, count, s));
      break;
    default:
      return g_last_cudaError = cudaErrorInvalidMemcpyDirection;
  }
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
    int *numBlocks, const char *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  printf(
      "GPGPU-Sim PTX: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags "
      "%p\n",
      hostFunc);
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFunc);
  printf(
      "Calculate Maxium Active Block with function ptr=%p, blockSize=%d, "
      "SMemSize=%lu\n",
      hostFunc, blockSize, dynamicSMemSize);
  if (flags == cudaOccupancyDefault) {
    // create kernel_info based on entry
    dim3 gridDim(context->get_device()->get_gpgpu()->max_cta_per_core() *
                 context->get_device()->get_gpgpu()->get_config().num_shader());
    dim3 blockDim(blockSize);
    // because this fuction is only checking for resource requirements, we do
    // not care which stream this kernel runs at, just picked -1
    kernel_info_t result(gridDim, blockDim, entry, -1);
    // if(entry == NULL){
    //	*numBlocks = 1;
    //	return g_last_cudaError = cudaErrorUnknown;
    //}
    *numBlocks = context->get_device()->get_gpgpu()->get_max_cta(result);
    printf("Maximum size is %d with gridDim %d and blockDim %d\n", *numBlocks,
           gridDim.x, blockDim.x);
    return g_last_cudaError = cudaSuccess;
  }
  return g_last_cudaError = cudaErrorInvalidValue;
}

__host__ cudaError_t CUDARTAPI cudaMemsetInternal(
    void *mem, int c, size_t count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI
cudaMemsetAsyncInternal(void *mem, int c, size_t count, cudaStream_t stream = 0,
                        gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)stream;
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaHostAllocInternal(void **pHost, size_t bytes,
                                  unsigned int flags,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (!pHost || bytes == 0 || (flags & ~0x07U) != 0)
    return g_last_cudaError = cudaErrorInvalidValue;
  *pHost = malloc(bytes);
  // need to track the size allocated so that cudaHostGetDevicePointer() can
  // function properly.
  // TODO: vary this function behavior based on flags value (following nvidia
  // documentation)
  if (*pHost) {
    ctx->api->pinned_memory_size[*pHost] = bytes;
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

size_t getMaxThreadsPerBlock(struct cudaFuncAttributes *attr,
                             gpgpu_context *ctx) {
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  struct cudaDeviceProp prop;

  prop = *dev->get_prop();

  size_t max = prop.maxThreadsPerBlock;

  if (attr->numRegs && (prop.regsPerBlock / attr->numRegs) < max)
    max = prop.regsPerBlock / attr->numRegs;

  return max;
}

cudaError_t CUDARTAPI cudaFuncGetAttributesInternal(
    struct cudaFuncAttributes *attr, const char *hostFun,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  if (entry == nullptr) {
    return g_last_cudaError = cudaErrorInvalidDeviceFunction;
  }
  memset(attr, 0, sizeof(*attr));
  const struct gpgpu_ptx_sim_info *kinfo = entry->get_kernel_info();
  attr->sharedSizeBytes = kinfo->smem;
  attr->constSizeBytes = kinfo->cmem;
  attr->localSizeBytes = kinfo->lmem;
  attr->numRegs = kinfo->regs;
  if (kinfo->maxthreads > 0)
    attr->maxThreadsPerBlock = kinfo->maxthreads;
  else
    attr->maxThreadsPerBlock = getMaxThreadsPerBlock(attr, ctx);
  attr->ptxVersion = kinfo->ptx_version;
  attr->binaryVersion = kinfo->sm_target;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaDeviceGetAttributeInternal(int *value, enum cudaDeviceAttr attr, int device,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  if (value == nullptr) {
    return g_last_cudaError = cudaErrorInvalidValue;
  }

  const struct cudaDeviceProp *prop;
  _cuda_device_id *dev = ctx->GPGPUSim_Init();

  if (device >= 0 && device < dev->num_devices()) {
    prop = dev->get_prop();
    switch (attr) {
      case cudaDevAttrMaxThreadsPerBlock:
        *value = prop->maxThreadsPerBlock;
        break;
      case cudaDevAttrMaxBlockDimX:
        *value = prop->maxThreadsDim[0];
        break;
      case cudaDevAttrMaxBlockDimY:
        *value = prop->maxThreadsDim[1];
        break;
      case cudaDevAttrMaxBlockDimZ:
        *value = prop->maxThreadsDim[2];
        break;
      case cudaDevAttrMaxGridDimX:
        *value = prop->maxGridSize[0];
        break;
      case cudaDevAttrMaxGridDimY:
        *value = prop->maxGridSize[1];
        break;
      case cudaDevAttrMaxGridDimZ:
        *value = prop->maxGridSize[2];
        break;
      case cudaDevAttrMaxSharedMemoryPerBlock:
        *value = prop->sharedMemPerBlock;
        break;
      case cudaDevAttrTotalConstantMemory:
        *value = prop->totalConstMem;
        break;
      case cudaDevAttrWarpSize:
        *value = prop->warpSize;
        break;
      case cudaDevAttrMaxRegistersPerBlock:
        *value = prop->regsPerBlock;
        break;
      case cudaDevAttrClockRate:
        *value = prop->clockRate;
        break;
      case cudaDevAttrTextureAlignment:
        *value = prop->textureAlignment;
        break;
      case cudaDevAttrMultiProcessorCount:
        *value = prop->multiProcessorCount;
        break;
      case cudaDevAttrMaxThreadsPerMultiProcessor:
        *value = dev->get_gpgpu()->threads_per_core();
        break;
      case cudaDevAttrComputeCapabilityMajor:
        *value = prop->major;
        break;
      case cudaDevAttrComputeCapabilityMinor:
        *value = prop->minor;
        break;
      case cudaDevAttrMaxSharedMemoryPerMultiprocessor:
        *value = prop->sharedMemPerMultiprocessor;
        break;
      case cudaDevAttrMaxRegistersPerMultiprocessor:
        *value = prop->regsPerMultiprocessor;
        break;
      case cudaDevAttrComputeMode:
        *value = prop->computeMode;
        break;
      default:
        return g_last_cudaError = cudaErrorInvalidValue;
    }
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernelInternal(
    const char *hostFun, dim3 gridDim, dim3 blockDim, const void **args,
    size_t sharedMem, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }

  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  if (entry == NULL || (entry->num_args() != 0 && args == NULL))
    return g_last_cudaError = cudaErrorInvalidDeviceFunction;
  cudaError_t error =
      cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream, ctx);
  if (error != cudaSuccess) return error;
  for (unsigned i = 0; i < entry->num_args(); i++) {
    std::pair<size_t, unsigned> p = entry->get_param_config(i);
    error = cudaSetupArgumentInternal(args[i], p.first, p.second, ctx);
    if (error != cudaSuccess) return error;
  }

  return cudaLaunchInternal(hostFun, ctx);
}

__host__ cudaError_t CUDARTAPI cudaStreamCreateInternal(
    cudaStream_t *stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: cudaStreamCreate\n");
  *stream = new struct CUstream_st();
  ctx->the_gpgpusim->g_stream_manager->add_stream(*stream);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroyInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // per-stream synchronization required for application using external
  // libraries without explicit synchronization in the code to avoid the
  // stream_manager from spinning forever to destroy non-empty streams without
  // making any forward progress.
  stream->synchronize();
  ctx->the_gpgpusim->g_stream_manager->destroy_stream(stream);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronizeInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (stream == NULL)
    ctx->synchronize();
  else
    stream->synchronize();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfigInternal(const char *func, enum cudaFuncCache cacheConfig,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  context->get_device()->get_gpgpu()->set_cache_config(
      context->get_kernel(func)->get_name(), (FuncCache)cacheConfig);
  return g_last_cudaError = cudaSuccess;
}

CUevent_st *get_event(cudaEvent_t event) {
  unsigned event_uid;
  event_uid = event->get_uid();
  event_tracker_t::iterator e = g_timer_events.find(event_uid);
  if (e == g_timer_events.end()) return NULL;
  return e->second;
}

__host__ cudaError_t CUDARTAPI cudaEventRecordInternal(
    cudaEvent_t event, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (!e) return g_last_cudaError = cudaErrorUnknown;
  struct CUstream_st *s = (struct CUstream_st *)stream;
  stream_operation op(e, s);
  e->issue();
  ctx->the_gpgpusim->g_stream_manager->push(op);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamWaitEventInternal(
    cudaStream_t stream, cudaEvent_t event, unsigned int flags,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // reference:
  // https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__STREAM_gfe68d207dc965685d92d3f03d77b0876.html
  CUevent_st *e = get_event(event);
  if (!e) {
    printf(
        "GPGPU-Sim API: Error at cudaStreamWaitEvent. Event is not created "
        ".\n");
    return g_last_cudaError = cudaErrorInvalidResourceHandle;
  } else if (e->num_issued() == 0) {
    printf(
        "GPGPU-Sim API: Warning: cudaEventRecord has not been called on event "
        "before calling cudaStreamWaitEvent.\nNothin    g to be done.\n");
    return g_last_cudaError = cudaSuccess;
  }
  if (!stream) {
    ctx->the_gpgpusim->g_stream_manager->pushCudaStreamWaitEventToAllStreams(
        e, flags);
  } else {
    struct CUstream_st *s = (struct CUstream_st *)stream;
    stream_operation op(s, e, flags);
    ctx->the_gpgpusim->g_stream_manager->push(op);
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadExitInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  ctx->exit_simulation();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Called on host side
  ctx->synchronize();
  copy_mapped_memory_to_host(ctx);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI
cudaDeviceSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Blocks until the device has completed all preceding requested tasks
  ctx->synchronize();
  copy_mapped_memory_to_host(ctx);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaPeekAtLastError(void) { return g_last_cudaError; }

__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  return cudaMallocInternal(devPtr, size);
}

__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
  return cudaMallocHostInternal(ptr, size);
}
__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch,
                                               size_t width, size_t height) {
  return cudaMallocPitchInternal(devPtr, pitch, width, height);
}

__host__ cudaError_t CUDARTAPI cudaMallocArray(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height, unsigned int flags) {
  (void)flags;
  return cudaMallocArrayInternal(array, desc, width, height);
}

__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
}
__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  gpgpu_context *ctx = GPGPU_Context();
  if (!ptr || ctx->api->pinned_memory_size.find(ptr) ==
                  ctx->api->pinned_memory_size.end())
    return g_last_cudaError = cudaErrorInvalidValue;
  ctx->synchronize();
  copy_mapped_memory_to_host(ctx);
  ctx->api->pinned_memory.erase(ptr);
  ctx->api->pinned_memory_size.erase(ptr);
  free(ptr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  free(array);
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
                                          size_t count,
                                          enum cudaMemcpyKind kind) {
  return cudaMemcpyInternal(dst, src, count, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst,
                                                 size_t wOffset, size_t hOffset,
                                                 const void *src, size_t count,
                                                 enum cudaMemcpyKind kind) {
  return cudaMemcpyToArrayInternal(dst, wOffset, hOffset, src, count, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch,
                                            const void *src, size_t spitch,
                                            size_t width, size_t height,
                                            enum cudaMemcpyKind kind) {
  return cudaMemcpy2DInternal(dst, dpitch, src, spitch, width, height, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
  return cudaMemcpy2DToArrayInternal(dst, wOffset, hOffset, src, spitch, width,
                                     height, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol,
                                                  const void *src, size_t count,
                                                  size_t offset,
                                                  enum cudaMemcpyKind kind) {
  return cudaMemcpyToSymbolInternal(static_cast<const char *>(symbol), src,
                                    count, offset, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst,
                                                    const void *symbol,
                                                    size_t count, size_t offset,
                                                    enum cudaMemcpyKind kind) {
  return cudaMemcpyFromSymbolInternal(dst, static_cast<const char *>(symbol),
                                      count, offset, kind);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                               size_t count,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  return cudaMemcpyAsyncInternal(dst, src, count, kind, stream);
}

cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
      numBlocks, static_cast<const char *>(hostFunc), blockSize,
      dynamicSMemSize, flags);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count) {
  return cudaMemsetInternal(mem, c, count);
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI cudaMemsetAsync(void *mem, int c, size_t count,
                                               cudaStream_t stream) {
  return cudaMemsetAsyncInternal(mem, c, count, stream);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
  return cudaGetDeviceCountInternal(count);
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  return cudaGetDevicePropertiesInternal(prop, device);
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value,
                                                      enum cudaDeviceAttr attr,
                                                      int device) {
  return cudaDeviceGetAttributeInternal(value, attr, device);
}

__host__ cudaError_t CUDARTAPI
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
  return cudaChooseDeviceInternal(device, prop);
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
  return cudaSetDeviceInternal(device);
}

__host__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  return cudaGetDeviceInternal(device);
}

__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer,
                                                       int device,
                                                       int peerDevice) {
  if (canAccessPeer == nullptr) {
    return g_last_cudaError = cudaErrorInvalidValue;
  }

  _cuda_device_id *dev = GPGPU_Context()->GPGPUSim_Init();
  if (device < 0 || device >= dev->num_devices() || peerDevice < 0 ||
      peerDevice >= dev->num_devices()) {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }

  // The simulator currently exposes one device, so no peer mapping exists.
  *canAccessPeer = 0;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue,
                                                  cudaLimit limit) {
  return cudaDeviceGetLimitInternal(pValue, limit);
}

__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(cudaLimit limit,
                                                  size_t value) {
  return cudaDeviceSetLimitInternal(limit, value);
}

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(
    struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *desc = array->desc;
  return g_last_cudaError = cudaSuccess;
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(
    int x, int y, int z, int w, enum cudaChannelFormatKind f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct cudaChannelFormatDesc dummy;
  dummy.x = x;
  dummy.y = y;
  dummy.z = z;
  dummy.w = w;
  dummy.f = f;
  return dummy;
}

__host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t error = g_last_cudaError;
  g_last_cudaError = cudaSuccess;
  return error;
}

__host__ const char *cudaGetErrorName(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";
    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady:
      return "cudaErrorNotReady";
    default:
      return "cudaErrorUnknown";
  }
}

__host__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaGetErrorName(error);
}

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size,
                                                 size_t offset) {
  return cudaSetupArgumentInternal(arg, size, offset);
}

__host__ cudaError_t CUDARTAPI cudaLaunch(const char *hostFun) {
  return cudaLaunchInternal(hostFun);
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *hostFun,
                                                dim3 gridDim, dim3 blockDim,
                                                void **args, size_t sharedMem,
                                                cudaStream_t stream) {
  return cudaLaunchKernelInternal(static_cast<const char *>(hostFun), gridDim,
                                  blockDim, const_cast<const void **>(args),
                                  sharedMem, stream);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream) {
  return cudaStreamCreateInternal(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *stream,
                                                         unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (flags != cudaStreamDefault && flags != cudaStreamNonBlocking) {
    return g_last_cudaError = cudaErrorInvalidValue;
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  return cudaStreamDestroyInternal(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  return cudaStreamSynchronizeInternal(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (stream == NULL) return g_last_cudaError = cudaErrorInvalidResourceHandle;
  return g_last_cudaError = stream->empty() ? cudaSuccess : cudaErrorNotReady;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = new CUevent_st(false);
  g_timer_events[e->get_uid()] = e;
  *event = e;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                               cudaStream_t stream) {
  return cudaEventRecordInternal(event, stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,
                                                   cudaEvent_t event,
                                                   unsigned int flags) {
  return cudaStreamWaitEventInternal(stream, event, flags);
}

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (e == NULL) {
    return g_last_cudaError = cudaErrorInvalidValue;
  } else if (e->done()) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorNotReady;
  }
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim API: cudaEventSynchronize ** waiting for event\n");
  fflush(stdout);
  CUevent_st *e = (CUevent_st *)event;
  while (!e->done())
    ;
  printf("GPGPU-Sim API: cudaEventSynchronize ** event detected\n");
  fflush(stdout);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  unsigned event_uid = e->get_uid();
  event_tracker_t::iterator pe = g_timer_events.find(event_uid);
  if (pe == g_timer_events.end())
    return g_last_cudaError = cudaErrorInvalidValue;
  g_timer_events.erase(pe);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms,
                                                    cudaEvent_t start,
                                                    cudaEvent_t end) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  time_t elapsed_time;
  CUevent_st *s = get_event(start);
  CUevent_st *e = get_event(end);
  if (s == NULL || e == NULL) return g_last_cudaError = cudaErrorUnknown;
  elapsed_time = e->clock() - s->clock();
  *ms = 1000 * elapsed_time;
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaThreadExit(void) {
  return cudaThreadExitInternal();
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
  return cudaThreadSynchronizeInternal();
}

int CUDARTAPI __cudaSynchronizeThreads(void **, void *) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaThreadExit();
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

// extracts all ptx files from binary and dumps into
// prog_name.unique_no.sm_<>.ptx files
void cuda_runtime_api::extract_ptx_files_using_cuobjdump_internal(
    CUctx_st *context, std::string &app_binary) {
  char command[1000];

  char ptx_list_file_name[1024];
  snprintf(ptx_list_file_name, 1024, "_cuobjdump_list_ptx_XXXXXX");
  int fd2 = mkstemp(ptx_list_file_name);
  close(fd2);

  // only want file names
  snprintf(command, 1000,
           GPGPUSIM_CUOBJDUMP
           " -lptx %s  | cut -d \":\" -f 2 | "
           "awk '{$1=$1}1' > %s",
           app_binary.c_str(), ptx_list_file_name);
  if (system(command) != 0) {
    printf("WARNING: Failed to execute cuobjdump to get list of ptx files \n");
    exit(0);
  }

  {
    // based on the list above, dump ptx files individually. Format of dumped
    // ptx file is prog_name.unique_no.sm_<>.ptx
    std::ifstream infile(ptx_list_file_name);
    std::string line;
    while (std::getline(infile, line)) {
      // int pos = line.find(std::string(get_app_binary_name(app_binary)));
      const char *ptx_file = line.c_str();
      printf("Extracting specific PTX file named %s \n", ptx_file);
      snprintf(command, 1000, GPGPUSIM_CUOBJDUMP " -xptx %s %s", ptx_file,
               app_binary.c_str());
      if (system(command) != 0) {
        printf("ERROR: command: %s failed \n", command);
        exit(0);
      }
      context->no_of_ptx++;
    }
  }

  if (!context->no_of_ptx) {
    printf(
        "WARNING: executable contains no extractable PTX (CDP may be "
        "enabled)\n");
  }

  std::ifstream infile(ptx_list_file_name);
  std::string line;
  while (std::getline(infile, line)) {
    // int pos = line.find(std::string(get_app_binary_name(app_binary)));
    int pos1 = line.find("sm_");
    int pos2 = line.find_last_of(".");
    if (pos1 == std::string::npos && pos2 == std::string::npos) {
      printf("ERROR: PTX list is not in correct format");
      exit(0);
    }
    std::string vstr = line.substr(pos1 + 3, pos2 - pos1 - 3);
    int version = atoi(vstr.c_str());
    if (version_filename.find(version) == version_filename.end()) {
      version_filename[version] = std::set<std::string>();
    }
    version_filename[version].insert(line);
  }
}

void cuda_runtime_api::cuobjdumpInit() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  std::string app_binary = get_app_binary();
  extract_ptx_files_using_cuobjdump_internal(context, app_binary);
}

//! Load the PTX files selected by CUDA 11.8 cuobjdump.
void gpgpu_context::load_fatbin_ptx() {
  CUctx_st *context = GPGPUSim_Context(this);
  if (context->has_binary()) return;
  symbol_table *symtab = NULL;

  // loops through all ptx files from smallest sm version to largest
  std::map<unsigned, std::set<std::string> >::iterator itr_m;
  for (itr_m = api->version_filename.begin();
       itr_m != api->version_filename.end(); itr_m++) {
    std::set<std::string>::iterator itr_s;
    for (itr_s = itr_m->second.begin(); itr_s != itr_m->second.end(); itr_s++) {
      std::string ptx_filename = *itr_s;
      printf("GPGPU-Sim PTX: Parsing %s\n", ptx_filename.c_str());
      symtab = gpgpu_ptx_sim_load_ptx_from_filename(ptx_filename.c_str());
    }
  }
  if (symtab == NULL) {
    fprintf(stderr, "GPGPU-Sim PTX: no PTX symbol table was loaded\n");
    exit(EXIT_FAILURE);
  }
  context->add_binary(symtab);
  api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                           context->get_device()->get_gpgpu());
  api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                      context->get_device()->get_gpgpu());
  for (itr_m = api->version_filename.begin();
       itr_m != api->version_filename.end(); itr_m++) {
    std::set<std::string>::iterator itr_s;
    for (itr_s = itr_m->second.begin(); itr_s != itr_m->second.end(); itr_s++) {
      std::string ptx_filename = *itr_s;
      printf("GPGPU-Sim PTX: Loading PTXInfo from %s\n", ptx_filename.c_str());
      gpgpu_ptx_info_load_from_filename(ptx_filename.c_str(), itr_m->first);
    }
  }
}

extern "C" {

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaRegisterFatBinaryInternal(fatCubin);
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem,
                                               struct CUstream_st *stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
  return 0;
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_runtime_api *api = GPGPU_Context()->api;
  if (api->g_cuda_launch_stack.empty()) {
    return g_last_cudaError = cudaErrorMissingConfiguration;
  }
  kernel_config config = api->g_cuda_launch_stack.back();
  api->g_cuda_launch_stack.pop_back();
  *gridDim = config.grid_dim();
  *blockDim = config.block_dim();
  *sharedMem = config.shared_mem();
  *static_cast<cudaStream_t *>(stream) = config.get_stream();
  return g_last_cudaError = cudaSuccess;
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim, int *wSize) {
  (void)wSize;
  cudaRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun, deviceName,
                               thread_limit, tid, bid, bDim, gDim);
}

extern void __cudaRegisterVar(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global) {
  cudaRegisterVarInternal(fatCubinHandle, hostVar, deviceAddress, deviceName,
                          ext, size, constant, global);
}

__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                                 size_t sharedMem,
                                                 cudaStream_t stream) {
  return cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
  return cudaDeviceSynchronizeInternal();
}

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterShared\n");
}

void CUDARTAPI __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
                                       size_t size, size_t alignment,
                                       int storage) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterSharedVar\n");
}

char __cudaInitModule(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  (void)fatCubinHandle;
  return 0;
}

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t bytes,
                                    unsigned int flags) {
  return cudaHostAllocInternal(pHost, bytes, flags);
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                               unsigned int flags) {
  return cudaHostGetDevicePointerInternal(pDevice, pHost, flags);
}

cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size,
                                       unsigned int flags) {
  if (!ptr || size == 0 || (flags & ~0x0fU) != 0)
    return g_last_cudaError = cudaErrorInvalidValue;
  gpgpu_context *ctx = GPGPU_Context();
  if (ctx->api->pinned_memory_size.find(ptr) !=
      ctx->api->pinned_memory_size.end())
    return g_last_cudaError = cudaErrorHostMemoryAlreadyRegistered;
  ctx->api->pinned_memory_size[ptr] = size;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaHostUnregister(void *ptr) {
  if (!ptr) return g_last_cudaError = cudaErrorInvalidValue;
  gpgpu_context *ctx = GPGPU_Context();
  if (ctx->api->pinned_memory_size.find(ptr) ==
      ctx->api->pinned_memory_size.end())
    return g_last_cudaError = cudaErrorHostMemoryNotRegistered;
  ctx->synchronize();
  copy_mapped_memory_to_host(ctx);
  ctx->api->pinned_memory.erase(ptr);
  ctx->api->pinned_memory_size.erase(ptr);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags) {
  unsigned int schedule = flags & cudaDeviceScheduleMask;
  if ((flags & ~cudaDeviceMask) != 0 || schedule == 3 || schedule > 4)
    return g_last_cudaError = cudaErrorInvalidValue;
  GPGPU_Context()->api->device_flags = flags;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                            const void *hostFun) {
  return cudaFuncGetAttributesInternal(attr,
                                       static_cast<const char *>(hostFun));
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event,
                                               unsigned int flags) {
  CUevent_st *e = new CUevent_st(flags == cudaEventBlockingSync);
  g_timer_events[e->get_uid()] = e;
  *event = e;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *driverVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *runtimeVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) {
  return cudaFuncSetCacheConfigInternal(static_cast<const char *>(func),
                                        cacheConfig);
}
}

////////

/// static functions

int cuda_runtime_api::load_static_globals(symbol_table *symtab,
                                          unsigned min_gaddr,
                                          unsigned max_gaddr, gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading globals with explicit initializers... \n");
  fflush(stdout);
  int ng_bytes = 0;
  symbol_table::iterator g = symtab->global_iterator_begin();

  for (; g != symtab->global_iterator_end(); g++) {
    symbol *global = *g;
    if (global->has_initializer()) {
      printf("GPGPU-Sim PTX:     initializing '%s' ... ",
             global->name().c_str());
      unsigned addr = global->get_address();
      const type_info *type = global->type();
      type_info_key ti = type->get_key();
      size_t size;
      int t;
      ti.type_decode(size, t);
      int nbytes = size / 8;
      int offset = 0;
      std::list<operand_info> init_list = global->get_initializer();
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value;
        if (op.is_function_address())
          value.u64 = op.get_symbol()->get_pc()->get_start_PC();
        else
          value = op.get_literal_value();
        assert((addr + offset + nbytes) <
               min_gaddr);  // min_gaddr is start of "heap" for cudaMalloc
        gpu->get_global_memory()->write(addr + offset, nbytes, &value, NULL,
                                        NULL);  // assuming little endian here
        offset += nbytes;
        ng_bytes += nbytes;
      }
      printf(" wrote %u bytes\n", offset);
    }
  }
  printf("GPGPU-Sim PTX: finished loading globals (%u bytes total).\n",
         ng_bytes);
  fflush(stdout);
  return ng_bytes;
}

int cuda_runtime_api::load_constants(symbol_table *symtab, addr_t min_gaddr,
                                     gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading constants with explicit initializers... ");
  fflush(stdout);
  int nc_bytes = 0;
  symbol_table::iterator g = symtab->const_iterator_begin();

  for (; g != symtab->const_iterator_end(); g++) {
    symbol *constant = *g;
    if (constant->is_const() && constant->has_initializer()) {
      // get the constant element data size
      int basic_type;
      size_t num_bits;
      constant->type()->get_key().type_decode(num_bits, basic_type);

      std::list<operand_info> init_list = constant->get_initializer();
      int nbytes_written = 0;
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        int nbytes = num_bits / 8;
        switch (op.get_type()) {
          case int_t:
            assert(nbytes >= 1);
            break;
          case float_op_t:
            assert(nbytes == 4);
            break;
          case double_op_t:
            assert(nbytes >= 4);
            break;  // account for double DEMOTING
          default:
            abort();
        }
        unsigned addr = constant->get_address() + nbytes_written;
        assert(addr + nbytes < min_gaddr);

        gpu->get_global_memory()->write(
            addr, nbytes, &value, NULL,
            NULL);  // assume little endian (so u8 is the first byte in u32)
        nc_bytes += nbytes;
        nbytes_written += nbytes;
      }
    }
  }
  printf(" done.\n");
  fflush(stdout);
  return nc_bytes;
}

kernel_info_t *cuda_runtime_api::gpgpu_cuda_ptx_sim_init_grid(
    const char *hostFun, gpgpu_ptx_sim_arg_list_t args, struct dim3 gridDim,
    struct dim3 blockDim, CUctx_st *context) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  function_info *entry = context->get_kernel(hostFun);
  assert(entry != NULL);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  /*
  Passing a snapshot of the GPU's current texture mapping to the kernel's info
  as kernels should use texture bindings present at the time of their launch.
  */
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  unsigned argcount = args.size();
  unsigned argn = 1;
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    argn++;
  }

  entry->finalize(result->get_param_memory());
  gpgpu_ctx->func_sim->g_ptx_kernel_count++;
  fflush(stdout);

  if (g_debug_execution >= 4) {
    entry->ptx_jit_config(g_mallocPtr_Size, result->get_param_memory(),
                          (gpgpu_t *)context->get_device()->get_gpgpu(),
                          gridDim, blockDim);
  }

  return result;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
