#ifndef __cuda_api_object_h__
#define __cuda_api_object_h__

#include <list>
#include <map>
#include <set>
#include <string>

#include <builtin_types.h>

#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/gpgpu-sim/gpu-sim.h"

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

struct _cuda_device_id {
  _cuda_device_id(gpgpu_sim *gpu) {
    m_id = 0;
    m_next = NULL;
    m_gpgpu = gpu;
  }
  struct _cuda_device_id *next() {
    return m_next;
  }
  unsigned num_shader() const { return m_gpgpu->get_config().num_shader(); }
  int num_devices() const {
    if (m_next == NULL)
      return 1;
    else
      return 1 + m_next->num_devices();
  }
  struct _cuda_device_id *get_device(unsigned n) {
    assert(n < (unsigned)num_devices());
    struct _cuda_device_id *p = this;
    for (unsigned i = 0; i < n; i++) p = p->m_next;
    return p;
  }
  const struct cudaDeviceProp *get_prop() const { return m_gpgpu->get_prop(); }
  unsigned get_id() const { return m_id; }

  gpgpu_sim *get_gpgpu() { return m_gpgpu; }

 private:
  unsigned m_id;
  class gpgpu_sim *m_gpgpu;
  struct _cuda_device_id *m_next;
};

struct CUctx_st {
  CUctx_st(_cuda_device_id *gpu) : m_gpu(gpu), m_symbol_table(NULL) {
    m_binary_info.cmem = 0;
    m_binary_info.gmem = 0;
    no_of_ptx = 0;
  }

  _cuda_device_id *get_device() { return m_gpu; }

  bool has_binary() const { return m_symbol_table != NULL; }

  void add_binary(symbol_table *symtab) {
    assert(symtab != NULL);
    assert(m_symbol_table == NULL);
    m_symbol_table = symtab;
  }

  void add_ptxinfo(const char *deviceFun,
                   const struct gpgpu_ptx_sim_info &info) {
    assert(m_symbol_table != NULL);
    symbol *s = m_symbol_table->lookup(deviceFun);
    assert(s != NULL);
    function_info *f = s->get_pc();
    assert(f != NULL);
    f->set_kernel_info(info);
  }

  void add_ptxinfo(const struct gpgpu_ptx_sim_info &info) {
    m_binary_info = info;
  }

  void register_function(const char *hostFun, const char *deviceFun) {
    if (m_symbol_table != NULL) {
      symbol *s = m_symbol_table->lookup(deviceFun);
      if (s != NULL) {
        function_info *f = s->get_pc();
        assert(f != NULL);
        m_kernel_lookup[hostFun] = f;
      } else {
        printf("Warning: cannot find deviceFun %s\n", deviceFun);
        m_kernel_lookup[hostFun] = NULL;
      }
    } else {
      m_kernel_lookup[hostFun] = NULL;
    }
  }

  void register_hostFun_function(const char *hostFun, function_info *f) {
    m_kernel_lookup[hostFun] = f;
  }

  function_info *get_kernel(const char *hostFun) {
    std::map<const void *, function_info *>::iterator i =
        m_kernel_lookup.find(hostFun);
    assert(i != m_kernel_lookup.end());
    return i->second;
  }

  int no_of_ptx;

 private:
  _cuda_device_id *m_gpu;  // selected gpu
  symbol_table *m_symbol_table;
  // unique id (CUDA app function address) => kernel entry point
  std::map<const void *, function_info *> m_kernel_lookup;
  struct gpgpu_ptx_sim_info m_binary_info;
};

class kernel_config {
 public:
  kernel_config(dim3 GridDim, dim3 BlockDim, size_t sharedMem,
                struct CUstream_st *stream) {
    m_GridDim = GridDim;
    m_BlockDim = BlockDim;
    m_sharedMem = sharedMem;
    m_stream = stream;
  }
  kernel_config() {
    m_GridDim = dim3(-1, -1, -1);
    m_BlockDim = dim3(-1, -1, -1);
    m_sharedMem = 0;
    m_stream = NULL;
  }
  void set_arg(const void *arg, size_t size, size_t offset) {
    m_args.push_front(gpgpu_ptx_sim_arg(arg, size, offset));
  }
  dim3 grid_dim() const { return m_GridDim; }
  dim3 block_dim() const { return m_BlockDim; }
  size_t shared_mem() const { return m_sharedMem; }
  void set_grid_dim(dim3 *d) { m_GridDim = *d; }
  void set_block_dim(dim3 *d) { m_BlockDim = *d; }
  gpgpu_ptx_sim_arg_list_t get_args() { return m_args; }
  struct CUstream_st *get_stream() {
    return m_stream;
  }

 private:
  dim3 m_GridDim;
  dim3 m_BlockDim;
  size_t m_sharedMem;
  struct CUstream_st *m_stream;
  gpgpu_ptx_sim_arg_list_t m_args;
};

class cuda_runtime_api {
 public:
  cuda_runtime_api(gpgpu_context *ctx) {
    g_active_device = 0;  // active gpu that runs the code
    gpgpu_ctx = ctx;
  }
  // global list
  std::list<kernel_config> g_cuda_launch_stack;
  bool fatbin_registered = false;
  void *fatbin_handle_token = NULL;
  void **fatbin_handle() { return &fatbin_handle_token; }
  std::map<unsigned long long, size_t> g_mallocPtr_Size;
  // maps sm version number to set of filenames
  std::map<unsigned, std::set<std::string> > version_filename;
  std::map<void *, void *> pinned_memory;
  std::map<void *, size_t> pinned_memory_size;
  std::map<int, size_t> device_limits;
  unsigned int device_flags = 0;
  int g_active_device;  // active gpu that runs the code
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // member function list

  void cuobjdumpInit();
  void extract_ptx_files_using_cuobjdump_internal(CUctx_st *context,
                                                  std::string &app_binary);
  kernel_info_t *gpgpu_cuda_ptx_sim_init_grid(const char *kernel_key,
                                              gpgpu_ptx_sim_arg_list_t args,
                                              struct dim3 gridDim,
                                              struct dim3 blockDim,
                                              struct CUctx_st *context);
  int load_static_globals(symbol_table *symtab, unsigned min_gaddr,
                          unsigned max_gaddr, gpgpu_t *gpu);
  int load_constants(symbol_table *symtab, addr_t min_gaddr, gpgpu_t *gpu);
};
#endif /* __cuda_api_object_h__ */
