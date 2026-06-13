// UMA-LLM native layer: shared declarations.
//
// Route B ("real UMA in the allocator layer"): the KV cache pool is backed
// by cudaMallocManaged so its physical pages can live on either Hopper HBM3
// or the Grace LPDDR5X NUMA node, and a "tier demotion" is a *residency
// hint* (cudaMemAdvise SetPreferredLocation + cudaMemPrefetchAsync) rather
// than a copy. Because NVLink-C2C is cache-coherent, the attention kernel
// keeps the same virtual address whether a block currently resides on HBM
// or Grace -- T0<->T1 becomes genuinely zero-copy.
//
// This header is shared by:
//   * uma_managed_alloc.cu -- the torch CUDAPluggableAllocator C-ABI
//     (uma_malloc / uma_free) so vLLM's KV pool allocates managed memory.
//   * uma_residency.cu      -- the advise / prefetch / query wrappers.
//   * bindings.cpp          -- the pybind module umallm._uma_native.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace uma {

// Residency target for a managed range. We keep this as a small enum at the
// ABI boundary and translate to a CUDA device ordinal inside the .cu:
//   GRACE -> cudaCpuDeviceId (the Grace LPDDR5X host NUMA node on GH200)
//   HBM   -> the current GPU device ordinal (Hopper HBM3)
enum class Node : int {
  GRACE = 0,
  HBM = 1,
};

// Per-process pool statistics (managed bytes currently reserved by uma_malloc).
struct PoolStats {
  std::size_t live_bytes = 0;     // sum of outstanding uma_malloc sizes
  std::size_t peak_bytes = 0;     // high-water mark
  std::size_t live_allocs = 0;    // outstanding allocation count
};

// True only when built with CUDA *and* the device reports
// concurrentManagedAccess (required for cudaMemPrefetchAsync to Grace).
bool available();

// True only on a *coherent* CPU-GPU link (GH200 NVLink-C2C), where the host
// can access device-resident managed memory without a migration
// (cudaDevAttrDirectManagedMemAccessFromHost). False on discrete CUDA (e.g.
// A100 over PCIe): there a "demotion" is a real page migration, so such a run
// is the non-coherent / discrete reference, NOT coherent UMA.
bool coherent();

// CUDA device name (e.g. "NVIDIA A100-SXM4-80GB", "GH200 ..."), for labeling.
std::string device_name();

// CUDA device ordinals behind the Node enum (HBM = current device).
int grace_device_id();   // == cudaCpuDeviceId (-1) in the classic API
int hbm_device_id();

// Apply a residency hint to [ptr, ptr+nbytes): set the preferred location to
// `node` and (asynchronously, on `stream`) prefetch the pages there. Returns
// 0 on success or the cudaError_t code on failure. `stream` is a cudaStream_t
// reinterpreted as uintptr_t (0 -> default stream).
int advise_range(std::uintptr_t ptr, std::size_t nbytes, Node node,
                 std::uintptr_t stream);

// Query where the pages of [ptr, ptr+nbytes) were last prefetched.
// Returns the device ordinal (>=0 for an HBM device, cudaCpuDeviceId for
// Grace) or a negative cudaError-derived sentinel on failure.
int query_node(std::uintptr_t ptr, std::size_t nbytes);

PoolStats pool_stats();

}  // namespace uma

// ---- CUDAPluggableAllocator C ABI ------------------------------------- //
// torch.cuda.memory.CUDAPluggableAllocator dlopen()s these by name. They are
// extern "C" with default visibility so they survive the torch-extension
// build. Routing vLLM's KV allocation through a MemPool bound to this
// allocator is what lands the KV pool on managed memory (and *only* the KV
// pool -- weights/activations keep the fast default caching allocator).
extern "C" {
__attribute__((visibility("default"))) void* uma_malloc(std::size_t size,
                                                         int device,
                                                         void* stream);
__attribute__((visibility("default"))) void uma_free(void* ptr,
                                                      std::size_t size,
                                                      int device, void* stream);
}
