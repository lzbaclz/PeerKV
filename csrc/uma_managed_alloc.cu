// cudaMallocManaged-backed allocator exposed via the torch
// CUDAPluggableAllocator C ABI.
//
// Every allocation routed here (i.e. vLLM's KV cache, allocated inside a
// torch.cuda.MemPool bound to this allocator) becomes Unified Memory whose
// physical pages CUDA can migrate between Hopper HBM3 and the Grace LPDDR5X
// NUMA node. We seed each allocation with PreferredLocation = the GPU
// (so a fresh KV block is HBM-hot) and AccessedBy = the GPU device, which
// suppresses fault-driven migration when the *CPU/Grace* side reads a block
// that has been demoted -- on a coherent C2C link the GPU reads the Grace
// page in place. Demotion later flips PreferredLocation to Grace.
#include "uma.h"

#include <cuda_runtime_api.h>

#include <mutex>
#include <unordered_map>

namespace {

struct Registry {
  std::mutex mu;
  std::unordered_map<void*, std::size_t> live;  // ptr -> size
  uma::PoolStats stats;
};

Registry& registry() {
  static Registry r;
  return r;
}

// Seed residency advice on a freshly allocated managed block. Best-effort:
// on a GPU without managed-access support these are no-ops / errors we
// deliberately ignore so allocation still succeeds.
void seed_advice(void* ptr, std::size_t size, int device) {
  cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
  cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
}

}  // namespace

extern "C" void* uma_malloc(std::size_t size, int device, void* /*stream*/) {
  if (size == 0) return nullptr;
  // The allocator is invoked with the device torch wants the tensor on.
  cudaSetDevice(device);
  void* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
  if (err != cudaSuccess || ptr == nullptr) {
    // Surface OOM to torch by returning null; torch raises a clear error.
    return nullptr;
  }
  seed_advice(ptr, size, device);

  Registry& r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  r.live.emplace(ptr, size);
  r.stats.live_bytes += size;
  r.stats.live_allocs += 1;
  if (r.stats.live_bytes > r.stats.peak_bytes) {
    r.stats.peak_bytes = r.stats.live_bytes;
  }
  return ptr;
}

extern "C" void uma_free(void* ptr, std::size_t /*size*/, int /*device*/,
                         void* /*stream*/) {
  if (ptr == nullptr) return;
  Registry& r = registry();
  std::size_t size = 0;
  {
    std::lock_guard<std::mutex> lock(r.mu);
    auto it = r.live.find(ptr);
    if (it != r.live.end()) {
      size = it->second;
      r.live.erase(it);
      r.stats.live_bytes -= size;
      r.stats.live_allocs -= 1;
    }
  }
  cudaFree(ptr);
}

namespace uma {

PoolStats pool_stats() {
  Registry& r = registry();
  std::lock_guard<std::mutex> lock(r.mu);
  return r.stats;
}

}  // namespace uma
