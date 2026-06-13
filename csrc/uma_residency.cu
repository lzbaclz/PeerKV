// Residency control for managed KV pages: the actual "tier move".
//
// to_grace / to_hbm in Python land here. A move is:
//   1. cudaMemAdvise(SetPreferredLocation, dst)  -- where pages *want* to be
//   2. cudaMemPrefetchAsync(dst, stream)         -- migrate them now, async
// No bytes are copied through a staging buffer; the page's physical home
// changes while its virtual address (and thus the attention block table) is
// untouched. On GH200 a not-yet-migrated access is served coherently over
// C2C, so even before the prefetch completes the kernel reads correct data.
//
// NOTE on precise Grace targeting: the classic API used here routes "host"
// to cudaCpuDeviceId, which on a single-socket GH200 is the Grace node. On
// multi-socket / multi-NUMA Grace systems, switch to cudaMemAdvise_v2 /
// cudaMemPrefetchAsync_v2 with cudaMemLocation{cudaMemLocationTypeHostNuma,
// <numa_id>} to pin a specific Grace node. Left as classic for portability
// (the _v2 API requires CUDA >= 12.2).
#include "uma.h"

#include <cuda_runtime_api.h>

namespace uma {

namespace {

int current_device() {
  int dev = 0;
  cudaGetDevice(&dev);
  return dev;
}

int device_for(Node node) {
  return node == Node::GRACE ? cudaCpuDeviceId : current_device();
}

}  // namespace

bool available() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  int dev = current_device();
  int concurrent = 0;
  // concurrentManagedAccess == 1 is required for cudaMemPrefetchAsync and for
  // the CPU to touch managed memory while the GPU runs (the coherent path).
  if (cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess,
                             dev) != cudaSuccess) {
    return false;
  }
  return concurrent == 1;
}

bool coherent() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  int direct = 0;
  // DirectManagedMemAccessFromHost == 1 only on a coherent CPU-GPU link
  // (GH200 NVLink-C2C): the host reaches device-resident managed pages with
  // no migration. On discrete CUDA (A100/PCIe) it is 0 -- a demoted page must
  // physically migrate, i.e. the run is the discrete / non-coherent reference.
  if (cudaDeviceGetAttribute(&direct, cudaDevAttrDirectManagedMemAccessFromHost,
                             current_device()) != cudaSuccess) {
    return false;
  }
  return direct == 1;
}

std::string device_name() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return "no-cuda-device";
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, current_device()) != cudaSuccess) {
    return "unknown-cuda-device";
  }
  return std::string(prop.name);
}

int grace_device_id() { return cudaCpuDeviceId; }

int hbm_device_id() { return current_device(); }

int advise_range(std::uintptr_t ptr, std::size_t nbytes, Node node,
                 std::uintptr_t stream) {
  if (ptr == 0 || nbytes == 0) return 0;
  void* p = reinterpret_cast<void*>(ptr);
  int dst = device_for(node);

  cudaError_t err =
      cudaMemAdvise(p, nbytes, cudaMemAdviseSetPreferredLocation, dst);
  if (err != cudaSuccess) return static_cast<int>(err);

  err = cudaMemPrefetchAsync(p, nbytes, dst,
                             reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) return static_cast<int>(err);
  return 0;
}

int query_node(std::uintptr_t ptr, std::size_t nbytes) {
  if (ptr == 0 || nbytes == 0) return -2;
  void* p = reinterpret_cast<void*>(ptr);
  int location = 0;
  cudaError_t err = cudaMemRangeGetAttribute(
      &location, sizeof(location), cudaMemRangeAttributeLastPrefetchLocation, p,
      nbytes);
  if (err != cudaSuccess) {
    // Negative sentinel distinct from cudaCpuDeviceId (-1).
    return -100 - static_cast<int>(err);
  }
  return location;  // device ordinal, or cudaCpuDeviceId (-1) for Grace
}

}  // namespace uma
