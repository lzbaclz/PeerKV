// pybind module: umallm._uma_native
//
// Exposes the residency control surface to Python. The CUDAPluggableAllocator
// C symbols (uma_malloc / uma_free) live in the *same* shared object and are
// resolved by torch via dlopen by name -- see uma_alloc.py.
#include "uma.h"

#include <torch/extension.h>

namespace {

int advise_grace(std::uintptr_t ptr, std::size_t nbytes,
                 std::uintptr_t stream) {
  return uma::advise_range(ptr, nbytes, uma::Node::GRACE, stream);
}

int advise_hbm(std::uintptr_t ptr, std::size_t nbytes, std::uintptr_t stream) {
  return uma::advise_range(ptr, nbytes, uma::Node::HBM, stream);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "UMA-LLM native residency control for GH200 managed KV pages";

  m.def("available", &uma::available,
        "True if CUDA + concurrentManagedAccess are present.");
  m.def("coherent", &uma::coherent,
        "True only on coherent CPU-GPU UMA (GH200 NVLink-C2C); False on "
        "discrete CUDA (A100/PCIe), where a demotion is a real migration.");
  m.def("device_name", &uma::device_name, "CUDA device name string.");
  m.def("grace_device_id", &uma::grace_device_id);
  m.def("hbm_device_id", &uma::hbm_device_id);

  m.def("advise_grace", &advise_grace, py::arg("ptr"), py::arg("nbytes"),
        py::arg("stream") = 0,
        "Prefer+prefetch [ptr, ptr+nbytes) onto the Grace LPDDR5X node.");
  m.def("advise_hbm", &advise_hbm, py::arg("ptr"), py::arg("nbytes"),
        py::arg("stream") = 0,
        "Prefer+prefetch [ptr, ptr+nbytes) onto Hopper HBM3.");
  m.def("query_node", &uma::query_node, py::arg("ptr"), py::arg("nbytes"),
        "Device ordinal of the last prefetch location (-1 == Grace).");

  m.def(
      "pool_stats",
      []() {
        uma::PoolStats s = uma::pool_stats();
        py::dict d;
        d["live_bytes"] = s.live_bytes;
        d["peak_bytes"] = s.peak_bytes;
        d["live_allocs"] = s.live_allocs;
        return d;
      },
      "Managed bytes currently reserved by the KV allocator.");
}
