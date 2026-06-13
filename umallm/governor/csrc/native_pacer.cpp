// NativePacer -- the governor's pacing hot path, off the GIL.
//
// Replaces umallm/governor/pacer.py's per-chunk loop for production use.
// The Python pacer's measured costs (adversarial review, 2026-06-10):
//   (1) commanded-rate undershoot of 10-27% above 0.5x link (sleep
//       granularity + GIL scheduling jitter accumulate into every gap), so
//       the s1 census topped out at 0.72x link;
//   (2) a host-launch/GIL confound collinear with the census x-axis
//       (~800-3000 GIL-holding launches/s next to the victim's timed loop).
// This module runs the whole chunk train in C++ with the GIL RELEASED:
// cudaMemcpyAsync chunks on a dedicated non-blocking stream, depth-N
// cudaEvent pipelining (launch chunk i after waiting on chunk i-depth, so
// the DMA engine never drains), and precise gap pacing (coarse 100us sleeps
// down to 150us-from-deadline, then a steady_clock spin).
//
// Control-plane policy stays in Python; the only shared state is ControlBox
// (atomic rate + cancel flag) which Python may update at any time and the
// C++ loop reads lock-free before each launch.  Pacing-clock epoch is
// CLOCK_MONOTONIC on Linux for both std::chrono::steady_clock and Python's
// time.monotonic(), so next_due threads transparently across the two pacers.
//
// Build: JIT via torch.utils.cpp_extension.load (see ../native.py).  Pure
// host C++ (no kernels) -- compiles with g++, links cudart only.

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("native_pacer: ") + #call +     \
                               " failed: " + cudaGetErrorString(_e));      \
    }                                                                      \
  } while (0)

namespace {

double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct ControlBox {
  std::atomic<double> rate_gbs{0.0};   // <=0 or inf => no gap for that chunk
  std::atomic<bool> cancelled{false};

  void set_rate(double r) { rate_gbs.store(r, std::memory_order_relaxed); }
  double get_rate() const { return rate_gbs.load(std::memory_order_relaxed); }
  void cancel() { cancelled.store(true, std::memory_order_relaxed); }
  bool is_cancelled() const {
    return cancelled.load(std::memory_order_relaxed);
  }
  void reset() {
    cancelled.store(false, std::memory_order_relaxed);
  }
};

// Sleep coarsely until ~150us before the deadline, then spin.  Polls the
// cancel flag every coarse slice; returns false if cancelled.
bool wait_until(double deadline, const ControlBox& box) {
  constexpr double kSpinWindow = 150e-6;
  constexpr double kSlice = 100e-6;
  while (true) {
    if (box.is_cancelled()) return false;
    double remaining = deadline - now_s();
    if (remaining <= 0) return true;
    if (remaining > kSpinWindow) {
      double nap = std::min(kSlice, remaining - kSpinWindow);
      std::this_thread::sleep_for(std::chrono::duration<double>(nap));
    } else {
      while (now_s() < deadline) {            // ~<=150us busy spin
        if (box.is_cancelled()) return false;
      }
      return true;
    }
  }
}

class NativePacer {
 public:
  NativePacer(int device, int depth)
      : device_(device), depth_(std::max(1, depth)) {
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    events_.resize(depth_);
    for (auto& e : events_) {
      CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
    // make sure peer paths are direct (NVLink), not host-staged
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    for (int d = 0; d < ndev; ++d) {
      if (d == device_) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, device_, d) == cudaSuccess && can) {
        cudaError_t e = cudaDeviceEnablePeerAccess(d, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
          CUDA_CHECK(e);
        }
        (void)cudaGetLastError();   // clear AlreadyEnabled sticky state
      }
    }
  }

  ~NativePacer() {
    // best-effort: the CUDA runtime may already be torn down at interpreter
    // exit; never throw from a destructor
    for (auto& e : events_) cudaEventDestroy(e);
    cudaStreamDestroy(stream_);
  }

  NativePacer(const NativePacer&) = delete;
  NativePacer& operator=(const NativePacer&) = delete;

  py::dict run(std::uintptr_t dst, std::uintptr_t src, int64_t nbytes,
               int64_t chunk_bytes, ControlBox& box, bool unpaced,
               double next_due0) {
    if (nbytes <= 0 || chunk_bytes <= 0) {
      throw std::runtime_error("native_pacer: nbytes/chunk_bytes must be >0");
    }
    int64_t launched_bytes = 0;
    int64_t chunks = 0;
    double gap_total = 0.0;
    bool cancelled = false;
    double start, done, next_due;

    {
      py::gil_scoped_release release;          // the whole train is GIL-free
      CUDA_CHECK(cudaSetDevice(device_));
      const int64_t n_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
      start = now_s();
      next_due = (next_due0 > 0.0) ? next_due0 : start;

      for (int64_t ci = 0; ci < n_chunks; ++ci) {
        if (box.is_cancelled()) {
          cancelled = true;
          break;
        }
        if (ci >= depth_) {
          // bound the in-flight queue: wait for chunk ci-depth (DMA-time
          // bounded; we are off the GIL so nobody is starved)
          CUDA_CHECK(cudaEventSynchronize(events_[ci % depth_]));
        }
        if (!unpaced) {
          double t = now_s();
          if (t < next_due) {
            gap_total += next_due - t;
            if (!wait_until(next_due, box)) {
              cancelled = true;
              break;
            }
          }
        }
        const int64_t lo = ci * chunk_bytes;
        const int64_t hi = std::min(nbytes, lo + chunk_bytes);
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(dst) + lo,
                                   reinterpret_cast<const char*>(src) + lo,
                                   static_cast<size_t>(hi - lo),
                                   cudaMemcpyDefault, stream_));
        CUDA_CHECK(cudaEventRecord(events_[ci % depth_], stream_));
        launched_bytes += hi - lo;
        ++chunks;
        if (!unpaced) {
          const double r = box.get_rate();
          if (r > 0.0 && std::isfinite(r)) {
            // schedule from max(now, next_due): a late chunk does not earn
            // back its delay as a burst credit (same rule as the Python pacer)
            const double base = std::max(now_s(), next_due);
            next_due = base + static_cast<double>(hi - lo) / (r * 1e9);
          }
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(stream_));   // stream-scoped drain only
      done = now_s();
    }

    py::dict out;
    out["start_t"] = start;
    out["done_t"] = done;
    out["chunks_launched"] = chunks;
    out["bytes_launched"] = launched_bytes;
    out["paced_gap_s"] = gap_total;
    out["next_due_final"] = next_due;
    out["cancelled"] = cancelled;
    return out;
  }

 private:
  int device_;
  int depth_;
  cudaStream_t stream_{};
  std::vector<cudaEvent_t> events_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PeerKV governor native pacing hot path (GIL-free chunk trains)";
  py::class_<ControlBox>(m, "ControlBox")
      .def(py::init<>())
      .def("set_rate", &ControlBox::set_rate)
      .def("get_rate", &ControlBox::get_rate)
      .def("cancel", &ControlBox::cancel)
      .def("is_cancelled", &ControlBox::is_cancelled)
      .def("reset", &ControlBox::reset);
  py::class_<NativePacer>(m, "NativePacer")
      .def(py::init<int, int>(), py::arg("device"), py::arg("depth") = 2)
      .def("run", &NativePacer::run, py::arg("dst"), py::arg("src"),
           py::arg("nbytes"), py::arg("chunk_bytes"), py::arg("box"),
           py::arg("unpaced") = false, py::arg("next_due0") = -1.0);
  m.def("monotonic_s", &now_s,
        "steady_clock seconds (same epoch as time.monotonic on Linux)");
}
