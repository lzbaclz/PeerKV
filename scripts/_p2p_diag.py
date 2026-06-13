"""Careful P2P / tier bandwidth diagnostic to corroborate (or correct) e15.

Measures, with CUDA events and several buffer sizes, the effective copy bandwidth
of every relevant path, and prints whether torch actually enabled peer access.
"""
import torch, time, statistics

def bw_events(fn, nbytes, trials=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(trials):
        torch.cuda.synchronize()
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)  # s
    return nbytes / statistics.median(ts) / 1e9

def bw_wall(fn, nbytes, trials=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return nbytes / statistics.median(ts) / 1e9

print("torch", torch.__version__, "ndev", torch.cuda.device_count())
print("can_access_peer 0->1:", torch.cuda.can_device_access_peer(0,1),
      " 1->0:", torch.cuda.can_device_access_peer(1,0))

# Try to explicitly enable peer access at the driver level via cudart.
import ctypes
try:
    cudart = ctypes.CDLL("libcudart.so")
    def enable(dev, peer):
        cudart.cudaSetDevice(dev)
        rc = cudart.cudaDeviceEnablePeerAccess(peer, 0)
        return rc  # 0 ok, 704 already enabled
    r01 = enable(0,1); r10 = enable(1,0)
    print("cudaDeviceEnablePeerAccess 0->1 rc:", r01, " 1->0 rc:", r10,
          "(0=ok, 704=already-enabled)")
    torch.cuda.set_device(0)
except Exception as ex:
    print("explicit enable failed:", ex)

for mb in (64, 256, 512):
    n = mb*1024*1024//2
    nbytes = n*2
    a0 = torch.ones(n, dtype=torch.float16, device="cuda:0")
    b1 = torch.ones(n, dtype=torch.float16, device="cuda:1")
    h  = torch.ones(n, dtype=torch.float16, device="cpu").pin_memory()
    d0 = torch.empty(n, dtype=torch.float16, device="cuda:0")
    d1 = torch.empty(n, dtype=torch.float16, device="cuda:1")
    torch.cuda.set_device(0)
    loc  = bw_events(lambda: d0.copy_(a0), nbytes)                       # local D2D (read+write HBM)
    peer = bw_events(lambda: d0.copy_(b1), nbytes)                       # peer GPU -> cuda:0
    host = bw_events(lambda: d0.copy_(h, non_blocking=True), nbytes)     # host pinned -> cuda:0
    h2 = torch.empty(n, dtype=torch.float16, device="cpu").pin_memory()
    d2h  = bw_events(lambda: h2.copy_(a0, non_blocking=True), nbytes)    # cuda:0 -> host pinned
    print(f"mb={mb:4d}  local_D2D={loc:8.1f}  peer->0={peer:8.2f}  host->0={host:7.2f}  0->host={d2h:7.2f} GB/s")

# Also: cudaMemcpyPeer directly (bypass torch copy kernel), 512MB
mb=512; n=mb*1024*1024//2; nbytes=n*2
a0 = torch.ones(n, dtype=torch.float16, device="cuda:0")
b1 = torch.ones(n, dtype=torch.float16, device="cuda:1")
def memcpy_peer():
    cudart.cudaMemcpyPeer(ctypes.c_void_p(a0.data_ptr()), 0,
                          ctypes.c_void_p(b1.data_ptr()), 1, ctypes.c_size_t(nbytes))
try:
    bwp = bw_wall(memcpy_peer, nbytes)
    print(f"cudaMemcpyPeer 1->0 (512MB): {bwp:.2f} GB/s")
except Exception as ex:
    print("cudaMemcpyPeer failed:", ex)
