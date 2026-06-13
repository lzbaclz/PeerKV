import sys
def p(*a): print(*a, flush=True)

import torch
p("1 torch", torch.__version__, "cuda avail", torch.cuda.is_available())
import umallm._uma_native as n
p("2 native available", n.available(), "coherent", n.coherent(), "dev", n.device_name())

from umallm.uma_alloc import UMAManagedAllocator
p("3 constructing UMAManagedAllocator")
alloc = UMAManagedAllocator()
p("4 got allocator obj")
a = alloc.allocator()
p("5 .allocator() ok ->", type(a))
pool = alloc.mem_pool()
p("6 mem_pool() ok ->", type(pool))

p("7 allocating tensor inside use_mem_pool scope")
with torch.cuda.use_mem_pool(pool):
    buf = torch.empty(64*1024*1024//2, dtype=torch.float16, device="cuda")
    p("8 alloc done, data_ptr", hex(buf.data_ptr()))
    buf.fill_(1.0)
    p("9 fill done")
torch.cuda.synchronize()
p("10 sync after scope ok")

# residency hint via native directly
ptr = buf.data_ptr(); nbytes = buf.numel()*buf.element_size()
p("11 advise_grace rc", n.advise_grace(ptr, nbytes, 0))
torch.cuda.synchronize()
p("12 query_node", n.query_node(ptr, nbytes))
p("13 advise_hbm rc", n.advise_hbm(ptr, nbytes, 0))
torch.cuda.synchronize()
p("14 done; sum", float(buf.sum().item()))
