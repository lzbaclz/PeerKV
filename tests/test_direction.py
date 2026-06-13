"""Direction selector: direction is a non-effect, so the chosen initiator must not
change the bytes (the load-bearing correctness property), and the policy returns a
fixed convenience default rather than a performance-optimizing choice."""
import pytest

from umallm.transfer.direction import Endpoint, choose_initiator

try:
    import torch
    _HW = torch.cuda.device_count() >= 2
except Exception:  # pragma: no cover
    torch = None
    _HW = False


def test_policy_returns_convenience_default_regardless_of_membound():
    # Direction is a non-effect (retracted "always-PUSH" law): the initiator is a
    # convenience default (holder-issued PUSH, matching MoRIIO), independent of which
    # endpoint is memory-bound. The victim cost is identical either way.
    h_busy = Endpoint(device=1, memory_bound=True)
    h_idle = Endpoint(device=1, memory_bound=False)
    c_idle = Endpoint(device=0, memory_bound=False)
    c_busy = Endpoint(device=0, memory_bound=True)
    assert choose_initiator(h_busy, c_idle) == 1          # holder issues (default)
    assert choose_initiator(h_idle, c_idle) == 1          # same default, not load-driven
    assert choose_initiator(h_idle, c_busy) == 1          # same default
    assert choose_initiator(h_busy, c_busy) == 1          # same default


@pytest.mark.skipif(not _HW, reason="needs 2 GPUs")
def test_bit_identical_push_vs_pull():
    from umallm.transfer.direction import transfer_push, transfer_pull
    src = torch.randn(1 << 20, dtype=torch.float16, device="cuda:1")
    d_pull = torch.empty_like(src, device="cuda:0")
    d_push = torch.empty_like(src, device="cuda:0")
    transfer_pull(src, d_pull)
    transfer_push(src, d_push)
    assert torch.equal(d_pull, d_push)                     # initiator must not change bytes
    assert torch.equal(d_pull.cpu(), src.cpu())
