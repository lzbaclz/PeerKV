"""Direction-neutral inter-GPU KV transfer shim.

Earlier versions of this module encoded a "read-port arbitration law" / "always-PUSH
rule": the claim that a remotely issued read (a consumer PULL) of a memory-bound
peer's HBM is penalized, so the holder should PUSH. **That claim was a measurement
artifact and has been retracted.** Under controlled measurement (locked clocks,
overlap-safe per-iteration timing, DCGM counters) on dual-A100 NVLink3 and dual-H100
NVLink4, PUSH and PULL are indistinguishable in both effective bandwidth and the
busy holder's decode slowdown: the bytes physically traverse the same holder->consumer
NVLink path regardless of who issues the copy, so the holder's memory system sees the
same load either way.

The real lever is the copy's HBM-read-bandwidth footprint, set by the
**destination/link**, not by the direction:

  - NVLink peer handoff  -> gentle on a busy holder (do-no-harm)
  - PCIe host offload     -> gentler still but far slower (into slow memory)
  - same-GPU local repack -> brutal (full-rate HBM read on the busy GPU)

So the transfer-time placement guidance is: prefer an NVLink peer handoff over a
full-rate local repack on a busy GPU, and chunk a large handoff to bound the tail.
The *initiator* (push vs pull) is a correctness/convenience choice only -- pick it for
protocol or software-pipelining reasons, never as a performance knob. This module is a
direction-neutral connector shim: same bytes, freely chosen initiator.
"""
from __future__ import annotations
from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only CI (the direction policy is pure logic)
    torch = None  # type: ignore[assignment]

# Software-convenience default initiator. Direction is a non-effect, so we follow
# MoRIIO's shipped default (write-mode / holder-issued PUSH), which composes with
# layer pipelining. Flip this only for protocol reasons, not for performance.
DEFAULT_INITIATOR_IS_HOLDER = True


@dataclass
class Endpoint:
    """Live state of one GPU endpoint the connector reads cheaply from the
    scheduler. ``memory_bound`` (a GPU draining a decode queue is HBM-read-bound)
    is retained because it drives real placement decisions (peer vs local repack),
    not the now-retired direction choice."""
    device: int
    memory_bound: bool          # True iff this GPU is currently HBM-read-bound (decoding)


def choose_initiator(holder: Endpoint, consumer: Endpoint) -> int:
    """Return the device id that issues the copy.

    Direction is a non-effect (see module docstring), so this returns a fixed
    convenience default rather than a performance-optimizing choice: the holder
    issues (PUSH) by default, matching MoRIIO's shipped write-mode. Callers may
    override for protocol reasons; the bytes and the victim cost are identical
    either way.
    """
    return holder.device if DEFAULT_INITIATOR_IS_HOLDER else consumer.device


def transfer(src: torch.Tensor, dst: torch.Tensor, holder: Endpoint,
             consumer: Endpoint, block_index: torch.Tensor | None = None) -> None:
    """Move ``src`` (resident on ``holder.device``) into ``dst`` (on
    ``consumer.device``). The initiator is the convenience default; it does not
    affect the bytes or the holder's decode slowdown. If ``block_index`` is given,
    gather those paged rows on the issuing side first. Bit-identical to a plain copy.
    """
    init = choose_initiator(holder, consumer)
    with torch.cuda.device(init):
        stream = torch.cuda.Stream(device=init)
        with torch.cuda.stream(stream):
            if block_index is not None:
                if init == holder.device:
                    gathered = src.index_select(0, block_index)
                    dst.copy_(gathered, non_blocking=True)
                else:
                    dst.copy_(src.index_select(0, block_index.to(src.device)),
                              non_blocking=True)
            else:
                dst.copy_(src, non_blocking=True)
        stream.synchronize()


# Baselines for the evaluation (the two shipping policies; both are equivalent here).
def transfer_pull(src, dst):
    """NIXL / read-mode: consumer issues the read. Equivalent to push (direction
    is a non-effect); kept as a baseline to demonstrate the equivalence."""
    with torch.cuda.device(dst.device):
        s = torch.cuda.Stream(device=dst.device)
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)
        s.synchronize()


def transfer_push(src, dst):
    """MoRIIO / write-mode: holder issues the write. Equivalent to pull (direction
    is a non-effect); the shipped default we follow for software convenience."""
    with torch.cuda.device(src.device):
        s = torch.cuda.Stream(device=src.device)
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)
        s.synchronize()
