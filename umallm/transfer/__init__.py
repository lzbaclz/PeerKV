"""Inter-GPU KV transfer primitives.

Direction is a non-effect (see ``direction.py``); the initiator is a convenience
choice, and the real placement lever is the copy's HBM-read-bandwidth footprint.
"""
from .direction import (
    Endpoint,
    choose_initiator,
    transfer,
    transfer_push,
    transfer_pull,
)

__all__ = [
    "Endpoint",
    "choose_initiator",
    "transfer",
    "transfer_push",
    "transfer_pull",
]
