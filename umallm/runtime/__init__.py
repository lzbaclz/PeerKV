"""PeerKV online runtime (Track C P2/P3). PLANNED -- mostly stubs.

This package marks the planned-but-unbuilt online hot path so the design in
``collaboration_plan/`` maps to real file paths (risk 9.C.1: "docs >> code").
What exists today is the *offline* algorithm core in ``umallm.elastic_policy``
(``select_point``) and ``umallm.multigpu`` (cost model). Modules here are where
the online versions land:

  - ``selector``     -- online cost-model-gated corner selector + enforce_do_no_harm (04 SS1)
  - ``kv_manager``   -- online block placement state machine (MASTER_PLAN SS3.2)
  - ``calibration``  -- CUDA startup probe + online predicted-vs-measured loop (M3)

Each raises ``NotImplementedError`` with a pointer to its spec until built.
"""
