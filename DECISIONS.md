# DECISIONS

Append-only. Format: D<n> — <date> — <decision>. <why>.

- D1 — 2026-07-11 — Test framework is two-layer: `scripts/check.py`
  (Python stdlib only, no installs) runs (a) grep-based architectural
  boundary checks and (b) an nvcc-compiled CUDA smoke test
  (`tests/test_smoke.cu`) executed on the GPU. Why: pytest/matplotlib
  are not installed and installs are user-only on this machine; the
  real correctness signal for this project is "kernel compiles and
  runs on the GPU", which needs no third-party framework. pytest can
  be layered on later for the Python CLI/plot code (requirements.txt
  already lists it).

- D2 — 2026-07-11 — GPU is RTX 4070 **Laptop** (driver 576.80), not
  desktop 4070. CLAUDE.md's ~29.15 TFLOP/s FP32 / ~504.2 GB/s are
  desktop numbers; laptop part is roughly 15–20 TFLOP/s and ~256 GB/s.
  Decision: keep CLAUDE.md numbers as the documented fallback (per its
  own instruction) but treat runtime NVML query (milestone 2) as the
  source of truth for roofs; ridge point will differ from the blog
  post's 57.8 FLOP/byte on this machine.

- D3 — 2026-07-11 — Layering rule: kernels -> profiler -> cli -> plot,
  forward-only. kernels/*.cu self-contained (no profiler/CUPTI/NVML
  includes); profiler/ independent of Python layers; plot/ never
  imports cli/. Why: CLAUDE.md mandates independently compilable
  kernels, and CUPTI isolation (milestone 4) only works if kernels
  stay CUPTI-unaware. Enforced executably in scripts/check.py, not as
  prose.

- D4 — 2026-07-11 — No venv yet; system Python 3.13.5 pinned via
  .python-version. Why: creating a venv is pointless before the user
  installs requirements.txt; revisit at milestone 6 when cli/ starts.

- D5 — 2026-07-11 — Pre-existing root scratch files (chp3_2_2.cu,
  a.exe, add.cu, ...) left untouched; binaries covered by .gitignore
  patterns so they stop appearing in git status. Moving/deleting them
  is the user's call.

- D6 — 2026-07-11 — harness-init's "checkpoint commit" step is
  intentionally left to the user (machine rule: agent never commits).
  Handover includes the exact commit command instead.
