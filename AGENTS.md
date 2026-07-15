# AGENTS.md — CUDA Roofline Profiler

Entry point for agent sessions. Full scope, milestones, and acceptance
criteria live in `CLAUDE.md` — read it first; it is the source of truth.
This file covers how to run and verify things, plus hard constraints.

## First-run commands

```powershell
# Verify toolchain (all required)
nvcc --version        # expect CUDA 12.6
python --version      # expect 3.13.x
cmake --version       # expect >= 3.24

# Full check: architectural boundaries + CUDA smoke test (compile + run on GPU)
python scripts/check.py

# Boundary check only (no GPU / no compile needed)
python scripts/check.py --no-smoke
```

## Stack + versions (tested)

- CUDA Toolkit 12.6 (V12.6.85), driver 576.80
- GPU: NVIDIA GeForce RTX 4070 **Laptop** (Ada, sm_89) — see DECISIONS.md D2:
  desktop-4070 numbers in CLAUDE.md are fallbacks only; NVML query is truth.
- Python 3.13.5 (system install, no venv yet)
- CMake 4.3.0-rc2
- Windows 11, PowerShell 5.1

## Python dependencies

Not yet installed. Needed from milestone 6 onward (`cli/`, `plot/`):
see `requirements.txt`. Agents must not install — hand the command to the
user:

```powershell
python -m pip install -r requirements.txt
```

## Hard constraints

1. Never `git commit/push/add/reset/stash` — user commits after review.
2. Never install packages — hand exact command to user and stop.
3. Kernel files (`kernels/*.cu`) stay self-contained: no includes from
   `profiler/`, no CUPTI/NVML headers. Enforced by `scripts/check.py`.
4. `profiler/` never depends on `cli/` or `plot/`.
5. `plot/` never imports from `cli/` (cli calls plot, not the reverse).
6. Layering is forward-only: kernels -> profiler -> cli -> plot.
7. Every feature starts as a `features.json` entry with a verification
   command defined BEFORE implementation; state moves to `passing` only
   when that command succeeds.
8. One commit per milestone (CLAUDE.md order). Update PROGRESS.md and
   check off the milestone in the same change.
9. Don't hardcode GPU specs in code — query NVML, fall back to CLAUDE.md
   numbers (they are desktop-4070; this machine is laptop-4070, see D2).
10. Record non-obvious choices in DECISIONS.md as D<n> entries.
11. Root-level scratch files (`chp3_2_2.cu`, `a.exe`, etc.) are the user's
    pre-existing WIP — never delete, move, or clean them.
12. Results JSON goes in `results/` (gitignored except `results/examples/`).
13. Keep `scripts/check.py` stdlib-only so it runs before any install.
