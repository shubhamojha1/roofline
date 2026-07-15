#!/usr/bin/env python3
"""Repo check: architectural boundaries + CUDA smoke test.

Stdlib only (must run before any pip install). Exit 0 = all pass.

Boundary rules (see DECISIONS.md D3, forward-only layering
kernels -> profiler -> cli -> plot):
  B1: kernels/*.cu must not include profiler/, cupti*, or nvml* headers.
  B2: profiler/* must not reference cli/ or plot/.
  B3: plot/*.py must not import cli.

Smoke test: compiles tests/test_smoke.cu with nvcc and runs it on the
GPU. Skip with --no-smoke (e.g. on a machine without a GPU).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def check_boundaries() -> None:
    kernel_bad = re.compile(
        r'#include\s+["<](?:\.\./)?(?:profiler/(?!timing\.cuh)|cupti|nvml)', re.IGNORECASE
    )
    kernels = sorted((ROOT / "kernels").glob("*.cu")) if (ROOT / "kernels").is_dir() else []
    for f in kernels:
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if kernel_bad.search(line):
                fail(f"B1 {f.relative_to(ROOT)}:{i} kernel includes profiler/CUPTI/NVML: {line.strip()}")
    ok(f"B1 kernels self-contained ({len(kernels)} file(s) checked)")

    profiler_bad = re.compile(r'(?:#include\s+["<]|import\s+)(?:\.\./)?(?:cli/|plot/|cli\b|plot\b)')
    prof_files = (
        sorted((ROOT / "profiler").glob("*.c*")) + sorted((ROOT / "profiler").glob("*.cu"))
        if (ROOT / "profiler").is_dir()
        else []
    )
    for f in prof_files:
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if profiler_bad.search(line):
                fail(f"B2 {f.relative_to(ROOT)}:{i} profiler references cli/plot: {line.strip()}")
    ok(f"B2 profiler independent of cli/plot ({len(prof_files)} file(s) checked)")

    plot_bad = re.compile(r"^\s*(?:from\s+cli|import\s+cli)\b")
    plot_files = sorted((ROOT / "plot").glob("*.py")) if (ROOT / "plot").is_dir() else []
    for f in plot_files:
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if plot_bad.search(line):
                fail(f"B3 {f.relative_to(ROOT)}:{i} plot imports cli: {line.strip()}")
    ok(f"B3 plot does not import cli ({len(plot_files)} file(s) checked)")


def run_smoke() -> None:
    src = ROOT / "tests" / "test_smoke.cu"
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    exe = build / "test_smoke.exe"

    r = subprocess.run(
        ["nvcc", "-arch=sm_89", str(src), "-o", str(exe)],
        capture_output=True, text=True, cwd=ROOT,
    )
    if r.returncode != 0:
        fail(f"smoke: nvcc compile failed:\n{r.stderr.strip()}")
        return
    ok("smoke: nvcc compile")

    r = subprocess.run([str(exe)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        fail(f"smoke: run failed (exit {r.returncode}): {r.stdout.strip()} {r.stderr.strip()}")
        return
    ok(f"smoke: GPU run ({r.stdout.strip()})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-smoke", action="store_true", help="skip nvcc compile + GPU run")
    args = p.parse_args()

    check_boundaries()
    if not args.no_smoke:
        run_smoke()

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
