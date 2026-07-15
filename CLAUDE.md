# CUDA Roofline Profiler — Project Guide

This file is loaded automatically by Claude Code at the start of every
session in this repo. Treat it as the source of truth for scope, structure,
and build order — check off milestones here as they're completed rather
than tracking progress elsewhere.

## Goal
A standalone, open-source CLI tool that profiles CUDA kernels and plots them
on a roofline chart: arithmetic intensity (FLOP/byte) on the x-axis vs.
achieved throughput (FLOP/s) on the y-axis, overlaid with the GPU's compute
and memory-bandwidth roofs. Purpose: portfolio piece demonstrating CUDA
profiling depth, culminating in a Flash Attention kernel case study.

Target hardware: RTX 4070 (Ada Lovelace, sm_89), ~29.15 TFLOP/s FP32,
~504.2 GB/s HBM bandwidth (ridge point ≈ 57.8 FLOP/byte — reuse these
numbers from the FlashAttention post, don't re-derive).

## Repo layout
```
roofline-profiler/
├── kernels/
│   ├── 00_elementwise_add.cu
│   ├── 01_naive_gemm.cu
│   ├── 02_tiled_gemm.cu
│   ├── 03_softmax.cu
│   ├── 04_naive_attention.cu
│   └── 05_flash_attention.cu
├── profiler/
│   ├── timing.cu          # CUDA Events wrapper
│   ├── nvml_specs.cpp      # queries peak FLOPs/bandwidth via NVML
│   └── cupti_counters.cpp  # DRAM byte counters via CUPTI
├── cli/
│   └── roofline.py         # orchestrates runs, collects JSON, calls plotter
├── plot/
│   └── plot_roofline.py    # matplotlib roofline chart from JSON results
├── results/                # JSON output per kernel run, gitignored except examples
├── CMakeLists.txt
└── README.md
```

## Milestones (build in order, one commit per milestone)

1. **Scaffolding + CUDA Events timing.** `timing.cu` — a reusable
   `GpuTimer` class (cudaEventRecord start/stop, elapsed ms). Wire up
   `00_elementwise_add.cu` as the smoke test kernel.

2. **NVML hardware spec query.** `nvml_specs.cpp` reads peak FP32 TFLOP/s
   and peak memory bandwidth for the current device at runtime (don't
   hardcode — but fall back to the known 4070 numbers if NVML query fails).

3. **Naive + tiled GEMM kernels.** `01_naive_gemm.cu`, `02_tiled_gemm.cu`.
   Each kernel reports FLOPs analytically (2*M*N*K) so the profiler has a
   ground truth to compare against measured throughput.

4. **CUPTI DRAM byte counters.** `cupti_counters.cpp` — wraps CUPTI's
   Profiling API to capture `dram__bytes.sum` (or equivalent event) per
   kernel launch. This is the hard part — isolate it behind a clean
   `measure_dram_bytes(kernel_launch_fn)` interface so kernel code doesn't
   need CUPTI awareness.

5. **Softmax + naive attention kernels.** `03_softmax.cu`,
   `04_naive_attention.cu`. These establish the memory-bound baseline that
   Flash Attention will be compared against.

6. **CLI orchestration.** `cli/roofline.py` — runs each kernel via a thin
   ctypes/pybind11 binding or subprocess, collects (arithmetic_intensity,
   achieved_flops, kernel_name) tuples, writes `results/<kernel>.json`.

7. **Roofline plotter.** `plot/plot_roofline.py` — matplotlib, log-log
   axes, draws the two roofs (compute ceiling, bandwidth ceiling) using
   NVML-queried specs, scatters each kernel's point, labels ridge point.

8. **Flash Attention capstone.** `05_flash_attention.cu` — port the
   existing FA2 tiled kernel (online softmax, causal masking) from the
   FlashAttention Substack post. Profile it alongside naive attention on
   the same roofline plot as the headline result.

## Explicit non-goals (v1)
- No backward pass, no FA3/Hopper-specific kernels.
- No multi-GPU support.
- No web UI — CLI + static PNG/SVG plot output only.

## Acceptance criteria
- `python cli/roofline.py --all` runs every kernel in `kernels/` and
  produces a single `roofline.png` with all points plotted.
- Naive attention and Flash Attention are visibly at different points on
  the plot, with Flash Attention closer to the compute roof — this is the
  money shot for the portfolio README.
- README includes the roofline plot image and a one-paragraph explanation
  of what arithmetic intensity means and why FA moves the point right.

## Notes for Claude Code
- Reuse ridge-point math and SRAM/block-size reasoning from the existing
  FlashAttention CUDA blog post draft rather than re-deriving.
- Pin CUDA toolkit / driver version in README once tested (same discipline
  as the seccomp kernel-version pin — specificity is a credibility signal).
- Keep kernel files self-contained and independently compilable for
  readability; CLI just orchestrates.