# Task 17.12: GPU Performance Benchmarks

**Role:** Test engineer  
**Status:** Not started  
**Depends on:** 17.11 (correctness confirmed before benchmarking)  
**Unlocks:** Nothing — final deliverable

---

## What to implement

### `benchmarks/benchmark_cuda_ctxbgth.py` (new)

Uses the CTX-BG-TH model from `benchmarks/ctxbgth_model.py`. Measures wall-clock time per simulation millisecond for CPU serial, CPU OpenMP, and CUDA.

```python
import time
import hodgkin_huxley as hh
from benchmarks.ctxbgth_model import build_ctxbgth

N_VALUES  = [10, 20, 50, 100, 200, 500]  # neurons per population
DURATION  = 1000.0   # ms
N_REPEATS = 3
DT        = 0.01

def run_mode(rn, device, n_threads=0):
    rn.to(device)
    if n_threads:
        rn.set_num_threads(n_threads)
    # warm-up
    rn.simulate(100.0, DT, ...)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        rn.simulate(DURATION, DT, ...)
        times.append(time.perf_counter() - t0)
    return min(times)

# Run all modes for each N; collect into DataFrame
# Modes: cpu_serial, cpu_omp4, cuda_0

# Plots → benchmarks/figures/
# 1. cuda_ctxbgth_time.png   — time vs N, 3 curves
# 2. cuda_ctxbgth_speedup.png — speedup vs N (cpu_omp4/serial, cuda/serial)
```

### `benchmarks/benchmark_cuda_scaling.py` (new)

Scaling study across neuron count and network topology.

```python
# Topologies: fully_connected, random_sparse (p=0.1)
# N_NEURONS: 50, 100, 200, 500, 1000, 2000 (single population)
# Modes: cpu_serial, cuda
# Duration: 500ms
# Measure: time_per_sim_ms, neurons_per_second

# Plots → benchmarks/figures/
# 1. cuda_scaling_time.png      — time vs N, both modes, both topologies
# 2. cuda_scaling_throughput.png — neurons/s vs N
# 3. cuda_vs_cpu_crossover.png  — find N where CUDA first beats CPU (per topology)
```

### `benchmarks/benchmark_cuda_memory.py` (new)

Profiles VRAM usage vs. network size. Uses `torch.cuda.memory_allocated()` or `pynvml` if available; falls back to `nvidia-smi` subprocess.

```python
# For N = [100, 500, 1000, 5000, 10000, 50000] neurons:
#   measure VRAM before + after rn.to(cuda)
#   measure VRAM during simulate() peak
#   compute theoretical: N*8 (V) + N*4*8 (HH gates) + S*2*8 (syn g+A) + S*delay*1 (ring)

# Plots → benchmarks/figures/
# 1. cuda_memory_usage.png  — measured vs theoretical VRAM (MB) vs N
# 2. cuda_memory_budget.png — stacked bar: neuron state / synapse state / delay ring / overhead
```

### Output

All plots saved to `benchmarks/figures/`. Print a summary table to stdout with columns:
`N | cpu_serial_ms | cpu_omp4_ms | cuda_ms | speedup_omp | speedup_cuda`.

---

## Key files

| File | Change |
|---|---|
| `benchmarks/benchmark_cuda_ctxbgth.py` | New — CTX-BG-TH GPU vs CPU benchmark |
| `benchmarks/benchmark_cuda_scaling.py` | New — scaling study |
| `benchmarks/benchmark_cuda_memory.py` | New — VRAM profiling |

---

## Notes

- All benchmark files are standalone scripts (`python benchmarks/benchmark_cuda_ctxbgth.py`), not pytest tests.
- Skip gracefully if `not hh.cuda_is_available()`: print a message and exit 0.
- Use `torch.cuda.synchronize()` or `cudaDeviceSynchronize()` (via a `rn.synchronize_cuda()` Python binding from 17.10) before stopping the timer to avoid measuring only kernel-launch latency.
- Record the GPU model name in the plot title (`hh.cuda_device_count()` + `nvidia-smi --query-gpu=name` subprocess).
