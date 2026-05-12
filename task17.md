# Task 17: CUDA GPU Acceleration

## Priority: 3 — Depends on task13 (CUDAPrinter for SymPy → `__device__`)

## Status: Not started

---

## Subtask Progress

| # | Role | Description | Done |
|---|---|---|---|
| [17.1](completed/task17.1.md) | Team lead | PoolBase CUDA interface + Device struct | [x] |
| [17.2](task17.2.md) | Team lead | CMake CUDA build configuration + stub `.cu` files | [ ] |
| [17.3](task17.3.md) | Team lead | PoolManager + Network CUDA routing + pinned memory + `RegionalNetwork::to()` | [ ] |
| [17.4](task17.4.md) | Team lead | Python Device API, `hh.device("cuda:0")`, pybind11 bindings, `.pyi` stubs | [ ] |
| [17.5](task17.5.md) | Codegen | CUDAPrinter: SymPy → `__device__` codegen + `compile_gate_cuda()` | [ ] |
| [17.6](task17.6.md) | CUDA eng | CudaHHPool + CudaIzPool — device SoA, step kernels, scatter/gather | [ ] |
| [17.7](task17.7.md) | CUDA eng | On-device synapse kernels + per-neuron spike delay ring buffer | [ ] |
| [17.8](task17.8.md) | CUDA eng | CudaComposablePool — device SoA + pattern-matched gate kernels | [ ] |
| [17.9](task17.9.md) | CUDA eng + Codegen | CudaComposablePool — CUSTOM_EXPR gates, intracellular dynamics, modulation | [ ] |
| [17.10](task17.10.md) | VRAM eng | Async double-buffered recording pipeline to pinned host memory | [ ] |
| [17.11](task17.11.md) | Test eng | CUDA correctness test suite (device API, pool correctness, recording, robustness) | [ ] |
| [17.12](task17.12.md) | Test eng | GPU performance benchmarks (CTX-BG-TH, scaling study, VRAM profiling) | [ ] |

**Dependency order:** 17.1 → 17.2 → {17.6, 17.7, 17.8} → 17.9 → 17.10 → 17.11 → 17.12  
**Parallel tracks:** 17.3 → 17.4 (Python API available once network routing lands); 17.5 (CUDAPrinter) branches from 17.1 independently; 17.8 depends on 17.6; 17.9 depends on 17.5 + 17.8.

---

## Overview

Add an optional CUDA backend for GPU-accelerated simulation of large networks.
The CPU path (Eigen pools + SoA synapses) remains the default and is never
affected by this task.  CUDA activates **only when explicitly requested** by the
user — no automatic threshold, no silent fallback.

Target scale: the fruit fly connectome (~135K neurons, ~50M synapses) on a
single consumer 24 GB GPU, with a clear path to multi-GPU for larger models
via the delay-decomposition abstraction already established in task16.

Expected speedup at scale: 50–200× vs C++ CPU for networks >5 000 neurons.

> **Environment note:** conda is required for CUDA work — the CUDA toolkit,
> cuDNN, and NCCL are not pip/UV-installable.  Always build with the `rebuild`
> conda env active (`conda install -c nvidia cuda-toolkit cudnn`).  The UV
> `.venv` is for pure-Python testing on non-GPU machines only.

---

## 17.1 Design Constraints and Resolutions

These constraints were established before implementation to avoid expensive
redesigns later.

### C1 — Step-to-step dependency ("GPU bubbling")

Each time step depends on the previous; parallelism is **within** a step (all N
neurons simultaneously), not across steps.  This is not a problem: at N ≥ ~500
neurons a single step saturates a modern GPU.  Below that threshold GPU overhead
exceeds any gain — the GPU path should be disabled (or warn) for small networks.
The sequential structure of the time loop is unchanged; only what executes inside
each step changes.

### C2 — Izhikevich voltage resets and other conditionals

`if (v >= 30) { v = c; u += d; }` causes warp divergence in naive CUDA.
Resolution: replace with predicated assignment — identical to the Eigen `select`
pattern already in `IzPool`:

```cuda
bool fired = v[i] >= 30.0;
v[i] = fired ? c : v[i];
u[i] = fired ? u[i] + d : u[i];
```

This compiles to `SETP` + `SEL` — one predicated execution path, no true
divergence, one extra instruction per thread.  All other conditionals in the
simulation (spike detection, synapse gating) follow the same pattern.

### C3 — Recording without CPU↔GPU round-trips

Recording cannot accumulate full traces in VRAM (see §17.5 for the VRAM budget
— a 135K-neuron 1 s simulation at dt=0.01 ms would need 108 GB for V alone).

**Required strategy: async streaming to pinned host memory.**  Every
`record_interval` steps, the simulation kernel writes a slice into a small
on-device staging buffer; a secondary CUDA stream fires a `cudaMemcpyAsync` to
pinned host memory concurrently with the next compute batch.  Double-buffering
the staging buffer eliminates any synchronization stall.

This is the **only** recording path for the GPU backend.  VRAM usage for
recording is bounded at `2 × N × record_interval × 8` bytes regardless of
simulation length.

### C4 — VRAM minimization / no CPU↔GPU round-trips during the hot loop

All simulation state must be in VRAM for the entire loop.  Data that crosses the
PCIe bus mid-loop kills throughput.

Rules:
- Upload all neuron/synapse state **once** before the loop starts.
- Free all setup temporaries (unsorted weight lists, index maps used to build
  CSR) **immediately after** the on-device CSR structure is built.
- Spike delay buffers live **on device** as ring buffers (see §17.6); no
  per-step spike vector upload from CPU.
- The only intentional PCIe traffic during the loop is the async recording
  stream — and it runs on a separate CUDA stream so it never stalls compute.

---

## 17.2 Device Model

Devices are first-class objects following PyTorch conventions.

```cpp
// src/cpp/include/hodgkin_huxley/device.hpp
struct Device {
    enum class Type { CPU, CUDA };
    Type type  = Type::CPU;
    int  index = 0;          // CUDA device index; ignored for CPU

    static Device cpu()             { return {Type::CPU,  0}; }
    static Device cuda(int idx = 0) { return {Type::CUDA, idx}; }
    bool operator==(const Device&) const = default;
    std::string str() const;   // "cpu", "cuda:0", "cuda:1", ...
};

int  cuda_device_count();          // 0 if CUDA unavailable / not built
bool cuda_is_available();          // true iff at least one CUDA device present
```

```cpp
// On RegionalNetwork / Network
void   to(Device d);               // move to device; raises if !cuda_is_available()
Device device() const;             // current device
```

---

## 17.3 Python API

```python
import hodgkin_huxley as hh

# --- Device query (PyTorch-style) ---
hh.cuda.is_available()       # bool — standard guard for CUDA-conditional code
hh.cuda.device_count()       # int
hh.cuda.get_device_name(0)   # str — "NVIDIA RTX 4090" etc.

# --- Device objects ---
dev_gpu  = hh.device("cuda")     # default GPU (index 0)
dev_gpu0 = hh.device("cuda:0")   # explicit index
dev_cpu  = hh.device("cpu")

# --- Assign network to device (user-explicit, never automatic) ---
rn = hh.RegionalNetwork()
# ... build network ...
if hh.cuda.is_available():
    rn.to(hh.device("cuda:0"))

# Device is retained for the lifetime of the object
print(rn.device())               # hh.device("cuda:0")

# simulate() dispatches based on rn.device() — no extra arguments
result = rn.simulate(2000.0, 0.01, {"A": 10.0})

# --- Move back to CPU (e.g. to inspect state or use serial tools) ---
rn.to(hh.device("cpu"))
```

**No user action is required for custom SymPy equations** — CUDAPrinter (task13)
generates `__device__` functions automatically when the CUDA path is active.

---

## 17.4 VRAM Budget Analysis

At the fruit fly connectome scale (135K neurons, 50M synapses, 1 s simulation,
dt = 0.01 ms, record_interval = 100 steps):

| Item | Formula | Size |
|---|---|---|
| Neuron state (V + ~15 gates/vars) | 135K × 16 × 8 B | ~17 MB |
| Synapse CSR arrays (weight, delay, pre, post, g_syn) | 50M × 24 B | ~1.2 GB |
| I_syn scratch | 135K × 8 B | ~1 MB |
| Delay ring buffers (on-device, max 5 ms delay) | 50M × 500 B | ~25 GB ⚠ |
| Recording staging (double-buffered, async) | 2 × 135K × 100 × 8 B | ~216 MB |
| **Total without delay buffers** | | **~1.4 GB** |

The delay ring buffer calculation above uses a per-synapse-per-step byte flag,
which is the Phase 2 cross-group buffer layout.  On a single GPU this collapses:
since all neurons run in the same kernel, the delay buffer can be stored as a
**per-neuron circular spike history** (1 bit per neuron per step) rather than
per-synapse:

| Item (revised delay) | Formula | Size |
|---|---|---|
| Spike history ring (5 ms @ dt=0.01 ms = 500 steps) | 135K × 500 bits | ~8 MB |
| **Total (revised)** | | **~1.4 GB** |

This comfortably fits on a 24 GB consumer GPU.  The simulation state of the
entire fruit fly connectome is viable on a single GPU.  Recording is the only
variable that could grow without bound — the async streaming strategy caps it.

---

## 17.5 CUDA Kernels

### Neuron Pool Kernel

One thread per neuron.  All state in registers / global memory; no cross-thread
dependencies within a step.

```cuda
__global__ void hh_pool_step_kernel(
    double* __restrict__ V,
    double* __restrict__ m, double* __restrict__ h, double* __restrict__ n,
    const double* __restrict__ I_ext, const double* __restrict__ I_syn,
    HHParams params, double dt, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Euler step in registers; RK4 after validation
    // ...
}
```

Grid: `<<<(N + 255) / 256, 256>>>`.

For Izhikevich neurons, voltage reset uses predicated assignment (§17.1 C2) —
no warp divergence.

For composable neurons, `__device__` gate functions generated by CUDAPrinter
(task13) are called directly per thread — no JIT, no runtime compilation at
simulate time.

### Synaptic Current Accumulation

One thread per synapse.  Multiple pre-synaptic neurons → same post-synaptic
neuron requires atomic accumulation.

```cuda
__global__ void isyn_accumulate_kernel(
    const size_t* __restrict__ post,
    const double* __restrict__ g,
    const double* __restrict__ E_syn,
    const double* __restrict__ V,
    double* I_syn, size_t n_synapses)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;
    atomicAdd(&I_syn[post[i]], g[i] * (V[post[i]] - E_syn[i]));
}
```

`atomicAdd(double*)` requires sm_60 (Pascal) or newer — document minimum GPU
requirement.

### Synapse Conductance Update

Embarrassingly parallel — one thread per synapse.

```cuda
__global__ void synapse_decay_kernel(
    double* __restrict__ g,
    const double* __restrict__ decay_factors,
    size_t n_synapses)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;
    g[i] *= decay_factors[i];
}
```

---

## 17.6 Delay Buffers on Device

Spike history is stored as a **per-neuron circular buffer of binary spike
flags** (`uint8_t fired[N][max_delay_steps]`), indexed by
`step % max_delay_steps`.  For a 5 ms max delay at dt=0.01 ms this is a 500-slot
ring per neuron — 135K × 500 bytes ≈ 64 MB.

The `isyn_accumulate_kernel` reads from `fired[pre[s]][(step - delay_steps[s])
% capacity]` directly in device memory.  No CPU involvement per step.

Compare to the Phase 2 ring buffer design (task16): that uses per-synapse
buffers for cross-group isolation.  On single GPU the per-neuron design is more
memory-efficient since all synapses from the same pre-neuron share the same
spike history.

---

## 17.7 Stream Concurrency

The GPU simulation uses **two permanent CUDA streams**:

- **Stream 0 (compute)**: all neuron step, synapse, and spike-detection kernels.
- **Stream 1 (record)**: async `cudaMemcpyAsync` from staging buffer to pinned
  host memory, fired every `record_interval` steps.

These streams run concurrently — PCIe transfer overlaps with compute at no
throughput cost.

**Optional — multi-kernel concurrency for small networks:**
When `N < occupancy_threshold` (profiler-determined), each population group's
kernel is launched on its own compute stream.  The GPU's hardware scheduler
packs multiple groups in-flight simultaneously, improving SM utilization.
At N ≥ ~10K per group this gives no benefit (each group saturates the GPU
alone), so it auto-disables above threshold.  This is an internal optimization,
invisible to the user.

---

## 17.8 Phase 2 / Multi-GPU Bridge

Phase 2 delay-decomposition (task16) is **the correct partition primitive for
multi-GPU**.  Each GPU holds one or more population groups.  Inter-GPU spike
communication travels through delay buffers over NVLink (~600 GB/s) or PCIe.

This is deferred to task19, but task17 must not close the door on it:

- The `Device` struct (§17.2) is designed to extend to multi-device assignment.
- `CUDASimContext` is scoped per-device (not global) so multiple instances can
  coexist.
- The on-device delay ring buffer layout (§17.6) is the same layout used for
  inter-GPU ring buffers in task19 — only the backing memory location changes
  (device-local vs NVLink-mapped peer memory).

The user-facing multi-GPU API (task19):

```python
# Not implemented in task17 — shown for design reference
rn.assign("CTX_e", device=hh.device("cuda:0"))
rn.assign("STN",   device=hh.device("cuda:1"))
rn.simulate(...)  # inter-device communication via delay buffers over NVLink
```

---

## 17.9 Limitations and Non-Goals

- CUDA support is an opt-in build option; the default CPU build is completely
  unaffected.
- Windows: CUDA Toolkit requires MSVC + nvcc; document the exact conda setup.
- Minimum GPU: sm_60 (Pascal, GTX 1080 / Tesla P100) for `atomicAdd(double*)`.
- Mixed precision (float32 pools, float64 accumulation) is out of scope.
- STDP / STP weight updates on GPU: not in scope for task17 (cross-synapse
  dependency patterns complicate kernel design); document as limitation.
- Multi-GPU: task19.  The architecture here supports it; the implementation
  does not.
- The `auto` device selection mode (choose GPU when N > threshold) is
  intentionally **not implemented** — the user always opts in explicitly.

---

## 17.10 Implementation Checklist

### Build System
- [ ] Add `USE_CUDA` CMake option with `enable_language(CUDA)` guard
- [ ] Separate compilation: `.cu` files via nvcc, `.cpp` files via host compiler
- [ ] `HH_USE_CUDA` compile definition; all CUDA code guarded behind it
- [ ] CI: optional CUDA build job, skipped on machines without GPU

### C++ Device Abstraction
- [ ] `device.hpp` — `Device` struct, `cuda_device_count()`, `cuda_is_available()`
- [ ] `Network::to(Device)`, `Network::device()`
- [ ] `CUDASimContext` — owns all device pointers, scoped per simulation run

### CUDA Kernels
- [ ] `hh_pool_step_kernel` — Euler (validate), then RK4
- [ ] `iz_pool_step_kernel` — with predicated voltage reset (no branching)
- [ ] `isyn_accumulate_kernel` — `atomicAdd`, sm_60 guard
- [ ] `synapse_decay_kernel`
- [ ] `spike_detect_kernel` — writes per-neuron spike flags to on-device ring buffer
- [ ] On-device delay ring buffer: `fired[N][max_delay_steps]`, indexed by step mod capacity
- [ ] Recording staging: double-buffered device array + `cudaMemcpyAsync` to pinned host

### Stream Setup
- [ ] Allocate stream 0 (compute) and stream 1 (record) at simulation start
- [ ] Recording: async copy on stream 1, every `record_interval` steps
- [ ] Optional multi-kernel concurrency: per-group compute streams when N < occupancy threshold

### Simulation Loop Dispatch
- [ ] `RegionalNetwork::simulate()` detects `device().type == CUDA` and routes to GPU path
- [ ] GPU path: upload all state once → hot loop (all kernels on device) → final download
- [ ] No CPU↔GPU data movement inside the hot loop except stream 1 recording copies
- [ ] Free all setup temporaries (unsorted weight/index buffers) immediately after CSR build

### Python Bindings
- [ ] `hh.cuda` submodule: `is_available()`, `device_count()`, `get_device_name(n)`
- [ ] `hh.device(str)` factory — parses "cpu", "cuda", "cuda:N"
- [ ] `RegionalNetwork.to(device)`, `RegionalNetwork.device()`

### Validation
- [ ] CPU vs CUDA produce identical V traces for HH network (small, atol=1e-6)
- [ ] CPU vs CUDA identical for Izhikevich network (validates predicated reset)
- [ ] CPU vs CUDA identical for composable neuron with CUSTOM_EXPR gate
- [ ] Recording correctness: async-streamed traces match full in-memory traces
- [ ] VRAM accounting: verify setup temporaries are freed before hot loop

### Benchmarks
- [ ] Speedup vs CPU serial: N = 500, 2K, 10K, 50K, 135K neurons
- [ ] VRAM usage at each N (verify budget analysis in §17.4)
- [ ] Async recording overhead: record_interval sweep (1, 10, 100, 1000 steps)
