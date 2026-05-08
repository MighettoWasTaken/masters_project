# Task 17.10: VRAM Recording Pipeline — Async Streaming to Pinned Host

**Role:** VRAM / memory engineer  
**Status:** Not started  
**Depends on:** 17.3 (pinned buffers), 17.6 (CudaHHPool/CudaIzPool device arrays), 17.7 (DeviceSynapseArrays), 17.8 (CudaComposablePool `device_ptr_*` accessors)  
**Unlocks:** 17.11 (tests verify recording output)

---

## What to implement

On CPU, recording is done by calling `scatter_gates`, `scatter_calcium`, etc. every `interval` steps — they write directly into numpy-backed buffers. On CUDA, pool state lives on-device; copying every step is too slow. Use double-buffering and a dedicated copy stream.

### Design

Two streams per simulation:
- `compute_stream_`: runs step kernels (already in each CUDA pool)
- `copy_stream_`: runs `cudaMemcpyAsync` for recording

Double-buffer on host (pinned):
- `rec_buf_A_[metric][t]`, `rec_buf_B_[metric][t]` — ping-pong between intervals

### `src/cpp/include/hodgkin_huxley/cuda_recording.hpp` (new)

```cpp
#pragma once
#ifdef HH_USE_CUDA
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

namespace hodgkin_huxley {

struct CudaRecordingBuffers {
    // Pinned host double-buffers for async recording
    double* V_ping    = nullptr;   // [N * rec_steps/2 + 1]
    double* V_pong    = nullptr;
    double* gate_ping = nullptr;   // [N * n_gates * rec_steps/2 + 1]
    double* gate_pong = nullptr;
    double* ca_ping   = nullptr;
    double* ca_pong   = nullptr;

    size_t N, n_gates, n_substances, rec_steps;
    cudaStream_t copy_stream  = nullptr;
    cudaEvent_t  copy_done_A, copy_done_B;

    void init(size_t N, size_t n_gates, size_t n_subst, size_t rec_steps);
    void destroy();

    // Called every interval steps from compute_stream_ after scatter_voltages:
    // schedules copy of device V → current ping buffer on copy_stream_
    void record_step(int t_rec,
                     const double* d_V, size_t N,
                     const double** d_gates, size_t n_gates,
                     const double* d_ca);

    // Blocks until all async copies done; assembles final output buffers
    void finalize(double* out_V, double* out_gates, double* out_ca);
};

} // namespace hodgkin_huxley
#endif
```

### Integration in `Network::simulate_with_descriptors`

After the existing `#ifdef HH_USE_CUDA` check, when building recording buffers:

```cpp
#ifdef HH_USE_CUDA
CudaRecordingBuffers cuda_rec;
if (pool_mgr_.on_cuda() && V_buf != nullptr) {
    cuda_rec.init(N, max_gates, n_subst, n_rec);
}
#endif
```

In the hot loop, replace the CPU `scatter_gates` calls with:

```cpp
#ifdef HH_USE_CUDA
if (pool_mgr_.on_cuda()) {
    cuda_rec.record_step(t_rec, d_V_device, N, d_gates_device, n_gates, d_ca_device);
} else
#endif
{
    // existing CPU recording path
}
```

After the loop:

```cpp
#ifdef HH_USE_CUDA
if (pool_mgr_.on_cuda()) {
    cuda_rec.finalize(V_buf, gate_buf, calcium_buf);
    cuda_rec.destroy();
}
#endif
```

### VRAM budget note

For large networks (135K neurons, 50M synapses — fruit fly scale):
- V state: 135K × 8B = ~1 MB/snapshot, trivial
- Spike delay ring: see task 17.7 — using per-neuron ring (135K × 500 bits ≈ 8 MB) avoids the per-synapse explosion

Mitigation already established in 17.7: store only the **spike events** (1 bit per pre-neuron per step) rather than per-synapse bits. This task adds a comment in `cuda_recording.hpp` referencing the design.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/cuda_recording.hpp` | New — double-buffer recording struct |
| `src/cpp/src/cuda_recording.cu` | `init`, `record_step`, `finalize` implementations |
| `src/cpp/src/network.cpp` | Integrate `CudaRecordingBuffers` into `simulate_with_descriptors` |
| `src/cpp/CMakeLists.txt` | Add `src/cuda_recording.cu` to CUDA sources list |

---

## Baseline tests (before PR to testing branch)

Requires: 17.4 (Python API) and 17.8 (CudaComposablePool with device_ptr_* accessors) merged.

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] Single HH population, 200ms on CUDA with recording enabled: `result["V"].shape == (n_neurons, n_rec_steps)`, all values finite
- [ ] Recorded V traces (CUDA) match CPU reference traces within `atol=1e-6`
- [ ] `record_every=10` produces the correct number of recorded steps
- [ ] `cuda_rec.finalize()` does not hang; `cudaStreamSynchronize` completes

---

## Contract for downstream tasks

- Task 17.11's correctness tests compare `cuda_rec.finalize()` output against CPU reference V traces — the values must match to within floating-point tolerance.
- Task 17.12's benchmarks measure the overhead of recording vs. no recording on GPU — use `CudaRecordingBuffers::copy_done_*` events for precise timing.
