# Task 17.9: CudaComposablePool — CUSTOM_EXPR Gates + Intracellular + Modulation

**Role:** CUDA engineer + Codegen engineer  
**Status:** Not started  
**Depends on:** 17.5 (CUDAPrinter — `__device__` codegen), 17.8 (CudaComposablePool structure + `is_custom_gate_` skip path)  
**Unlocks:** 17.10 (recording pipeline can now rely on fully functional composable pool)

---

## What to implement

Task 17.8 built the device memory layout and pattern-matched gate kernels for `CudaComposablePool`, but skips any gate where `is_custom_gate_[g] == true`. This task fills that gap by generating `__device__` functions at build time and dispatching to them, and adds support for intracellular substance dynamics and synapse-conductance modulation.

### `src/cpp/src/cuda_composable_pool.cu` — custom gate dispatch

After the pattern-matched gate loop in `launch_gate_kernels()`, add a second pass for custom gates:

```cuda
// Custom gate kernel — calls generated __device__ functions by gate index
__global__ void gate_step_custom(
    double* gate, const double* V,
    int gate_idx, int N, double dt,
    const double* inf_lut,   // unused — here for ABI consistency
    const double* tau_lut)
{
    // This kernel is never called directly — instead, one specialized kernel
    // per custom gate is JIT-compiled (see below) or pre-generated per model.
    // The generic fallback below is used when no pre-generated kernel exists.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // inf and tau computed by __device__ fn pointers set per-gate at alloc time
    // (compile-time specialization path described in the codegen section below)
}
```

### Build-time `__device__` function generation

When `CudaComposablePool` is constructed with a `NeuronModelSpec` that contains `CUSTOM_EXPR` gates, the pool constructor calls `compile_gate_cuda()` (task 17.5) to generate device function strings, then writes and compiles a per-model `.cu` file:

```cpp
// In CudaComposablePool constructor, per CUSTOM_EXPR gate g:
#ifdef HH_USE_CUDA
if (spec_.gates[g].is_custom()) {
    std::string device_fns = compile_gate_cuda(spec_.gates[g], gate_fn_prefix(g));
    // Write to ~/.cache/hodgkin_huxley/cuda/model_<hash>.cu
    // Compile via nvrtc (CUDA runtime compilation) → device function pointer
    // Store in custom_gate_fns_[g] for launch_gate_kernels()
}
#endif
```

Use NVRTC (`nvrtc.h`) for runtime compilation of custom gate kernels. NVRTC is included with the CUDA Toolkit — no additional dependency.

```cpp
struct CustomGateFn {
    CUmodule  module   = nullptr;
    CUfunction inf_fn  = nullptr;  // __device__ double {prefix}_inf(double V)
    CUfunction tau_fn  = nullptr;  // __device__ double {prefix}_tau(double V)
};
CustomGateFn custom_gate_fns_[MAX_CUSTOM_GATES];
```

Add to private members of `CudaComposablePool`:
```cpp
CustomGateFn* custom_gate_fns_ = nullptr;   // [n_custom_gates]
```

### Intracellular substance dynamics

Each intracellular substance `s` follows an ODE: `ds/dt = f(s, V, gate_states)`. If the spec ODE is pattern-matched (linear decay + gate-driven production), use the analytical kernel. If it is a `CUSTOM_EXPR`, generate via CUDAPrinter.

```cuda
// Pattern-matched: ds/dt = -s/tau_s + alpha_s * gate^p * (1 - s)
__global__ void substance_step_linear(
    double* substance, const double* gate_driver, const double* V,
    double tau_s, double alpha_s, int power, int N, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double g = gate_driver[i];
    double gpow = 1.0;
    for (int p = 0; p < power; ++p) gpow *= g;
    substance[i] += dt * (-substance[i] / tau_s + alpha_s * gpow * (1.0 - substance[i]));
}
```

### Synapse-conductance modulation

`SYNAPSE_G` modulation scales the effective synapse conductance based on intracellular substance level. Device-side modulation is applied in `accumulate_isyn` (task 17.7) via a per-neuron modulation factor array:

```cuda
// Add d_mod_factor_[N] to CudaComposablePool; updated each step after substance step
__global__ void compute_mod_factor(
    double* mod_factor, const double* substance,
    double mod_scale, double mod_offset, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    mod_factor[i] = mod_offset + mod_scale * substance[i];
}
```

The modulation factor array is exposed via a new accessor:
```cpp
const double* device_ptr_mod_factor() const { return d_mod_factor_; }
```

Task 17.7's `accumulate_isyn` kernel receives a `mod_factor` pointer; when nullptr (no modulation), it skips the multiplication.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/src/cuda_composable_pool.cu` | Custom gate NVRTC compilation + dispatch, substance kernels, modulation factor kernel |
| `src/cpp/include/hodgkin_huxley/cuda_composable_pool.hpp` | Add `custom_gate_fns_`, `d_mod_factor_`, `device_ptr_mod_factor()` |
| `src/cpp/src/cuda_synapse.cu` | Accept `mod_factor` pointer in `accumulate_isyn` |

---

## Baseline tests (before PR to testing branch)

Requires: 17.4 (Python API) and 17.8 (CudaComposablePool standard gates) merged.

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] Single composable population with one `CUSTOM_EXPR` gate (e.g. `1 / (1 + exp(-(V + 40) / 10))` with non-Boltzmann tau), 200ms on CUDA: V finite, gate values finite
- [ ] NVRTC compilation of custom gate: no error, no crash
- [ ] Composable pool with calcium intracellular substance + `SYNAPSE_G` modulation on CUDA: mod_factor non-trivial, I_syn affected, no crash
- [ ] `rn.to(cpu)` after composable CUDA simulate with custom gates — no crash, CPU re-simulate produces finite V

---

## Contract for downstream tasks

- Task 17.10's recording pipeline accesses `device_ptr_mod_factor()` via `cudaMemcpyAsync` if recording modulation state.
- The NVRTC-compiled kernels are cached to `~/.cache/hodgkin_huxley/cuda/model_<hash>.cubin` — re-used if spec hash matches.
- `accumulate_isyn` in task 17.7 must accept a nullable `mod_factor` pointer — `nullptr` skips modulation, pointer present applies it per-post-neuron.
