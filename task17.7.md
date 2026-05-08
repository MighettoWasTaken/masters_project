# Task 17.7: CudaComposablePool — Device SoA + Standard Gate Kernels

**Role:** CUDA engineer  
**Status:** Not started  
**Depends on:** 17.1 (PoolBase), 17.2 (CMake), 17.5 (establishes device SoA patterns)  
**Unlocks:** 17.8 (CUSTOM_EXPR kernels extend this class)

---

## What to implement

`ComposablePool` is the most general pool — supports arbitrary gate counts, intracellular substances, and modulation. This task covers the device memory layout and the kernels for gates whose expressions are **pattern-matched** (Boltzmann, 6 tau forms, 4 rate forms) — i.e., the standard forms that already map to pre-compiled C++ paths.

Custom (VM) expressions are handled in task 17.8.

### `src/cpp/include/hodgkin_huxley/cuda_composable_pool.hpp` (new)

```cpp
#pragma once
#ifdef HH_USE_CUDA
#include "hodgkin_huxley/pool/pool_base.hpp"
#include "hodgkin_huxley/model/neuron_model_spec.hpp"
#include <cuda_runtime.h>

namespace hodgkin_huxley {

class CudaComposablePool : public PoolBase {
public:
    CudaComposablePool(const NeuronModelSpec& spec, size_t capacity, int device_id = 0);
    ~CudaComposablePool() override;

    void scatter_voltages(double* V_buf) const override;
    void gather_currents(const double* I_buf) override;
    void step(double dt) override;
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>&) const override;

    bool is_cuda()                const override { return true; }
    int  device_id()              const override { return device_id_; }
    void synchronize()                  override;
    bool requires_pinned_memory() const override { return true; }
    void migrate_to_device(int new_id)  override;

    size_t size() const override { return n_; }
    int    n_gates() const override;
    int    n_substances() const override;
    bool   has_synapse_g_mods() const;

    void add(size_t net_idx, double V0,
             const std::vector<double>& gate_states,
             const std::vector<double>& substance_states);

    bool contains_neuron(size_t global_idx) const;
    double get_substance_at(size_t global_idx, size_t subst_idx) const;  // host copy

private:
    NeuronModelSpec spec_;
    int    device_id_;
    size_t n_ = 0;
    size_t capacity_;
    cudaStream_t stream_ = nullptr;

    // Device SoA — layout: [n_neurons] per variable
    double* d_V_   = nullptr;
    double* d_I_   = nullptr;
    double** d_gates_ = nullptr;      // [n_gates][n_neurons]
    double** d_substances_ = nullptr; // [n_subst][n_neurons]
    size_t*  d_net_idx_ = nullptr;

    // Gate parameter arrays (constant across neurons — broadcast in kernel)
    // These hold per-gate scalars from the spec.
    struct GateParams { double v_half, k, tau_scale, tau_min; /* etc. */ };
    GateParams* d_gate_params_ = nullptr;

    void alloc(size_t cap);
    void free_device();
    void launch_gate_kernels(double dt);
    void launch_V_kernel(double dt);
};

} // namespace hodgkin_huxley
#endif
```

### `src/cpp/src/cuda_composable_pool.cu`

#### Membrane voltage kernel

```cuda
__global__ void composable_V_step(
    double* V, const double* I_ext, const double* I_ion,
    const double* Cm, int N, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    V[i] += dt * (I_ext[i] - I_ion[i]) / Cm[i];
}
```

`I_ion` is accumulated by the gate kernels into a temporary device array before this kernel runs.

#### Standard gate kernel — Boltzmann inf, pattern-matched tau

Each gate type (Boltzmann, alpha/beta, etc.) gets its own kernel or a dispatch via a per-gate `gate_type` enum stored alongside `GateParams`:

```cuda
__global__ void gate_step_boltzmann(
    double* gate,      // [N] gate variable
    const double* V,
    double v_half, double k, double tau_scale, double tau_min,
    int N, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double inf = 1.0 / (1.0 + exp(-(V[i] - v_half) / k));
    double tau = tau_scale + tau_min;  // simplified — full form depends on tau_type
    gate[i] += dt * (inf - gate[i]) / tau;
}
```

The full set of pattern-matched kernels covers:
1. Boltzmann inf + constant tau
2. Boltzmann inf + Boltzmann-product tau (tau_A / (1 + exp(...)))
3. Alpha/beta rate function form (calls device helper functions)

For gates that are NOT pattern-matched (CUSTOM_EXPR), skip in this task — add a `bool is_custom_[n_gates]` flag and skip the kernel call. Task 17.8 fills that gap.

#### `step()` implementation

```cpp
void CudaComposablePool::step(double dt) {
    // 1. Launch gate kernels (pattern-matched gates only; custom gates skipped)
    launch_gate_kernels(dt);
    // 2. Launch V update kernel
    launch_V_kernel(dt);
    // Async — caller calls synchronize() before reading
}
```

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/cuda_composable_pool.hpp` | New class |
| `src/cpp/src/cuda_composable_pool.cu` | Device SoA alloc + gate/V kernels |

---

## Contract for downstream tasks

- Task 17.8 extends `CudaComposablePool` by implementing the `is_custom_` gate path — it calls device functions generated by `CUDAPrinter` (task 17.4).
- Task 17.9's recording interacts with `d_gates_` and `d_substances_` via `cudaMemcpyAsync` — ensure these device arrays are accessible from outside the class through a `device_ptr_gates(int gate_idx)` getter or similar accessor.
- `migrate_to_device(new_id)`: free all device arrays, `cudaSetDevice(new_id)`, re-alloc, copy from a host-side snapshot. Keep a host mirror (`h_V_`, `h_gates_`, etc.) updated after each `synchronize()` call, or just copy back on demand.
