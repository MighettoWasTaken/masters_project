# Task 17.8: CudaComposablePool — Device SoA + Standard Gate Kernels

**Role:** CUDA engineer  
**Status:** Not started  
**Depends on:** 17.1 (PoolBase), 17.2 (CMake), 17.6 (establishes device SoA patterns)  
**Unlocks:** 17.9 (CUSTOM_EXPR kernels extend this class)

---

## What to implement

`ComposablePool` is the most general pool — supports arbitrary gate counts, intracellular substances, and modulation. This task covers the device memory layout and kernels for **pattern-matched** gates (Boltzmann, 6 tau forms, 4 rate forms). Custom (VM) expressions are handled in task 17.9.

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

    bool   contains_neuron(size_t global_idx) const;
    double get_substance_at(size_t global_idx, size_t subst_idx) const;

    // Accessors for task 17.10 recording pipeline
    const double* device_ptr_V()               const { return d_V_; }
    const double* device_ptr_gates(int g_idx)  const;
    const double* device_ptr_substance(int s_idx) const;

private:
    NeuronModelSpec spec_;
    int    device_id_;
    size_t n_ = 0;
    size_t capacity_;
    cudaStream_t stream_ = nullptr;

    double*  d_V_            = nullptr;
    double*  d_I_            = nullptr;
    double** d_gates_        = nullptr;      // [n_gates][n_neurons]
    double** d_substances_   = nullptr;      // [n_subst][n_neurons]
    size_t*  d_net_idx_      = nullptr;
    bool*    is_custom_gate_ = nullptr;      // [n_gates] — skip custom gates until 17.9

    struct GateParams { double v_half, k, tau_scale, tau_min; };
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

#### Standard gate kernel — Boltzmann inf + pattern-matched tau

```cuda
__global__ void gate_step_boltzmann(
    double* gate, const double* V,
    double v_half, double k, double tau_scale, double tau_min,
    int N, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double inf = 1.0 / (1.0 + exp(-(V[i] - v_half) / k));
    double tau = tau_scale + tau_min;
    gate[i] += dt * (inf - gate[i]) / tau;
}
```

Implement the full set of pattern-matched kernels:
1. Boltzmann inf + constant tau
2. Boltzmann inf + Boltzmann-product tau
3. Alpha/beta rate function form

For `is_custom_gate_[g] == true`, skip the kernel call for that gate — task 17.9 fills this gap.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/cuda_composable_pool.hpp` | New class |
| `src/cpp/src/cuda_composable_pool.cu` | Device SoA alloc + gate/V kernels |

---

## Baseline tests (before PR to testing branch)

Requires: 17.4 (Python API) merged.

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] Single composable population using `NeuronModelSpec.hh_default()` (Boltzmann gates), 200ms on CUDA: V finite, no crash
- [ ] `rn.to(cpu)` after composable CUDA simulate — no crash
- [ ] Population with a gate that has `CUSTOM_EXPR` on CUDA — skips gracefully (no incorrect output, no crash)

---

## Contract for downstream tasks

- Task 17.9 extends `CudaComposablePool` by implementing the `is_custom_gate_` path using device functions from `CUDAPrinter` (task 17.5).
- Task 17.10's recording pipeline accesses `device_ptr_gates()` and `device_ptr_substance()` via `cudaMemcpyAsync`.
- `migrate_to_device(new_id)`: free device arrays, `cudaSetDevice(new_id)`, re-alloc, copy from host mirror. Keep a host mirror updated after each `synchronize()`.
