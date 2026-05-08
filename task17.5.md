# Task 17.5: CudaHHPool + CudaIzPool

**Role:** CUDA engineer  
**Status:** Not started  
**Depends on:** 17.1 (PoolBase virtual methods), 17.2 (CMake CUDA wired up)  
**Unlocks:** 17.9 (recording interacts with CUDA pool memory)

---

## What to implement

Two new pool classes in the CUDA sources created by 17.2.

### `src/cpp/include/hodgkin_huxley/cuda_hh_pool.hpp` (new)

```cpp
#pragma once
#ifdef HH_USE_CUDA
#include "hodgkin_huxley/pool/pool_base.hpp"
#include "hodgkin_huxley/neuron.hpp"

namespace hodgkin_huxley {

class CudaHHPool : public PoolBase {
public:
    explicit CudaHHPool(size_t capacity, int device_id = 0);
    ~CudaHHPool() override;

    // PoolBase overrides
    void scatter_voltages(double* V_buf) const override;   // cudaMemcpyAsync → pinned
    void gather_currents(const double* I_buf) override;    // cudaMemcpyAsync ← pinned
    void step(double dt) override;                         // launch HH kernel
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>&) const override;

    bool is_cuda()               const override { return true; }
    int  device_id()             const override { return device_id_; }
    void synchronize()                 override;  // cudaStreamSynchronize(stream_)
    bool requires_pinned_memory() const override { return true; }
    void migrate_to_device(int new_id)  override;

    size_t size() const override { return n_; }

    // Build-time: populate from CPU neuron objects (called by PoolManager)
    void add(size_t net_idx, const HHNeuron::Parameters& p,
             const HHNeuron::State& s);

private:
    int    device_id_;
    size_t n_ = 0;
    size_t capacity_;
    cudaStream_t stream_ = nullptr;

    // Device SoA arrays — one double* per state variable
    double* d_V_   = nullptr;
    double* d_m_   = nullptr;
    double* d_h_   = nullptr;
    double* d_n_   = nullptr;
    double* d_I_   = nullptr;   // gathered currents
    double* d_net_idx_ = nullptr;  // size_t array for scatter/gather indexing

    // Per-neuron fixed parameters (constant memory or device arrays)
    double* d_gNa_ = nullptr;
    double* d_gK_  = nullptr;
    double* d_gL_  = nullptr;
    double* d_ENa_ = nullptr;
    double* d_EK_  = nullptr;
    double* d_EL_  = nullptr;
    double* d_Cm_  = nullptr;

    void alloc(size_t cap);
    void free_device();
};

} // namespace hodgkin_huxley
#endif // HH_USE_CUDA
```

### `src/cpp/src/cuda_hh_pool.cu`

Implement `CudaHHPool`. Key kernel:

```cuda
__global__ void hh_step_kernel(
    double* V, double* m, double* h, double* n,
    const double* I_ext,
    const double* gNa, const double* gK, const double* gL,
    const double* ENa, const double* EK,  const double* EL,
    const double* Cm,
    double dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double v = V[i];
    double mi = m[i], hi = h[i], ni = n[i];

    // Standard HH alpha/beta rate functions (same formulas as hh_pool.cpp)
    double alpha_m = (v == -40.0) ? 1.0 : 0.1*(v+40.0)/(1.0-exp(-(v+40.0)/10.0));
    double beta_m  = 4.0*exp(-(v+65.0)/18.0);
    double alpha_h = 0.07*exp(-(v+65.0)/20.0);
    double beta_h  = 1.0/(1.0+exp(-(v+35.0)/10.0));
    double alpha_n = (v == -55.0) ? 0.1 : 0.01*(v+55.0)/(1.0-exp(-(v+55.0)/10.0));
    double beta_n  = 0.125*exp(-(v+65.0)/80.0);

    double I_Na = gNa[i]*mi*mi*mi*hi*(v-ENa[i]);
    double I_K  = gK[i]*ni*ni*ni*ni*(v-EK[i]);
    double I_L  = gL[i]*(v-EL[i]);

    V[i] = v + dt*(I_ext[i] - I_Na - I_K - I_L) / Cm[i];
    m[i] = mi + dt*(alpha_m*(1.0-mi) - beta_m*mi);
    h[i] = hi + dt*(alpha_h*(1.0-hi) - beta_h*hi);
    n[i] = ni + dt*(alpha_n*(1.0-ni) - beta_n*ni);
}
```

`scatter_voltages` uses `cudaMemcpyAsync(V_buf + net_idx[0], d_V_, n_*sizeof(double), cudaMemcpyDeviceToHost, stream_)` — but net_idx may not be contiguous, so use a gather kernel or `cudaMemcpy2D` pattern. Simpler: a small kernel that writes `V_buf[d_net_idx_[i]] = d_V_[i]`.

### `CudaIzPool` — same pattern

Follow identical structure for `cuda_iz_pool.hpp` / `cuda_iz_pool.cu`. The Izhikevich step is simpler (2 state variables: V, u; reset condition handled with `__ballot_sync` or conditional write).

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/cuda_hh_pool.hpp` | New class declaration |
| `src/cpp/src/cuda_hh_pool.cu` | Kernel + PoolBase overrides |
| `src/cpp/include/hodgkin_huxley/cuda_iz_pool.hpp` | New class declaration |
| `src/cpp/src/cuda_iz_pool.cu` | Kernel + PoolBase overrides |

---

## Contract for downstream tasks

- `requires_pinned_memory()` returns `true` — task 17.3 uses this to allocate `V_cache_pinned_`.
- `synchronize()` calls `cudaStreamSynchronize(stream_)` — task 17.3's hot loop calls this after `step_all()`.
- `scatter_voltages(double* V_buf)` writes into the pinned host buffer via `cudaMemcpyAsync`; caller must call `synchronize()` before reading.
- Use forward Euler to match the CPU `HHPool` implementation exactly — correctness tests in 17.11 compare against CPU traces.
