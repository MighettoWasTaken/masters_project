# Task 17.7: On-Device Synapse Kernels + Spike Delay Ring Buffer

**Role:** CUDA engineer  
**Status:** Not started  
**Depends on:** 17.2 (CMake), 17.6 (device memory layout established)  
**Unlocks:** 17.10 (recording needs device-side synapse state)

---

## What to implement

The existing CPU synapse update loop lives in `network.cpp` inside `update_synapses_grouped()`. This task moves that work onto the GPU when CUDA pools are active.

### `src/cpp/src/cuda_synapse.cu` + `src/cpp/include/hodgkin_huxley/cuda_synapse.hpp`

#### Device-side synapse SoA

```cpp
struct DeviceSynapseArrays {
    double* d_weight;
    double* d_g;           // [S] conductance state
    double* d_A;           // [S] rise variable (alpha/double-exp)
    double* d_delay_buf;   // [S * max_delay_steps] ring buffer
    int*    d_pre;
    int*    d_post;
    int*    d_delay_steps;
    int*    d_syn_type;
    double* d_E_rev;
    double* d_tau_rise;
    double* d_tau_decay;
    size_t  S;
    size_t  max_delay;
};
```

Provide:
- `DeviceSynapseArrays alloc_device_synapses(const Network&, int device_id)`
- `void free_device_synapses(DeviceSynapseArrays&)`

#### Spike detection + delay ring kernel

```cuda
__global__ void update_spike_delay_ring(
    const double* V_cache, uint8_t* spike_ring,
    const int* pre, const int* delay_steps,
    int S, int current_step, int max_delay, double threshold)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S) return;
    bool spiked = (V_cache[pre[k]] >= threshold);
    spike_ring[k * max_delay + (current_step % max_delay)] = spiked ? 1 : 0;
}
```

#### Synapse conductance update kernel

```cuda
__global__ void update_synapses(
    double* g, double* A, const uint8_t* spike_ring,
    const int* delay_steps, const double* tau_rise, const double* tau_decay,
    const int* syn_type, int S, int current_step, int max_delay, double dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S) return;
    int read_slot = (current_step - delay_steps[k] + max_delay) % max_delay;
    bool spike_arrived = spike_ring[k * max_delay + read_slot];
    g[k] *= exp(-dt / tau_decay[k]);
    if (syn_type[k] > 0) A[k] *= exp(-dt / tau_rise[k]);
    if (spike_arrived) {
        g[k] += 1.0;
        if (syn_type[k] > 0) A[k] += 1.0;
    }
}
```

#### I_syn accumulation kernel

```cuda
__global__ void accumulate_isyn(
    double* I_syn, const double* g, const double* weight,
    const double* E_rev, const double* V_cache, const int* post, int S)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S) return;
    atomicAdd(&I_syn[post[k]], -weight[k] * g[k] * (V_cache[post[k]] - E_rev[k]));
}
```

`atomicAdd(double*)` requires sm_60+ — enforced by `CMAKE_CUDA_ARCHITECTURES` in 17.2.

### Integration in `Network`

```cpp
#ifdef HH_USE_CUDA
if (pool_mgr_.on_cuda()) {
    launch_cuda_synapse_update(dev_syn_, V_cache_pinned_, I_syn_pinned_, step, dt);
    return;
}
#endif
// existing CPU path
```

`DeviceSynapseArrays dev_syn_` added as a `Network` private member.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/cuda_synapse.hpp` | New — `DeviceSynapseArrays`, launch declarations |
| `src/cpp/src/cuda_synapse.cu` | Kernels + alloc/free helpers |
| `src/cpp/include/hodgkin_huxley/network.hpp` | `DeviceSynapseArrays dev_syn_` private member |
| `src/cpp/src/network.cpp` | Conditional CUDA dispatch in synapse update hot loop |

---

## Baseline tests (before PR to testing branch)

Requires: 17.4 (Python API) and 17.6 (CudaHHPool) merged.

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] Two HH populations connected with AMPA synapses (delay=5ms), 200ms: CUDA simulate produces finite V, no crash
- [ ] Two Iz populations connected with GABA_A synapses, 200ms: finite V, no crash
- [ ] Network with delay=0 synapses on CUDA — no crash
- [ ] STDP/STP network on CUDA falls back to CPU synapse update without crashing (no silent wrong answer)

---

## Contract for downstream tasks

- `I_syn_pinned_` is cleared to zero each step via `cudaMemsetAsync` before accumulate kernel. Task 17.3 allocates this pinned buffer.
- PLASTICITY (STDP/STP): if `has_stdp_` or `has_stp_` is true and `on_cuda()` is true, fall back to CPU synapse update with a prior CUDA stream sync. Log a warning. Full CUDA plasticity is out of scope for task17.
