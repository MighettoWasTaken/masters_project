# Task 17.6: On-Device Synapse Kernels + Spike Delay Ring Buffer

**Role:** CUDA engineer  
**Status:** Not started  
**Depends on:** 17.2 (CMake), 17.5 (device memory layout established)  
**Unlocks:** 17.9 (recording g_syn needs device-side synapse state)

---

## What to implement

The existing CPU synapse update loop lives in `network.cpp` inside `update_synapses_grouped()`. This task moves that work onto the GPU when CUDA pools are active.

### `src/cpp/src/cuda_synapse.cu` + `src/cpp/include/hodgkin_huxley/cuda_synapse.hpp`

#### Device-side synapse SoA

Allocate on-device copies of the synapse arrays that `Network` currently holds on the host:

```cpp
struct DeviceSynapseArrays {
    // Mirrors the host SoA in network.cpp — allocated on device
    double* d_weight;      // [S]
    double* d_g;           // [S] conductance state
    double* d_A;           // [S] rise variable (alpha/double-exp)
    double* d_delay_buf;   // [S * max_delay_steps] ring buffer (spike history)
    int*    d_pre;         // [S] pre-neuron global index
    int*    d_post;        // [S] post-neuron global index
    int*    d_delay_steps; // [S]
    int*    d_syn_type;    // [S] encodes AMPA/GABA_A/etc enum value
    double* d_E_rev;       // [S]
    double* d_tau_rise;    // [S]
    double* d_tau_decay;   // [S]
    size_t  S;             // number of synapses
    size_t  max_delay;     // ring buffer depth
};
```

Provide:
- `DeviceSynapseArrays alloc_device_synapses(const Network&, int device_id)` — copies host SoA to device
- `void free_device_synapses(DeviceSynapseArrays&)`

#### Spike detection + delay ring kernel

```cuda
__global__ void update_spike_delay_ring(
    const double* V_cache,  // [N] current voltages (pinned host — use UVA or explicit copy)
    uint8_t* spike_ring,    // [S * max_delay] — the delay ring buffer
    const int* pre,         // [S]
    const int* delay_steps, // [S]
    int S, int current_step, int max_delay,
    double threshold)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S) return;
    int pre_idx = pre[k];
    bool spiked = (V_cache[pre_idx] >= threshold);
    int slot = (current_step) % max_delay;
    spike_ring[k * max_delay + slot] = spiked ? 1 : 0;
}
```

#### Synapse conductance update kernel

```cuda
__global__ void update_synapses(
    double* g, double* A,
    const uint8_t* spike_ring,
    const int* delay_steps,
    const double* tau_rise, const double* tau_decay,
    const int* syn_type,
    int S, int current_step, int max_delay, double dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S) return;

    int read_slot = (current_step - delay_steps[k] + max_delay) % max_delay;
    bool spike_arrived = spike_ring[k * max_delay + read_slot];

    // Exponential decay
    g[k] *= exp(-dt / tau_decay[k]);
    if (syn_type[k] > 0) A[k] *= exp(-dt / tau_rise[k]);  // double-exp / alpha

    if (spike_arrived) {
        g[k] += 1.0;   // weight scaling done at accumulate step
        if (syn_type[k] > 0) A[k] += 1.0;
    }
}
```

#### I_syn accumulation kernel

```cuda
__global__ void accumulate_isyn(
    double* I_syn,        // [N] output — atomicAdd into post-neuron slot
    const double* g,
    const double* weight,
    const double* E_rev,
    const double* V_cache,
    const int* post,
    int S)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S) return;
    int j = post[k];
    double i_k = weight[k] * g[k] * (V_cache[j] - E_rev[k]);
    atomicAdd(&I_syn[j], -i_k);  // inhibitory sign convention matches CPU
}
```

`atomicAdd(double*)` requires sm_60+ — enforced by `CMAKE_CUDA_ARCHITECTURES` in 17.2.

### Integration in `Network`

Add `#ifdef HH_USE_CUDA` blocks in `network.cpp` around `update_synapses_grouped()`:

```cpp
#ifdef HH_USE_CUDA
if (pool_mgr_.on_cuda()) {
    launch_cuda_synapse_update(dev_syn_, V_cache_pinned_, I_syn_pinned_, step, dt);
    return;
}
#endif
// existing CPU path
```

`DeviceSynapseArrays dev_syn_` added as a `Network` private member, allocated in `simulate_with_descriptors` build phase when `on_cuda()` is true.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/cuda_synapse.hpp` | New — `DeviceSynapseArrays`, launch declarations |
| `src/cpp/src/cuda_synapse.cu` | Kernels + alloc/free helpers |
| `src/cpp/include/hodgkin_huxley/network.hpp` | `DeviceSynapseArrays dev_syn_` private member |
| `src/cpp/src/network.cpp` | Conditional CUDA dispatch in synapse update hot loop |

---

## Contract for downstream tasks

- `d_delay_buf` ring is the device-side equivalent of the CPU `spike_detected_` + delay buffer. Task 17.9 does not record spike events directly from this; it uses the post-scatter V values.
- `I_syn_pinned_` is cleared to zero on device each step (add a `cudaMemsetAsync` before accumulate kernel). Task 17.3 allocates this pinned buffer.
- PLASTICITY (STDP/STP): defer — keep the `has_stdp_` / `has_stp_` guard. If either is true and `on_cuda()` is true, fall back to CPU synapse update with a CUDA stream sync first. Log a warning. Full CUDA plasticity is out of scope for task17.
