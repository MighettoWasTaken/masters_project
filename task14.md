# Task 14: CUDA GPU Acceleration

## Priority: 3

## Overview

Add an optional CUDA backend for GPU-accelerated simulation of large networks (>1000 neurons). The CPU path (Eigen pools + SoA synapses) remains the default; CUDA activates only when explicitly requested or when the network exceeds a configurable threshold. The primary bottleneck at scale is the neuron pool `step()` function and synaptic current accumulation; these are the first targets for GPU offload.

Expected speedup: 50–200× vs C++ CPU for networks >5000 neurons where memory bandwidth is not the bottleneck.

---

## 14.1 Architecture

### Build Configuration

CUDA support is opt-in at build time:

```cmake
option(USE_CUDA "Enable CUDA GPU backend" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_CUDA)
endif()
```

The Python package builds normally without CUDA; a separate `hodgkin_huxley_cuda` wheel can be distributed for GPU-capable environments.

### Backend Selection

```cpp
enum class Backend { CPU, CUDA };

// In Network:
void set_backend(Backend b);  // explicit override
Backend backend() const;

// Auto-select threshold
static constexpr size_t CUDA_AUTO_THRESHOLD = 1000; // neurons
```

### Pool Abstraction

Introduce a `PoolBase` interface (currently implicit):

```cpp
class PoolBase {
public:
    virtual ~PoolBase() = default;
    virtual void step(double dt,
                      const double* I_ext,
                      const double* I_syn,
                      double* V_out,
                      size_t n) = 0;
    virtual void get_V(double* out) const = 0;
    virtual void reset() = 0;
};
```

CPU pools (`HHPool`, `IzPool`, `ComposablePool`) implement this interface. CUDA pools (`CUDAHHPool`, `CUDAComposablePool`) provide equivalent GPU implementations.

---

## 14.2 CUDA Kernels

### Neuron Pool Kernel (one thread per neuron)

```cuda
__global__ void hh_pool_step_kernel(
    double* V, double* m, double* h, double* n,
    const double* I_ext, const double* I_syn,
    const HHParams params, double dt, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // RK4 step using registers — no shared memory needed
    // (embarrassingly parallel; no dependencies between neurons)
    // ...
}
```

Grid configuration: `<<<(N + 255) / 256, 256>>>` — standard 1D grid.

### Synaptic Current Accumulation

The main challenge: multiple pre-synaptic neurons contribute to the same post-synaptic neuron's current. This requires parallel reduction with atomic operations.

```cuda
__global__ void isyn_accumulate_kernel(
    const size_t* post, const double* g, const double* E_syn,
    const double* V, double* I_syn, size_t n_synapses)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;
    double contrib = g[i] * (V[post[i]] - E_syn[i]);
    atomicAdd(&I_syn[post[i]], contrib);  // race condition: use atomicAdd
}
```

Note: `atomicAdd` for `double` requires `sm_60` (Pascal) or newer. For older GPUs, fall back to segmented reduction.

### Synapse Update Kernel

Synapse conductance decay is embarrassingly parallel (no cross-neuron dependencies):

```cuda
__global__ void synapse_decay_kernel(
    double* g, const double* decay_factors, size_t n_synapses)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_synapses) return;
    g[i] *= decay_factors[i];  // pre-computed decay = exp(-dt/tau)
}
```

---

## 14.3 Memory Management

### Device Memory Layout

All simulation state is transferred to device at simulation start; results transferred back at recording intervals or simulation end.

```cpp
class CUDASimContext {
public:
    CUDASimContext(size_t n_neurons, size_t n_synapses);
    ~CUDASimContext();

    void upload_neuron_state(const NeuronState& state);
    void upload_synapse_state(const SynArrays& sa);
    void download_V(double* host_V);
    void download_I_syn(double* host_I_syn);

private:
    double *d_V_, *d_m_, *d_h_, *d_n_;   // neuron state
    double *d_g_;                          // synapse conductances
    size_t *d_pre_, *d_post_;             // synapse connectivity
    double *d_E_syn_, *d_decay_;          // synapse params
    double *d_I_syn_;                     // accumulation buffer (zeroed each step)
    double *d_I_ext_;                     // external current
};
```

### Transfer Strategy

- **Upload once** at `simulate()` start
- **Stay on device** throughout the hot loop
- **Download** only when recording buffers require it (every `interval` steps)
- Use `cudaMemcpyAsync` + streams for overlap with CPU post-processing of previous interval

---

## 14.4 Delay Buffers on GPU

Synaptic delays complicate the GPU path. Options:

1. **Pre-transfer**: maintain spike history on CPU, transfer pre-synaptic spike vector to device each step (simple, low bandwidth cost for typical networks)
2. **On-device ring buffer**: circular buffer in device memory per synapse (requires careful atomic index management)

**Recommended starting point:** option 1. Transfer spike vectors CPU→GPU each step (N_neurons bytes = negligible). Implement option 2 only if profiling shows this is a bottleneck.

---

## 14.5 Python API

No user-facing API change is required for auto-selected backend:

```python
# Auto-select (GPU if N > 1000 and CUDA available)
net = RegionalNetwork()
net.add_population("STN", 5000, NeuronModelSpec.stn())
# ... network builds normally ...
result = net.simulate(2000.0, 0.01, ...)  # transparently uses GPU

# Explicit override
net.set_backend("cuda")   # force GPU
net.set_backend("cpu")    # force CPU
print(net.backend())      # "cuda" or "cpu"
```

---

## 14.6 Limitations and Non-Goals

- CUDA support is a separate build option; the default CPU build is unaffected
- Windows CUDA Toolkit requires MSVC + separate nvcc compiler; document setup carefully
- Mixed-precision (float32 for pools, float64 for accumulation) is out of scope for first implementation
- Multi-GPU is out of scope

---

## 14.7 Implementation Checklist

### Build System
- [ ] Add `USE_CUDA` CMake option with proper `enable_language(CUDA)` guard
- [ ] Separate compilation: `.cu` files compiled with nvcc, `.cpp` files with host compiler
- [ ] CI: add optional CUDA build job (skip on machines without GPU)

### C++ Abstraction
- [ ] Define `PoolBase` virtual interface
- [ ] Refactor `HHPool`, `IzPool`, `ComposablePool` to implement `PoolBase`
- [ ] Add `Backend` enum and `Network::set_backend()` / `Network::backend()`

### CUDA Kernels
- [ ] Implement `hh_pool_step_kernel` (Euler first, RK4 later)
- [ ] Implement `isyn_accumulate_kernel` with atomicAdd
- [ ] Implement `synapse_decay_kernel` for exponential synapses
- [ ] Implement `CUDASimContext` with upload/download methods

### Integration
- [ ] `Network::simulate_with_descriptors()`: dispatch to CUDA path when backend=CUDA
- [ ] Recording download: async transfer every `interval` steps
- [ ] Delay handling: CPU-managed spike vectors uploaded each step

### Tests
- [ ] Verify CPU and CUDA produce identical voltage traces for HH neuron (small network)
- [ ] Verify synaptic transmission accuracy (spike propagation) on GPU
- [ ] Benchmark: measure speedup at N=100, 1000, 5000, 10000 neurons
