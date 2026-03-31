# Task 16: Parallelism

## Priority: 3

## Overview

Add CPU-level parallelism via OpenMP for multi-core speedup on large networks, and implement synaptic delay decomposition for population-level parallel simulation.

Two independent sub-features:
1. **OpenMP thread parallelism** (short term, easier) — 2–4× speedup on a 4-core machine for N>100
2. **Delay-based population parallelism** (longer term, harder) — near-linear scaling with number of populations when inter-population delays ≥ communication latency

---

## 16.1 OpenMP Parallelism

### Concept

Each neuron pool's `step()` loop iterates over N neurons independently. With OpenMP, this loop is distributed across available threads. The critical race condition is I_syn accumulation: multiple synapses write to the same post-neuron's current buffer.

### Pool Loops

```cpp
// In HHPool::step_rk4()
#pragma omp parallel for schedule(static) if(n_ > OMP_THRESHOLD)
for (size_t i = 0; i < n_; ++i) {
    // RK4 step for neuron i — no cross-neuron dependencies
}
```

### I_syn Accumulation (Race Condition Fix)

**Option A — Thread-local partial sums + reduce** (preferred for dense connectivity, n_synapses > 10 * n_neurons):
```cpp
#pragma omp parallel
{
    std::vector<double> local_I(n_neurons, 0.0);
    #pragma omp for nowait
    for (size_t i = 0; i < n_synapses; ++i)
        local_I[post[i]] += g[i] * (V_cache_[pre[i]] - E_syn[i]);
    #pragma omp critical
    for (size_t j = 0; j < n_neurons; ++j)
        I_syn_buffer_[j] += local_I[j];
}
```

**Option B — Atomic adds** (preferred for sparse networks):
```cpp
#pragma omp atomic
I_syn_buffer_[post[i]] += contribution;
```

Strategy selected at simulation start based on connectivity density.

### Build Configuration

```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hodgkin_huxley_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_OPENMP)
endif()
```

OpenMP is auto-detected; builds without it fall back to single-threaded silently.

### Python API

```python
net.set_num_threads(4)   # explicit thread count
net.set_num_threads(0)   # auto (use all available cores)
print(net.num_threads())
```

---

## 16.2 Delay-Based Population Decomposition

### Concept

If the minimum synaptic delay between populations A and B is `D` ms, population A can be advanced `D / dt` steps before its spikes reach B. Multiple populations can run concurrently if their mutual delays form a valid partial order (Morrison et al. 2005).

For the CTX-BG-TH benchmark with typical inter-area delays of 4–6 ms and dt=0.01 ms, each population can run 400–600 steps ahead of its downstream targets.

### Algorithm

1. Build delay graph: nodes = populations, edges = (src, dst, min_delay_steps)
2. Topological level assignment: populations at level k can advance min(outgoing_delay) steps ahead of level k+1
3. Per-population spike event queues (time-stamped, thread-safe)
4. Parallel advance loop: each thread owns one population level; events delivered across boundaries at each synchronisation point

### Transport Abstraction

`PopulationQueue` uses a pluggable transport layer so the same delay-decomposition algorithm works whether the sync boundary is between threads (task16 OpenMP), CUDA devices (task19 P2P), or an NCCL collective (task19 multi-GPU). Concrete implementations are selected automatically based on context.

```cpp
// Defined in spike_transport.hpp — shared by task16 and task19.
class SpikeTransport {
public:
    virtual ~SpikeTransport() = default;
    virtual void send(int src_device, int dst_device,
                      const SpikeEvent* events, size_t count) = 0;
    virtual size_t recv(int src_device, SpikeEvent* buffer, size_t capacity) = 0;
    virtual void flush() = 0;  // barrier/sync — mutex for threads, cudaStreamSync for CUDA
};

// Default: shared-memory, mutex-protected (used by OpenMP thread-parallel path in task16)
class LocalTransport : public SpikeTransport {
    void send(...) override;   // enqueue to mutex-protected in-process queue
    size_t recv(...) override;
    void flush() override;     // no-op; threads share an address space
};

// CUDA device-to-device transports are implemented in task19:
// class CUDAP2PTransport : public SpikeTransport { ... };  // cudaMemcpyPeer, NVLink/PCIe
// class NCCLTransport    : public SpikeTransport { ... };  // ncclAllGather, >4 GPUs
```

### Population Event Queue

```cpp
struct SpikeEvent {
    double time;
    size_t neuron;
};

// SpikeTransport* is injected at construction; defaults to LocalTransport.
// Swapping to CUDAP2PTransport or NCCLTransport (task19) requires no other changes.
class PopulationQueue {
public:
    explicit PopulationQueue(SpikeTransport* transport = nullptr,
                             int src_device = 0, int dst_device = 0);
    void push(SpikeEvent e);
    std::vector<SpikeEvent> drain_before(double t);
private:
    SpikeTransport* transport_;
    int src_device_, dst_device_;
    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>,
                        std::greater<SpikeEvent>> queue_;
    std::mutex mutex_;   // used by LocalTransport path only
};
```

### Python API

```python
net.enable_delay_decomposition(True)
net.set_num_threads(8)
```

### Caveats

- Benefit realised only when delays are long relative to dt and populations are large
- Populations with self-connections at zero delay cannot be decomposed
- Profile before implementing: overhead may dominate for small N

---

## 16.3 Implementation Checklist

### OpenMP (Phase 1 — implement first)
- [ ] Add `find_package(OpenMP)` to `CMakeLists.txt`
- [ ] Add `#pragma omp parallel for` to `HHPool::step_rk4()`, `IzPool::step()`, `ComposablePool::step()`
- [ ] Implement thread-safe I_syn accumulation (option A or B based on connectivity density)
- [ ] Add `Network::set_num_threads()` and `Network::num_threads()`
- [ ] Bind `set_num_threads()` in Python
- [ ] Benchmark: measure speedup at 1, 2, 4, 8 threads for N=1000

### Delay Decomposition (Phase 2)
- [ ] Define `SpikeTransport` abstract class and `LocalTransport` in `spike_transport.hpp`
- [ ] Update `PopulationQueue` constructor to accept `SpikeTransport*`; default to `LocalTransport`
- [ ] Build delay graph from `RegionalNetwork` population structure
- [ ] Implement topological level assignment
- [ ] Implement per-population `PopulationQueue` with thread-safe push/drain
- [ ] Implement parallel advance loop in `RegionalNetwork::simulate()`
- [ ] Add `enable_delay_decomposition()` API
- [ ] Validate: identical output to single-threaded simulation
- [ ] Benchmark: speedup for CTX-BG-TH 8-population model at varying N
- [ ] Note: `CUDAP2PTransport` and `NCCLTransport` are implemented in task19 and slot in without further changes here
