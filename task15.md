# Task 15: Parallelism

## Priority: 3

## Overview

Add CPU-level parallelism to accelerate large-scale simulations on multi-core machines without requiring CUDA (task 14). Two complementary mechanisms are targeted:

1. **OpenMP thread parallelism**: parallelize the neuron pool `step()` loops across hardware cores — low implementation cost, 2–4× speedup on a typical 4-core laptop
2. **Delay-based population decomposition**: exploit synaptic delays to advance independent sub-networks concurrently — architectural change, near-linear scaling with number of populations when delays ≥ communication latency

These are independent and can be shipped separately; OpenMP should be completed first.

---

## 15.1 OpenMP Thread Parallelism

### Concept

Each neuron pool's `step()` loop iterates over N neurons independently. With OpenMP, this loop is distributed across available threads. The critical race condition is I_syn accumulation: multiple synapses write to the same post-neuron's current buffer.

### Implementation

**Pool loops** (embarrassingly parallel — add `#pragma omp parallel for`):
```cpp
// In HHPool::step_rk4()
#pragma omp parallel for schedule(static) if(n_ > OMP_THRESHOLD)
for (size_t i = 0; i < n_; ++i) {
    // RK4 step for neuron i — no cross-neuron dependencies
}
```

**I_syn accumulation** (race condition on shared buffer — requires fix):

Option A — Thread-local partial sums, then reduce:
```cpp
// Thread-private buffer
std::vector<double> I_syn_local(n_neurons, 0.0);
#pragma omp parallel private(I_syn_local) reduction(+: I_syn_buffer_)
{
    // Each thread accumulates into I_syn_local
    // Reduction merges across threads
}
```

Option B — Atomic adds (simpler but slower for dense connectivity):
```cpp
#pragma omp atomic
I_syn_buffer_[post[i]] += contribution;
```

Option A is preferred for networks with O(N²) synapses; the reduction cost is O(N * T) where T is thread count. Option B is better for sparse networks.

**Recommended strategy:** Use option A when connectivity is dense (n_synapses > 10 * n_neurons), option B otherwise. Threshold determined at simulation start.

### Build Configuration

```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hodgkin_huxley_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_OPENMP)
endif()
```

OpenMP is detected automatically; builds without it fall back to single-threaded.

### Python API

```python
net.set_num_threads(4)   # explicit thread count
net.set_num_threads(0)   # auto (use all available cores)
print(net.num_threads()) # → int
```

Thread count can also be set via environment variable `OMP_NUM_THREADS` (standard OpenMP convention).

---

## 15.2 Delay-Based Population Decomposition

### Concept

If the minimum synaptic delay between two populations A and B is `D` ms, then population A can be advanced `D / dt` steps before its spikes reach B. With D = 5 ms (typical inter-area delay in the CTX-BG-TH model) and dt = 0.01 ms, population A can run 500 steps ahead of B. Multiple populations can thus run concurrently if their mutual delays form a valid partial order.

This is the core insight behind NEST's "communication period" and Henker et al.'s delay decomposition.

### Algorithm

```
1. Build delay graph: nodes = populations, edges = (src, dst, min_delay_steps)
2. Find topological levels: populations at level k can be advanced to the
   minimum incoming delay from populations at level k-1
3. At each global time step:
   a. Advance level-0 populations n_steps (= min outgoing delay)
   b. In parallel: advance all level-k populations their n_steps
   c. Synchronize: deliver spike events across level boundaries
   d. Advance all populations together for remainder of global step
```

This requires:
- Per-population spike event queues (time-stamped)
- A "virtual time" per population that can run ahead of global time
- Safe merge of spike events from ahead-running populations

### Population Event Queue

```cpp
struct SpikeEvent {
    double time;      // simulation time (ms)
    size_t neuron;    // global neuron index
};

class PopulationQueue {
public:
    void push(SpikeEvent e);
    std::vector<SpikeEvent> drain_before(double t);
private:
    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>,
                        std::greater<SpikeEvent>> queue_;
    std::mutex mutex_;
};
```

### Python API

```python
net.enable_delay_decomposition(True)
net.set_num_threads(8)   # one thread per population level (ideally)
```

### Caveats

- Benefit is only realized when delays are long relative to dt and populations are large
- Shared-memory populations (self-connections with zero delay) cannot be decomposed
- Implementation complexity is significantly higher than OpenMP; plan for careful testing
- Profile before implementing: for the CTX-BG-TH benchmark at N=10 per population, overhead may dominate

---

## 15.3 Implementation Checklist

### OpenMP (Phase 1 — implement first)
- [ ] Add `find_package(OpenMP)` to `CMakeLists.txt`
- [ ] Add `#pragma omp parallel for` to `HHPool::step_rk4()`, `IzPool::step()`, `ComposablePool::step()`
- [ ] Implement thread-safe I_syn accumulation (choose option A or B based on connectivity density)
- [ ] Add `Network::set_num_threads()` and `Network::num_threads()`
- [ ] Bind `set_num_threads()` in Python
- [ ] Benchmark: measure speedup at 1, 2, 4, 8 threads for N=1000

### Delay Decomposition (Phase 2 — after OpenMP validated)
- [ ] Build delay graph from `RegionalNetwork` population structure
- [ ] Implement topological level assignment
- [ ] Implement per-population `PopulationQueue` with thread-safe push/drain
- [ ] Implement parallel advance loop in `RegionalNetwork::simulate()`
- [ ] Add `enable_delay_decomposition()` API
- [ ] Validate: verify identical output to single-threaded simulation
- [ ] Benchmark: measure speedup for CTX-BG-TH 8-population model at varying N
