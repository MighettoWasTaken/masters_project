# Task 16: Parallelism

## Priority: 3

## Status: Complete

---

## Overview

CPU-level parallelism in two phases:

1. **Phase 1 — OpenMP pool parallelism** (complete): different pool types (HHPool, IzPool, ComposablePool instances) step concurrently using `#pragma omp parallel sections` / `parallel for` in `PoolManager::step_all()`. Controlled via `set_num_threads()`.

2. **Phase 2 — Delay-decomposition thread groups** (complete): user assigns populations to named groups; each group runs in its own `std::thread`; inter-group spike communication uses per-synapse ring buffers sized by the synaptic delay, with two-counter synchronisation (`step_done` / `read_done`). Controlled via `set_thread_groups()`.

Benchmarks on CTX-BG-TH (8 populations):
- Phase 1: marginal benefit at typical network sizes (pool type diversity limited; overhead ~= gain)
- Phase 2: **~2.9× speedup** at 1000 ms simulation (0.388s vs 1.24s serial)

---

## 16.1 OpenMP Pool Parallelism (Phase 1)

### What was built

Parallelism at pool-type granularity in `PoolManager::step_all()` (`src/cpp/src/network/pool_manager.cpp`):

```cpp
void PoolManager::step_all(double dt) {
#ifdef HH_USE_OPENMP
    #pragma omp parallel sections num_threads(num_threads_)
    {
        #pragma omp section
        hh_pool_.step(dt);
        #pragma omp section
        iz_pool_.step(dt);
    }
    std::vector<ComposablePool*> cpools;
    for (auto& kv : comp_pools_) cpools.push_back(&kv.second);
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (int i = 0; i < (int)cpools.size(); ++i)
        cpools[i]->step(dt);
#else
    hh_pool_.step(dt);
    iz_pool_.step(dt);
    for (auto& kv : comp_pools_) kv.second.step(dt);
#endif
}
```

Each pool owns its own temporaries so no shared-state issues. I_syn accumulation and synapse updates remain sequential (serial path unchanged).

### Build

```cmake
# CMakeLists.txt (root)
find_package(OpenMP)

# src/cpp/CMakeLists.txt
if(OpenMP_CXX_FOUND)
    target_link_libraries(hodgkin_huxley_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_OPENMP)
endif()
```

### Python API

```python
rn.set_num_threads(4)    # explicit thread count
rn.set_num_threads(0)    # auto (let OpenMP decide)
rn.num_threads()         # returns current setting

# Also exposed on the underlying _Network binding:
net = hh._core._Network()
net.set_num_threads(2)
net.num_threads          # property
```

---

## 16.2 Delay-Decomposition Thread Groups (Phase 2)

### What was built

One `std::thread` per user-defined group. Inter-group spike delivery uses per-synapse ring buffers (`CrossGroupBuffer`) sized by the maximum synaptic delay across all connections in that channel.

**Key data structures** (`src/cpp/include/hodgkin_huxley/parallel_sim.hpp`):

```cpp
struct CrossGroupBuffer {
    std::vector<uint8_t> buf;           // [syn_local * capacity + step % capacity]
    std::atomic<size_t>  step_done{0};  // incremented by writer after scatter
    std::atomic<size_t>  read_done{0};  // incremented by reader after consuming
    size_t n_synapses = 0;
    size_t capacity   = 0;             // max_delay_steps for this pair
};

struct GroupDef {
    int group_id;
    std::vector<size_t>  neuron_indices; // global neuron indices owned by this group
    std::vector<size_t>  intra_syn;      // synapses where BOTH pre and post are in group
    // Out / in channels indexed by partner group id
    std::map<int, OutChannel> out_channels;
    std::map<int, InChannel>  in_channels;
};
```

Thread synchronisation: writer increments `step_done` after scattering voltages; reader busy-waits on `step_done` before reading delayed spike state, then increments `read_done` so writer can overwrite the ring slot.

**Recording in Phase 2**: all metrics supported — V, gates, calcium, u (Izhikevich), g_syn, I_syn, spike_events. Each group writes only to its own neuron/synapse index set; no locks needed. Spike events count only intra-group arrivals (cross-group spike detection via `spike_detected_` is unsafe without extra sync; documented limitation).

### Python API

```python
# Assign populations to groups — string keys, list of pop names
rn.set_thread_groups({"g0": ["CTX_e", "CTX_i"], "g1": ["STN"], "g2": ["GPe"]})

# All populations not listed: treated as their own single-pop group internally.
# Constraint: inter-group connections must have delay > 0.
# Zero-delay cross-group raises ValueError at set_thread_groups() time.

rn.has_thread_groups()   # True / False
rn.clear_thread_groups() # return to serial
rn.set_thread_groups(None)  # alias for clear

# Combine with Phase 1:
rn.set_num_threads(4)
rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})
```

### Routing

`RegionalNetwork.simulate()` selects Phase 2 when:
- `has_thread_groups()` is True
- `stim_plan` is available (all I_ext values are scalars and stimulators are `DBSStimulator`)
- `set(cfg.metrics) ⊆ {V, gates, calcium, u, g_syn, I_syn, spike_events} ∪ DERIVED_FROM_V`

Falls back to serial automatically when any condition is unmet.

### Known limitations

- STDP / STP synapse updates are only processed for intra-group synapses. Cross-group plastic synapses will not update weights during Phase 2 simulation.
- Spike events recorded via `spike_events` metric count only intra-group arrivals.
- Populations with zero-delay connections to another group must be co-grouped (enforced by validation at `set_thread_groups()` time).

---

## 16.3 Files Changed

| File | Change |
|---|---|
| `CMakeLists.txt` | `find_package(OpenMP)` |
| `src/cpp/CMakeLists.txt` | Link `OpenMP::OpenMP_CXX`; add `parallel_sim.cpp` |
| `src/cpp/include/hodgkin_huxley/parallel_sim.hpp` | **New** — `CrossGroupBuffer`, `GroupDef`, `OutChannel`, `InChannel` |
| `src/cpp/include/hodgkin_huxley/network/pool_manager.hpp` | `set_num_threads()`; recording-subset declarations |
| `src/cpp/src/network/pool_manager.cpp` | `step_all()` with OpenMP; `scatter_gates_for_names`, `scatter_calcium_for_names`, `scatter_recoveries_for_iz` |
| `src/cpp/include/hodgkin_huxley/network.hpp` | `set_num_threads()`, `num_threads()`, `simulate_with_descriptors_parallel()` |
| `src/cpp/src/network.cpp` | Phase 2 thread-group simulation loop; `intra_syn` construction |
| `src/cpp/include/hodgkin_huxley/hh_pool.hpp` | Phase 2 recording subset helpers |
| `src/cpp/include/hodgkin_huxley/iz_pool.hpp` | `scatter_recoveries_subset()` declaration |
| `src/cpp/src/iz_pool.cpp` | `scatter_recoveries_subset()` implementation |
| `src/cpp/include/hodgkin_huxley/regional_network.hpp` | `set_thread_groups()`, `clear_thread_groups()`, `has_thread_groups()`, `simulate_parallel()` |
| `src/cpp/src/regional_network.cpp` | Thread-group bookkeeping; `simulate_parallel()` routing |
| `src/python/bindings.cpp` | Bind `set_num_threads`, `num_threads`, `_simulate_parallel` |
| `src/hodgkin_huxley/_network/__init__.py` | `set_thread_groups()`, `clear_thread_groups()`, `has_thread_groups()`, `set_num_threads()`, `num_threads()`; Phase 2 routing in `simulate()` |
| `src/hodgkin_huxley/_core.pyi` | Stubs for all new bindings |

---

## 16.4 Tests

| File | What is tested |
|---|---|
| `tests/python/test_parallel.py` | Full Phase 1 + Phase 2 test suite (Section A–D): `set_num_threads` API, serial vs OpenMP identity, thread-group API stubs, delay-decomposition correctness, timing |
| `tests/python/test_recording.py` | `use_parallel` fixture parametrizes existing recording tests; 4 correctness tests compare serial vs Phase 2 traces for gates, calcium, g_syn, I_syn |
| `tests/python/test_plasticity.py` | `use_parallel` fixture on 10 no-crash / finite-value tests |
| `tests/python/test_dbs_stimulator.py` | `use_parallel` fixture on 8 stimulation correctness tests |
| `tests/python/test_sympy_gates.py` | `use_parallel` fixture on 7 CUSTOM_EXPR gate simulation tests |

---

## 16.5 Benchmarks

| Script | Output |
|---|---|
| `benchmarks/benchmark_parallel_ctxbgth.py` | Phase 1 vs Phase 2 vs serial on CTX-BG-TH model → `benchmarks/figures/parallel_ctxbgth_time.png`, `parallel_ctxbgth_speedup.png` |
| `examples/benchmark_parallel_scaling.py` | Scaling across chain / ring topologies → `examples/figs/parallel_scaling_chain.png`, `parallel_scaling_ring.png`, `parallel_speedup_heatmap.png` |

---

## 16.6 Implementation Checklist

### Phase 1 — OpenMP
- [x] `find_package(OpenMP)` in root `CMakeLists.txt`
- [x] `OpenMP::OpenMP_CXX` linked in `src/cpp/CMakeLists.txt`
- [x] `HH_USE_OPENMP` compile definition guarding all pragmas
- [x] `#pragma omp parallel sections` over HHPool / IzPool in `PoolManager::step_all()`
- [x] `#pragma omp parallel for` over composable pool instances
- [x] `PoolManager::set_num_threads()` / private `num_threads_` member
- [x] `Network::set_num_threads()` / `Network::num_threads()`
- [x] Python binding `set_num_threads` / `num_threads` on `_Network`
- [x] `RegionalNetwork.set_num_threads()` / `num_threads()` Python wrappers
- [x] Tests: `TestPhase1OpenMP` (serial vs OpenMP identity, race check, timing)

### Phase 2 — Delay Decomposition
- [x] `parallel_sim.hpp` — `CrossGroupBuffer`, `GroupDef`, channel structs
- [x] `parallel_sim.cpp` — thread loop, ring buffer read/write, two-counter sync
- [x] `RegionalNetwork::set_thread_groups()` with zero-delay cross-group validation
- [x] `RegionalNetwork::clear_thread_groups()` / `has_thread_groups()`
- [x] Python `set_thread_groups()` with string-key groups, spec auto-rename, zero-delay guard
- [x] Phase 2 routing condition in `RegionalNetwork.simulate()`
- [x] Full recording: V, gates, calcium, u, g_syn, I_syn, spike_events
- [x] `intra_syn` field in `GroupDef` for safe spike_event accumulation
- [x] Tests: `TestPhase2ThreadGroups`, `TestDelayDecompositionCorrectness` (serial vs parallel identity, race probe)
- [x] `use_parallel` fixture extended to recording, plasticity, DBS, and sympy gate tests
- [x] Benchmarks: CTX-BG-TH speedup (~2.9× at 1000 ms); scaling heatmaps
