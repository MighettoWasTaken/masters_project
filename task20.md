# Task 20: ML Framework Integration

## Priority: 2 — Depends on task12 (stable public API), task13 (serialisable SymPy state)

## Overview

Four related features that make the framework usable as a data source and stateful component within machine learning pipelines:

1. **Continuable simulations** — `simulate()` preserves neuron and synapse state across multiple calls; `state_dict()` / `load_state_dict()` for explicit state management following PyTorch `nn.Module` conventions.
2. **Sparse spike encoding** — memory-efficient binary spike tensor output indexed by time step, in addition to the existing timestamp event list. Target format for SNN training pipelines.
3. **Tensor and DataFrame export** — zero-copy (where possible) conversion of recorded data to PyTorch tensors, TensorFlow tensors, and pandas DataFrames.
4. **Pickling** — full network serialisation via `pickle` / `joblib`; required for multiprocessing, hyperparameter sweeps, and checkpoint/restore.

The framework does not depend on PyTorch or TensorFlow at import time — all ML exports are lazy-imported and raise a clear `ImportError` if the target library is not installed.

---

## 20.1 Continuable Simulations

### Current Behaviour

Each call to `simulate()` is currently self-contained: it rebuilds synapse group indices, recomputes decay factors, and does not guarantee that pool state from a previous call is used as the initial condition.

### Target Behaviour

State (membrane potentials, gate variables, intracellular concentrations, synapse conductances, delay ring buffers) persists across `simulate()` calls by default. This is the natural semantics for online learning and multi-phase protocols.

```python
net = hh.RegionalNetwork()
# ... build network ...

# Phase 1: baseline
result1 = net.simulate(1000.0, 0.01, I_ext={"STN": 2.0})

# Phase 2: stimulation — picks up from end of phase 1
result2 = net.simulate(1000.0, 0.01, I_ext={"STN": 2.0},
                        stimulators={"STN": dbs})

# Explicit reset to initial conditions
net.reset_state()
result3 = net.simulate(1000.0, 0.01, ...)  # starts fresh
```

The `time_offset` of each result reflects elapsed simulation time:

```python
result1.time_axis   # [0.0, 0.01, ..., 999.99]  ms
result2.time_axis   # [1000.0, 1000.01, ..., 1999.99]  ms
```

### State Dict API (PyTorch-style)

```python
# Snapshot state
sd = net.state_dict()
# Returns a flat dict of numpy arrays:
# {
#   "populations.STN.V":       np.ndarray (n,),
#   "populations.STN.gates.m": np.ndarray (n,),
#   "populations.STN.Ca":      np.ndarray (n,),
#   "synapses.g":              np.ndarray (n_syn,),
#   "synapses.delay_buf":      np.ndarray (n_syn, buf_len),
#   "elapsed_ms":              float,
#   ...
# }

# Restore state (e.g. after parameter sweep clone)
net.load_state_dict(sd)
net.load_state_dict(sd, strict=False)  # ignore missing keys (partial restore)

# Reset to construction-time initial conditions
net.reset_state()

# Query elapsed simulated time
net.elapsed_ms   # float — total ms simulated since last reset_state()
```

### C++ Changes

```cpp
// network.hpp additions
struct SimulationState {
    std::vector<std::vector<double>> pool_V;       // per pool
    std::vector<std::vector<double>> pool_gates;   // per pool × per gate
    std::vector<std::vector<double>> pool_X;       // per pool × per substance
    std::vector<double> syn_g;                     // SoA g
    std::vector<double> syn_delay_buf;             // flattened ring buffers
    std::vector<size_t> syn_buf_heads;
    double elapsed_ms = 0.0;
};

SimulationState Network::get_state() const;
void            Network::set_state(const SimulationState&);
void            Network::reset_state();
```

The hot loop no longer calls `reset_pools()` at the start of `simulate_with_descriptors()`. Decay factors and synapse groups are cached and only rebuilt when topology or `dt` changes (dirty flags `groups_dirty_`, `decay_dirty_`).

---

## 20.2 Sparse Spike Encoding

The existing output (`result["spikes"]`) is a list of `(time_ms, neuron_idx)` event tuples — efficient for plotting and ISI analysis but not directly usable as SNN input tensors. This section adds a second representation: a binary array indexed by time step.

### Format Options

| Format | Shape | Use case |
|--------|-------|----------|
| `"numpy"` | `(T, N)` bool | General; baseline |
| `"scipy_csr"` | `(T, N)` CSR sparse | Memory-efficient; batch indexing |
| `"torch_dense"` | `(T, N)` BoolTensor | PyTorch SNN layers (spikingjelly, Norse) |
| `"torch_sparse"` | `(T, N)` sparse COO | Memory-efficient PyTorch; large N |
| `"tensorflow"` | `(T, N)` SparseTensor | TF/Keras SNN pipelines |

`T = round(duration / dt)`, `N` = total number of neurons in the population / network.

### Python API

```python
# From a completed result
spike_tensor = result.to_spike_tensor(
    format="torch_dense",   # see table above
    population="STN",       # None = all neurons
    dt=None,                # None = use simulation dt; or coarser bin width
    time_axis=0,            # 0 → (T, N),  1 → (N, T)
)

# Streaming iterator — avoids allocating the full T×N matrix at once
for chunk in result.spike_tensor_chunks(chunk_ms=100.0, format="torch_sparse"):
    snn_layer(chunk)

# During simulation: record in spike-tensor format from the start
# (avoids storing raw timestamps then converting)
recording = hh.RecordingConfig(
    spike_format="sparse_coo",   # store directly as COO indices
    populations=["STN", "GPe"],
)
result = net.simulate(5000.0, 0.01, recording=recording)
```

### Memory Accounting

For N=1000 neurons, 5000 ms, dt=0.01 ms (500,000 steps):
- Dense bool: `500,000 × 1000 × 1 byte = 500 MB` — impractical
- Sparse COO (10 Hz mean rate): `500,000 × 1000 × 0.01 × 8 bytes × 2 = 80 MB` — acceptable
- Streaming chunks of 100 ms: `10,000 × 1000 × 1 byte = 10 MB` per chunk — comfortable

Default `format` when N > 500 or T > 100,000 is `"torch_sparse"` / `"scipy_csr"` to avoid accidental OOM.

---

## 20.3 Tensor and DataFrame Export

### `MetricsResult.to_torch()` / `to_tensorflow()` / `to_numpy()` / `to_dataframe()`

```python
# PyTorch
tensors = result.to_torch()
# Returns dict[str, torch.Tensor]:
# {
#   "V":            Tensor shape (N, T) float32
#   "firing_rates": Tensor shape (N,) float32
#   "spikes":       sparse COO Tensor shape (T, N) bool  (if spikes recorded)
#   "gates.m":      Tensor shape (N, T) float32          (if gates recorded)
# }

# TensorFlow
tensors = result.to_tensorflow()
# Same keys; values are tf.Tensor (V, rates) or tf.SparseTensor (spikes)

# Pandas DataFrame (long format by default)
df = result.to_dataframe(population="STN", format="long")
# columns: time_ms, neuron_idx, V, firing_rate, ...
df = result.to_dataframe(format="wide")
# columns: time_ms, V_0, V_1, ..., V_N

# For RegionalNetwork results (PopulationMetricsResult)
tensors = pop_result.to_torch()
# dict[population_name, dict[metric, Tensor]]
tensors["STN"]["V"]            # (N_STN, T) float32
tensors["GPe"]["firing_rates"] # (N_GPe,) float32
```

### Zero-Copy on GPU

When the network is on a CUDA device (task17), `to_torch()` returns tensors that share device memory — no host round-trip:

```python
net.to(hh.device("cuda:0"))
result = net.simulate(...)
tensors = result.to_torch()
tensors["V"].device   # device(type='cuda', index=0)
```

This uses `torch.as_tensor()` on a DLPack capsule or `cudaIpcMemHandle` rather than a copy. Falls back to host-copy if sharing is not possible.

### Dtype Conventions

| Metric | Default dtype | Rationale |
|--------|--------------|-----------|
| V (membrane potential) | float32 | Sufficient for ~0.01 mV precision; halves memory vs float64 |
| Gate states | float32 | Values in [0, 1] |
| Firing rates | float32 | |
| Spike tensor | bool / uint8 | Binary |
| Synapse weights | float32 | |
| Time axis | float64 | Preserves ms precision over long simulations |

Override: `result.to_torch(dtype=torch.float64)` for full precision.

---

## 20.4 Pickling and Serialisation

Full support for `pickle`, `pickle.dumps` / `pickle.loads`, `joblib.dump`, and `copy.deepcopy`. Required for:
- `multiprocessing.Pool` parameter sweeps
- `joblib.Parallel` hyperparameter searches
- Checkpoint / restore of trained network states
- `copy.deepcopy` for cloning a network before a destructive experiment

### Protocol

pybind11-bound classes expose `__getstate__` / `__setstate__`:

```python
import pickle

# Round-trip
data = pickle.dumps(net)
net2 = pickle.loads(data)
assert net2.n_neurons() == net.n_neurons()

# Joblib (compressed)
import joblib
joblib.dump(net, "checkpoint.pkl", compress=3)
net2 = joblib.load("checkpoint.pkl")

# Deep copy (same process, different object)
import copy
net_clone = copy.deepcopy(net)
```

`__getstate__` returns a dict of Python primitives and numpy arrays — no C++ pointers. `__setstate__` reconstructs the full object from this dict, going through the same construction path as the normal API (so validation is preserved).

**What is serialised:**
- Network topology (pre/post/weight/type arrays, all SoA fields)
- Population definitions (name, start, count, NeuronModelSpec)
- Current simulation state (V, gates, intracellular, synapse g, delay buffers, elapsed_ms)
- Stimulator and recording configuration
- SymPy expression cache references (expressions are re-compiled on first use after load, not stored as compiled `.so` paths — these are machine-specific)

**What is NOT serialised:**
- Compiled `.so` JIT artefacts (regenerated from SymPy expr hash on load)
- CUDA device memory (state is downloaded to host before serialisation; re-uploaded on `simulate()` if device was set)

### `save` / `load` Convenience Methods

```python
net.save("checkpoint.hh")             # gzip-compressed pickle
net2 = hh.RegionalNetwork.load("checkpoint.hh")

# Explicit state-only save (smaller; topology separate)
net.save_state("state_1000ms.hh")
net.load_state("state_1000ms.hh")
```

---

## 20.5 Limitations

- Zero-copy GPU tensor export (§20.3) requires PyTorch ≥ 2.0 and CUDA-enabled build (task17)
- TensorFlow SparseTensor export requires TF ≥ 2.0
- Sparse spike tensors larger than available RAM: use the streaming iterator (§20.2)
- SymPy JIT cache is not portable across machines or Python versions; networks saved with custom equations must recompile on load (~0.5–2 s per novel expression)
- Pickle protocol 5 (buffer protocol, zero-copy large arrays) requires Python ≥ 3.8

---

## 20.6 Implementation Checklist

### Continuable Simulations
- [ ] Define `SimulationState` struct in `network.hpp`
- [ ] Implement `Network::get_state()` / `set_state()` / `reset_state()`
- [ ] Add `elapsed_ms_` counter to `Network`; increment each `simulate_with_descriptors()` call
- [ ] Cache `groups_dirty_` / `decay_dirty_` flags; skip rebuild when topology and dt are unchanged between calls
- [ ] Remove implicit pool reset at start of `simulate_with_descriptors()` — only reset if `reset_state()` was called
- [ ] Expose `state_dict()` / `load_state_dict()` / `reset_state()` / `elapsed_ms` in Python
- [ ] Return `time_axis` starting at `elapsed_ms` (before advance) in each result
- [ ] Tests: two consecutive `simulate()` calls produce identical output to one call of double the duration

### Sparse Spike Encoding
- [ ] Add `spike_format` option to `RecordingConfig`; implement COO storage path in C++ recording loop
- [ ] Implement `MetricsResult.to_spike_tensor(format, population, dt, time_axis)`
- [ ] Implement `MetricsResult.spike_tensor_chunks(chunk_ms, format)` iterator
- [ ] Support `"numpy"`, `"scipy_csr"`, `"torch_dense"`, `"torch_sparse"`, `"tensorflow"` formats
- [ ] Auto-select sparse format when dense would exceed 200 MB; warn user
- [ ] Tests: spike tensor matches event-list representation; round-trip fidelity

### Tensor and DataFrame Export
- [ ] Implement `MetricsResult.to_torch()` — dict of tensors; lazy-import `torch`
- [ ] Implement `MetricsResult.to_tensorflow()` — dict of tensors; lazy-import `tensorflow`
- [ ] Implement `MetricsResult.to_dataframe(format)` — lazy-import `pandas`
- [ ] Implement `MetricsResult.to_numpy()` — returns dict of ndarrays (no optional dep)
- [ ] Propagate to `PopulationMetricsResult.to_torch()` / `to_tensorflow()` / `to_dataframe()`
- [ ] GPU zero-copy path: use DLPack (`torch.utils.dlpack`) when recording buffers are on-device
- [ ] Tests: shape/dtype assertions for all formats; GPU zero-copy path (task17 dep)

### Pickling
- [ ] Implement `__getstate__` / `__setstate__` on all pybind11-bound classes: `Network`, `RegionalNetwork`, `NeuronModelSpec`, `SynArrays`, stimulator types
- [ ] Ensure SymPy expression fields serialise as expression trees (not compiled pointers)
- [ ] Ensure CUDA state is downloaded to host in `__getstate__` and re-uploaded lazily on next `simulate()`
- [ ] Implement `RegionalNetwork.save()` / `RegionalNetwork.load()` using gzip pickle
- [ ] Implement `save_state()` / `load_state()` for state-only checkpoint
- [ ] Tests: `pickle.loads(pickle.dumps(net))` round-trip; `copy.deepcopy`; `joblib` round-trip
- [ ] Tests: pickle of network mid-simulation; resume produces identical output
