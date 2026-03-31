# Roadmap

## Vision

A fast, biologically accurate, and ergonomic Python framework for constructing and simulating conductance-based neural network models, backed by a high-performance C++ core. The framework should be capable of reproducing published computational neuroscience models with minimal code and running them at >10× the speed of pure-Python equivalents, while remaining extensible to novel neuron types, synapse dynamics, and learning rules without recompilation.

---

## Development Phases

### Phase 1 — Core Framework (Complete)

All foundational features required to reproduce the Hahn/Kumaravelu CTX-BG-TH Parkinson's disease benchmark.

| Task | Title | Status |
|------|-------|--------|
| task1 | HH Neuron Core | Complete |
| task2 | Izhikevich Neuron | Complete |
| task3 | Exponential Synapses | Complete |
| task4 | Network Class | Complete |
| task5 | Composable Neuron / Ion Channel System | Complete |
| task6 | Alpha + Double-Exp Synapses | Complete |
| task7 | Regional Network & Populations | Complete |
| task8 | Recording System | Complete |
| task9 | Pulse & DBS Stimulators | Complete |
| task10 | Spectral Analysis | Complete |
| task11 | CTX-BG-TH Benchmark Validation | Complete |

**Key outcomes:** 5 neuron presets (TH, STN, GPe, GPi, Striatum), 4 synapse types, population-level connectivity API, zero-allocation simulation hot loop, >10× speedup vs. pure Python, beta-band power validated against reference.

---

### Phase 2 — API Cleanup and Unified Equation System (Current)

task12 (structural cleanup) has no dependencies and should be done first to clear the surface before any new feature work. task13 (SymPy equation system) is the highest-priority feature and unblocks everything else in phases 2–4.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task12](../task12.md) | API Structural Cleanup | 2 | None — **start here** |
| [task13](../task13.md) | Unified Equation System (SymPy) | 2 | task12 recommended first |
| [task14](../task14.md) | Generalized Intracellular Dynamics | 2 | task13 (ODE fields use SymPy) |

task13 also includes its own API cleanup section (13.9) which retires the legacy equation types (`BoltzmannParams`, `TauParams`, form enums) to `hodgkin_huxley.legacy` once SymPy is in place.

**Key outcomes:** One canonical network-building workflow; one equation language (SymPy) across all model definitions; legacy equation types retired; arbitrary intracellular substances (dopamine, cAMP, etc.).

---

### Phase 3 — Synaptic Plasticity (After Phase 2)

Synaptic learning rules using SymPy update-rule expressions (task13) and optionally intracellular substances (task14) for neuromodulator-gated plasticity.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task15](../task15.md) | Plasticity Support (STDP, STP) | 2 | task13 (SymPy update rules), task14 (for gated plasticity) |

**Key outcomes:** STDP, short-term depression/facilitation, dopamine-gated STDP, weight recording.

---

### Phase 4 — Performance Scaling (Parallel with Phase 3)

Infrastructure for simulating networks with >1000 neurons efficiently. task16 (OpenMP) can begin as soon as the codebase is stable post-task13; task17 (CUDA) depends on both task16 (PoolBase interface) and task13 (CUDAPrinter). task19 (Multi-GPU) depends on task17's `Device` model and task16's `SpikeTransport` abstraction — target for extremely large models (N > 50,000) or full-scale cortical column simulations where single-GPU memory is a bottleneck.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task16](../task16.md) | CPU Parallelism (OpenMP) | 3 | None |
| [task17](../task17.md) | CUDA GPU Acceleration | 3 | task16 (PoolBase), task13 (CUDAPrinter) |
| [task19](../task19.md) | Multi-GPU Parallelism | 3 | task17 (Device API), task16 (SpikeTransport) |

**Key outcomes:** 2–4× speedup via OpenMP on multi-core CPU; 50–200× speedup via CUDA for N>1000; custom SymPy equations work transparently on GPU; near-linear multi-GPU scaling for N>50,000 via CUDA P2P or NCCL.

---

### Phase 5 — Ecosystem (Parallel with Phases 3–4)

Documentation and ML integration tooling. task18 (docs) can proceed as features stabilise post-task13. task20 (ML integration) depends on task12 for a stable public API and task13 for serialisable SymPy state, but its continuable-simulation and pickling sub-features can begin earlier in parallel with task14/15.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task18](../task18.md) | Web Documentation | 3 | Stable API (tasks 12–13 complete) |
| [task20](../task20.md) | ML Framework Integration | 2 | task12 (stable API), task13 (serialisable state) |

**Key outcomes:** MkDocs site on GitHub Pages, full API reference, tutorial notebooks, contributor guide; continuable simulations with `state_dict()` / `load_state_dict()`; sparse spike tensors for SNN pipelines; PyTorch / TensorFlow / pandas export; full `pickle` / `joblib` support.

---

## Architectural Principles

1. **Data over code**: neuron and synapse models are described by parameter structs, not class hierarchies. This enables Eigen vectorization across neuron pools and preserves composability. Equations within those structs are expressed as SymPy expressions — compiled to Eigen-compatible C++ lambdas (CPU) or `__device__` functions (CUDA) with standard forms recognized at no JIT cost.

2. **One canonical workflow**: users should reach any simulation through the same sequence — define model → build network → simulate → analyze. Internal routing (descriptor path, pool dispatch, backend selection) is invisible.

3. **Performance by default**: the hot loop should be zero-allocation and SIMD-vectorized for all supported neuron types. Accuracy is not traded for speed; the polynomial fast-exp and dense-vs-descriptor routing are opt-in or automatically selected.

4. **Biological accuracy first**: preset parameters and synapse kinetics match published values. Convenience should never compromise fidelity.

5. **Extensibility without recompilation**: custom neuron models, synapse kinetics, and intracellular dynamics are expressible through Python-level spec objects without writing C++.

---

## Known Limitations (Deferred)

- **RK45 adaptive integration**: enum exists, implementation incomplete. Fixed dt (0.01 ms) is sufficient for current use cases.
- **Multi-GPU / distributed simulation**: planned for task19 (depends on task17 Device API + task16 SpikeTransport). Not out of scope — deferred until single-GPU path is stable.
- **Non-conductance-based neuron models** (LIF, AdEx): not planned; use Izhikevich as the lightweight alternative.
- **Python-defined neuron subclasses** (pybind11 trampolines): possible but carries performance penalty; composable spec system is the preferred extension mechanism.
