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

### Phase 2 — API Cleanup and Unified Equation System (Complete)

task12 (structural cleanup) has no dependencies and should be done first to clear the surface before any new feature work. task13 (SymPy equation system) is the highest-priority feature and unblocks everything else in phases 2–4.

| Task | Title | Priority | Status |
|------|-------|----------|--------|
| [task12](../completed/task12.md) | API Structural Cleanup | 2 | **Complete** |
| [task13](../task13.md) / [task13.5](../task13.5.md) | Unified Equation System (SymPy) + Synapse Overhaul | 2 | **Complete** |
| [task14](../task14.md) | Generalized Intracellular Dynamics | 2 | **Complete** |

task13 delivered: unified `SynapseSpec` (7 UpdateForms replacing the old class hierarchy), `SynapseModel` Python builder with SymPy pattern matching, VM bytecode for `CUSTOM_EXPR` forms, JIT compilation for gate kinetics, `NeuronModel` builder, and full API cleanup — legacy equation types (`BoltzmannParams`, `TauParams`, form enums) retired to `hodgkin_huxley.legacy`. See `task13.5.md` for the detailed implementation record.

task14 delivered: `CalciumSpec` / `Ca_` / `E_Ca_` replaced by a general `std::vector<IntracellularSpec>` system. Calcium is index 0 by convention. New Python builders — `IntracellularDynamics` (with SymPy ODE + Nernst pattern matching) and `Modulation` (7 target types: CHANNEL_G, CHANNEL_EREV, GATE_INF_SHIFT, GATE_INF_SCALE, GATE_TAU_SCALE, GATE_INF_EXPR, SYNAPSE_G). `RegionalNetwork.add_intracellular()` attaches dynamics to named populations. Hot-loop allocation-free: modulation scratch arrays pre-allocated as members; per-channel currents cached and reused; E_rev stored as pre-filled member arrays; gate modulation ops skipped entirely when no modulations are active. Legacy `CalciumSpec` API emits `DeprecationWarning` and delegates to the new system.

**Key outcomes achieved:** One canonical network-building workflow; SymPy equation language across all model definitions; unified synapse architecture with branch-free SoA hot loops; arbitrary intracellular substances expressible in Python without C++ changes; benchmark restores to pre-task14 performance (1.1–1.2 s / 1000 ms at n=10).

---

### Phase 3 — Synaptic Plasticity (Complete)

Synaptic learning rules using SymPy update-rule expressions (task13) and optionally intracellular substances (task14) for neuromodulator-gated plasticity.

| Task | Title | Priority | Status |
|------|-------|----------|--------|
| [task15](../task15.md) | Plasticity Support (STDP, STP) | 2 | **Complete** |

task15 delivered: STDP with configurable A+/A− windows and τ+/τ−; short-term depression and facilitation (STP) expressed as SymPy update rules; dopamine-gated STDP via intracellular substance modulation; per-synapse weight recording (`record_weights`); plasticity rules thread-safe under Phase-2 delay-decomposition.

**Key outcomes:** STDP, short-term depression/facilitation, dopamine-gated STDP, weight recording.

---

### Phase 4 — Performance Scaling (In Progress)

Infrastructure for simulating networks with >1000 neurons efficiently. task16 (CPU parallelism) is complete. task17 (CUDA) depends on task16 (PoolBase interface) and task13 (CUDAPrinter). task19 (Multi-GPU) depends on task17's `Device` model and task16's `SpikeTransport` abstraction — target for extremely large models (N > 50,000) or full-scale cortical column simulations where single-GPU memory is a bottleneck.

| Task | Title | Priority | Status |
|------|-------|----------|--------|
| [task16](../task16.md) | CPU Parallelism | 3 | **Complete** |
| [task17](../task17.md) | CUDA GPU Acceleration | 3 | Not started |
| [task19](../task19.md) | Multi-GPU Parallelism | 3 | Not started |

task16 delivered: **Phase-1 OpenMP** — parallel pool stepping (`set_num_threads(n)`); all HHPool/IzPool/ComposablePool steps run in `omp parallel sections` / `parallel for`. **Phase-2 delay-decomposition** — `set_thread_groups({"g0": ["popA"], "g1": ["popB"], ...})` assigns populations to `std::thread` groups; inter-group spikes travel through per-synapse SPSC ring buffers (sized to the synaptic delay) with two-counter step synchronization; serial path completely unmodified when no groups are set. Plasticity (STDP/STP), SymPy CUSTOM_EXPR gates, and intracellular dynamics all work correctly under Phase-2 parallelism. Benchmarks: `examples/figs/benchmark_threading_*.png` (serial vs Phase-2 vs NumPy across 4 topologies); `benchmarks/figures/parallel_ctxbgth_*.png` (CTX-BG-TH scaling).

**Key outcomes:** Phase-1 OpenMP pool parallelism; Phase-2 delay-decomposition threading with 1.5–3× additional speedup for N>1000; zero overhead on serial path; SpikeTransport abstraction ready for task17 CUDA backend; 50–250× NumPy/C++ speedup maintained.

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

1. **Data over code**: neuron and synapse models are described by parameter structs, not class hierarchies. This enables Eigen vectorization across neuron pools and preserves composability. Equations within those structs are expressed as SymPy expressions — compiled to Eigen-compatible C++ (CPU via `EigenPrinter` + JIT, or `__device__` functions for CUDA in task17) with standard forms recognized via structural pattern matching at zero compilation cost.

2. **One canonical workflow**: users should reach any simulation through the same sequence — define model → build network → simulate → analyze. Internal routing (descriptor path, pool dispatch, backend selection) is invisible.

3. **Performance by default**: the hot loop should be zero-allocation and SIMD-vectorized for all supported neuron types. Accuracy is not traded for speed; the polynomial fast-exp and dense-vs-descriptor routing are opt-in or automatically selected.

4. **Biological accuracy first**: preset parameters and synapse kinetics match published values. Convenience should never compromise fidelity.

5. **Extensibility without recompilation**: custom neuron models, synapse kinetics, and intracellular dynamics are expressible through Python-level spec objects without writing C++.

---

---

### Phase 6 — Multi-Compartment Neuron Models

Full spatial neuron models with per-compartment gates, channels, and intracellular dynamics. The Hines (1984) algorithm provides O(C) cable-equation solves for arbitrary tree topologies. Point-neuron and multi-compartment populations coexist in the same `RegionalNetwork`. Morphologies can be defined programmatically or imported from SWC files (NeuroMorpho.org format).

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task21](../task21.md) | `CompartmentSpec` + `MorphologySpec` Data Structures | 3 | task14 (IntracellularSpec) |
| [task22](../task22.md) | Hines Cable Solver + `MCPassivePool` | 3 | task21 |
| [task23](../task23.md) | `MCPool` — Active Multi-Compartment Pool | 3 | task22, task14 |
| [task24](../task24.md) | Python API + `RegionalNetwork` Integration + Recording | 3 | task23, task21 |
| [task25](../task25.md) | SWC Morphology Import | 3 | task24 |

**Key outcomes:** backpropagating action potentials; compartment-specific channel complements (somatic Na, dendritic Ca); per-compartment intracellular substance tracking; Rall equivalent-cylinder reduction; SWC import for NeuroMorpho.org reconstructions; recording shape `(N, C, T)` for compartment-resolved voltage traces.

---

## Known Limitations (Deferred)

- **RK45 adaptive integration**: enum exists, implementation incomplete. Fixed dt (0.01 ms) is sufficient for current use cases.
- **Multi-GPU / distributed simulation**: planned for task19 (depends on task17 Device API + task16 SpikeTransport). Not out of scope — deferred until single-GPU path is stable.
- **Non-conductance-based neuron models** (LIF, AdEx): not planned; use Izhikevich as the lightweight alternative.
- **Python-defined neuron subclasses** (pybind11 trampolines): possible but carries performance penalty; composable spec system is the preferred extension mechanism.
