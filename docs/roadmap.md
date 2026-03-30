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

### Phase 2 — API Clarity and Model Expressiveness (Current)

Improve the user-facing API and extend the biological modeling capabilities. These tasks are largely independent and can proceed in parallel.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task17](../task17.md) | API Streamlining | 2 | None |
| [task16](../task16.md) | Better Equation Support | 2 | None |
| [task12](../task12.md) | Generalized Intracellular Dynamics | 2 | None |

**Recommended order:** task17 first (clears the API surface), task16 and task12 in parallel (both extend existing internal systems without user-facing conflicts).

**Key outcomes:** One canonical workflow for all simulations, named equation parameters, arbitrary intracellular substances (dopamine, cAMP, etc.).

---

### Phase 3 — Synaptic Plasticity (After Phase 2)

Synaptic learning rules that depend on a clean API (task17) and optionally on intracellular substances (task12) for neuromodulator-gated plasticity.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task13](../task13.md) | Plasticity Support (STDP, STP) | 2 | task12 (for gated plasticity) |

**Key outcomes:** STDP, short-term depression/facilitation, dopamine-gated STDP, weight recording.

---

### Phase 4 — Performance Scaling (Parallel with Phase 3)

Infrastructure for simulating networks with >1000 neurons efficiently. These are largely independent of Phase 2/3 feature work and can be developed in parallel.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task15](../task15.md) | CPU Parallelism (OpenMP) | 3 | None |
| [task14](../task14.md) | CUDA GPU Acceleration | 3 | task15 (PoolBase interface) |

**Recommended order:** task15 OpenMP phase first (introduces PoolBase abstraction needed by CUDA), then task14.

**Key outcomes:** 2–4× speedup via OpenMP on multi-core CPU, 50–200× speedup via CUDA for N>1000.

---

### Phase 5 — Ecosystem (Parallel with Phases 3–4)

Documentation and community tooling. This can proceed independently as features stabilise.

| Task | Title | Priority | Dependencies |
|------|-------|----------|--------------|
| [task18](../task18.md) | Web Documentation | 3 | Stable API (task17) |

**Key outcomes:** MkDocs site on GitHub Pages, full API reference, tutorial notebooks, contributor guide.

---

## Architectural Principles

These principles guide all design decisions across phases:

1. **Data over code**: neuron and synapse models are described by parameter structs, not class hierarchies. This enables Eigen vectorization across neuron pools and preserves composability.

2. **One canonical workflow**: users should reach any simulation through the same sequence — define model → build network → simulate → analyze. Internal routing (descriptor path, pool dispatch, backend selection) is invisible.

3. **Performance by default**: the hot loop should be zero-allocation and SIMD-vectorized for all supported neuron types. Accuracy is not traded for speed; the polynomial fast-exp and dense-vs-descriptor routing are opt-in or automatically selected.

4. **Biological accuracy first**: preset parameters and synapse kinetics match published values. Convenience should never compromise fidelity.

5. **Extensibility without recompilation**: custom neuron models, synapse kinetics, and intracellular dynamics are expressible through Python-level spec objects without writing C++.

---

## Known Limitations (Deferred)

- **RK45 adaptive integration**: enum exists, implementation incomplete. Fixed dt (0.01 ms) is sufficient for current use cases.
- **Multi-GPU / distributed simulation**: out of scope for all current tasks.
- **Non-conductance-based neuron models** (LIF, AdEx): not planned; use Izhikevich as the lightweight alternative.
- **Python-defined neuron subclasses** (pybind11 trampolines): possible but carries performance penalty; composable spec system is the preferred extension mechanism.
