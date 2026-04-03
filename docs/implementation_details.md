# Implementation Details

This document is the primary architectural reference for the project. It describes the current state of every source file, the key design decisions and trade-offs, how data flows through the system, and what changes are required to implement each planned feature. It is intended to allow an AI assistant or returning developer to understand the full picture quickly after a context reset.

---

## Repository Layout

```
Masters Project/
├── src/
│   ├── cpp/
│   │   ├── include/hodgkin_huxley/
│   │   │   ├── model/               ← spec + kinetics headers (split from ion_channels.hpp)
│   │   │   │   ├── gate_spec.hpp
│   │   │   │   ├── channel_spec.hpp
│   │   │   │   ├── synapse_spec.hpp
│   │   │   │   ├── neuron_spec.hpp
│   │   │   │   └── kinetics.hpp     ← single authoritative math (scalar + vec + fast_exp)
│   │   │   ├── pool/
│   │   │   │   └── pool_base.hpp    ← abstract pool interface (PoolBase)
│   │   │   ├── network/
│   │   │   │   └── pool_manager.hpp ← owns HHPool + IzPool + ComposablePools
│   │   │   └── [other existing headers]
│   │   └── src/
│   │       ├── network/
│   │       │   └── pool_manager.cpp
│   │       └── [other existing .cpp files]
│   ├── python/
│   │   └── bindings.cpp              ← pybind11 glue layer
│   └── hodgkin_huxley/               ← Python package
│       ├── __init__.py               ← public API re-exports
│       ├── recording.py              ← RecordingConfig, MetricsResult
│       ├── spectral.py               ← multitaper spectrum, beta power
│       ├── pulse.py                  ← PulseStimulator
│       └── noise.py                  ← NoiseInjector
├── tests/python/                     ← pytest test suite (15 files)
│   └── neuron_specs.py              ← circuit-specific NeuronModelSpec builders
├── benchmarks/                       ← CTX-BG-TH model + comparison scripts
├── examples/                         ← usage examples
├── completed/                        ← finished task files (task1–11)
├── docs/                             ← design documentation (this folder)
├── CMakeLists.txt
└── pyproject.toml
```

---

## C++ File Reference

### `neuron_base.hpp` / `neuron_base.cpp`

**Role:** Abstract base class for all neuron types.

**Key interface:**
```cpp
class NeuronBase {
    virtual double membrane_potential() const = 0;
    virtual void   set_membrane_potential(double V) = 0;
    virtual void   reset() = 0;
    virtual void   step(double dt, double I_ext) = 0;
    virtual std::string type_name() const = 0;
};
```

**Integration:** `IntegrationMethod` enum (EULER, RK4, RK45_ADAPTIVE) is stored here. RK45_ADAPTIVE is defined but the adaptive step-size loop is not fully implemented — the fixed dt path (RK4) is used for all current simulations.

**Note:** pybind11 trampoline support is present in the header (for Python-defined subclasses), but is not exercised; composable specs are the preferred extension mechanism.

---

### `neuron.hpp` / `neuron.cpp` (HHNeuron)

**Role:** Classic 4-state Hodgkin-Huxley neuron (V, m, h, n). Squid giant axon default parameters.

**Key internals:**
- `euler_step()` and `rk4_step()` are the integration methods
- `alpha_m()`, `beta_m()`, etc. are the standard HH rate functions
- `m_inf()`, `h_inf()`, `n_inf()` provide steady-state values for initialization

**Usage context:** Primarily a legacy/reference model; new models use `ComposableNeuron` with `NeuronModelSpec`. `HHPool` handles vectorized simulation of HH populations.

---

### `izhikevich.hpp` / `izhikevich.cpp`

**Role:** Reduced 2-state spiking neuron (v, u recovery variable).

**Parameters:** `a, b, c, d` with five presets (REGULAR_SPIKING, FAST_SPIKING, INTRINSICALLY_BURSTING, CHATTERING, LOW_THRESHOLD_SPIKING) plus CUSTOM.

**Integration:** Always Euler — the spike reset `if v >= 30: v=c, u+=d` is discontinuous and incompatible with RK4.

**Usage:** CTX_e (RS) and CTX_i (FS) populations in the CTX-BG-TH benchmark.

---

### `ion_channels.hpp` (Umbrella re-export header)

**Role:** Thin backwards-compatible umbrella. Re-exports all types and math functions from the `model/` sub-headers so that existing `#include "ion_channels.hpp"` sites continue to compile without change.

**Content:** `#include model/gate_spec.hpp`, `channel_spec.hpp`, `synapse_spec.hpp`, `neuron_spec.hpp`, `kinetics.hpp`

---

### `model/gate_spec.hpp` · `model/channel_spec.hpp` · `model/synapse_spec.hpp` · `model/neuron_spec.hpp`

**Role:** Pure data structs — no computation. Split out of the former monolithic `ion_channels.hpp`.

| Header | Types |
|--------|-------|
| `gate_spec.hpp` | `BoltzmannParams`, `TauParams` (6 forms), `RateFuncParams` (4 forms), `GateSpec`, `CalciumSpec` |
| `channel_spec.hpp` | `ChannelSpec` |
| `synapse_spec.hpp` | `KineticSynapseSpec` + presets (`gaba_kinetic`, `nmda_kinetic`, `gaba_b`) |
| `neuron_spec.hpp` | `NeuronModelSpec` + static presets (`hh_default`, `izhikevich`) |

**Presets** on `NeuronModelSpec`: `hh_default()`, `izhikevich(Type)`, `izhikevich(Parameters)`. Circuit-specific presets (thalamic, STN, GPe, GPi, striatum) live in `tests/python/neuron_specs.py` and `benchmarks/ctxbgth_model.py`.

---

### `model/kinetics.hpp`

**Role:** Single authoritative implementation of all kinetic math. Eliminates the former 4-site duplication (scalar in ComposableNeuron, vectorized in ComposablePool, free functions in ion_channels.hpp, inline in network.cpp synapse update).

**Functions:**

| Function | Form |
|----------|------|
| `boltzmann_scalar(x, BoltzmannParams)` | scalar sigmoid |
| `compute_tau_scalar(V, TauParams)` | scalar, 6 forms |
| `compute_rate_scalar(V, RateFuncParams)` | scalar, 4 forms |
| `fast_exp(src, dst, tmp_r)` | range-reduction + degree-7 Taylor + 5 squarings; `src==dst` safe |
| `boltzmann_vec(x, BoltzmannParams)` | vectorized (Eigen::ArrayXd) |
| `compute_tau_vec(V, TauParams, tmp)` | vectorized, 6 forms |
| `compute_rate_vec(V, RateFuncParams, tmp)` | vectorized, 4 forms |

**Insertion point (task13):** Swap this header with a generated `kinetics_sympy.hpp` that overloads the same function names using SymPy-codegen'd expressions.

**Future change (task14):** `CalciumSpec` will be replaced by `std::vector<IntracellularSpec>` to support arbitrary intracellular substances. Calcium becomes a special case.

---

### `composable_neuron.hpp` / `composable_neuron.cpp`

**Role:** Single-neuron instantiation of a `NeuronModelSpec`. Maintains gate states, calcium concentration, and reversal potential. Used for API access and single-neuron operations; hot-loop simulation uses `ComposablePool`.

**Key methods:**
- `reset_gates_to_steady_state()`: initialize gates to x_inf(V_rest)
- `set_gate_states(vector<double>)` / `set_calcium(double)`: manual state override
- `step(dt, I_ext, I_syn)`: delegate to composable step logic

---

### `composable_pool.hpp` / `composable_pool.cpp`

**Role:** Vectorized (Eigen) batch step for a population of neurons sharing the same `NeuronModelSpec`. Derives from `PoolBase`.

**State layout (SoA):**
```
Eigen::ArrayXd V_;
vector<Eigen::ArrayXd> gate_states_;   // one per gate in model
Eigen::ArrayXd Ca_;                    // if CalciumSpec::enabled
Eigen::ArrayXd E_Ca_;                  // if CalciumSpec::use_nernst
```

**Pre-allocated working buffers:** `I_total_`, `tmp_`, `tmp2_`, `tmp_exp_r_` — no heap allocation in hot loop.

**step() algorithm:**
1. For each gate: evaluate `x_inf(V)` and `tau_x(V)` via `boltzmann_vec` / `compute_tau_vec` / `compute_rate_vec` (from `model/kinetics.hpp`)
2. Update gate states (INF_TAU / ALPHA_BETA / INSTANT / DERIVED)
3. For each channel: compute `g * gate_product * (V - E_rev)`, accumulate I_total
4. `V += dt * (-I_total + I_ext) / C_m`
5. Update calcium if enabled; recompute E_Ca if Nernst

**Math:** Calls `hodgkin_huxley::fast_exp(src, dst, tmp_exp_r_)` (free function from `kinetics.hpp`) via thin `fast_exp()` member wrapper.

**Future change (task14):** Replace `Ca_` / `E_Ca_` with `vector<ArrayXd> X_` and `vector<ArrayXd> E_nernst_` for N substances.

---

### `hh_pool.hpp` / `hh_pool.cpp`

**Role:** Eigen-vectorized batch step for pure HH populations (fixed 4-state model). Derives from `PoolBase`.

**State:** `V_, m_, h_, n_` (Eigen::ArrayXd)

**Performance:** Pre-allocated k-vectors for RK4 stages; polynomial fast-exp (7th-degree Taylor + range reduction, ~8 digits precision, ~2× faster than std::exp) via `fast_exp()` member wrapper → `hodgkin_huxley::fast_exp()` free function in `model/kinetics.hpp`. The `fast_math_` flag in `Network` controls this.

**PoolBase interface:** `step(dt)` delegates to `step_rk4(dt)`.

---

### `iz_pool.hpp` / `iz_pool.cpp`

**Role:** Eigen-vectorized batch step for Izhikevich populations.

**State:** `v_, u_` (Eigen::ArrayXd). Spike reset vectorized using Eigen masked operations.

**PoolBase interface:** `step(dt)` delegates to `step_euler(dt)`.

**Critical gotcha (documented in MEMORY.md):** The spike reset `v_ = (v_ >= 30).select(c, v_)` must materialize the boolean mask with `.eval()` before modifying `v_`, otherwise the lazy expression reads the modified values:
```cpp
auto fired = (v_ >= 30.0).eval();  // materialize first
v_ = fired.select(params_.c, v_);
u_ = fired.select(u_ + params_.d, u_);
```

---

### `pool/pool_base.hpp`

**Role:** Abstract interface shared by `HHPool`, `IzPool`, and `ComposablePool`.

**Key virtual methods:**
```cpp
void scatter_voltages(double* V_buf) const   // pool state → global V cache
void gather_currents(const double* I_buf)    // global I buffer → pool I_ext
void step(double dt)                          // HH→step_rk4, Iz→step_euler, Composable→step
void sync_to_neurons(vector<unique_ptr<NeuronBase>>&)  // pool → API objects
// Recording (default no-ops):
void scatter_gate_states_into(...)
void scatter_calcium_into(...)
void scatter_recoveries(...)
```

**Insertion point (task16):** Add `virtual void step_parallel(double dt)` and override in each pool with `#pragma omp parallel for` loops.

**Insertion point (task17):** Add CUDA pool subclasses (`CudaComposablePool`) — override `step(dt)` with CUDA kernel launch.

---

### `network/pool_manager.hpp` / `src/network/pool_manager.cpp`

**Role:** Owns `HHPool`, `IzPool`, and one `ComposablePool` per model spec name. Provides a uniform hot-loop interface and manages pool lifetime across `simulate()` calls.

**Key method — `build_from_neurons(neurons, fast_math)`:**
- Dynamic-casts to classify neurons by type
- Constructs/resets pools sized to capacity
- Populates from API neuron state (`parameters()`, `state()`, `gate_states()`, `calcium()`)
- Called at most once between `reset()` / `add_neuron()` calls — `pools_dirty_` flag gates it

**Hot-loop delegates:** `scatter_all_voltages`, `gather_all_currents`, `step_all`, `sync_all_to_neurons`, `scatter_gates`, `scatter_calcium`, `scatter_recoveries`

---

### `synapse_base.hpp` / `synapse_base.cpp`

**Role:** Abstract base for the polymorphic synapse objects kept for API access. Not used in the hot loop — the SoA in `SynArrays` is the hot-path source of truth.

**Spike detection:** rising-edge threshold crossing (`V_prev <= thresh < V_curr`). Per-synapse `V_pre_prev` field in SoA.

---

### `synapse.hpp`

**Role:** Concrete synapse types and the `KineticSynapseSpec` definition.

| Type | Model | Parameters |
|------|-------|------------|
| `ExponentialSynapse` | `g *= exp(-dt/tau)` on each step, `g += weight` on spike | `tau` |
| `AlphaSynapse` | Coupled ODE: `dx/dt=-x/tau, dg/dt=(x-g)/tau` | `tau` |
| `DoubleExponentialSynapse` | Separate rise/decay: normalized peak = weight | `tau_rise, tau_decay` |
| `KineticSynapse` | Gating variable S with ALPHA_BETA, TANH_GATE, or BOLTZMANN_GATE dynamics | `KineticSynapseSpec` |

Receptor presets (on `Network`): `add_ampa_synapse()`, `add_nmda_synapse()`, `add_gaba_a_synapse()` — use double-exp with biologically accurate kinetics.

---

### `network.hpp` / `network.cpp`

**Role:** Core simulation class. Manages polymorphic neurons + SoA synapse data + Eigen pools.

**Data layout:**

```
neurons_: vector<unique_ptr<NeuronBase>>   // polymorphic, for API access
synapses_: vector<unique_ptr<SynapseBase>> // polymorphic, API-only; lazy-synced from SoA

sa_ (SynArrays):
  Common:     pre, post, weight, E_syn, g, type, V_pre_prev
  Delay:      delay, spike_buf (ring buffer), buf_head, delay_init
  Exp:        exp_tau, exp_decay (cached)
  Alpha:      alpha_x, alpha_inv_tau
  DExp:       dexp_g_rise/decay, tau_rise/decay, rise_decay/fall_decay (cached), dexp_norm
  Kinetic:    kin_S, kin_spec_idx → kinetic_specs_

syn_groups_: type-separated index lists (exp, alpha, dexp, kinetic)
```

**Simulation entry points:**
- `simulate_into_buffers()`: dense I_ext path; accepts 2D input matrix
- `simulate_with_descriptors()`: compact StimPlan path (preferred for scalar/pulse/DBS)
- `step()`: single time step (used by `RegionalNetwork.simulate()` for Python-managed loops)

**Pool management (`PoolManager pool_mgr_`):**
Pools are held as a `Network` member and reused across `simulate()` calls. A `pools_dirty_` flag controls rebuild:
- `pools_dirty_ = true` after `add_neuron()`, `reset()`, or at construction
- At start of `simulate_into_buffers` / `simulate_with_descriptors`: if dirty, call `pool_mgr_.build_from_neurons(neurons_, fast_math_)`; clear flag
- Consecutive `simulate()` calls with no intervening `reset()` continue from pool state at end of prior run (task20 continuable simulation insertion point)

**Hot-loop internal sequence per time step:**
1. `pool_mgr_.scatter_all_voltages(V_cache_)` — pool state → V_cache_
2. Recording block (if `t % interval == 0`): V, gates, calcium, u, g_syn via `pool_mgr_.*`
3. Seed `I_syn_buffer_` from I_ext (dense or descriptor)
4. Accumulate synaptic currents: `I_buf[post[i]] += g[i] * (E_syn[i] - V[post[i]])`
5. `pool_mgr_.gather_all_currents(I_syn_buffer_)` → pools
6. `pool_mgr_.step_all(dt)` — HH: RK4, Iz: Euler, Composable: Euler
7. `pool_mgr_.scatter_all_voltages(V_cache_)` — re-scatter for spike detection
8. `update_synapses_grouped(dt)` — spike detection + type-separated kinetics

**Lazy sync:** `SynArrays` (SoA) is the authoritative state during simulation. `SynapseBase` objects are synced lazily on `synapse(idx)` access (`soa_dirty_` flag).

**Insertion point (task16):** `pool_mgr_.step_all(dt)` — replace with parallel version adding `#pragma omp parallel for` inside each pool's step.

**Insertion point (task17):** Swap `HHPool` / `ComposablePool` for CUDA variants in `PoolManager::build_from_neurons()` when a CUDA device is selected.

**Insertion point (task20):** `pools_dirty_` + `PoolManager` persistence already enables continuable simulation. Add `get_state()` / `set_state()` wrappers over `pool_mgr_` + `sa_` for full pickle support.

---

### `regional_network.hpp` / `regional_network.cpp`

**Role:** Population-level wrapper around `Network`. Named populations with bulk connectivity.

**Population bookkeeping:**
```cpp
struct Population { string name; size_t start_idx, count; };
vector<Population> populations_;
map<string, size_t> pop_index_;   // name → index in populations_
```

**Connectivity dispatch:** `connect()` routes to `generate_connections()` which implements each `ConnectivityPattern` in C++ with a seeded RNG. For patterns not expressible via presets, `add_connection()` adds one synapse by local population index.

**Stimulation routing (Python layer):** `RegionalNetwork.simulate()` (Python) auto-selects descriptor vs dense path based on whether all I_ext values are scalars and all stimulators are `DBSStimulator`.

**Future change (task12):** `Network` will be prefixed `_Network` and removed from public API; `RegionalNetwork` becomes the sole public network class.

**Future change (task19):** `device_map_` will track per-population device assignment; `simulate()` will dispatch to `MultiDeviceSimContext` when > 1 unique device is present.

---

### `dbs_stimulator.hpp` / `dbs_stimulator.cpp`

**Role:** Periodic DBS pulse train with on-the-fly evaluation.

**Key:** `current_at(step, dt)` evaluates via modulo arithmetic — O(1), zero allocation. `generate(duration, dt)` produces the full waveform (for plotting only). The descriptor path uses `current_at()` internally.

---

## Python Layer Reference

### `src/hodgkin_huxley/__init__.py`

Re-exports all public symbols from `_core` (pybind11 extension) plus Python-native utilities. `__all__` controls what appears on `from hodgkin_huxley import *`. Currently exports ~50 symbols; task12 will trim this to the core workflow set.

### `recording.py`

**`RecordingConfig`**: specifies which metrics to record. Preset constructors: `voltage_only()`, `spikes_only()`, `all_neuron_metrics()`, `for_population(name)`. Internally maps metric names to buffer allocation decisions.

**`MetricsResult`**: dict-like container. Stores time axis, V traces, gate states, spike times, firing rates, ISI stats. `summary()` prints human-readable table.

**`PopulationMetricsResult`**: multiplexed result for `RegionalNetwork.simulate()`. Indexed by population name; slices per-neuron metrics by population index range.

**`_StimPlan`**: Python-side compact stimulation descriptor. Passed to `_simulate_with_descriptors()`. Created automatically by `RegionalNetwork.simulate()` when all inputs are scalars or `DBSStimulator` objects.

### `spectral.py`

**`mtspectrumpt(spike_times_list, duration, Fs, fpass, tapers)`**: Chronux-compatible multitaper point-process spectrum. Matches benchmark's `make_Spectrum()` exactly. DPSS tapers, FFT-based NUFFT, bias correction. Critical for GPi beta-band validation.

**`analyze_beta_power(result)`**: convenience wrapper — reads spike times from `MetricsResult`, converts ms→s, runs mtspectrumpt, integrates 7–35 Hz band.

### `pulse.py` — `PulseStimulator`

Rectangular or biphasic pulses. Constructors: `single()`, `train()`, `burst()`, `from_onsets()`. Biphasic: cathodic + anodic phase with configurable interphase gap. `apply_to(base, ...)` adds pulse to a base current array.

### `noise.py` — `NoiseInjector`

Adapts pre-generated noise arrays (from any source) into the I_ext interface. Supports `(n_steps,)` (broadcast) or `(n_neurons, n_steps)` (per-neuron). No noise generation is done here — bring your own noise.

---

## Key Architectural Patterns

### 1. Data-Driven Model Specification

All neuron and synapse models are described by **parameter structs**, not class hierarchies. A `NeuronModelSpec` is pure data — a list of `GateSpec`, `ChannelSpec`, and `CalciumSpec` objects with numeric parameters. This has two benefits:

- The same struct drives both the scalar `ComposableNeuron` and the vectorized `ComposablePool` — no code duplication, single source of truth
- The pool can vectorize across N neurons simultaneously (all share the same template; only V and state differ per neuron)

### 2. Structure-of-Arrays (SoA) Synapses

Synapse data is stored as flat parallel arrays (not a vector of objects). A "synapse" is just an index. This layout enables:
- SIMD auto-vectorization of the conductance-decay inner loop
- Cache efficiency (all `exp_decay[]` values accessed sequentially)
- Branch-free update loops (type-separated `syn_groups_`)

The polymorphic `SynapseBase` objects exist only for API convenience (inspecting individual synapses). They are lazy-synced from SoA on demand.

### 3. Eigen SIMD Pools

Each neuron type gets a dedicated pool class (`HHPool`, `IzPool`, `ComposablePool`) that stores all state as `Eigen::ArrayXd` and computes updates with Eigen expressions. Key properties:
- Entire population updated in one vectorized call
- Pre-allocated working buffers: no heap allocation in hot loop
- Polynomial fast-exp for ~2× speedup on exp-heavy HH computations

### 4. StimPlan Compact Descriptors

For typical simulations (constant baseline + DBS or rectangular pulses), the dense `(n_neurons × n_steps)` I_ext matrix is replaced by a `StimPlan` with three components:
- `I_const`: per-neuron scalar baseline (N floats)
- `pulses`: list of rectangular events with onset/end steps
- `dbs`: list of periodic pulse trains evaluated via modulo arithmetic

This eliminates the 128 MB allocation that was the dominant memory cost for the CTX-BG-TH benchmark at dt=0.01 ms.

### 5. Lazy SoA Sync

During simulation, SoA is the ground truth. The polymorphic synapse objects are not updated on every step. The `soa_dirty_` flag is set after simulation; accessing `network.synapse(idx)` triggers a one-time sync pass. This avoids O(n_synapses) copy overhead on every step.

### 6. Device Model (task17+)

Following PyTorch conventions, compute devices are represented by a `Device` struct:

```cpp
struct Device {
    enum class Type { CPU, CUDA };
    Type type  = Type::CPU;
    int  index = 0;   // CUDA device index
};
```

`Network::to(device)` moves all pool state to the target device. For multi-GPU simulation (task19), `RegionalNetwork::assign(population, device)` maps individual populations to devices; the `SpikeTransport` abstraction (task16) handles inter-device spike delivery without changing the delay-decomposition algorithm.

---

## Data Flow: Simulation

```
Python: net.simulate(duration, dt, I_ext, recording)
  │
  ├─ RegionalNetwork.simulate() [Python]
  │   ├─ Build _StimPlan (or dense I_ext matrix)
  │   ├─ Allocate numpy recording buffers
  │   └─ Call _simulate_with_descriptors() [C++]
  │
  └─ Network::simulate_with_descriptors() [C++]
      ├─ ensure_buffers()         → allocate I_syn_buffer_, V_cache_
      ├─ build_synapse_groups()   → populate syn_groups_ (exp/alpha/dexp/kin)
      ├─ update_decay_factors(dt) → cache exp(-dt/tau) per synapse
      └─ Hot loop (n_steps iterations):
          ├─ Evaluate I_ext from StimPlan (O(n_neurons))
          ├─ cache_voltages()     → V_cache_ = current V
          ├─ compute_synaptic_currents() → I_syn from SoA g[]
          ├─ Pool.step(dt, I_ext, I_syn) → update V, gates, calcium
          ├─ update_synapses_grouped(dt) → update g for all synapse types
          ├─ Spike detection     → threshold crossing on V_cache_ vs V
          └─ Record              → fill V_buf, gate_buf, spike_event_buf etc.
```

---

## Known Limitations

| Issue | Impact | Resolution |
|-------|--------|------------|
| `TauParams` uses positional `double params[8]` | Poor readability | task13: SymPy equation system |
| `CalciumSpec` supports only calcium | Cannot model dopamine, cAMP etc. | task14: generalized intracellular |
| Multiple overlapping neuron-add APIs | Confusing entry points | task12: API streamlining |
| Single-threaded hot loop | Limits large-N performance | task16: OpenMP |
| No GPU support | Limits N>5000 simulations | task17: CUDA |
| No multi-GPU support | Limits extremely large models (N>50,000) | task19: Multi-GPU (CUDA P2P / NCCL) |
| RK45 adaptive integration incomplete | Stiff systems require tiny fixed dt | Low priority; RK4 at dt=0.01 ms is sufficient |
| No plasticity | Weights are fixed at init | task15: STDP, STP |
| `simulate()` resets state each call | Cannot continue simulation across calls | task20: continuable simulations |
| No pickle / deepcopy support | Blocks multiprocessing parameter sweeps | task20: `__getstate__` / `__setstate__` |
| No ML tensor export | Manual conversion required for PyTorch/TF | task20: `to_torch()`, `to_tensorflow()`, sparse spike tensors |
| No web documentation | Onboarding friction | task18: MkDocs site |

---

## Extension Points

### Adding a New Neuron Type

1. Define a `NeuronModelSpec` via gate/channel specs (no C++ code required for most cases)
2. If the model requires new gate update forms: add a new variant to `TauParams::Form` or `GateSpec::UpdateForm` in `ion_channels.hpp`, implement in `composable_pool.cpp`
3. Add a static preset factory to `NeuronModelSpec` in `ion_channels.hpp`
4. Bind the factory in `bindings.cpp`
5. Add tests in `tests/python/test_composable_neuron.py`

### Adding a New Synapse Type

1. Add a new `SynType` variant to the `SynType` enum in `network.hpp`
2. Extend `SynArrays` with the new type's state fields
3. Implement the update rule in `Network::update_synapses_grouped()` within a new type-specific loop
4. Add `add_XSynapse()` method to `Network` and `RegionalNetwork`
5. Add to `syn_groups_` in `Network::build_synapse_groups()`
6. Bind in `bindings.cpp` and test

### Adding a New Intracellular Substance (after task14)

1. Create an `IntracellularSpec` with the substance dynamics and modulation targets
2. Add it to the `NeuronModelSpec` via `model.add_intracellular(spec)`
3. `ComposablePool` will automatically allocate a state array and update it per step
4. Gates referencing the substance use `GateDependency::INTRACELLULAR` with the correct `intracellular_idx`
