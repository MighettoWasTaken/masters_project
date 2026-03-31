# Implementation Details

This document is the primary architectural reference for the project. It describes the current state of every source file, the key design decisions and trade-offs, how data flows through the system, and what changes are required to implement each planned feature. It is intended to allow an AI assistant or returning developer to understand the full picture quickly after a context reset.

---

## Repository Layout

```
Masters Project/
├── src/
│   ├── cpp/
│   │   ├── include/hodgkin_huxley/   ← C++ headers (public interface)
│   │   └── src/                      ← C++ implementation files
│   ├── python/
│   │   └── bindings.cpp              ← pybind11 glue layer
│   └── hodgkin_huxley/               ← Python package
│       ├── __init__.py               ← public API re-exports
│       ├── recording.py              ← RecordingConfig, MetricsResult
│       ├── spectral.py               ← multitaper spectrum, beta power
│       ├── pulse.py                  ← PulseStimulator
│       └── noise.py                  ← NoiseInjector
├── tests/python/                     ← pytest test suite (15 files)
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

### `ion_channels.hpp` (Composable System Specs)

**Role:** Data structs defining parameterized gate kinetics and ion channels. These are **pure data** — no computation happens here. Computation is in the pools.

**Key types:**

| Type | Purpose |
|------|---------|
| `BoltzmannParams` | Steady-state x_inf: `1/(1+exp(-(V-v_half)/k))` |
| `TauParams` | Time constant tau_x: 6 forms (CONSTANT, BOLTZMANN, DOUBLE_EXP_SUM, OFFSET_DOUBLE_EXP, SCALED_EXP, COMPOUND_AB) |
| `RateFuncParams` | Alpha/beta rate functions: 4 forms (LINEAR_OVER_EXP, EXP_DECAY, LINEAR_OVER_EXPM1, SIGMOID) |
| `GateSpec` | One gating variable: update form (INF_TAU / ALPHA_BETA / INSTANT / DERIVED), dependency (VOLTAGE / CALCIUM), kinetic params |
| `ChannelSpec` | One ion channel: maximal conductance, reversal potential, gate power list, AHP flag |
| `CalciumSpec` | Calcium dynamics: simple decay or Nernst mode, source channel list |
| `NeuronModelSpec` | Complete model: C_m, list of GateSpecs, list of ChannelSpecs, CalciumSpec |

**Presets** (static factories on `NeuronModelSpec`): `thalamic()`, `stn()`, `gpe()`, `gpi()`, `striatum(pd_factor)`.

**Future change (task13):** `TauParams` and `RateFuncParams` will switch from positional `params[8]` arrays to SymPy expressions. Legacy types move to `hodgkin_huxley.legacy`.

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

**Role:** Vectorized (Eigen) batch step for a population of neurons sharing the same `NeuronModelSpec`.

**State layout (SoA):**
```
Eigen::ArrayXd V_;
vector<Eigen::ArrayXd> gate_states_;   // one per gate in model
Eigen::ArrayXd Ca_;                    // if CalciumSpec::enabled
Eigen::ArrayXd E_Ca_;                  // if CalciumSpec::use_nernst
```

**Pre-allocated working buffers:** `I_total_`, `inf_cache_`, `tau_cache_` — no heap allocation in hot loop.

**step() algorithm:**
1. For each gate: evaluate `x_inf(V)` and `tau_x(V)` via vectorized Eigen expressions
2. Update gate states (INF_TAU / ALPHA_BETA / INSTANT / DERIVED)
3. For each channel: compute `g * gate_product * (V - E_rev)`, accumulate I_total
4. `V += dt * (-I_total + I_ext + I_syn) / C_m`
5. Update calcium if enabled; recompute E_Ca if Nernst

**Future change (task14):** Replace `Ca_` / `E_Ca_` with `vector<ArrayXd> X_` and `vector<ArrayXd> E_nernst_` for N substances.

---

### `hh_pool.hpp` / `hh_pool.cpp`

**Role:** Eigen-vectorized batch step for pure HH populations (fixed 4-state model).

**State:** `V_, m_, h_, n_` (Eigen::ArrayXd)

**Performance:** Pre-allocated k-vectors for RK4 stages; polynomial fast-exp (7th-degree Taylor + range reduction, ~8 digits precision, ~2× faster than std::exp). The `fast_math_` flag in `Network` controls this.

**Contiguity optimization:** `step_rk4()` checks if state arrays are contiguous in memory before using vectorized path; falls back to scalar for non-contiguous layouts (rare in practice).

---

### `iz_pool.hpp` / `iz_pool.cpp`

**Role:** Eigen-vectorized batch step for Izhikevich populations.

**State:** `v_, u_` (Eigen::ArrayXd). Spike reset vectorized using Eigen masked operations.

**Critical gotcha (documented in MEMORY.md):** The spike reset `v_ = (v_ >= 30).select(c, v_)` must materialize the boolean mask with `.eval()` before modifying `v_`, otherwise the lazy expression reads the modified values:
```cpp
auto fired = (v_ >= 30.0).eval();  // materialize first
v_ = fired.select(params_.c, v_);
u_ = fired.select(u_ + params_.d, u_);
```

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

**Hot-loop internal sequence per time step:**
1. `cache_voltages()` — copy V to `V_cache_`
2. `compute_synaptic_currents()` — accumulate I_syn from SoA g values
3. Pool `step()` — update each neuron pool (HH, Iz, Composable) with I_ext + I_syn
4. `update_synapses_grouped()` — update g for each synapse type (branch-free, type-grouped)
5. Spike detection via V_cache_ vs current V
6. Record to buffers if this step falls on a recording interval

**Lazy sync:** `SynArrays` (SoA) is the authoritative state during simulation. `SynapseBase` objects are synced lazily on `synapse(idx)` access (`soa_dirty_` flag). Avoids copying conductance values on every step.

**Future change (task16):** `step()` pool loops will gain `#pragma omp parallel for`; I_syn accumulation will need thread-local partial sums or atomics.

**Future change (task17):** `to(Device)` will route pool `step()` to CUDA kernels when `device().type == CUDA`.

**Future change (task20):** `simulate_with_descriptors()` will preserve pool and synapse state across calls (persistent by default). A `SimulationState` struct (`get_state()` / `set_state()` / `reset_state()`) will enable `state_dict()` / `load_state_dict()` and pickle support. Decay factors and synapse groups will be cached via `groups_dirty_` / `decay_dirty_` flags to avoid redundant rebuilds on repeated calls.

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
