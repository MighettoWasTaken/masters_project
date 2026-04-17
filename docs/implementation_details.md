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
│       ├── __init__.py               ← public API re-exports + deprecation shims
│       ├── _equations/               ← NeuronModel + SynapseModel + IntracellularDynamics + Modulation SymPy builders
│       ├── _codegen.py               ← EigenPrinter, VM compiler, JIT cache, symbols
│       ├── _network/                 ← RegionalNetwork Python wrapper
│       ├── legacy.py                 ← deprecated API shims (emits DeprecationWarning)
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
| `gate_spec.hpp` | `BoltzmannParams`, `TauParams` (6 forms), `RateFuncParams` (4 forms), `GateSpec`, `IntracellularSpec`, `IntracellularModulation`, `VmExpr` / `VmOp` (VM bytecode types) |
| `channel_spec.hpp` | `ChannelSpec` |
| `synapse_spec.hpp` | `SynapseSpec` with `UpdateForm` (7 values) and `CurrentForm` (3 values); static factories: `exponential`, `alpha_function`, `double_exponential`, `gaba_kinetic`, `nmda_kinetic`, `gaba_b`, `ampa`, `nmda`, `gaba_a` |
| `neuron_spec.hpp` | `NeuronModelSpec` + static presets (`hh_default`, `izhikevich`) |

**`GateSpec::Dependency`**: `VOLTAGE` or `INTRACELLULAR` (replaces `CALCIUM`). `intracellular_idx` selects which substance (0 = calcium by convention). The `CALCIUM` alias is deprecated Python-only — not present in C++.

**`IntracellularSpec`**: Describes one intracellular substance. Key fields: `name`, `initial`, `UpdateForm` (DECAY / DRIVEN_DECAY / DRIVEN_DECAY_NERNST / CUSTOM_EXPR), scalar params (`epsilon`, `k_decay`, `source_channels`), Nernst params (enabled flag + R/T/z/F/Ca_o constants), `ode_vm` (VM bytecode for CUSTOM_EXPR), `nernst_vm` (VM bytecode for non-standard Nernst), `modulations` (list of `IntracellularModulation`).

**`IntracellularModulation`**: Pairs a `Target` enum (CHANNEL_G, CHANNEL_EREV, GATE_INF_SHIFT, GATE_INF_SCALE, GATE_TAU_SCALE, GATE_INF_EXPR, SYNAPSE_G) with `target_idx`, `substance_idx`, `mod_vm` (VM bytecode), and `shift_scale` (scalar for GATE_INF_SHIFT).

**`ChannelSpec`**: `nernst_substance_idx` (≥0 to use `E_nernst_[idx]` instead of constant `E_rev`; -1 = none) and `ahp_substance_idx` replace the former `use_calcium_nernst` bool. Legacy `use_calcium_nernst` is a deprecated Python property that sets `nernst_substance_idx = 0`.

**VM opcodes**: `PUSH_X = 19` is the opcode for cross-substance references in ODE VMs (`X_[operand]`). `PUSH_S` (17) is reused as "self concentration" inside substance ODE context.

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

**SymPy integration (task13, done):** Gate kinetics generated by SymPy expressions are either pattern-matched to existing `TauParams`/`BoltzmannParams`/`RateFuncParams` forms (zero compilation cost) or JIT-compiled to `.so` via `EigenPrinter` output and loaded at runtime. The kinetics header is unchanged; the generated code calls the same scalar/vectorized functions.

**`vm_eval_substance`** (added task14): Vectorized evaluator for substance ODEs. Accepts `dep` (I_source ArrayXd), `self` (current concentration ArrayXd), and `X_all` (all substance arrays). Dispatches `PUSH_DEP → dep`, `PUSH_S → self`, `PUSH_X(n) → X_all[n]`.

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
vector<Eigen::ArrayXd> X_;             // one per IntracellularSpec (X_[0] = Ca by convention)
vector<Eigen::ArrayXd> E_nernst_;      // one per spec with nernst_enabled
Eigen::ArrayXd synapse_g_scale_;       // per-neuron synaptic g multiplier (SYNAPSE_G mods)
```

**Pre-allocated working buffers:** `I_total_`, `tmp_`, `tmp2_`, `tmp_exp_r_` — no heap allocation in hot loop.

**Pre-allocated modulation scratch arrays** (reset each step, no per-step malloc):
```
vector<Eigen::ArrayXd> mod_ch_g_;       // nc × N, ones — channel g multipliers
vector<Eigen::ArrayXd> mod_ch_E_ovr_;   // nc, empty = not overridden
vector<Eigen::ArrayXd> mod_gate_shift_; // ng × N, zeros — gate V shifts
vector<Eigen::ArrayXd> mod_gate_scale_; // ng × N, ones — gate inf scales
vector<Eigen::ArrayXd> mod_gate_tau_;   // ng × N, ones — gate tau scales
vector<Eigen::ArrayXd> mod_gate_expr_;  // ng, empty = not overridden
vector<Eigen::ArrayXd> ch_E_rev_;       // nc × N, pre-filled at finalize() — zero malloc per step
vector<Eigen::ArrayXd> I_channel_;      // nc × N, cached during channel loop, reused by update_substances
bool has_any_mods_       = false;        // any substance has modulations — gates all mod overhead
bool has_synapse_g_mods_ = false;        // SYNAPSE_G modulation present
```

**step() algorithm:**
1. If `has_any_mods_`: reset scratch arrays, call `apply_modulations()` (writes gate/channel mods from substance state)
2. For each gate: evaluate `x_inf(dep)` and `tau_x(V)` via `boltzmann_vec` / `compute_tau_vec` (dep = V or X_[intracellular_idx]). Apply `mod_gate_scale_`, `mod_gate_tau_`, `mod_gate_expr_` only when `has_any_mods_`.
3. For each channel: `E_rev` resolved from `ch_E_rev_[ci]` (pre-filled constant) or `E_nernst_[idx]` (Nernst) or `mod_ch_E_ovr_[ci]` (modulation override). Write `I_channel_[ci] = g * [mod_ch_g_] * gate_prod * (V - E_rev)`. Accumulate `I_total_`.
4. `V += dt * (-I_total_ + I_ext) / C_m`
5. `update_substances(dt)`: for each `IntracellularSpec`, sum `I_channel_[src]` into `tmp2_` (I_src reuse), compute dX into `tmp_`, update `X_[i]`, clamp to 0. Nernst update reuses `tmp2_` as scratch.
6. If `has_synapse_g_mods_`: scatter `synapse_g_scale_` (written by `apply_modulations`)

**Math:** Calls `hodgkin_huxley::fast_exp(src, dst, tmp_exp_r_)` via thin `fast_exp()` member wrapper.

**Performance notes:** `has_any_mods_ = false` on all standard benchmark populations — gate/channel mod multiplications are completely skipped. Channel E_rev pre-allocated in `ch_E_rev_[ci]` eliminates `Eigen::ArrayXd::Constant(N_, E_rev)` malloc per channel per step. `I_channel_` eliminates gate-product recomputation in `update_substances`.

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
void scatter_substance_into(size_t subst_idx, double* buf, size_t n_rec, size_t t_rec)
void scatter_synapse_g_scale(double* buf) const
void scatter_recoveries(...)
```
`scatter_calcium_into` is a deprecated wrapper that delegates to `scatter_substance_into(0, ...)`. `HHPool` and `IzPool` inherit no-op defaults for all substance/synapse-g methods.

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

**Hot-loop delegates:** `scatter_all_voltages`, `gather_all_currents`, `step_all`, `sync_all_to_neurons`, `scatter_gates`, `scatter_calcium`, `scatter_recoveries`, `scatter_substances`, `scatter_synapse_g_scale`

**`has_synapse_g_mods_`** flag: set after `build_from_neurons()` by scanning `comp_pools_`. Exposed via `has_synapse_g_mods()` accessor. Used by `network.cpp` to gate the fill/scatter/multiply machinery for `synapse_g_scale_`.

---

### `synapse_base.hpp` / `synapse_base.cpp`

**Role:** Lightweight, non-virtual view of a single synapse stored in `SynArrays`. No virtual dispatch, no per-object heap allocation. Holds an `(idx, const Network*)` pair and reads state directly from SoA arrays on demand.

**Interface:** `conductance()`, `reversal_potential()`, `weight()`, `pre_idx()`, `post_idx()`, `delay()`, `type_name()`, `spec()`. All methods are const and read from `net_->syn_arrays()` — always in sync with SoA, no lazy sync flag needed.

**Storage in Network:** `std::vector<SynapseBase> synapses_` (value objects, not `unique_ptr`).

**Spike detection:** rising-edge threshold crossing (`V_pre_prev <= thresh < V_curr`). Per-synapse `V_pre_prev` field in `SynArrays`.

---

### `model/synapse_spec.hpp` / `src/synapse_spec.cpp`

**Role:** Unified `SynapseSpec` struct — pure data, no computation. All update logic lives in `Network::update_synapses_grouped()`.

**Discriminants:**

| `UpdateForm` | Model |
|---|---|
| `EXP_DECAY` | Spike-driven: `S += delta_S` on spike; `S *= exp(-dt/tau_S)` each step |
| `ALPHA_FUNC` | Spike-driven: `dS/dt=(A-S)/tau_A`, `dA/dt=-A/tau_A`; `A += delta_A` on spike (Euler) |
| `DOUBLE_EXP` | Spike-driven: `S *= exp(-dt/tau_S)`, `A *= exp(-dt/tau_A)`; `g = norm*(S-A)` |
| `TANH_GATE` | Voltage-gated: `dS/dt = tanh_amp*(1+tanh((V-vh)/k))*(1-S) - S/tau_decay` |
| `BOLTZMANN_GATE` | Voltage-gated: `dS/dt = (S_inf(V) - S) / tau(V)` |
| `ALPHA_BETA` | Voltage-gated: `dS/dt = alpha(V)*(1-S) - beta(V)*S` |
| `CUSTOM_EXPR` | VM bytecode `dS_dt_vm`; optional `dA_dt_vm` for two-variable novel forms |

`CurrentForm`: `LINEAR` (I = g·S^power·(E_syn − V)), `MG_BLOCK` (NMDA Mg²⁺ voltage-dependent block), `CUSTOM_EXPR` (VM bytecode `current_vm`).

**Static factories** (implementations in `synapse_spec.cpp`):
- Spike-driven: `exponential(tau_S, g, E_syn)`, `alpha_function(tau, g, E_syn)`, `double_exponential(tau_rise, tau_decay, g, E_syn)`
- Voltage-gated: `gaba_kinetic()`, `nmda_kinetic()`, `gaba_b()`
- Receptor presets: `ampa()`, `nmda()`, `gaba_a()`

---

### `network.hpp` / `network.cpp`

**Role:** Core simulation class. Manages polymorphic neurons + SoA synapse data + Eigen pools.

**Data layout:**

```
neurons_: vector<unique_ptr<NeuronBase>>   // polymorphic, for API access
synapses_: vector<SynapseBase>             // lightweight views — always in sync

sa_ (SynArrays):                           // all synapse types unified
  Common:     pre, post, weight, E_syn, g, V_pre_prev
  Delay:      delay, spike_buf (ring buffer per synapse), buf_head, delay_init
  Unified S:  S, A, delta_S, delta_A       // primary + auxiliary state; spike-jump increments
  Time consts: tau_S, tau_A, inv_tau_A     // inv_tau_A cached for Euler ALPHA_FUNC
  Derived:    norm, decay_S, decay_A       // DOUBLE_EXP normalization; cached exp(-dt/tau)
  Spec ref:   spec_idx → synapse_specs_    // index into deduped SynapseSpec vector

syn_groups_: type-separated index lists
  exp_decay, alpha_func, double_exp, voltage_gated

synapse_g_scale_: vector<double>           // per-neuron synaptic g multiplier (size = n_neurons)
                                            // reset to 1.0 each step; populated by pools via
                                            // pool_mgr_.scatter_synapse_g_scale() only when
                                            // pool_mgr_.has_synapse_g_mods() is true
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
2. Recording block (if `t % interval == 0`): V, gates, substances, u, g_syn via `pool_mgr_.*`
3. Seed `I_syn_buffer_` from I_ext (dense or descriptor)
4. Accumulate synaptic currents: `I_buf[post[i]] += g[i] * [synapse_g_scale_[post[i]]] * (E_syn[i] - V[post[i]])`
   — the `synapse_g_scale_` multiply is present only when `pool_mgr_.has_synapse_g_mods()` is true
5. `pool_mgr_.gather_all_currents(I_syn_buffer_)` → pools
6. `pool_mgr_.step_all(dt)` — HH: RK4, Iz: Euler, Composable: Euler + substance update + modulation
7. If `pool_mgr_.has_synapse_g_mods()`: fill `synapse_g_scale_` to 1.0, then `pool_mgr_.scatter_synapse_g_scale()`
8. `pool_mgr_.scatter_all_voltages(V_cache_)` — re-scatter for spike detection
9. `update_synapses_grouped(dt)` — spike detection + type-separated kinetics

**SynapseBase sync:** `SynArrays` (SoA) is the authoritative state. `SynapseBase` view objects read from SoA directly on each method call — always current, no lazy sync pass required.

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

**task12 (done):** `_Network` is prefixed and removed from the public API; `RegionalNetwork` is the sole public network class.

**`update_population_spec(name, spec)`**: Sets a new `NeuronModelSpec` on a named population and marks `pools_dirty_ = true` so pools rebuild at the next `simulate()` call. Also propagates the spec to stored `ComposableNeuron` objects via `set_model()`. Called by `RegionalNetwork.add_intracellular()`.

**`_pop_specs` dict (Python layer):** Populated at `add_population()` time — `self._pop_specs[name] = spec`. Used by `add_intracellular()` to mutate the spec before pushing it back to C++.

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
- `result["STN"].substances` — `dict[str, ndarray(n_neurons, n_steps)]` for each recorded substance
- `result["STN"].calcium` — alias for `substances["Ca"]` when calcium is at index 0 (backward compat)
- Recorded when `RecordingConfig.intracellular=True` (or `record_calcium=True` for backward compat)

**`_StimPlan`**: Python-side compact stimulation descriptor. Passed to `_simulate_with_descriptors()`. Created automatically by `RegionalNetwork.simulate()` when all inputs are scalars or `DBSStimulator` objects.

### `spectral.py`

**`mtspectrumpt(spike_times_list, duration, Fs, fpass, tapers)`**: Chronux-compatible multitaper point-process spectrum. Matches benchmark's `make_Spectrum()` exactly. DPSS tapers, FFT-based NUFFT, bias correction. Critical for GPi beta-band validation.

**`analyze_beta_power(result)`**: convenience wrapper — reads spike times from `MetricsResult`, converts ms→s, runs mtspectrumpt, integrates 7–35 Hz band.

### `pulse.py` — `PulseStimulator`

Rectangular or biphasic pulses. Constructors: `single()`, `train()`, `burst()`, `from_onsets()`. Biphasic: cathodic + anodic phase with configurable interphase gap. `apply_to(base, ...)` adds pulse to a base current array.

### `noise.py` — `NoiseInjector`

Adapts pre-generated noise arrays (from any source) into the I_ext interface. Supports `(n_steps,)` (broadcast) or `(n_neurons, n_steps)` (per-neuron). No noise generation is done here — bring your own noise.

### `_equations/` — `NeuronModel`, `SynapseModel`, `IntracellularDynamics`, `Modulation` builders

Python-level builders that accept SymPy expressions and produce C++ spec objects.

**`NeuronModel`**: `add_gate(name, update_form, inf=, tau=, alpha=, beta=, ...)`, `add_channel(name, g, E_rev, gating=)`, `to_spec() → NeuronModelSpec`. Gate kinetic expressions can be raw SymPy or `TaggedExpr` from `Boltzmann(...)`, `Tau.*`, `RateFunc.*`.

**`SynapseModel`**: Named constructors (classmethods) covering all 7 `UpdateForm` values:
- Spike-driven: `exponential(tau=, g=, E_syn=)`, `alpha_function(tau=, g=, E_syn=)`, `double_exponential(tau_rise=, tau_decay=, g=, E_syn=)`
- Voltage-gated: `tanh_gate(amp=, v_half=, k=, tau_decay=, g=, E_syn=)`, `boltzmann_gate(v_half=, k=, tau=, g=, E_syn=)`, `alpha_beta(alpha=, beta=, g=, E_syn=)`
- General constructor with SymPy dS_dt/dA_dt → pattern-matched or `CUSTOM_EXPR` path
- `.to_spec() → SynapseSpec`

`KineticSynapseModel` is a deprecated alias for `SynapseModel`.

**`IntracellularDynamics`**: `IntracellularDynamics(name, ode=<sympy>, source_channels=[], nernst=<sympy|None>, initial=0.0, modulations=[])`. Pattern-matches `ode` against:
- `DECAY`: `-k*X`
- `DRIVEN_DECAY`: `ε*(-I_source - k*X)`
- `DRIVEN_DECAY_NERNST`: DRIVEN_DECAY + standard Nernst `(R*T/z/F)*log(X_o/X)` in `nernst=`
- `CUSTOM_EXPR`: anything else → VM bytecode

Non-standard `nernst=` expressions compile to `nernst_vm` VM. `to_spec(substance_map) → IntracellularSpec`.

**`Modulation`**: dataclass with classmethods:
- `channel_g(channel_name, expr)` — scale channel conductance
- `channel_erev(channel_name, expr)` — override reversal potential
- `gate_inf_shift(gate_name, scale)` — shift gate V_half by `scale * X`
- `gate_inf_scale(gate_name, expr)` — scale x_inf
- `gate_tau_scale(gate_name, expr)` — scale tau_x
- `gate_inf_expr(gate_name, expr)` — fully replace x_inf
- `synapse_g(expr)` — scale all incoming synaptic conductance for this population

Target channel/gate names are resolved to indices at `add_intracellular()` time.

### `_codegen.py` — SymPy compilation pipeline

**Symbols:** `V`, `Ca`, `V_pre`, `V_post`, `S`, `A`, `x`, `x_pre`, `x_post`, `w`, `I_source`, `Da`, `cAMP`, `IP3`, `NO`, `X_ic` — pre-defined SymPy symbols, re-exported from `hodgkin_huxley`. `substance(name)` helper returns a cached symbol by name (re-uses pre-defined symbols).

**`EigenPrinter`**: SymPy expression → C++ string. Two modes:
- Vectorized: symbols → `Symbol_` (Eigen `ArrayXd`); `exp(x)` → `(x).exp()` etc. — SIMD-preserving.
- Scalar: `std::exp`, `std::tanh` etc. — used for JIT scalar2 functions.

**Pattern matching (`try_pattern_match`)**: structural coefficient extraction via `sympy.Poly` against a catalog of 10 forms (1 Boltzmann, 6 Tau, 4 RateFunc). Returns pre-populated `BoltzmannParams`/`TauParams`/`RateFuncParams` when matched; no compilation needed.

**`TaggedExpr`**: SymPy expression + pre-matched params dict. Returned by `Boltzmann(...)`, `Tau.*`, `RateFunc.*` helpers to short-circuit pattern matching.

**JIT compilation (`jit_compile`)**: for gate kinetics that fail pattern matching. Two fn_types: `"vec"` (`void(double*, double*, int)` — Eigen vectorized) and `"scalar2"` (`double(double, double)` — per-neuron). Pipeline: `srepr(expr)` → SHA-256 → `~/.cache/hodgkin_huxley/<hash>.so`; compiles with `g++ -O3 -fPIC -shared -std=c++17`. Raises `HHEquationError` on failure.

**VM bytecode (`compile_to_vm_bytecode`)**: compiles arbitrary SymPy expressions to `VmExpr` (flat instruction list + constants vector) for use in `CUSTOM_EXPR` synapse forms. No subprocess spawned — always available. Opcodes: `PUSH_DEP`, `PUSH_S`, `PUSH_A`, `PUSH_CONST`, `ADD`, `MUL`, `NEG`, `RCP`, `EXP`, `LOG`, `TANH`, `SIN`, `COS`, `SQRT`, `ABS`, `POW_INT`, `POW_HALF`, `POW_GEN`.

### `legacy.py` — deprecated API shims

Accessed via `hodgkin_huxley.__getattr__` when a deprecated name is imported. Emits `DeprecationWarning`. Contains shims for:
- Old neuron/network types: `HHNeuron`, `IzhikevichNeuron`, `HHParameters`, `IzhikevichParameters`, `ExponentialSynapse`, `AlphaSynapse`, `DoubleExponentialSynapse`
- Old equation struct types: `BoltzmannParams`, `TauParams`, `TauForm`, `RateFuncParams`, `RateFuncForm`, `GateUpdateForm`, `GateDependency`
- Old enum aliases: `KineticUpdateForm` → `SynapseUpdateForm`, `KineticCurrentForm` → `SynapseCurrentForm`
- `CalciumSpec`: constructs an `IntracellularDynamics` with standard calcium params (DRIVEN_DECAY_NERNST) and attaches it via `add_intracellular()`
- `GateSpec(dependency=CALCIUM)`: maps to `INTRACELLULAR` with `intracellular_idx=0`
- `ChannelSpec(use_calcium_nernst=True)`: sets `nernst_substance_idx=0`

---

## Key Architectural Patterns

### 1. Data-Driven Model Specification

All neuron and synapse models are described by **parameter structs**, not class hierarchies. A `NeuronModelSpec` is pure data — a list of `GateSpec`, `ChannelSpec`, and `IntracellularSpec` objects with numeric parameters. This has two benefits:

- The same struct drives both the scalar `ComposableNeuron` and the vectorized `ComposablePool` — no code duplication, single source of truth
- The pool can vectorize across N neurons simultaneously (all share the same template; only V and state differ per neuron)

### 2. Structure-of-Arrays (SoA) Synapses

Synapse data is stored as flat parallel arrays (not a vector of objects). A "synapse" is just an index. This layout enables:
- SIMD auto-vectorization of the conductance-decay inner loop
- Cache efficiency (all `exp_decay[]` values accessed sequentially)
- Branch-free update loops (type-separated `syn_groups_`)

`SynapseBase` view objects exist only for API convenience (inspecting individual synapses). They are lightweight non-virtual structs that read from SoA directly — always current.

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

### 5. Intracellular Dynamics

Arbitrary intracellular substances are expressed as `IntracellularSpec` objects attached to a `NeuronModelSpec` via `RegionalNetwork.add_intracellular()`. Calcium is index 0 by convention; additional substances (dopamine, cAMP, etc.) are appended.

The Python `IntracellularDynamics` builder pattern-matches the user's SymPy ODE against standard forms (DECAY / DRIVEN_DECAY / DRIVEN_DECAY_NERNST). Only expressions that fail matching compile to VM bytecode — standard calcium ODE in benchmark populations costs zero VM overhead.

Modulation effects (gate shifts, channel g scaling, synapse g scaling) are applied through pre-allocated member arrays reset each step. The `has_any_mods_` flag skips all modulation machinery entirely for populations without any modulations — the benchmark case.

### 6. SynapseBase as SoA View

`SynapseBase` is a plain value type (not a virtual class) that holds an `(idx, const Network*)` pair. Every accessor reads from `sa_` on demand. No sync flag, no copy — inspection is always live. The `soa_dirty_` flag (still present in the private section) is no longer load-bearing for SynapseBase sync; it may be used by future task20 state-serialisation work.

### 7. Device Model (task17+)

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
      ├─ build_synapse_groups()   → populate syn_groups_ (exp_decay/alpha_func/double_exp/voltage_gated)
      ├─ update_decay_factors(dt) → cache exp(-dt/tau) per synapse
      └─ Hot loop (n_steps iterations):
          ├─ Evaluate I_ext from StimPlan (O(n_neurons))
          ├─ cache_voltages()     → V_cache_ = current V
          ├─ compute_synaptic_currents() → I_syn from SoA g[]
          ├─ Pool.step(dt, I_ext, I_syn) → update V, gates, substances, modulations
          ├─ scatter_synapse_g_scale()   → only when SYNAPSE_G mods present
          ├─ update_synapses_grouped(dt) → update g for all synapse types
          ├─ Spike detection     → threshold crossing on V_cache_ vs V
          └─ Record              → fill V_buf, gate_buf, substance_buf, spike_event_buf etc.
```

---

## Known Limitations

| Issue | Impact | Resolution |
|-------|--------|------------|
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

**Most custom kinetics require no C++ changes.** Use the `CUSTOM_EXPR` VM path:

```python
import sympy as sp
import hodgkin_huxley as hh

dS_dt = sp.Float(2.0) * (1 + sp.tanh(hh.V_pre / 4)) * (1 - hh.S) - hh.S / 13.0
my_syn = hh.SynapseModel("my_syn", dS_dt=dS_dt, g=0.1, E_syn=-80.0)
net.connect("A", "B", "all_to_all", synapse=my_syn, weight=0.5)
```

If the expression matches a known ODE pattern (EXP_DECAY, ALPHA_FUNC, DOUBLE_EXP), `SynapseModel.to_spec()` pattern-matches it automatically and selects the fast pre-compiled path.

**For a new dedicated C++ `UpdateForm`** (only when performance requires it — e.g., a novel exact-integration formula):
1. Add a new `UpdateForm` variant to `SynapseSpec::UpdateForm` in `model/synapse_spec.hpp`
2. Add a new static factory method to `SynapseSpec` in `synapse_spec.cpp`
3. Add the corresponding index list to `SynapseGroups` in `network.hpp` and populate in `build_synapse_groups()`
4. Implement a branch-free inner loop in `Network::update_synapses_grouped()` — using only the existing unified SoA fields (`S`, `A`, `delta_S`, `decay_S`, etc.); add new SoA fields to `SynArrays` only if the existing set cannot express the form
5. Bind the new `UpdateForm` value in `bindings.cpp`
6. Add a corresponding `SynapseModel` named constructor in `_equations/__init__.py` that calls `.to_spec()` with the new form
7. Test in `tests/python/test_synapses.py`

### Adding a New Intracellular Substance

No C++ required. The full path is Python-only:

```python
import sympy as sp
import hodgkin_huxley as hh

Ca = hh.Ca
I_source = hh.I_source

ca_dyn = hh.IntracellularDynamics(
    "Ca",
    ode=sp.Float(5.182e-6) * (-I_source - sp.Float(386.0) * Ca),
    source_channels=["Ca_L"],
    nernst=(hh.R * hh.T / (2 * hh.F)) * sp.log(sp.Float(2000.0) / Ca),
    initial=5e-5,
)
net.add_intracellular(ca_dyn, populations=["STN"])
```

`add_intracellular()`:
1. Builds `substance_map` (existing substance names → indices)
2. Validates modulation target names against channel/gate names in the spec
3. Resolves indices, calls `ca_dyn.to_spec(substance_map)` → `IntracellularSpec`
4. Appends to `spec.intracellular` and calls `update_population_spec()` → C++

`ComposablePool` allocates `X_[i]` and (if Nernst) `E_nernst_[i]` at the next `finalize()` call. Gates referencing the substance by `GateSpec.Dependency.INTRACELLULAR` with `intracellular_idx=i` automatically receive `X_[i]` as their dep input.
