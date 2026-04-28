# Task 15: Plasticity Support

## Priority: 2 — Depends on task13 (SymPy update rules), task14 (for modulator-gated rules)

## Overview

Add configurable synaptic plasticity rules that modify connection weights or synapse parameters during simulation. The design splits into two sub-problems with different implementation costs:

**Sub-problem A — Neuromodulatory gain (task14 SYNAPSE_G, no new C++):**  
Slow, population-wide modulation of synaptic efficacy via intracellular substance → SYNAPSE_G pipeline. Covers dopamine-gated synaptic scaling (Frank, Humphries/Gurney), cholinergic gain control, and homeostatic activity-dependent scaling. Delivered as `IntracellularDynamics` presets. The multiplier is applied per-neuron on top of weights without modifying them in-place.

**Sub-problem B — Per-synapse plasticity (new C++ required):**  
True STDP and Tsodyks-Markram STP, which require per-synapse state not expressible by the per-neuron intracellular system. Covered in section 15.1 below.

**What task14 already covers (sub-problem A):**

| Rule | Mechanism | Status |
|---|---|---|
| Dopamine-gated synaptic gain | `IntracellularDynamics("DA", ode=...)` + `Modulation.synapse_g(...)` | Ready (presets needed) |
| Homeostatic activity scaling | Slow DECAY substance tracking firing rate → SYNAPSE_G | Ready (presets needed) |
| ACh / serotonin tone | Same pattern | Ready (presets needed) |

**What still requires C++ (sub-problem B):**

| Rule | Why task14 can't cover it |
|---|---|
| STDP | x_pre is per-synapse (not per-neuron); weight changes are spike-triggered |
| Tsodyks-Markram STP | u, x are per-synapse and spike-triggered; SYNAPSE_G is per-neuron and step-driven |

**Memory/performance design principle (applies to both sub-problems):**  
All new state is opt-in. Populations and synapses without plasticity must pay zero memory and zero runtime cost. This is enforced by the same flag pattern used in task14 (`has_any_mods_`, `has_synapse_g_mods_`, `has_intracellular_`). New flags for sub-problem B follow the same pattern.

The feature covers three rule families:
1. **Neuromodulator-gated gain** (sub-problem A, task14 infrastructure)
2. **STDP** (Spike-Timing Dependent Plasticity, sub-problem B)
3. **STP** (Short-Term Plasticity, Tsodyks-Markram, sub-problem B)

---

## 15.0 Sub-problem A — Neuromodulatory Presets (Python-only, no C++)

These presets wrap `IntracellularDynamics` + `Modulation.synapse_g()` into named convenience constructors. No C++ changes needed.

### `DopamineGating` preset

Models tonic dopamine concentration driving a sigmoidal gain on incoming synaptic conductance:

```python
# Usage
net.add_intracellular(hh.DopamineGating(da_level=0.5, k=10.0), populations=["Str_D1"])

# Internally constructs:
# IntracellularDynamics("DA",
#     ode = -k_decay * DA,
#     initial = da_level,
#     modulations = [Modulation.synapse_g(1 / (1 + exp(-k * (DA - 0.5))))]
# )
```

### `HomeostaticScaling` preset

Tracks a slow exponential average of post-synaptic voltage and scales incoming conductance inversely:

```python
net.add_intracellular(hh.HomeostaticScaling(target_rate=20.0, tau=5000.0), populations=["STN"])
```

### Checklist (sub-problem A)

- [x] Add `DopamineGating` preset class in `_equations/__init__.py`
- [x] Add `HomeostaticScaling` preset class in `_equations/__init__.py`
- [x] Export from `__init__.py`
- [x] Tests: verify gain scales correctly with DA level; verify homeostatic convergence toward target rate

---

## 15.1 Architecture (Sub-problem B)

### Design Principles

- Plasticity state lives in `SynArrays` (SoA), consistent with existing synapse data
- Only synapses with plasticity incur memory cost
- Plasticity update runs inside the simulation hot loop, after synapse conductance update
- Modulator gating reads from `ComposablePool` intracellular substance arrays (task14)
- Weight bounds enforced after every update

### C++ Structs

```cpp
enum class PlasticityType { NONE, STDP, STP };

struct STDPParams {
    double A_plus    = 0.005;
    double A_minus   = 0.006;
    double tau_plus  = 20.0;
    double tau_minus = 20.0;
    double w_min     = 0.0;
    double w_max     = 1.0;

    // Neuromodulator gating (optional; requires task14)
    int   modulator_pop_idx       = -1;
    int   modulator_substance_idx = -1;
    double modulator_threshold    = 0.5;
};

struct STPParams {
    double U     = 0.5;
    double tau_u = 1000.0;
    double tau_x = 100.0;
};

struct PlasticitySpec {
    PlasticityType type = PlasticityType::NONE;
    STDPParams stdp;
    STPParams  stp;
};
```

### SynArrays Extensions

```cpp
std::vector<PlasticityType> plast_type;
std::vector<double> plast_x_pre;    // STDP pre-synaptic trace
std::vector<double> plast_x_post;   // STDP post-synaptic trace
std::vector<double> stp_u;          // STP utilization
std::vector<double> stp_x;          // STP depression
std::vector<size_t> plast_spec_idx; // index into Network::plasticity_specs_
```

### Update Rules

**STDP:**
```
On pre-synaptic spike:  w += A_plus * x_post;  x_pre += 1.0
On post-synaptic spike: w -= A_minus * x_pre;  x_post += 1.0
Each step:  x_pre  *= exp(-dt / tau_plus)
            x_post *= exp(-dt / tau_minus)
            w = clamp(w, w_min, w_max)
```

Neuromodulator gating: multiply `A_plus` / `A_minus` by a factor derived from modulator concentration before applying weight updates.

**STP:**
```
Each step (no spike):
    u += dt * (U - u) / tau_u
    x += dt * (1 - x) / tau_x

On pre-synaptic spike:
    effective_weight = weight * u * x
    u += U * (1 - u)
    x -= u * x
```

---

## 15.2 Python API

```python
from hodgkin_huxley import STDPRule, STPRule

# STDP on a single synapse
net.add_ampa_synapse(pre=0, post=1, weight=0.5,
    plasticity=STDPRule(A_plus=0.005, A_minus=0.006,
                        tau_plus=20.0, tau_minus=20.0,
                        w_min=0.0, w_max=1.0))

# STP (depression)
net.add_alpha_synapse(pre=2, post=3, weight=1.0,
    plasticity=STPRule(U=0.6, tau_u=1000.0, tau_x=100.0))

# STDP on a projection (applies to all synapses created by connect())
rn.connect("CTX_e", "STN",
    pattern=ConnectivityPattern.ONE_TO_ONE,
    synapse=SynapseSpec.ampa(),
    weight=WeightDistribution.uniform(0.1, 0.5),
    plasticity=STDPRule(A_plus=0.005, A_minus=0.006,
                        tau_plus=20, tau_minus=25))

# Dopamine-gated STDP (requires task14)
rn.connect("CTX_e", "Str_D1",
    pattern=ConnectivityPattern.ONE_TO_ONE,
    synapse=SynapseSpec.ampa(),
    weight=WeightDistribution.constant(0.3),
    plasticity=STDPRule(A_plus=0.01, A_minus=0.008,
                        modulator_population="Str_D1",
                        modulator_substance="dopamine",
                        modulator_threshold=0.5))
```

---

## 15.3 Recording Integration

```python
# Record weight traces for plastic synapses
recording = RecordingConfig.with_plasticity()
result = net.simulate(...)
# result["weights"] → shape (n_plastic_synapses, n_rec)

# Read final weights without recording traces
weights = net.get_synapse_weights()  # np.ndarray shape (n_synapses,)
```

---

## 15.4 Implementation Checklist

### Sub-problem A — Neuromodulatory presets (Python-only)
- [x] `DopamineGating` preset in `_equations/__init__.py`
- [x] `HomeostaticScaling` preset in `_equations/__init__.py`
- [x] Export from `hodgkin_huxley/__init__.py`
- [x] Tests in `tests/python/test_plasticity.py`: gain scales with DA level; homeostatic convergence

### Sub-problem B — Per-synapse STDP / STP (C++ required)

**Memory / performance constraints (mandatory):**
- Plasticity state fields in `SynArrays` must be zero-length by default; resized only for synapses with `plasticity=` set
- A `has_stdp_` / `has_stp_` boolean flag on `Network` (same pattern as `has_synapse_g_mods_`) gates all per-step plasticity loops and weight-recording scatter
- Populations and synapse groups without plasticity must incur zero memory and zero loop iterations

**C++ Core:**
- [x] Define `STDPParams`, `STPParams`, `PlasticitySpec`, `PlasticityType` in new `plasticity.hpp`
- [x] Add opt-in plasticity state to `SynArrays`: `plast_type`, `plast_x_pre`, `plast_x_post`, `stp_u`, `stp_x`, `plast_spec_idx` — all empty by default
- [x] Add `has_stdp_`, `has_stp_` flags to `Network`; set in `build_synapse_groups()`
- [x] Extend `Network::add_synapse()` family to accept optional `PlasticitySpec`; only resize plasticity arrays when spec is non-NONE
- [x] Implement STDP trace decay + spike-triggered weight update after spike-detection block in hot loop; gated by `has_stdp_`
- [x] Implement STP step decay + spike-triggered `u`/`x` update; back-correct `S` on pre-spike (Tsodyks-Markram); gated by `has_stp_`
- [x] Modulator gating: read `X_[substance_idx]` from pool at post-spike time for gated STDP; `get_substance_at()` accessor on `ComposablePool`, `get_substance()` on `PoolManager`
- [x] Add `get_synapse_weights()` returning current weight vector

**Python Bindings:**
- [x] Bind `PlasticitySpec`, `STDPParams`, `STPParams`, `PlasticityType`
- [x] Expose `STDPRule` and `STPRule` as convenience constructors
- [x] Extend `RegionalNetwork.connect()` with `plasticity=` kwarg (follows g_syn/g_pre pattern)
- [x] Bind `get_synapse_weights()`

**Recording:**
- `RecordingConfig.with_plasticity()` deferred — weight traces accessible via `get_synapse_weights()` after each epoch

**Tests:**
- [x] STDP: verify LTP on pre-before-post spike pairs, LTD on post-before-pre
- [x] STDP: verify weight stays within `[w_min, w_max]`
- [x] STP: verify depression under high-frequency stimulation, recovery at rest
- [x] Gated STDP: verify weight update scales with modulator concentration
- [x] Zero-overhead: benchmark with `plasticity=None` (default) confirms no timing regression vs pre-task15

## Status: COMPLETE (2026-04-27)

34/34 plasticity tests pass. 1035/1035 total tests pass.

### Key implementation notes

- **STP back-correction**: `apply_stp()` corrects `S` (not `g`) to make depression persistent across steps: removes unscaled `delta_S` jump and replaces with `u*x`-scaled version, then recomputes `g = spec.g * weight * S`.
- **Pool name collision fix**: `add_intracellular()` renames spec to pop_name to ensure pools are not merged when multiple populations share the same base model name.
- **STP frequency discrimination**: HH neurons (Type II) are unsuitable — they fire near-constantly above rheobase. STP frequency tests use Izhikevich RS (Type I) neurons which span 5–80 Hz continuously.
