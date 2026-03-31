# Task 15: Plasticity Support

## Priority: 2 — Depends on task13 (SymPy update rules), task14 (for modulator-gated rules)

## Overview

Add configurable synaptic plasticity rules that modify connection weights or synapse parameters during simulation. Plasticity must integrate with the intracellular dynamics system (task14) to support neuromodulator-gated learning rules (e.g., dopamine-dependent STDP in striatum). The design extends the SoA synapse system with minimal memory overhead: plasticity state is appended to `SynArrays` only for synapses that request it.

Update rules are expressed as SymPy expressions using the pre-defined symbols from task13 (`hh.x_pre`, `hh.x_post`, `hh.w`, etc.), compiled via the same EigenPrinter/JIT pipeline.

The feature covers three rule families:
1. **STDP** (Spike-Timing Dependent Plasticity) — correlation-based Hebbian learning
2. **STP** (Short-Term Plasticity, Tsodyks-Markram) — depression and facilitation
3. **Homeostatic** — activity-dependent weight rescaling (lower priority)

---

## 15.1 Architecture

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

### C++ Core
- [ ] Define `STDPParams`, `STPParams`, `PlasticitySpec`, `PlasticityType` in `synapse.hpp` or new `plasticity.hpp`
- [ ] Extend `SynArrays` with plasticity state fields
- [ ] Extend `Network::add_synapse()` family to accept optional `PlasticitySpec`
- [ ] Implement STDP update in `Network::update_synapses_grouped()` hot loop
- [ ] Implement STP conductance scaling in synapse current computation
- [ ] Add modulator concentration lookup in `Network::simulate_with_descriptors()` for gated STDP
- [ ] Add `get_synapse_weights()` returning current weight vector

### Python Bindings
- [ ] Bind `PlasticitySpec`, `STDPParams`, `STPParams`, `PlasticityType`
- [ ] Expose `STDPRule` and `STPRule` as convenience constructors
- [ ] Extend synapse-add methods to accept `plasticity=` kwarg
- [ ] Extend `RegionalNetwork.connect()` to pass plasticity spec
- [ ] Bind `get_synapse_weights()`

### Recording
- [ ] Add weight recording buffer to `simulate_into_buffers()` / `simulate_with_descriptors()`
- [ ] Expose via `RecordingConfig.with_plasticity()` and `MetricsResult["weights"]`

### Tests
- [ ] STDP: verify LTP on pre-before-post spike pairs, LTD on post-before-pre
- [ ] STDP: verify weight stays within `[w_min, w_max]`
- [ ] STP: verify depression under high-frequency stimulation, recovery at rest
- [ ] Gated STDP: verify weight update scales with modulator concentration
- [ ] Weight recording: verify `result["weights"]` shape and values
