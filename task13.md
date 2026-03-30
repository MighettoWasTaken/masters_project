# Task 13: Plasticity Support

## Priority: 2

## Overview

Add configurable synaptic plasticity rules that modify connection weights or synapse parameters during simulation. Plasticity must integrate with the intracellular dynamics system (task 12) to support neuromodulator-gated learning rules (e.g., dopamine-dependent STDP in striatum). The design extends the SoA synapse system with minimal memory overhead: plasticity state is appended to `SynArrays` only for synapses that request it.

The feature covers three rule families:
1. **STDP** (Spike-Timing Dependent Plasticity) — correlation-based Hebbian learning
2. **STP** (Short-Term Plasticity, Tsodyks-Markram) — depression and facilitation
3. **Homeostatic** — activity-dependent weight rescaling (longer-term; lower priority)

---

## 13.1 Architecture

### Design Principles

- Plasticity state lives in `SynArrays` (SoA), consistent with existing synapse data
- Only synapses with plasticity incur memory cost
- Plasticity update runs inside the simulation hot loop, after synapse conductance update
- Modulator gating reads from `ComposablePool` intracellular substance arrays (task 12), requiring a reference from `Network` to the relevant pool's substance state
- Weight bounds enforced after every update

### C++ Structs

```cpp
enum class PlasticityType { NONE, STDP, STP };

struct STDPParams {
    double A_plus    = 0.005;  // LTP amplitude
    double A_minus   = 0.006;  // LTD amplitude (positive value; applied negatively)
    double tau_plus  = 20.0;   // pre-synaptic trace decay (ms)
    double tau_minus = 20.0;   // post-synaptic trace decay (ms)
    double w_min     = 0.0;    // weight lower bound
    double w_max     = 1.0;    // weight upper bound

    // Neuromodulator gating (optional; requires task 12)
    int   modulator_pop_idx = -1;       // population index (-1 = ungated)
    int   modulator_substance_idx = -1; // substance index within that pop's pool
    double modulator_threshold = 0.5;   // gate opens above this concentration
};

struct STPParams {
    double U       = 0.5;   // baseline utilization
    double tau_u   = 1000.0; // facilitation recovery (ms)
    double tau_x   = 100.0;  // depression recovery (ms)
};

struct PlasticitySpec {
    PlasticityType type = PlasticityType::NONE;
    STDPParams stdp;
    STPParams  stp;
};
```

### SynArrays Extensions

```cpp
struct SynArrays {
    // ... existing fields ...

    // Plasticity (allocated only where plast_type[i] != NONE)
    std::vector<PlasticityType> plast_type;
    std::vector<double> plast_x_pre;   // STDP pre-synaptic eligibility trace
    std::vector<double> plast_x_post;  // STDP post-synaptic eligibility trace
    std::vector<double> stp_u;         // STP utilization variable
    std::vector<double> stp_x;         // STP depression variable
    std::vector<size_t> plast_spec_idx;// index into Network::plasticity_specs_

    // Weight is already sa_.weight; plasticity updates it in place
};
```

### Update Rules

**STDP inner-loop update (per synapse with PlasticityType::STDP):**
```
On pre-synaptic spike (synapse fires):
    w += A_plus * x_post     // LTP: post trace at moment of pre spike
    x_pre += 1.0             // bump pre trace

On post-synaptic spike (post neuron threshold crossing):
    w -= A_minus * x_pre     // LTD: pre trace at moment of post spike
    x_post += 1.0            // bump post trace

Each step:
    x_pre  *= exp(-dt / tau_plus)
    x_post *= exp(-dt / tau_minus)
    w = clamp(w, w_min, w_max)
```

Neuromodulator gating: multiply `A_plus` and `A_minus` by a gating factor derived from the modulator concentration before applying weight updates.

**STP inner-loop update (per synapse with PlasticityType::STP):**
```
Each step (no spike):
    u += dt * (U - u) / tau_u      // u relaxes to U
    x += dt * (1 - x) / tau_x     // x recovers

On pre-synaptic spike:
    effective_weight = weight * u * x
    u += U * (1 - u)               // utilization jump
    x -= u * x                     // depletion

Apply effective_weight to conductance instead of weight.
```

---

## 13.2 Python API

### Adding Plasticity to a Synapse

```python
from hodgkin_huxley import STDPRule, STPRule, PlasticityType

# STDP on a single synapse
net.add_ampa_synapse(pre=0, post=1, weight=0.5,
    plasticity=STDPRule(
        A_plus=0.005, A_minus=0.006,
        tau_plus=20.0, tau_minus=20.0,
        w_min=0.0, w_max=1.0
    ))

# STP (short-term depression)
net.add_alpha_synapse(pre=2, post=3, weight=1.0,
    plasticity=STPRule(U=0.6, tau_u=1000.0, tau_x=100.0))
```

### Adding Plasticity to a Population Projection

```python
rn.connect(
    "CTX_e", "STN",
    pattern=ConnectivityPattern.ONE_TO_ONE,
    synapse=SynapseSpec.ampa(),
    weight=WeightDistribution.uniform(0.1, 0.5),
    plasticity=STDPRule(A_plus=0.005, A_minus=0.006,
                        tau_plus=20, tau_minus=25)
)
```

### Neuromodulator-Gated STDP (requires task 12)

```python
# Dopamine gates LTP in striatum D1 pathway
rn.connect(
    "CTX_e", "Str_D1",
    pattern=ConnectivityPattern.ONE_TO_ONE,
    synapse=SynapseSpec.ampa(),
    weight=WeightDistribution.constant(0.3),
    plasticity=STDPRule(
        A_plus=0.01, A_minus=0.008,
        modulator_population="Str_D1",
        modulator_substance="dopamine",
        modulator_threshold=0.5
    )
)
```

### Reading Final Weights

```python
result = net.simulate(...)
# Weights accessible via network object after simulation
weights = net.get_synapse_weights()  # returns np.ndarray shape (n_synapses,)
```

---

## 13.3 Recording Integration

Add optional weight recording to `RecordingConfig`:

```python
RecordingConfig.with_plasticity()  # records weight traces for plastic synapses
# result["weights"] → shape (n_plastic_synapses, n_rec)
```

---

## 13.4 Implementation Checklist

### C++ Core
- [ ] Define `STDPParams`, `STPParams`, `PlasticitySpec`, `PlasticityType` in `synapse.hpp` or new `plasticity.hpp`
- [ ] Extend `SynArrays` with plasticity state fields
- [ ] Extend `Network::add_synapse()` family to accept optional `PlasticitySpec`
- [ ] Implement STDP update in `Network::update_synapses_grouped()` hot loop
- [ ] Implement STP conductance scaling in synapse current computation
- [ ] Add `modulator_pop_idx` resolution in `Network::simulate_with_descriptors()` for gated STDP
- [ ] Add `get_synapse_weights()` method returning current weight vector

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
- [ ] STDP: verify weight convergence stays within `[w_min, w_max]`
- [ ] STP: verify depression under high-frequency stimulation, recovery at rest
- [ ] STP: verify facilitation when U is low and tau_u is long
- [ ] Gated STDP: verify weight update scales with modulator concentration
- [ ] Weight recording: verify `result["weights"]` shape and values
