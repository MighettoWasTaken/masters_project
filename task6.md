# Task 6: Composable Kinetic Synapse

## Priority: 1 (Critical)

## Overview

The existing spike-triggered synapses (Exponential, Alpha, DoubleExponential) cover discrete-event conductance shapes well. However, many biologically important synapse models are **kinetic** — their gating variable evolves continuously as a function of the presynaptic membrane potential rather than firing in response to a detected spike. The canonical example is the GABA intra-striatal synapse required by the benchmark, but NMDA voltage-dependence, GABA-B metabotropic synapses, and many novel research models fall into this category.

Rather than hardcoding `SYN_GABA_KINETIC` as a fourth fixed type, this task introduces a **composable kinetic synapse** system following the same design philosophy as the composable neuron (Task 5): parameterise the mathematical form, reuse existing structs (`RateFuncParams`, `BoltzmannParams`, `TauParams`), and let researchers configure novel synapses without touching C++.

---

## 6.1 Design: KineticSynapseSpec

A kinetic synapse has two independently configurable parts: how its gating variable **S** evolves, and how **S** is used to produce a postsynaptic current.

### 6.1.1 State Update Forms

```
S ∈ [0, 1]  — synaptic gating variable
```

| Form | Equation | When to use |
|---|---|---|
| `ALPHA_BETA` | `dS/dt = α(V_pre)·(1-S) − β(V_pre)·S` | Classical receptor kinetics; α, β are `RateFuncParams` |
| `TANH_GATE` | `dS/dt = amp·(1+tanh((V_pre−vh)/k))·(1-S) − S/τ_decay` | GABA intra-striatal (Kumaravelu 2016) |
| `BOLTZMANN_GATE` | `dS/dt = (S_inf(V_pre) − S) / τ(V_pre)` | Reuses `BoltzmannParams` + `TauParams`; general first-order |

All three use **exact exponential integration** (same fix applied to INF_TAU gates in Task 5) so numerical stability is guaranteed regardless of dt.

### 6.1.2 Current Computation Forms

| Form | Equation | When to use |
|---|---|---|
| `LINEAR` | `I = g · S^n · (V_post − E_syn)` | GABA-A, AMPA kinetic, GABA-B |
| `MG_BLOCK` | `I = g · S^n · (V_post − E_syn) / (1 + [Mg]·exp(−c·V_post)/d)` | NMDA with magnesium block |

### 6.1.3 C++ Struct

```cpp
// in ion_channels.hpp (natural home alongside composable neuron structs)

struct KineticSynapseSpec {
    std::string name;

    // ---- State update ------------------------------------------------
    enum class UpdateForm { ALPHA_BETA, TANH_GATE, BOLTZMANN_GATE };
    UpdateForm update_form = UpdateForm::TANH_GATE;

    // ALPHA_BETA: α(V) and β(V) as rate functions (reuse existing type)
    RateFuncParams alpha;
    RateFuncParams beta;

    // TANH_GATE: dS/dt = tanh_amp*(1+tanh((V-tanh_vh)/tanh_k))*(1-S) - S/tau_decay
    double tanh_amp   = 2.0;
    double tanh_vh    = 0.0;   // mV
    double tanh_k     = 4.0;   // mV
    double tau_decay  = 13.0;  // ms

    // BOLTZMANN_GATE: dS/dt = (S_inf(V) - S) / tau(V)
    BoltzmannParams s_inf;
    TauParams       tau;

    // ---- Current computation -----------------------------------------
    enum class CurrentForm { LINEAR, MG_BLOCK };
    CurrentForm current_form = CurrentForm::LINEAR;

    double g      = 0.1;   // max conductance (mS/cm²)
    double E_syn  = -80.0; // reversal potential (mV)
    int    power  = 1;     // S^power

    // MG_BLOCK parameters (NMDA)
    double mg_conc  = 1.0;    // [Mg²⁺] mM
    double mg_scale = 0.062;  // 1/mV
    double mg_denom = 3.57;

    // ---- Initial condition -------------------------------------------
    double S_init = 0.0;

    // ---- Preset factories -------------------------------------------
    static KineticSynapseSpec gaba_kinetic();  // Kumaravelu 2016 intra-striatal
    static KineticSynapseSpec nmda_kinetic();  // NMDA with Mg block
    static KineticSynapseSpec gaba_b();        // slow GABA-B metabotropic
};
```

### 6.1.4 Preset Implementations

```cpp
KineticSynapseSpec KineticSynapseSpec::gaba_kinetic() {
    KineticSynapseSpec s;
    s.name        = "GABA_kinetic";
    s.update_form = UpdateForm::TANH_GATE;
    s.tanh_amp    = 2.0;   s.tanh_vh = 0.0;   s.tanh_k = 4.0;
    s.tau_decay   = 13.0;
    s.current_form = CurrentForm::LINEAR;
    s.g = 0.1;   s.E_syn = -80.0;   s.power = 1;
    return s;
}

KineticSynapseSpec KineticSynapseSpec::nmda_kinetic() {
    KineticSynapseSpec s;
    s.name        = "NMDA_kinetic";
    s.update_form = UpdateForm::BOLTZMANN_GATE;
    s.s_inf       = {0.0, 16.0};    // S_inf = 1/(1+exp(-V/16))
    s.tau.form    = TauParams::Form::CONSTANT;
    s.tau.params[0] = 80.0;         // ms
    s.current_form = CurrentForm::MG_BLOCK;
    s.g = 0.3;   s.E_syn = 0.0;    s.power = 1;
    s.mg_conc = 1.0;  s.mg_scale = 0.062;  s.mg_denom = 3.57;
    return s;
}
```

---

## 6.2 Network Integration

### 6.2.1 SoA Storage

Kinetic synapses live in their own SoA block inside `Network`, separate from the spike-triggered synapses. Each kinetic synapse needs:

```cpp
// in Network (private):
struct KineticSynArrays {
    std::vector<size_t>             pre_idx;
    std::vector<size_t>             post_idx;
    std::vector<double>             weight;
    std::vector<double>             S;             // gating variable per synapse
    std::vector<size_t>             spec_idx;      // index into kinetic_specs_
    std::vector<size_t>             delay_steps;
    // delay ring buffers (same mechanism as spike-triggered)
    std::vector<std::vector<double>> delay_buf;
    std::vector<size_t>             delay_head;
};

std::vector<KineticSynapseSpec> kinetic_specs_;   // unique specs (deduped by name)
KineticSynArrays                kinetic_syns_;
```

### 6.2.2 Per-Step Update (in simulate loop)

```
for each kinetic synapse k:
    V_pre  = V_cache_[pre_idx[k]]
    V_post = V_cache_[post_idx[k]]
    spec   = kinetic_specs_[spec_idx[k]]

    // 1. Exact exponential integration of S
    switch spec.update_form:
        TANH_GATE:
            rate_open = spec.tanh_amp * (1 + tanh((V_pre - spec.tanh_vh) / spec.tanh_k))
            rate      = rate_open + 1.0/spec.tau_decay
            S_inf     = rate_open / rate
            S[k]      = S_inf + (S[k] - S_inf) * exp(-dt * rate)

        ALPHA_BETA:
            alpha = compute_rate(V_pre, spec.alpha)
            beta  = compute_rate(V_pre, spec.beta)
            rate  = alpha + beta
            S_inf = alpha / max(rate, 1e-10)
            S[k]  = S_inf + (S[k] - S_inf) * exp(-dt * rate)

        BOLTZMANN_GATE:
            S_inf = boltzmann(V_pre, spec.s_inf)
            tau_s = compute_tau(V_pre, spec.tau)
            S[k]  = S_inf + (S[k] - S_inf) * exp(-dt / tau_s)

    // 2. Current (added into I_syn_buffer_[post_idx[k]])
    g_eff = spec.g * weight[k] * S[k]^spec.power
    switch spec.current_form:
        LINEAR:   I = g_eff * (V_post - spec.E_syn)
        MG_BLOCK: I = g_eff * (V_post - spec.E_syn)
                       / (1 + spec.mg_conc * exp(-spec.mg_scale * V_post) / spec.mg_denom)
    I_syn_buffer_[post_idx[k]] += I
```

Note: V_cache_ is already populated before the synapse update in the existing loop, so V_pre access is free.

### 6.2.3 New Network Method

```cpp
size_t add_kinetic_synapse(size_t pre_idx, size_t post_idx,
                           double weight,
                           const KineticSynapseSpec& spec,
                           double delay = 0.0);
```

---

## 6.3 Python API

### 6.3.1 Bindings

```python
# Preset factories
KineticSynapseSpec.gaba_kinetic()
KineticSynapseSpec.nmda_kinetic()
KineticSynapseSpec.gaba_b()

# Manual construction (all fields exposed via def_readwrite)
spec = KineticSynapseSpec()
spec.name         = "my_kinetic"
spec.update_form  = KineticUpdateForm.TANH_GATE
spec.tanh_amp     = 2.0
spec.tau_decay    = 13.0
spec.g            = 0.1
spec.E_syn        = -80.0
```

### 6.3.2 Network.add_kinetic_synapse()

```python
net.add_kinetic_synapse(pre_idx=0, post_idx=1,
                        weight=0.025,
                        spec=KineticSynapseSpec.gaba_kinetic(),
                        delay=0.0)
```

### 6.3.3 RegionalNetwork.connect() — kinetic path

The existing `connect()` method already accepts `SynapseSpec` for spike-triggered synapses. Add an overload / kwarg to accept `KineticSynapseSpec`:

```python
rnet.connect("StrD2", "StrD2",
             pattern="random_permutation",
             weight=0.025,
             kinetic_spec=KineticSynapseSpec.gaba_kinetic())
```

When `kinetic_spec` is provided, connections are added via `add_kinetic_synapse()` instead of `add_synapse()`.

### 6.3.4 Python Builder (KineticSynapseModel)

For researchers who want ergonomic construction without dealing with struct fields:

```python
class KineticSynapseModel:
    def __init__(self, name):
        self._spec = KineticSynapseSpec()
        self._spec.name = name

    def tanh_gate(self, amp=2.0, v_half=0.0, k=4.0, tau_decay=13.0):
        self._spec.update_form = KineticUpdateForm.TANH_GATE
        self._spec.tanh_amp = amp
        self._spec.tanh_vh = v_half
        self._spec.tanh_k = k
        self._spec.tau_decay = tau_decay
        return self

    def boltzmann_gate(self, v_half, k, tau):
        self._spec.update_form = KineticUpdateForm.BOLTZMANN_GATE
        self._spec.s_inf = BoltzmannParams(v_half=v_half, k=k)
        self._spec.tau.form = TauForm.CONSTANT
        self._spec.tau.params[0] = tau
        return self

    def alpha_beta(self, alpha: RateFuncParams, beta: RateFuncParams):
        self._spec.update_form = KineticUpdateForm.ALPHA_BETA
        self._spec.alpha = alpha
        self._spec.beta = beta
        return self

    def linear_current(self, g, E_syn, power=1):
        self._spec.current_form = KineticCurrentForm.LINEAR
        self._spec.g = g
        self._spec.E_syn = E_syn
        self._spec.power = power
        return self

    def mg_block_current(self, g, E_syn, mg_conc=1.0, mg_scale=0.062, mg_denom=3.57):
        self._spec.current_form = KineticCurrentForm.MG_BLOCK
        self._spec.g = g
        self._spec.E_syn = E_syn
        self._spec.mg_conc = mg_conc
        self._spec.mg_scale = mg_scale
        self._spec.mg_denom = mg_denom
        return self

    def to_spec(self) -> KineticSynapseSpec:
        return self._spec
```

Usage:
```python
gaba = (KineticSynapseModel("my_GABA")
        .tanh_gate(amp=2.0, v_half=0.0, k=4.0, tau_decay=13.0)
        .linear_current(g=0.1, E_syn=-80.0)
        .to_spec())
```

---

## 6.4 What This Enables

| Model | How to express |
|---|---|
| GABA intra-striatal (Kumaravelu 2016) | `KineticSynapseSpec.gaba_kinetic()` |
| NMDA with Mg block | `KineticSynapseSpec.nmda_kinetic()` |
| GABA-B slow (τ=200ms) | `KineticSynapseModel().boltzmann_gate(...).linear_current(E_syn=-95)` |
| Novel first-order kinetic | `BOLTZMANN_GATE` with custom `BoltzmannParams` + `TauParams` |
| Classical receptor kinetics | `ALPHA_BETA` with `RateFuncParams` α and β |
| Custom Mg-like block | `MG_BLOCK` with tuned `mg_scale`, `mg_denom` |

---

## 6.5 Implementation Checklist

### C++
- [ ] Add `KineticSynapseSpec` struct to `ion_channels.hpp`
- [ ] Implement `gaba_kinetic()`, `nmda_kinetic()`, `gaba_b()` presets in `ion_channels.cpp`
- [ ] Add `KineticSynArrays` SoA block to `Network` (private)
- [ ] Implement `Network::add_kinetic_synapse()`
- [ ] Integrate kinetic synapse update into `Network::simulate()` loop
- [ ] Integrate kinetic synapse update into `Network::step()` loop
- [ ] Handle delay ring buffers for kinetic synapses (or defer: note delay=0 is valid for most kinetic models)
- [ ] `Network::reset()` resets kinetic S values to `spec.S_init`

### Python bindings
- [ ] Expose `KineticSynapseSpec` fields via `def_readwrite`
- [ ] Expose `KineticUpdateForm` and `KineticCurrentForm` enums
- [ ] Expose preset factories
- [ ] Expose `Network.add_kinetic_synapse()`
- [ ] Add `kinetic_spec` kwarg to `RegionalNetwork.connect()`
- [ ] Implement `KineticSynapseModel` builder in `__init__.py`
- [ ] Export all new symbols from `__init__.py`

### Tests
- [ ] S variable converges to correct steady state for each update form
- [ ] TANH_GATE: at V_pre=0, S_steady ≈ tanh_amp*2/(tanh_amp*2 + 1/tau_decay)
- [ ] MG_BLOCK: current is reduced relative to LINEAR at hyperpolarised V_post
- [ ] GABA kinetic preset suppresses postsynaptic firing
- [ ] NMDA preset passes current only when postsynaptic neuron is depolarised
- [ ] KineticSynapseModel builder round-trips to same spec as direct construction
- [ ] `reset()` returns S to S_init
