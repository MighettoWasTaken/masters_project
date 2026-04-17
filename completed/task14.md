# Task 14: Generalized Intracellular Dynamics — COMPLETED 2026-04-15

## Priority: 2 — Depends on task13 (VM, SymPy builders, pattern matching)

## Overview

Replace the single hardcoded `CalciumSpec` in `ComposablePool` with a fully general intracellular dynamics system. Any number of substances (calcium, dopamine, cAMP, IP3, etc.) can be defined using SymPy expressions, pattern-matched to common ODE forms, or compiled to VM bytecode for novel kinetics. Substances can modulate channel conductances, gate kinetics, and synaptic input weights — expressed as SymPy expressions that are pattern-matched or VM-compiled to the same pipeline used for gates and synapses.

A key architectural point: `IntracellularDynamics` objects are defined once and **attached to named populations at the network level**, so the same substance spec can apply to one, several, or all populations without duplication.

---

## 14.1 Survey of Published Models

The following intracellular phenomena appear in major published models and must be representable by the system:

| Phenomenon | Model | ODE form | Modulation target |
|---|---|---|---|
| Calcium decay + Nernst | Rubin-Terman 2004, Kumaravelu 2016 | `ε(-I_Ca - K_Ca * Ca)` | E_Ca on T/L channels |
| Calcium AHP gating | Bhattacharya, Traub | `Ca/(Ca+k1)` replaces gate product | Channel conductance |
| IP3 dynamics | Li-Rinzel 1994 | production-degradation ODE + h gate | Ca channel gating |
| Dopamine (D1/D2) | Wiecki & Frank 2010, Gurney | first-order decay, volume-transmitted | M-channel g, NMDA g |
| DA → cAMP cascade | Bhattacharya 2016 | `k_prod*DA - k_deg*cAMP` | PKA targets |
| Muscarinic ACh → M-current | Wang 1994 | steady-state gating by ACh | M-channel g |
| Nitric oxide (NO) | Bhattacharya, Bhattacharyya | diffusion + decay | cGMP targets |
| Synaptic receptor pool | Bhattacharya scaling | first-order | AMPA g (synaptic scaling) |

**Key design requirements extracted:**
1. Multiple substances per neuron model (Ca + DA + cAMP in striatum)
2. Substance-to-substance ODEs (`d[cAMP]/dt = f([DA]) - ...`)
3. Per-neuron substance state (calcium is local; but volume-transmitted DA is also representable as per-neuron if all neurons in a pop receive the same update)
4. Nernst reversal potential as a function of substance concentration
5. AHP-style gating: channel conductance gated by `f(Ca)` — expressed as a gate product SymPy expression
6. Modulating channel g, gate x_inf, gate tau, and synaptic input g
7. Attachable to multiple populations — define once, attach to several pops

---

## 14.2 Architecture

### Core design

`IntracellularSpec` is a **pure data struct** parallel to `GateSpec` and `SynapseSpec`. It contains:
- ODE update form (pattern-matched or VM bytecode)
- Optional Nernst expression
- List of source channels (for current-driven substances)
- List of modulation specs (what this substance affects)

`NeuronModelSpec` gains a `vector<IntracellularSpec> intracellular` field (replacing `CalciumSpec calcium`).

At the Python level, `IntracellularDynamics` is a builder (parallel to `NeuronModel` / `SynapseModel`) that accepts SymPy expressions, pattern-matches them to standard forms, and produces `IntracellularSpec` via `.to_spec()`.

`RegionalNetwork.add_intracellular(dynamics, populations=None)` attaches a dynamics spec to the named populations at add time (inserting into each pop's `NeuronModelSpec.intracellular`). If `populations=None`, it attaches to all populations.

---

## 14.3 C++ Structs

### `IntracellularSpec` (`model/gate_spec.hpp`)

```cpp
struct IntracellularSpec {
    std::string name;
    double      initial = 0.0;    // initial concentration

    // -------------------------------------------------------------------------
    // ODE update form
    // -------------------------------------------------------------------------
    enum class UpdateForm {
        DECAY,               // d[X]/dt = -k_decay * X
        DRIVEN_DECAY,        // d[X]/dt = epsilon * (-I_src - k_decay * X)
        DRIVEN_DECAY_NERNST, // as above + Nernst E_rev update
        CUSTOM_EXPR,         // VM bytecode (see ode_vm below)
    };
    UpdateForm update_form = UpdateForm::CUSTOM_EXPR;

    // Scalar params for standard forms
    double epsilon = 1e-4;
    double k_decay = 15.0;
    std::vector<int> source_channels;  // channel indices contributing to I_src

    // -------------------------------------------------------------------------
    // Nernst: E_rev = f(X)
    // -------------------------------------------------------------------------
    bool   nernst_enabled = false;
    double nernst_Ca_o = 2000.0;  // extracellular concentration (DRIVEN_DECAY_NERNST only)
    double nernst_z    = 2.0;
    double nernst_R    = 8314.0;
    double nernst_F    = 96485.0;
    double nernst_T    = 298.0;
    VmExpr nernst_vm;              // CUSTOM_EXPR Nernst: PUSH_DEP = X; returns E_rev

    // -------------------------------------------------------------------------
    // CUSTOM_EXPR ODE VM
    //   PUSH_DEP        → I_source sum (sum of source channel currents)
    //   PUSH_X (op=0)   → this substance's concentration
    //   PUSH_SUBST(idx) → other substance concentration X_[idx]
    // -------------------------------------------------------------------------
    VmExpr ode_vm;

    // -------------------------------------------------------------------------
    // Modulations: what this substance adjusts each step
    // -------------------------------------------------------------------------
    std::vector<IntracellularModulation> modulations;
};
```

### `IntracellularModulation` (`model/gate_spec.hpp`)

```cpp
struct IntracellularModulation {
    enum class Target {
        CHANNEL_G,           // g_eff = g * mod(X)  — mod_vm returns scale factor
        CHANNEL_EREV,        // E_rev = mod_vm(X)   — replaces channel E_rev
        GATE_INF_SHIFT,      // x_inf(V + mod_vm(X)) — V-shift by substance
        GATE_INF_SCALE,      // x_inf_eff = x_inf * mod_vm(X)
        GATE_TAU_SCALE,      // tau_eff = tau * mod_vm(X)
        GATE_INF_EXPR,       // x_inf fully replaced by mod_vm(V, X) — most general
        SYNAPSE_G,           // scale all incoming synaptic g to neurons in this pop
    };

    Target target;
    int    target_idx = -1;  // channel index (CHANNEL_*), gate index (GATE_*)
    int    substance_idx = 0; // which IntracellularSpec provides X (usually self = idx of this spec)

    // mod_vm:
    //   PUSH_DEP         → this substance concentration X
    //   PUSH_SUBSTANCE(n) → substance[n] concentration (for cascade-driven mods)
    //   PUSH_GATE(g)     → gate_states_[g] (for gate-product-style mods)
    // Returns: scale factor (CHANNEL_G, *_SCALE, SYNAPSE_G) or direct value (CHANNEL_EREV, GATE_INF_EXPR)
    VmExpr mod_vm;

    // For GATE_INF_SHIFT with scalar scale: set mod_vm = empty and use this
    double shift_scale = 0.0;
};
```

### VmOp additions (`model/gate_spec.hpp`)

Two new opcodes:

```cpp
PUSH_X    = 19,   // push X_[operand] (substance concentrations by index; operand=0 = self)
PUSH_SUBST = 20,  // alias clarity: same as PUSH_X but explicit about cross-substance reference
```

In practice `PUSH_X` with operand=0 is "self", and `PUSH_X` with operand=n is "substance n". The distinction is only in the Python compiler — C++ evaluates both the same way.

### `NeuronModelSpec` change (`model/neuron_spec.hpp`)

```cpp
struct NeuronModelSpec {
    std::string name;
    double C_m = 1.0;
    double V_init = -65.0;
    std::vector<GateSpec>             gates;
    std::vector<ChannelSpec>          channels;
    std::vector<IntracellularSpec>    intracellular;  // replaces CalciumSpec calcium
    // ...izhikevich fields unchanged...
};
```

### `GateSpec::Dependency` change (`model/gate_spec.hpp`)

```cpp
enum class Dependency {
    VOLTAGE,
    INTRACELLULAR   // replaces CALCIUM; intracellular_idx selects which substance
};
int intracellular_idx = 0;  // which IntracellularSpec (default 0 = calcium by convention)
```

`CALCIUM` is kept as a Python-level deprecated alias mapping to `INTRACELLULAR` with `intracellular_idx=0`.

### `ChannelSpec` change (`model/channel_spec.hpp`)

```cpp
struct ChannelSpec {
    std::string name;
    double g = 0.0;
    double E_rev = 0.0;
    int    nernst_substance_idx = -1;  // -1 = none; >=0 = use E_nernst_[idx]
                                        // replaces use_calcium_nernst
    std::vector<std::pair<int, int>> gates;
    bool   is_ahp = false;
    double ahp_k1 = 0.0;
    int    ahp_substance_idx = 0;      // which substance drives AHP (default = calcium)
    VmExpr gate_product_vm;
};
```

`use_calcium_nernst` is kept as a deprecated Python alias → `nernst_substance_idx=0`.

---

## 14.4 `ComposablePool` Changes

Replace `Ca_` / `E_Ca_` with:

```cpp
// Per-substance state (one ArrayXd per IntracellularSpec in model)
std::vector<Eigen::ArrayXd> X_;         // concentrations
std::vector<Eigen::ArrayXd> E_nernst_;  // Nernst reversals (if nernst_enabled)

// Per-neuron synaptic modulation output (written after step, read by Network)
Eigen::ArrayXd synapse_g_scale_;        // net synaptic g multiplier per neuron
```

`step()` sequence after the voltage update:

```
for each IntracellularSpec (in order):
  1. Sum source channel currents into I_src (reuse existing channel loop results)
  2. Evaluate ode_vm or standard form → dX/dt
  3. X_[i] += dt * dX/dt; clamp X_[i] >= 0
  4. If nernst_enabled: update E_nernst_[i] via nernst_vm or standard Nernst formula
  5. Apply modulations:
     - CHANNEL_G: multiply channel g by mod(X) for subsequent steps (or apply inline)
     - CHANNEL_EREV: replace E_rev in channel loop
     - GATE_*: modify gate state or shift input for next step
     - SYNAPSE_G: write to synapse_g_scale_
```

For `CHANNEL_G` and `GATE_*` modulations applied **during** channel/gate computation, the pool maintains a `per_channel_g_mod_` and `per_gate_shift_` vector updated at the end of each step and applied at the start of the next. This avoids re-ordering the step sequence.

### New `PoolBase` methods:

```cpp
// Scatter substance concentrations for recording
virtual void scatter_substance_into(size_t subst_idx, double* buf, size_t n_rec, size_t t_rec) const;
// Scatter per-neuron synaptic g scaling for network modulation
virtual void scatter_synapse_g_scale(double* buf) const;
```

---

## 14.5 Network-Level Synapse Modulation

`SYNAPSE_G` modulation crosses the pool → network boundary. `Network` maintains:

```cpp
std::vector<double> synapse_g_scale_;  // per-neuron multiplier (default 1.0)
```

After `pool_mgr_.step_all(dt)`, pools that have `SYNAPSE_G` modulations call `scatter_synapse_g_scale()` into this buffer. `compute_synaptic_currents()` then applies it:

```cpp
// Normal:  I_buf[post[i]] += g[i] * (E_syn[i] - V[post[i]])
// With mod: I_buf[post[i]] += g[i] * synapse_g_scale_[post[i]] * (E_syn[i] - V[post[i]])
```

`synapse_g_scale_` is initialized to 1.0 at simulation start and reset to 1.0 at each step before scatter. Populations without `SYNAPSE_G` modulations don't write to it.

---

## 14.6 Python API

### Pre-defined substance symbols (`_codegen.py`)

```python
Ca   # calcium (existing)
DA   # dopamine
cAMP # cyclic AMP
IP3  # inositol trisphosphate
NO   # nitric oxide
X_ic # generic intracellular symbol
I_source  # source channel current sum (used in ode expressions)
```

`hh.substance(name)` creates a named SymPy symbol, re-using a pre-defined one if the name matches.

### `IntracellularDynamics` builder (`_equations/__init__.py`)

```python
import hodgkin_huxley as hh
import sympy as sp

Ca = hh.Ca
I_src = hh.I_source

calcium = hh.IntracellularDynamics(
    "calcium",
    ode=1e-4 * (-I_src - 15.0 * Ca),
    source_channels=["T"],            # channel names resolved at add_intracellular time
    nernst=8314*298/(2*96485) * sp.log(2000 / Ca),
    initial=0.1,
)
```

Pattern matching for ODE:
- `ode = -k * X` → `UpdateForm::DECAY`, `k_decay=k`
- `ode = ε * (-I_src - k * X)` → `UpdateForm::DRIVEN_DECAY`, `epsilon=ε, k_decay=k`
- Same + Nernst standard form `(RT/zF)*ln(X_o/X)` → `UpdateForm::DRIVEN_DECAY_NERNST`
- Anything else → `UpdateForm::CUSTOM_EXPR`, compiled via `compile_to_vm_bytecode()`

Pattern matching for Nernst:
- Standard form `(RT/zF) * log(X_o / X)` → scalar params extracted
- Other form → `nernst_vm`

### `Modulation` helper class

```python
DA = hh.substance("DA")

# Channel conductance scale (Hill kinetics)
hh.Modulation.channel_g("M", expr=1 - DA / (0.3 + DA))

# Channel reversal potential (custom Nernst)
hh.Modulation.channel_erev("T", expr=8314*298/(2*96485) * sp.log(2000 / hh.Ca))

# Gate inf voltage shift: x_inf(V + α*[DA])
hh.Modulation.gate_inf_shift("m_NMDA", scale=-5.0)

# Gate inf scale: x_inf_eff = x_inf * f([DA])
hh.Modulation.gate_inf_scale("n_M", expr=1 + DA / (0.5 + DA))

# Gate tau scale
hh.Modulation.gate_tau_scale("h_Na", expr=1 - 0.3 * DA)

# Fully custom gate inf (SymPy with V and X)
hh.Modulation.gate_inf_expr("r", expr=1 / (1 + sp.exp((hh.V + 84 + 10*DA) / 4)))

# Synaptic g scale (all incoming synapses to post-synaptic neurons in this population)
hh.Modulation.synapse_g(expr=1 + DA / (0.3 + DA))
```

`Modulation` is a plain dataclass — just stores `(target, target_name, expr)`. Resolution of `target_name` to a channel/gate index happens in `add_intracellular()`.

### `IntracellularDynamics` with modulations

```python
DA = hh.substance("DA")

dopamine = hh.IntracellularDynamics(
    "dopamine",
    ode=-0.01 * DA,           # first-order decay, no channel sources
    initial=1.0,
    modulations=[
        hh.Modulation.channel_g("M", expr=1 - DA / (0.3 + DA)),
    ]
)
```

Cascade example (dopamine → cAMP):

```python
DA  = hh.substance("DA")
cAMP = hh.substance("cAMP")

dopamine = hh.IntracellularDynamics("dopamine", ode=-0.01 * DA, initial=1.0)

cAMP_dyn = hh.IntracellularDynamics(
    "cAMP",
    ode=0.5 * DA / (0.1 + DA) - 0.02 * cAMP,  # driven by DA, degrades
    # DA is resolved at to_spec() time to substance[0] = dopamine's index
    initial=0.0,
)
```

### `RegionalNetwork.add_intracellular()`

```python
net.add_intracellular(calcium, populations=["STN", "GPe"])
net.add_intracellular(calcium, populations="GPi")         # single string ok
net.add_intracellular(dopamine)                           # None = all populations
```

Resolution steps:
1. For each named population, retrieve its `NeuronModelSpec`
2. Validate channel/gate names in modulations exist in that spec
3. Resolve names to indices
4. Append the `IntracellularSpec` (with resolved indices) to `spec.intracellular`

Because `IntracellularSpec` is self-contained data, adding to multiple populations just stores copies (cheap — only scalar params and small VmExpr bytecode).

Cross-substance references in ODEs (e.g., `cAMP_dyn.ode` references `DA`) are resolved by name at `add_intracellular()` time: the `DA` symbol is matched to the substance named "dopamine" already in the spec's `intracellular` list. The `PUSH_X(idx)` opcode is populated with the correct index.

---

## 14.7 Standard Presets (Backwards Compatibility)

All five existing presets update internally; their Python-facing signatures are unchanged.

`NeuronModelSpec.stn()`, `.gpe()`, `.gpi()` — gain calcium `IntracellularSpec` with `DRIVEN_DECAY_NERNST` form (scalar params, no VM). Functionally identical to current `CalciumSpec` behavior.

`CalciumSpec` is kept in `legacy.py` as a Python shim that constructs `IntracellularDynamics` with the appropriate parameters. Emits `DeprecationWarning`.

`ChannelSpec.use_calcium_nernst = True` → `nernst_substance_idx = 0` in the binding. Deprecated at the binding level with a warning.

`GateSpec::Dependency::CALCIUM` → `INTRACELLULAR` with `intracellular_idx=0` at the C++ level. Python binding keeps `CALCIUM` as a deprecated alias.

---

## 14.8 Recording Extension

`RecordingConfig` gains an `intracellular=True` flag (defaults off — substances can be large arrays).

`PopulationMetricsResult` gains:
```python
result["STN"].substances          # dict: name → (n_neurons, n_steps) array
result["STN"].substances["Ca"]    # Ca time series
```

`PoolBase` virtual method:
```cpp
virtual void scatter_substance_into(
    size_t subst_idx, double* buf, size_t n_rec, size_t t_rec) const {}
```

`ComposablePool` overrides to scatter `X_[subst_idx]`.

---

## 14.9 Worked Examples

### Calcium (existing presets, new system)

```python
# Existing API — still works, routed through new system:
stn = hh.NeuronModelSpec.stn()           # internally uses IntracellularDynamics
gpe = hh.NeuronModelSpec.gpe()
net.add_population("STN", 10, stn)
net.add_population("GPe", 10, gpe)
# No change needed — CalciumSpec presets baked into the model spec.
```

### Dopamine modulation in striatum

```python
DA = hh.substance("DA")

dopamine = hh.IntracellularDynamics(
    "dopamine",
    ode=-0.01 * DA,   # first-order volume-transmission decay
    initial=1.0,
    modulations=[
        # D2 receptors reduce M-current
        hh.Modulation.channel_g("M", expr=1.0 - DA / (0.3 + DA)),
    ]
)

net.add_population("Str_D2", 10, hh.NeuronModelSpec.striatum(pd=1))
net.add_intracellular(dopamine, populations="Str_D2")
```

### IP3-gated calcium release (Li-Rinzel style)

```python
Ca, IP3 = hh.Ca, hh.substance("IP3")
X_h = hh.substance("h_IP3")  # inactivation variable

ip3_dyn = hh.IntracellularDynamics(
    "IP3",
    ode=-0.05 * IP3,                                        # simple decay
    initial=0.1,
)

h_dyn = hh.IntracellularDynamics(
    "h_IP3",
    ode=sp.Float(0.2) * (                                   # Li-Rinzel h gate
        (1 - X_h) / (1 + Ca / sp.Float(0.3)) -
        X_h * Ca / (Ca + sp.Float(0.1))
    ),
    initial=0.8,
)

ca_release = hh.IntracellularDynamics(
    "calcium",
    ode=1e-4 * (-hh.I_source - 15.0 * Ca)
        + sp.Float(0.3) * (IP3*Ca*X_h)**3 / ((IP3+sp.Float(0.1))*(Ca+sp.Float(0.3)))**3
          * (sp.Float(0.4) - Ca),  # IP3R flux term
    source_channels=["L_Ca"],
    nernst=8314*298/(2*96485) * sp.log(2000 / Ca),
    initial=0.05,
)

net.add_intracellular(ip3_dyn,     populations="CTX")
net.add_intracellular(h_dyn,       populations="CTX")
net.add_intracellular(ca_release,  populations="CTX")
```

---

## 14.10 Implementation Checklist

### C++ Core (`model/gate_spec.hpp`, `model/channel_spec.hpp`, `model/neuron_spec.hpp`)
- [ ] Define `IntracellularSpec::UpdateForm` enum and `IntracellularSpec` struct
- [ ] Define `IntracellularModulation::Target` enum and `IntracellularModulation` struct
- [ ] Add `PUSH_X = 19` to `VmOp` enum (operand = substance index; 0 = self)
- [ ] Add `intracellular: vector<IntracellularSpec>` to `NeuronModelSpec`; remove `CalciumSpec calcium`
- [ ] Change `GateSpec::Dependency::CALCIUM` → `INTRACELLULAR`; add `intracellular_idx` field
- [ ] Change `ChannelSpec::use_calcium_nernst` → `nernst_substance_idx: int`; add `ahp_substance_idx`

### `ComposablePool` (`composable_pool.hpp` / `.cpp`)
- [ ] Replace `Ca_`, `E_Ca_` with `vector<ArrayXd> X_`, `vector<ArrayXd> E_nernst_`
- [ ] Add `synapse_g_scale_` (ArrayXd) for SYNAPSE_G modulations
- [ ] Update `add()` to initialize `X_` per substance from `IntracellularSpec::initial`
- [ ] Update gate dependency resolution: `INTRACELLULAR` uses `X_[intracellular_idx]`
- [ ] Update channel Nernst lookup: use `E_nernst_[nernst_substance_idx]` instead of `E_Ca_`
- [ ] Update AHP channel: `X_[ahp_substance_idx]` instead of `Ca_`
- [ ] Implement substance update loop in `step()`:
  - [ ] Sum source channel currents for each substance
  - [ ] Evaluate DECAY / DRIVEN_DECAY / DRIVEN_DECAY_NERNST / CUSTOM_EXPR ODE
  - [ ] Clamp concentration >= 0
  - [ ] Update `E_nernst_` if enabled (standard or VM)
  - [ ] Apply modulations to per-channel g modifiers and gate shift arrays
  - [ ] Write `synapse_g_scale_` for SYNAPSE_G modulations
- [ ] Add `vm_eval_substance()` evaluation function (handles PUSH_X with concentration array access)
- [ ] Override `scatter_substance_into()` for recording
- [ ] Override `scatter_synapse_g_scale()` for network modulation
- [ ] Update `sync_to_neurons()` to sync `X_[0]` as calcium where applicable

### `Network` (`network.hpp` / `.cpp`)
- [ ] Add `synapse_g_scale_: vector<double>` (per neuron, default 1.0)
- [ ] Initialize to 1.0 at simulation start; reset to 1.0 each step before pool scatter
- [ ] Call `pool_mgr_.scatter_synapse_g_scale(synapse_g_scale_)` after `step_all()`
- [ ] Apply in `compute_synaptic_currents()`: `I_buf[post] += g * synapse_g_scale_[post] * (E_syn - V_post)`

### `PoolBase` (`pool_base.hpp`)
- [ ] Add `virtual scatter_substance_into(idx, buf, n_rec, t_rec)` (default no-op)
- [ ] Add `virtual scatter_synapse_g_scale(buf)` (default no-op — HH/Iz pools don't modify it)

### Python bindings (`bindings.cpp`)
- [ ] Bind `IntracellularSpec`, `IntracellularSpec::UpdateForm`, `IntracellularModulation`, `IntracellularModulation::Target`
- [ ] Bind `NeuronModelSpec.intracellular` (list of `IntracellularSpec`)
- [ ] Bind `GateSpec::Dependency::INTRACELLULAR` and `intracellular_idx`
- [ ] Bind `ChannelSpec.nernst_substance_idx`, `ahp_substance_idx`
- [ ] Keep `GateSpec::Dependency::CALCIUM`, `ChannelSpec::use_calcium_nernst` as deprecated aliases emitting Python DeprecationWarning via property setter

### Python API (`_equations/__init__.py`)
- [ ] Add `IntracellularDynamics` builder class with SymPy ODE + Nernst + modulations
- [ ] Add `Modulation` helper class with class methods for each target type
- [ ] Pattern matching for ODE: DECAY, DRIVEN_DECAY, DRIVEN_DECAY_NERNST
- [ ] Pattern matching for Nernst: standard `(RT/zF)*ln(X_o/X)` form
- [ ] `compile_to_vm_bytecode()` extended to handle `PUSH_X(idx)` for substance symbols
- [ ] `hh.substance(name)` symbol helper (caches by name; re-uses pre-defined Ca, DA etc.)

### `RegionalNetwork` (`_network/__init__.py`)
- [ ] Add `add_intracellular(dynamics, populations=None)` method
- [ ] Validate channel/gate names against population NeuronModelSpec
- [ ] Resolve names → indices; insert `IntracellularSpec` into spec's intracellular list
- [ ] Resolve cross-substance references (e.g., DA symbol in cAMP ODE → substance index)

### Recording (`recording.py`)
- [ ] Add `intracellular=False` flag to `RecordingConfig`
- [ ] Allocate per-substance recording buffers when `intracellular=True`
- [ ] Call `scatter_substance_into(i, ...)` for each substance during hot loop
- [ ] Populate `PopulationMetricsResult.substances` dict

### `legacy.py`
- [ ] Add `CalciumSpec` shim → `IntracellularDynamics` with standard calcium params
- [ ] Add deprecation for `use_calcium_nernst` kwarg in Python bindings

### Tests
- [ ] All five existing calcium presets produce numerically identical output (regression)
- [ ] Single decay substance (`DECAY` form): concentration decays at correct rate
- [ ] Driven substance (`DRIVEN_DECAY`): concentration tracks channel current correctly
- [ ] Nernst update: E_Ca tracks concentration in STN simulation
- [ ] Calcium-dependent gate (`INTRACELLULAR` dependency): AHP gate responds to Ca changes
- [ ] Dopamine modulation (`CHANNEL_G`): M-channel g scales with DA concentration
- [ ] Cross-substance ODE: cAMP concentration driven by DA in cascade
- [ ] `add_intracellular` to multiple populations: both populations track the substance
- [ ] `SYNAPSE_G` modulation: synaptic current scales with substance in post-synaptic pop
- [ ] Substance recording: `result.substances["Ca"]` has correct shape and values
- [ ] `CalciumSpec` deprecated alias still works (warns but functions)
