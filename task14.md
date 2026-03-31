# Task 14: Generalized Intracellular Dynamics

## Priority: 2 — Depends on task13 (ODE fields use SymPy expressions)

## Overview

Extend the current calcium-only intracellular tracking to a general-purpose system for arbitrary intracellular substances. Calcium is the only substance currently modeled; many biologically relevant phenomena require additional intracellular dynamics — notably dopamine (modulates M-current and plasticity in striatum), cAMP (second messenger), and IP3 (triggers calcium release from internal stores). The goal is a flexible `IntracellularSpec` system that makes calcium a special case of a broader mechanism, while preserving the existing calcium presets and pool performance.

---

## 14.1 Current State

`CalciumSpec` in `ion_channels.hpp` handles one substance with two modes:

- **Simple decay** (GPe/GPi): `d[Ca]/dt = epsilon * (-sum(I_sources) - K_Ca * Ca)`
- **Nernst** (STN): same ODE + dynamic `E_Ca = (RT/zF) * ln(Ca_o / Ca)`

`ComposablePool` maintains one `Eigen::ArrayXd Ca_` per pool and one `Eigen::ArrayXd E_Ca_` if Nernst is enabled. Calcium-dependent gates reference `Ca_` via `GateDependency::CALCIUM`.

**Limitations:**
- Hardcoded to one substance per neuron model
- No mechanism for one substance to modulate another's dynamics
- No way to express volume-transmitted modulators (dopamine, serotonin)
- `GateDependency::CALCIUM` must be generalised to reference any substance

---

## 14.2 Architecture

### Core Principle

Each `NeuronModelSpec` carries a list of `IntracellularSpec` objects. Each spec defines:
1. **Dynamics**: how the substance concentration changes each step — expressed as a SymPy ODE (see task13)
2. **Modulations**: what simulation parameters it adjusts (channel g, gate x_inf, gate tau)
3. **Nernst expression**: optional SymPy expression computing E_rev from concentration

Calcium is expressed as an `IntracellularSpec` with a Nernst expression. All existing presets continue to work via updated factory functions.

### C++ Structs

```cpp
/// Target types for concentration-dependent parameter modulation.
enum class ModulationTarget {
    CHANNEL_G_SCALE,      // g_eff = g * (1 + scale * X)
    GATE_INF_SHIFT,       // x_inf evaluated at (V + scale * X)
    GATE_TAU_SCALE,       // tau_eff = tau * (1 + scale * X)
};

/// A single modulation: substance X → parameter adjustment.
struct IntracellularModulation {
    int               substance_idx;  // index into NeuronModelSpec::intracellular
    ModulationTarget  target;
    int               target_idx;    // channel or gate index
    double            scale;         // linear coefficient
    // Note: non-linear modulations expressed via SymPy (task13) in future
};

/// Complete specification of one intracellular substance.
/// ODE and Nernst expressions are SymPy expressions compiled via task13 codegen.
struct IntracellularSpec {
    std::string name;
    double      initial;
    // SymPy-compiled function pointers (set at first simulate() call)
    CompiledFn  ode_fn;       // dX/dt = f(X, I_source_sum)
    CompiledFn  nernst_fn;    // E_rev = g(X), optional
    std::vector<int> source_channels;
    std::vector<IntracellularModulation> modulations;
};
```

### NeuronModelSpec Changes

Replace the single `CalciumSpec calcium` field:

```cpp
struct NeuronModelSpec {
    std::string name;
    double C_m = 1.0;
    std::vector<GateSpec>          gates;
    std::vector<ChannelSpec>       channels;
    std::vector<IntracellularSpec> intracellular;  // replaces CalciumSpec

    // Index of substance providing reversal potential to Nernst channels (-1 = none)
    int nernst_substance_idx = -1;
};
```

### GateDependency Extension

```cpp
enum class GateDependency {
    VOLTAGE,
    INTRACELLULAR   // replaces CALCIUM; gate_spec.intracellular_idx selects which substance
};

struct GateSpec {
    // ...existing fields...
    GateDependency dependency      = GateDependency::VOLTAGE;
    int            intracellular_idx = 0;  // which IntracellularSpec (ignored if VOLTAGE)
};
```

With SymPy (task13), the dependency is inferred automatically from which symbols appear in the `inf` expression — if `hh.Ca` appears, it is `INTRACELLULAR` referencing the substance named "calcium".

### ComposablePool Changes

```cpp
class ComposablePool {
private:
    // Per-substance state (one ArrayXd per IntracellularSpec in model)
    std::vector<Eigen::ArrayXd> X_;        // concentrations
    std::vector<Eigen::ArrayXd> E_nernst_; // Nernst reversals (if nernst_fn set)
};
```

The `step()` function iterates over substances in model order, updating each with its compiled ODE function after channel currents are computed.

---

## 14.3 Backwards Compatibility

`CalciumSpec` is kept as a Python-level convenience builder that constructs the equivalent `IntracellularSpec`. All five existing presets (`thalamic()`, `stn()`, `gpe()`, `gpi()`, `striatum()`) are updated internally; their Python-facing API is unchanged.

---

## 14.4 New Use Case: Dopamine Modulation

```python
Ca, I_src = hh.Ca, hh.I_source

# Calcium (standard Nernst mode)
calcium = hh.IntracellularSpec(
    name="calcium",
    ode=1e-4 * (-I_src - 2e-3 * Ca),
    source_channels=["L_Ca", "T_Ca"],
    nernst=8314*298/(2*96485) * log(2000 / Ca),
    initial=0.005
)

# Dopamine: externally driven, decays slowly, modulates Str M-channel
DA = hh.symbols("DA")
dopamine = hh.IntracellularSpec(
    name="dopamine",
    ode=-0.01 * DA,          # simple first-order decay; no channel source
    source_channels=[],
    initial=1.0,
    modulations=[
        hh.IntracellularModulation(
            substance="dopamine",
            target=hh.ModulationTarget.CHANNEL_G_SCALE,
            channel="M",
            scale=-1.1        # g_M_eff = g_M * (1 + (-1.1) * [DA])
        )
    ]
)

str_model.add_intracellular(calcium)
str_model.add_intracellular(dopamine)
```

---

## 14.5 Implementation Checklist

### C++ Core
- [ ] Define `IntracellularModulation`, `IntracellularSpec`, `ModulationTarget` in `ion_channels.hpp`
- [ ] Replace `CalciumSpec calcium` in `NeuronModelSpec` with `std::vector<IntracellularSpec> intracellular`
- [ ] Update `GateDependency::CALCIUM` → `GateDependency::INTRACELLULAR` + `intracellular_idx` in `GateSpec`
- [ ] Update `ComposablePool`: replace `Ca_` / `E_Ca_` with `std::vector<Eigen::ArrayXd>` for N substances
- [ ] Update `ComposablePool::step()`: iterate over substances, apply modulations before channel current summation
- [ ] Update existing presets to use `IntracellularSpec` internally

### Python Bindings
- [ ] Bind `IntracellularModulation`, `IntracellularSpec`, `ModulationTarget`
- [ ] Keep `CalciumSpec` as deprecated alias mapping to new system
- [ ] Expose `NeuronModelSpec.add_intracellular()`

### Tests
- [ ] Verify calcium dynamics unchanged for all five existing presets
- [ ] Test dopamine modulation: g_M_eff scales with dopamine concentration
- [ ] Test Nernst pathway: E_nernst updates correctly for generalised substance
- [ ] Test calcium-dependent gate still updates correctly via `intracellular_idx`
