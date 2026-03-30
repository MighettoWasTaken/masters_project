# Task 12: Generalized Intracellular Dynamics

## Priority: 2

## Overview

Extend the current calcium-only intracellular tracking to a general-purpose system for arbitrary intracellular substances. Calcium is the only substance currently modeled; many biologically relevant phenomena require additional intracellular dynamics — notably dopamine (modulates M-current and plasticity in striatum), cAMP (second messenger), and IP3 (triggers calcium release from internal stores). The goal is a flexible `IntracellularSpec` system that makes calcium a special case of a broader mechanism, while preserving the existing calcium presets and pool performance.

---

## 12.1 Current State

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

## 12.2 Architecture

### Core Principle

Each `NeuronModelSpec` carries a list of `IntracellularSpec` objects. Each spec defines:
1. **Dynamics**: how the substance concentration changes each step
2. **Modulations**: what simulation parameters it adjusts (channel g, gate x_inf, gate tau)
3. **Nernst flag**: if true, compute a reversal potential from this concentration

Calcium is expressed as an `IntracellularSpec` with `use_nernst = true`. All existing presets continue to work via updated factory functions.

### C++ Structs

```cpp
/// Dynamics of a single intracellular substance concentration X.
///   dX/dt = scale * (-sum(I_source_channels) - decay_rate * X)
struct IntracellularDynamicsSpec {
    double scale       = 1e-4;   // epsilon: conversion factor
    double decay_rate  = 15.0;   // K_X: first-order decay (ms^-1)
    double initial     = 0.1;    // initial concentration
    std::vector<int> source_channels;  // channel indices that contribute I to dX/dt

    // Nernst reversal potential (for ions, e.g. Ca2+)
    bool   use_nernst  = false;
    double X_o         = 2000.0; // extracellular concentration (Nernst denominator)
    double z           = 2.0;    // valence
    double F           = 96485.0, R = 8314.0, T = 298.0;
};

/// Target types for concentration-dependent parameter modulation.
enum class ModulationTarget {
    CHANNEL_G_SCALE,      // g_eff = g * (1 + scale * X)
    GATE_INF_SHIFT,       // x_inf computed at (V + scale * X) instead of V
    GATE_TAU_SCALE,       // tau_eff = tau * (1 + scale * X)
};

/// A single modulation: substance X → parameter adjustment.
struct IntracellularModulation {
    int               substance_idx;  // index into NeuronModelSpec::intracellular
    ModulationTarget  target;
    int               target_idx;    // channel or gate index
    double            scale;         // linear coefficient
};

/// Complete specification of one intracellular substance.
struct IntracellularSpec {
    std::string name;
    IntracellularDynamicsSpec dynamics;
    std::vector<IntracellularModulation> modulations;
};
```

### NeuronModelSpec Changes

Replace the single `CalciumSpec calcium` field:

```cpp
struct NeuronModelSpec {
    std::string name;
    double C_m = 1.0;
    std::vector<GateSpec>         gates;
    std::vector<ChannelSpec>      channels;
    std::vector<IntracellularSpec> intracellular;   // replaces CalciumSpec

    // Convenience: index of substance providing reversal potential to
    // channels with use_calcium_nernst=true (-1 = none)
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
    GateDependency dependency = GateDependency::VOLTAGE;
    int intracellular_idx = 0;  // which IntracellularSpec (ignored if dependency=VOLTAGE)
};
```

### ComposablePool Changes

```cpp
class ComposablePool {
    // ...
private:
    // Per-substance state (one ArrayXd per IntracellularSpec in model)
    std::vector<Eigen::ArrayXd> X_;       // concentrations
    std::vector<Eigen::ArrayXd> E_nernst_; // Nernst reversals (if use_nernst)
    // ...
};
```

The `step()` function iterates over substances in model order, updating each with its dynamics ODE after channel currents are computed.

---

## 12.3 Backwards Compatibility

`CalciumSpec` is kept as a Python-level convenience builder that constructs the equivalent `IntracellularSpec`:

```python
# Old (still works)
spec.set_calcium(mode="nernst", Ca_init=0.005, ...)

# New equivalent
spec.add_intracellular(IntracellularSpec(
    name="calcium",
    dynamics=IntracellularDynamicsSpec(use_nernst=True, initial=0.005, ...),
))
```

All five existing presets (`thalamic()`, `stn()`, `gpe()`, `gpi()`, `striatum()`) are updated to use the new system internally; their Python-facing API is unchanged.

---

## 12.4 New Use Case: Dopamine Modulation

The primary new use case is dopamine modulation of striatal M-current, replacing the PD scaling parameter with a dynamic dopamine concentration:

```python
# Dopamine: externally driven (no source channels), decays slowly
# Modulates Str M-channel conductance
dopamine = IntracellularSpec(
    name="dopamine",
    dynamics=IntracellularDynamicsSpec(
        scale=0.0,         # no ion-current source
        decay_rate=0.01,   # slow clearance
        initial=1.0,       # baseline tonic level
        source_channels=[]
    ),
    modulations=[
        IntracellularModulation(
            substance_idx=0,
            target=ModulationTarget.CHANNEL_G_SCALE,
            target_idx=3,    # M channel index in striatum spec
            scale=-1.1       # g_M_eff = g_M * (1 + (-1.1) * [DA])
        )
    ]
)

str_model.add_intracellular(dopamine)
```

---

## 12.5 Implementation Checklist

### C++ Core
- [ ] Define `IntracellularDynamicsSpec`, `IntracellularModulation`, `IntracellularSpec` in `ion_channels.hpp`
- [ ] Replace `CalciumSpec calcium` in `NeuronModelSpec` with `std::vector<IntracellularSpec> intracellular`
- [ ] Generalise `GateDependency::CALCIUM` → `GateDependency::INTRACELLULAR` + `intracellular_idx` field in `GateSpec`
- [ ] Update `ComposablePool`: replace `Ca_` / `E_Ca_` with `std::vector<Eigen::ArrayXd>` for N substances
- [ ] Update `ComposablePool::step()`: iterate over substances, apply modulations before channel current summation
- [ ] Update existing presets to use `IntracellularSpec` internally

### Python Bindings
- [ ] Bind `IntracellularDynamicsSpec`, `IntracellularModulation`, `IntracellularSpec`, `ModulationTarget`
- [ ] Keep `CalciumSpec` as deprecated alias mapping to new system
- [ ] Expose `NeuronModelSpec.add_intracellular()`

### Tests
- [ ] Verify calcium dynamics unchanged for all five existing presets
- [ ] Test dopamine modulation: g_M_eff scales with dopamine concentration
- [ ] Test Nernst pathway: E_nernst updates correctly for generalised substance
- [ ] Test calcium-dependent gate still updates correctly via `intracellular_idx`
