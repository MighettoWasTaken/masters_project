# Task 12: API Structural Cleanup

## Priority: 2 — First to implement; no dependencies

## Overview

The library has accumulated multiple overlapping entry points as features were added incrementally. This task removes all structural redundancy — network class ambiguity, duplicate neuron-add paths, exposed internal simulation methods, inconsistent I_ext interface — without touching equation-definition types (those are addressed in task13). Completing this first ensures all subsequent feature tasks build on a clean, consistent API surface.

---

## 12.1 Network Class Ambiguity

`Network` is the low-level class; `RegionalNetwork` wraps it with named populations. There is no documented guidance on when to use which, and `Network` is fully exposed in the public API.

**Change:** `Network` becomes `_Network` (internal). `RegionalNetwork` is the sole public network class. Single-population use cases work naturally with one population.

Migration:
```python
# Old
net = Network(10)
net.add_synapse(0, 1, 0.5)

# New
net = RegionalNetwork()
net.add_population("neurons", 10, NeuronModelSpec.hh_default())
net.add_connection("neurons", 0, "neurons", 1, 0.5, SynapseSpec.exponential(0.0, 2.0))
```

---

## 12.2 Redundant Neuron Add Methods

All of the following add a default HH neuron — four paths for one operation:
```python
net.add_neuron()
net.add_hh_neuron()
net.add_neuron(NetworkNeuronType.HH)
net.add_neuron(HHParameters())
```

**Change:** Remove from public API. Neuron type is always expressed via `NeuronModelSpec`. Add two new presets:

```python
NeuronModelSpec.hh_default()                             # replaces NetworkNeuronType.HH
NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING)  # replaces IZHIKEVICH_FS etc.
```

`NetworkNeuronType` enum is removed from the public API.

---

## 12.3 Scalar Neuron Classes

`HHNeuron` and `IzhikevichNeuron` are full public API classes with their own `simulate()`, `step()`, `parameters`, and `state` properties. They predate the composable system and represent a usage pattern (one neuron at a time) that is not how the library simulates networks.

**Change:** Move to `hodgkin_huxley.legacy`. Still importable with no breakage, but not in the top-level namespace. Emit `DeprecationWarning` on import from legacy.

```python
# Old (still works, raises DeprecationWarning)
from hodgkin_huxley import HHNeuron

# New location
from hodgkin_huxley.legacy import HHNeuron
```

---

## 12.4 Exposed Internal Simulation Methods

`_simulate_into_buffers()` and `_simulate_with_descriptors()` are prefixed `_` but still importable via bindings and appear in examples, creating confusion about which simulation path to use.

**Change:** Remove from public binding exports entirely. `RegionalNetwork.simulate()` routes to the correct internal path automatically; this is not user-visible.

---

## 12.5 I_ext Interface Unification

`RegionalNetwork.simulate()` currently accepts I_ext as a float, 1D array, 2D array, dict, or `_StimPlan` with no clear primary form.

**Change:** Per-population dict is the documented primary interface:

```python
I_ext = {"CTX": 10.0}                     # scalar constant for all neurons in pop
I_ext = {"CTX": DBSStimulator(...)}        # on-the-fly stimulator object
I_ext = {"CTX": np.array([10.0, 12.0])}   # per-neuron 1D array (length = pop size)
```

Dense 2D arrays (`n_neurons × n_steps`) remain accepted for backwards compatibility but are not documented as a primary interface.

---

## 12.6 Canonical Workflow

After this task, the intended flow for all simulations:

```python
import hodgkin_huxley as hh

# 1. Define models (presets or composable specs)
ctx_model = hh.NeuronModelSpec.izhikevich(hh.IzhikevichType.REGULAR_SPIKING)
stn_model = hh.NeuronModelSpec.stn()

# 2. Build network
net = hh.RegionalNetwork()
net.add_population("CTX", 10, ctx_model)
net.add_population("STN", 10, stn_model)
net.connect("CTX", "STN",
    pattern=hh.ConnectivityPattern.ONE_TO_ONE,
    synapse=hh.SynapseSpec.ampa(),
    weight=hh.WeightDistribution.constant(0.5))

# 3. Simulate
result = net.simulate(
    duration=2000.0, dt=0.01,
    I_ext={"CTX": 10.0, "STN": 0.0},
    recording=hh.RecordingConfig.all_neuron_metrics()
)

# 4. Analyse
print(result["CTX"].firing_rate.mean())
```

Note: equation-definition types (`GateSpec`, `TauParams`, etc.) are not changed by this task. That is task13.

---

## 12.7 Export Cleanup (Structural Only)

**`hodgkin_huxley` top-level after this task:**
`RegionalNetwork`, `NeuronModelSpec`, `SynapseSpec`, `WeightDistribution`, `ConnectivityPattern`, `GateSpec`, `ChannelSpec`, `KineticSynapseSpec`, `CalciumSpec`, `RecordingConfig`, `IzhikevichType`, `IntegrationMethod`, `DBSStimulator`, `PulseStimulator`, `NoiseInjector`, `BoltzmannParams`, `TauParams`, `TauForm`, `RateFuncParams`, `RateFuncForm`, `GateUpdateForm`, `GateDependency`, `KineticUpdateForm`, `KineticCurrentForm`, `analyze_beta_power`

*(Equation-definition types remain in top-level for now; moved to legacy in task13.)*

**`hodgkin_huxley.legacy` after this task:**
`HHNeuron`, `IzhikevichNeuron`, `Network`, `HHParameters`, `HHState`, `IzhikevichParameters`, `IzhikevichState`, `NetworkNeuronType`

**Removed entirely:**
`_simulate_into_buffers`, `_simulate_with_descriptors`, `add_hh_neuron()`, `add_izhikevich_neuron()`

---

## Implementation Checklist

### Bindings / C++
- [ ] Prefix `Network` as `_Network` in bindings (C++ class name unchanged)
- [ ] Remove `add_hh_neuron()`, `add_izhikevich_neuron()`, `NetworkNeuronType` from public binding exports
- [ ] Remove `_simulate_into_buffers` and `_simulate_with_descriptors` from public binding exports
- [ ] Add `NeuronModelSpec.hh_default()` and `NeuronModelSpec.izhikevich(type)` static factory methods

### Python Layer
- [ ] Create `hodgkin_huxley/legacy.py` re-exporting `HHNeuron`, `IzhikevichNeuron`, `HHParameters`, `HHState`, `IzhikevichParameters`, `IzhikevichState`, `Network`, `NetworkNeuronType` with `DeprecationWarning`
- [ ] Update `__init__.py` to the trimmed structural export list (keep equation types for now)
- [ ] Unify `I_ext` dispatch in `RegionalNetwork.simulate()`: scalar / stimulator / 1D array per population dict → auto-route to descriptor or dense path
- [ ] Remove legacy non-dict `I_ext` overloads from `RegionalNetwork.simulate()`

### Downstream
- [ ] Update all `examples/` to use `RegionalNetwork` and dict `I_ext`
- [ ] Update `benchmarks/ctxbgth_model.py` to new structural API

### Tests
- [ ] Test `NeuronModelSpec.hh_default()` and `NeuronModelSpec.izhikevich()` produce correct dynamics
- [ ] Test that `from hodgkin_huxley.legacy import HHNeuron` raises `DeprecationWarning`
- [ ] Test dict `I_ext` routing for scalar, stimulator, and 1D array forms
- [ ] Verify all existing tests pass (or update imports to legacy path)
