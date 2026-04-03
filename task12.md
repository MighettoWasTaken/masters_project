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
- [x] Prefix `Network` as `_Network` in bindings (C++ class name unchanged) — bound as `_Network`; `NeuronType` enum bound as `_NetworkNeuronType`
- [x] Remove `add_hh_neuron()`, `add_izhikevich_neuron()`, `NetworkNeuronType` from public binding exports — removed from `__all__`; `_NetworkNeuronType` accessible only via `legacy`
- [x] Remove `_simulate_into_buffers` and `_simulate_with_descriptors` from public binding exports — still defined on `_Network` internally but not exported in `__all__`
- [x] Add `NeuronModelSpec.hh_default()` and `NeuronModelSpec.izhikevich(type)` static factory methods — implemented in bindings

### Python Layer
- [x] Create `hodgkin_huxley/legacy.py` — re-exports `HHNeuron`, `IzhikevichNeuron`, `HHParameters`, `HHState`, `IzhikevichParameters`, `IzhikevichState`, `Network`, `NetworkNeuronType`, `SynapseBase`, `ExponentialSynapse`, `AlphaSynapse`, `DoubleExponentialSynapse`; all emit `DeprecationWarning` via `__getattr__`
- [x] Update `__init__.py` to trimmed structural export list — deprecated names removed from `__all__`; accessible via module `__getattr__` which delegates to `legacy`
- [x] `_HHNeuronWrapper`, `_IzhikevichNeuronWrapper`, `_NetworkWrapper` defined in `__init__.py` and returned by `legacy.__getattr__` for the three class names
- [x] dict `I_ext` routing — `RegionalNetwork.simulate()` accepts per-population dict with scalar / stimulator / 1D array per key; auto-routes to descriptor or dense path
- [x] `examples/` updated to `RegionalNetwork` and dict `I_ext` (verified: no raw `Network` or deprecated neuron-add methods in examples)
- [x] `benchmarks/ctxbgth_model.py` updated to new structural API (verified: no deprecated symbols in benchmarks)

### Tests
- [x] `tests/python/test_api_cleanup.py` added covering:
  - `NeuronModelSpec.hh_default()` gates/channels count and spiking behaviour
  - `NeuronModelSpec.izhikevich()` RS and FS variants fire correctly
  - `DeprecationWarning` for all 12 legacy names via `hodgkin_huxley.legacy`
  - `hh.Network` and `hh.HHNeuron` accessed from top-level also emit `DeprecationWarning`
  - `__all__` contains no removed symbols; `RegionalNetwork` and `NeuronModelSpec` present
  - dict `I_ext` routing for scalar, 1D array, stimulator, and missing-key-defaults-to-zero cases
- [x] All 751 existing tests pass
