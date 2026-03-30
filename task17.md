# Task 17: API Streamlining

## Priority: 2

## Overview

The library has accumulated multiple overlapping entry points for the same operations as features were added incrementally. A new user faces three ways to add a neuron, two network classes of ambiguous scope, and simulation methods that expose internal buffer management. This task removes redundancy, establishes a single clear workflow, and hides implementation details behind a clean interface — without sacrificing the flexibility needed for research use.

The goal: a user should be able to build and run any simulation following one canonical pattern with no need to consult internal implementation details.

---

## 17.1 Current Redundancies

### Neuron Creation (Network)
All of these add a default HH neuron — four paths for one operation:
```python
net.add_neuron()
net.add_hh_neuron()
net.add_neuron(NetworkNeuronType.HH)
net.add_neuron(HHParameters())
```

### Network Class Ambiguity
- `Network`: population-unaware; users must track index ranges manually
- `RegionalNetwork`: wraps `Network` with named populations
- No clear guidance on when to use which; `Network` is overly exposed

### Simulation Entry Points
```python
net.simulate(...)                      # Python wrapper
net._simulate_into_buffers(...)        # exposed via bindings
net._simulate_with_descriptors(...)    # exposed via bindings
```

The low-level buffer methods are prefixed `_` but still importable and used in examples, creating confusion about which is "correct".

### Scalar Neuron Classes
`HHNeuron` and `IzhikevichNeuron` are full public API classes with their own `simulate()`, `step()`, `parameters`, `state` properties. They exist primarily for unit testing and single-neuron exploration, but add surface area and imply a usage pattern (one-neuron-at-a-time) that is not how the library runs simulations.

---

## 17.2 Unified Workflow

The canonical pattern after this task:

```python
import hodgkin_huxley as hh

# 1. Define models (using composable specs or presets)
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

# 3. Stimulate and simulate
result = net.simulate(
    duration=2000.0,
    dt=0.01,
    I_ext={"CTX": 10.0, "STN": 0.0},
    recording=hh.RecordingConfig.all_neuron_metrics()
)

# 4. Analyse
print(result["CTX"].firing_rate.mean())
hh.analyze_beta_power(result["GPi"])
```

Everything else is either a preset convenience or an advanced option on the same objects.

---

## 17.3 Specific Changes

### 17.3.1 Deprecate/Remove `Network` as Public API

`Network` becomes `_Network` (internal). `RegionalNetwork` is the sole public network class.

Users who need single-population networks use `RegionalNetwork` with one population — the overhead is negligible.

**Migration path:**
```python
# Old
net = Network(10)
net.add_synapse(0, 1, 0.5)

# New
net = RegionalNetwork()
net.add_population("neurons", 10, NeuronModelSpec.hh_default())
net.add_connection("neurons", 0, "neurons", 1, 0.5, SynapseSpec.exponential(0.0, 2.0))
```

### 17.3.2 Collapse Neuron Add Methods

Remove `add_hh_neuron()`, `add_izhikevich_neuron()`, and the `NeuronType` enum from the public API. Neuron type is always specified via a `NeuronModelSpec`:

```python
# All neuron types expressed as specs:
NeuronModelSpec.hh_default()                            # was: HH
NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING) # was: IZHIKEVICH_FS
NeuronModelSpec.thalamic()                              # was: composable preset
```

The `NeuronModelSpec` becomes the single type-describing object.

### 17.3.3 Move Scalar Neuron Classes to `hodgkin_huxley.legacy`

`HHNeuron`, `IzhikevichNeuron`, `HHParameters`, `HHState`, `IzhikevichParameters`, `IzhikevichState` are moved to a `legacy` submodule. They remain importable (no breakage) but are no longer in the top-level namespace:

```python
# Old (still works with deprecation warning)
from hodgkin_huxley import HHNeuron

# New
from hodgkin_huxley.legacy import HHNeuron
```

These classes remain useful for unit testing, educational use, and single-neuron parameter tuning.

### 17.3.4 Hide Buffer Methods

`_simulate_into_buffers()` and `_simulate_with_descriptors()` are removed from public Python bindings. The `simulate()` method internally routes to the optimal path. Users who need direct buffer access for performance-critical analysis can use the `RecordingConfig` interface, which pre-allocates and returns the same buffers.

### 17.3.5 `I_ext` Interface Unification

Currently `I_ext` can be a float, 1-D array, 2-D array, dict, or `StimPlan`. Standardise to:

```python
# Preferred forms (all valid):
I_ext = {"CTX": 10.0, "STN": 0.0}           # per-population scalar
I_ext = {"CTX": pulse_stimulator, "STN": 0.0}  # stimulator per population
I_ext = {"CTX": np.array([...]), ...}        # per-neuron 1D array (heterogeneous)
```

Dense 2D arrays (n_neurons × n_steps) are still accepted but not advertised; they are the legacy path for non-scalar time-varying inputs that cannot be expressed as stimulator objects.

### 17.3.6 Clean Up `__init__.py` Exports

Current `__all__` exports ~50 symbols. After streamlining:

**Always exported (core workflow):**
`RegionalNetwork`, `NeuronModelSpec`, `SynapseSpec`, `WeightDistribution`, `ConnectivityPattern`, `RecordingConfig`, `IzhikevichType`, `IntegrationMethod`, `DBSStimulator`, `PulseStimulator`, `NoiseInjector`, `analyze_beta_power`

**Exported but in `hodgkin_huxley.model` submodule:**
`GateSpec`, `ChannelSpec`, `BoltzmannParams`, `Tau`, `RateFunc`, `CalciumSpec`, `IntracellularSpec` (and all composable-system types)

**Exported but in `hodgkin_huxley.legacy`:**
`HHNeuron`, `IzhikevichNeuron`, `Network`, `HHParameters`, `HHState`, etc.

---

## 17.4 Implementation Checklist

### C++ / Bindings
- [ ] Prefix `Network` as `_Network` in bindings; `RegionalNetwork` takes its place
- [ ] Mark `add_hh_neuron()`, `add_izhikevich_neuron()`, `NetworkNeuronType` deprecated
- [ ] Add `NeuronModelSpec.hh_default()` and `NeuronModelSpec.izhikevich(type)` presets
- [ ] Remove `_simulate_into_buffers` and `_simulate_with_descriptors` from public binding exports
- [ ] Unify `I_ext` dispatch in `RegionalNetwork.simulate()` to handle scalar/stimulator/array per population

### Python Layer
- [ ] Create `hodgkin_huxley/legacy.py` re-exporting scalar neuron classes
- [ ] Create `hodgkin_huxley/model.py` for composable-system types
- [ ] Update `__init__.py` to the trimmed export list
- [ ] Update all `examples/` to use unified workflow
- [ ] Add `DeprecationWarning` to old entry points

### Tests
- [ ] Verify benchmark model (`benchmarks/ctxbgth_model.py`) ports cleanly to new API
- [ ] Ensure all existing tests pass (or update for deprecated paths with warnings)
- [ ] Add tests for `NeuronModelSpec.hh_default()`, `NeuronModelSpec.izhikevich()` presets
