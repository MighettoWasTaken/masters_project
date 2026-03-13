# Hodgkin-Huxley Network Simulation Framework

A fast, generalized Python framework for building complex brain network simulations, powered by a high-performance C++ backend. Developed as a masters project targeting basal ganglia–thalamo–cortical models used in Parkinson's disease research.

**Author:** Edward Mighetto
**GitHub:** [MighettoWasTaken/masters_project](https://github.com/MighettoWasTaken/masters_project)

---

## Overview

This project provides modular, reusable components for computational neuroscience simulations. The C++ core with Python bindings delivers significant performance improvements over pure Python/NumPy implementations while maintaining a clean, readable API.

**Key features:**
- Three neuron models: Hodgkin-Huxley, Izhikevich, and a fully composable ion-channel system
- Five biological presets: Thalamic (TH), Subthalamic Nucleus (STN), GPe, GPi, Striatum
- Three synapse types: exponential, alpha-function, double-exponential, with receptor presets (AMPA, NMDA, GABA-A) and voltage-dependent kinetic synapses
- Synaptic delays via circular ring buffers
- Population-based `RegionalNetwork` with five connectivity patterns and heterogeneity support
- DBS stimulator (`DBSStimulator`) and flexible pulse stimulator (`PulseStimulator`)
- Compact `StimPlan` descriptor path — avoids dense I_ext matrix allocation
- Configurable recording system (`RecordingConfig`) with spike detection, ISI statistics, and gate/calcium traces
- Multitaper spectral analysis for beta-band power (Parkinson's biomarker)
- Selectable integration methods: Euler, RK4, RK45 adaptive
- Eigen-vectorized neuron pools (HHPool, IzPool, ComposablePool) for SIMD throughput
- Optional polynomial fast-exp approximation (~8-digit accuracy, ~2× faster)
- pybind11 Python bindings with NumPy zero-copy buffer interface

---

## Installation

**Requirements:** Python ≥ 3.8, a C++14-compatible compiler (MSVC, GCC, or Clang), and `uv` ([install uv](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
uv venv && uv pip install -e ".[examples]" --python .venv/Scripts/python.exe
```

Creates a virtual environment and installs the library in editable mode. `cmake` and `ninja` are fetched automatically as build dependencies — no separate CMake install required. 

---

## Verification

After installation, run these three commands to verify everything works:

### 1. Run the test suite

```bash
uv run pytest tests/python/
```

Runs 15 test files covering all neuron models, synapse types, the recording system, DBS/pulse stimulators, and spectral analysis. Expected output: all tests pass.

### 2. Run the network performance benchmark

```bash
uv run python examples/benchmark_network.py
```

Benchmarks the C++ backend against a pure-NumPy reference implementation across network sizes (10–200 neurons). Produces timing, speedup, and memory plots saved to `examples/figs/`. Typical speedup: 50–250× depending on network size.

### 3. Run the CTX-BG-TH model

```bash
uv run python benchmarks/ctxbgth_model.py
```

Builds and simulates the full 8-population cortex–basal ganglia–thalamus network (80 neurons, 2000 ms) using the library's composable neuron builder and population API. Prints per-population firing rates and GPi beta-band power. Reproduces the Parkinson's disease model from Hahn et al. (2019).

---

## Quick Start

### Single neuron

```python
from hodgkin_huxley import HHNeuron, IzhikevichNeuron, IzhikevichType, IntegrationMethod

# Classic Hodgkin-Huxley (squid giant axon)
neuron = HHNeuron()
neuron.integration_method = IntegrationMethod.RK4
trace = neuron.simulate(duration=100, dt=0.01, I_ext=10)

# Izhikevich regular-spiking cortical neuron
iz = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
trace = iz.simulate(duration=200, dt=0.1, I_ext=10)
```

### Composable ion-channel neuron

```python
from hodgkin_huxley import NeuronModelSpec, NeuronModel, Boltzmann, Tau, Network

# Use a biological preset
spec = NeuronModelSpec.thalamic()   # TH relay neuron (Rubin & Terman 2004)

# Or build one from scratch
m = NeuronModel("MyNeuron")
m.add_leak(g=0.05, E_rev=-70.0)
m.add_gate("m_Na", "INF_TAU", v_half=-37.0, k=7.0,
           tau=Tau.constant(0.5))
m.add_channel("Na", g=3.0, E_rev=50.0, gates=[("m_Na", 3)])
spec = m.to_spec()

net = Network()
net.add_neuron(spec)
traces = net.simulate(500.0, 0.1, [[5.0] * 5000])
```

### Network with synapses

```python
from hodgkin_huxley import Network, NeuronModelSpec

net = Network()
net.add_neuron(NeuronModelSpec.stn())
net.add_neuron(NeuronModelSpec.thalamic())

# Double-exponential AMPA synapse, 1 ms conduction delay
net.add_double_exp_synapse(0, 1, weight=0.5, E_syn=0.0,
                           tau_rise=0.5, tau_decay=3.0, delay=1.0)

n_steps = 2000
traces = net.simulate(200.0, 0.1, [[20.0] * n_steps, [5.0] * n_steps])
```

### Population-based network with DBS

```python
from hodgkin_huxley import (
    RegionalNetwork, NeuronModelSpec, SynapseSpec,
    DBSStimulator, DBSParameters, RecordingConfig
)

rnet = RegionalNetwork()
rnet.add_population("STN", 10, spec=NeuronModelSpec.stn())
rnet.add_population("GPe", 20, spec=NeuronModelSpec.gpe())

rnet.connect("STN", "GPe",
             pattern="random_sparse",
             weight=0.3, probability=0.1,
             synapse=SynapseSpec.ampa())

dbs_params = DBSParameters()
dbs_params.frequency  = 130.0   # Hz
dbs_params.amplitude  = 300.0   # µA/cm²
dbs_params.pulse_width = 0.06   # ms
rnet.attach_stimulator("STN", DBSStimulator(dbs_params))

result = rnet.simulate(1000.0, 0.01,
                       I_ext={"STN": 25.0, "GPe": 5.0},
                       record=RecordingConfig.spikes_only())
```

---

## Composable Neuron System

The composable ion-channel framework builds arbitrary conductance-based neurons without writing new C++ code.

### Gate update forms
| Form | Description |
|---|---|
| `INF_TAU` | `dX/dt = (X∞(V) − X) / τ(V)` — standard HH-style |
| `ALPHA_BETA` | `dX/dt = α(V)·(1−X) − β(V)·X` — classic rate-function form |
| `INSTANT` | `X = X∞(V)` always (infinitely fast gate) |
| `DERIVED` | `X = a·(b + c·X_src)` algebraic function of another gate |

### Tau function forms
`CONSTANT`, `BOLTZMANN`, `DOUBLE_EXP_SUM`, `OFFSET_DOUBLE_EXP`, `SCALED_EXP`, `COMPOUND_AB`

### Biological presets
| Name | Description |
|---|---|
| `NeuronModelSpec.thalamic()` | TH relay neuron — Na, K, T-type Ca, Leak (Rubin & Terman 2004) |
| `NeuronModelSpec.stn()` | STN neuron — 7 channels, calcium dynamics (Hahn et al. 2019) |
| `NeuronModelSpec.gpe()` | GPe neuron — AHP channel, calcium (Hahn et al. 2019) |
| `NeuronModelSpec.gpi()` | GPi neuron — same conductances as GPe |
| `NeuronModelSpec.striatum(pd=0.0)` | Striatum — Na, K, M-current, Leak |

---

## Feature Status

| Feature | Status |
|---|---|
| Hodgkin-Huxley neuron (squid giant axon) | Done |
| Izhikevich neuron (RS, FS, IB, CH, LTS presets + custom) | Done |
| Composable ion-channel neuron system | Done |
| Biological presets: TH, STN, GPe, GPi, Striatum | Done |
| Exponential, alpha, double-exponential synapses | Done |
| Receptor presets (AMPA, NMDA, GABA-A) | Done |
| Voltage-dependent kinetic synapses | Done |
| Synaptic delays (ring buffer) | Done |
| Network with mixed neuron types | Done |
| RegionalNetwork + 5 connectivity patterns | Done |
| Population heterogeneity (per-neuron parameter variation) | Done |
| Eigen-vectorized neuron pools (HHPool, IzPool, ComposablePool) | Done |
| Euler, RK4, RK45 adaptive integration | Done |
| Calcium dynamics + Nernst potential | Done |
| DBS stimulator with `current_at()` zero-copy path | Done |
| Pulse stimulator (monophasic / biphasic, trains, bursts) | Done |
| Compact StimPlan descriptor path (no dense I_ext matrix) | Done |
| RecordingConfig + PopulationMetricsResult | Done |
| Multitaper spectral analysis (beta-band power) | Done |
| Noise injection (`NoiseInjector`) | Done |
| CTX-BG-TH benchmark reproduction | Done |

---

## Project Structure

```
masters_project/
├── pyproject.toml               # Python package (scikit-build-core)
├── CMakeLists.txt               # Root CMake configuration
├── completed/                   # Completed task specifications (tasks 1–11)
├── src/
│   ├── cpp/
│   │   ├── include/hodgkin_huxley/
│   │   │   ├── neuron_base.hpp          # Abstract NeuronBase, IntegrationMethod
│   │   │   ├── neuron.hpp               # HHNeuron
│   │   │   ├── izhikevich.hpp           # IzhikevichNeuron
│   │   │   ├── ion_channels.hpp         # GateSpec, ChannelSpec, NeuronModelSpec
│   │   │   ├── composable_neuron.hpp    # ComposableNeuron (scalar step)
│   │   │   ├── composable_pool.hpp      # ComposablePool (Eigen-vectorized)
│   │   │   ├── hh_pool.hpp              # HHPool (Eigen-vectorized)
│   │   │   ├── iz_pool.hpp              # IzPool (Eigen-vectorized)
│   │   │   ├── synapse.hpp              # Exponential, Alpha, DoubleExp synapses
│   │   │   ├── network.hpp              # Network class + StimPlan descriptors
│   │   │   ├── regional_network.hpp     # RegionalNetwork class
│   │   │   └── dbs_stimulator.hpp       # DBSStimulator
│   │   └── src/                         # C++ implementations
│   ├── python/
│   │   └── bindings.cpp         # pybind11 module (_core)
│   └── hodgkin_huxley/
│       ├── __init__.py          # Python API + RegionalNetwork
│       ├── recording.py         # RecordingConfig, MetricsResult, _run_recording
│       ├── pulse.py             # PulseStimulator
│       ├── spectral.py          # Multitaper spectral analysis
│       └── noise.py             # NoiseInjector
├── tests/
│   └── python/                  # 15 test files
├── examples/
│   ├── benchmark_network.py         # C++ vs NumPy timing benchmark
│   ├── benchmark_network_averaged.py # Averaged benchmark + plots
│   └── verify_neuron.py / verify_izhikevich.py
└── benchmarks/
    ├── ctxbgth_model.py             # Full CTX-BG-TH model (8 populations)
    ├── compare_models.py            # Library vs benchmark comparison
    ├── variable_span.py             # Variable span readability metric
    └── flexibility_metric.py        # Configuration space metric
```

---

## Performance

The C++ backend achieves 50–250× speedup over pure NumPy through:
- **Eigen-vectorized pools** — batched SIMD neuron stepping in SoA layout
- **RK4 with pre-allocated buffers** — no heap allocation in the hot loop
- **Fast polynomial exp** — degree-7 Taylor/Horner approximation (~8-digit accuracy)
- **Compact StimPlan** — descriptor-based I_ext evaluation eliminates 128 MB dense matrix at tmax=2000 ms, dt=0.01 ms
- **Zero-copy NumPy buffers** — caller-allocated output arrays, GIL-free C++ execution

---

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117(4), 500–544.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569–1572.
- Rubin, J. E., & Terman, D. (2004). High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a computational model. *Journal of Computational Neuroscience*, 16(3), 211–235.
- Hahn, P. J., & McIntyre, C. C. (2010). Modeling shifts in the rate and pattern of subthalamopallidal network activity during deep brain stimulation. *Journal of Computational Neuroscience*, 28(3), 425–441.
- Kumaravelu, K., Brocker, D. T., & Grill, W. M. (2016). A biophysical model of the cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model of Parkinson's disease. *Journal of Computational Neuroscience*, 40(2), 207–229.

---

## License

MIT License — see LICENSE file.
