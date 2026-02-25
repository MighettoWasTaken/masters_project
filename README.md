# Hodgkin-Huxley Network Simulation Framework

A fast, generalized Python framework for building complex brain network simulations, powered by a high-performance C++ backend. Developed as a masters project targeting basal ganglia–thalamo–cortical models used in Parkinson's disease research.

**Author:** Edward Mighetto
**GitHub:** [MighettoWasTaken/masters_project](https://github.com/MighettoWasTaken/masters_project)

## Overview

This project provides modular, reusable components for computational neuroscience simulations. The C++ core with Python bindings offers significant performance improvements over pure Python/NumPy implementations while maintaining a clean, readable API.

**Key features:**
- Multiple neuron models: Hodgkin-Huxley, Izhikevich, and a fully composable ion-channel system
- Five biological neuron presets: Thalamic (TH), Subthalamic Nucleus (STN), GPe, GPi, Striatum
- Three synapse types: exponential, alpha-function, double-exponential, with receptor presets (AMPA, NMDA, GABA-A)
- Synaptic delays via circular ring buffers
- Population-based `RegionalNetwork` with five connectivity patterns
- Selectable integration methods: Euler, RK4, RK45 adaptive
- Eigen-vectorized neuron pools (HHPool, IzPool, ComposablePool) for SIMD throughput
- pybind11 Python bindings with NumPy integration

See [overview.md](overview.md) for project goals and evaluation plan.

---

## Installation

### Requirements

- CMake >= 3.15
- C++17 compatible compiler (MSVC, GCC, Clang)
- Python >= 3.8
- pybind11 >= 2.11
- Eigen (fetched automatically via CMake FetchContent)

### From source

```bash
git clone https://github.com/MighettoWasTaken/masters_project.git
cd masters_project

# Install build dependencies
pip install scikit-build-core pybind11 numpy

# Install in development mode
pip install -e .

# With dev extras (pytest, black, ruff)
pip install -e ".[dev]"
```

---

## Quick Start

### Single neuron

```python
from hodgkin_huxley import HHNeuron, IzhikevichNeuron, IntegrationMethod

# Classic Hodgkin-Huxley (squid giant axon)
neuron = HHNeuron()
neuron.integration_method = IntegrationMethod.RK4
trace = neuron.simulate(duration=100, dt=0.01, I_ext=10)

# Izhikevich regular-spiking neuron
iz = IzhikevichNeuron.regular_spiking()
trace = iz.simulate(duration=200, dt=0.1, I_ext=10)
```

### Composable ion-channel neuron

```python
from hodgkin_huxley import NeuronModelSpec, NeuronModel, Network

# Use a biological preset
spec = NeuronModelSpec.thalamic()   # TH relay neuron (Rubin & Terman 2004)

# Or build one from scratch
m = NeuronModel("MyNeuron")
m.add_leak(g=0.05, E_rev=-70.0)
m.add_gate("m_Na", "INF_TAU", v_half=-37.0, k=7.0, tau_scale=1.0, tau_vh=-41.0, tau_k=-4.0)
m.add_channel("Na", g=3.0, E_rev=50.0, gates=[(0, 3)])
spec = m.to_spec()

# Simulate via Network (pool-vectorized path)
net = Network()
net.add_neuron(spec)
n_steps = 5000
traces = net.simulate(500.0, 0.1, [[5.0] * n_steps])
```

### Network with synapses

```python
from hodgkin_huxley import Network, SynapseSpec

net = Network()
net.add_neuron(NeuronModelSpec.stn())
net.add_neuron(NeuronModelSpec.thalamic())

# Double-exponential AMPA synapse from neuron 0 → 1
spec = SynapseSpec.double_exponential(weight=0.5, tau_rise=0.5, tau_decay=3.0,
                                      E_syn=0.0, delay=1.0)
net.add_synapse(0, 1, spec)

n_steps = 2000
traces = net.simulate(200.0, 0.1, [[20.0, 5.0]] * n_steps)
```

### Regional (population-based) network

```python
from hodgkin_huxley import RegionalNetwork, NetworkNeuronType, ConnectivityPattern

rnet = RegionalNetwork()
rnet.add_population("STN",  50, NetworkNeuronType.COMPOSABLE, NeuronModelSpec.stn())
rnet.add_population("GPe", 100, NetworkNeuronType.COMPOSABLE, NeuronModelSpec.gpe())

rnet.connect("STN", "GPe",
             pattern=ConnectivityPattern.RANDOM_SPARSE,
             weight=0.3, probability=0.1,
             synapse_type="DOUBLE_EXP", tau_rise=0.5, tau_decay=3.0)

traces = rnet.simulate(500.0, 0.1, {"STN": 25.0, "GPe": 5.0})
```

---

## Current Status

| Feature | Status |
|---|---|
| Hodgkin-Huxley neuron (squid giant axon) | Done |
| Izhikevich neuron (RS, FS, IB, CH, LTS presets) | Done |
| Composable ion-channel neuron system | Done |
| Biological presets: TH, STN, GPe, GPi, Striatum | Done |
| Exponential synapse | Done |
| Alpha-function synapse | Done |
| Double-exponential synapse | Done |
| Receptor presets (AMPA, NMDA, GABA-A) | Done |
| Synaptic delays (ring buffer) | Done |
| Network with mixed neuron types | Done |
| RegionalNetwork + 5 connectivity patterns | Done |
| Eigen-vectorized neuron pools (HHPool, IzPool, ComposablePool) | Done |
| Euler, RK4, RK45 adaptive integration | Done |
| Calcium dynamics + Nernst potential | Done |
| Voltage-dependent GABA kinetic synapse | Planned |
| DBS stimulation pulse train | Planned |
| Multitaper spectral analysis (beta band power) | Planned |
| Cortical pulse stimulator | Planned |

---

## Composable Neuron System

Task 5 introduced a data-driven ion channel framework for building arbitrary conductance-based models without writing new C++ code.

### Gate update forms
| Form | Description |
|---|---|
| `INF_TAU` | `dX/dt = (X_inf(V) - X) / tau(V)` — standard HH-style |
| `ALPHA_BETA` | `dX/dt = α(V)·(1-X) - β(V)·X` — classic rate-function form |
| `INSTANT` | `X = X_inf(V)` always (infinitely fast) |
| `DERIVED` | `X = a·(b + c·X_src)` algebraic function of another gate |

### Tau function forms
`CONSTANT`, `BOLTZMANN`, `DOUBLE_EXP_SUM`, `OFFSET_DOUBLE_EXP`, `SCALED_EXP`, `COMPOUND_AB`

### Biological presets
| Name | Gates | Channels | Calcium |
|---|---|---|---|
| `NeuronModelSpec.thalamic()` | 5 | 4 (Na, K, T-type Ca, Leak) | No |
| `NeuronModelSpec.stn()` | 11 | 7 | Yes (Nernst) |
| `NeuronModelSpec.gpe()` | — | — (with AHP channel) | Yes |
| `NeuronModelSpec.gpi()` | — | — | Yes |
| `NeuronModelSpec.striatum(pd=0.0)` | — | 4 (Na, K, M-current, Leak) | No |

---

## Building

```bash
# Standard install
pip install .

# Development build with verbose output
pip install -v -e .

# Build C++ tests separately
cmake -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Running Tests

```bash
# All Python tests
pytest tests/python -v

# Specific suite
pytest tests/python/test_composable_neuron.py -v

# C++ unit tests (requires -DBUILD_TESTS=ON)
ctest --test-dir build
```

---

## Project Structure

```
masters_project/
├── CMakeLists.txt               # Root CMake configuration
├── pyproject.toml               # Python package (scikit-build-core)
├── overview.md                  # Project goals, evaluation plan
├── task6.md – task9.md          # Planned task specifications
├── completed/                   # Completed task specs (tasks 1–5)
├── src/
│   ├── cpp/
│   │   ├── include/hodgkin_huxley/
│   │   │   ├── neuron_base.hpp          # Abstract NeuronBase, IntegrationMethod
│   │   │   ├── neuron.hpp               # HHNeuron
│   │   │   ├── izhikevich.hpp           # IzhikevichNeuron
│   │   │   ├── ion_channels.hpp         # Composable structs: GateSpec, ChannelSpec, NeuronModelSpec
│   │   │   ├── composable_neuron.hpp    # ComposableNeuron (scalar step)
│   │   │   ├── composable_pool.hpp      # ComposablePool (Eigen-vectorized)
│   │   │   ├── hh_pool.hpp              # HHPool (Eigen-vectorized)
│   │   │   ├── iz_pool.hpp              # IzPool (Eigen-vectorized)
│   │   │   ├── synapse_base.hpp         # Abstract SynapseBase
│   │   │   ├── synapse.hpp              # Exponential, Alpha, DoubleExp synapses
│   │   │   ├── network.hpp              # Network class
│   │   │   └── regional_network.hpp     # RegionalNetwork class
│   │   └── src/
│   │       ├── ion_channels.cpp         # Preset factory implementations
│   │       ├── composable_neuron.cpp
│   │       ├── composable_pool.cpp
│   │       ├── hh_pool.cpp
│   │       ├── iz_pool.cpp
│   │       ├── network.cpp
│   │       └── regional_network.cpp
│   ├── python/
│   │   └── bindings.cpp         # pybind11 module (_core)
│   └── hodgkin_huxley/
│       └── __init__.py          # Python API: NeuronModel builder, exports
├── tests/
│   ├── cpp/                     # C++ unit tests (40+ tests)
│   └── python/                  # Python tests (6 test files)
│       ├── test_neuron.py
│       ├── test_izhikevich.py
│       ├── test_networks.py
│       ├── test_synapses.py
│       ├── test_regional_network.py
│       └── test_composable_neuron.py
└── examples/
    ├── basic_simulation.py
    ├── verify_neuron.py
    ├── verify_izhikevich.py
    └── visualize_network.py
```

---

## Performance

The C++ backend uses Eigen-vectorized pool stepping with Structure-of-Arrays (SoA) memory layout, RK4 with pre-allocated working buffers, and an optional polynomial fast-exp approximation (~8 digits accuracy). All neuron pools avoid heap allocation in the hot loop, and synaptic conductances are stored in cache-friendly SoA arrays.

---

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *The Journal of Physiology*, 117(4), 500–544.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569–1572.
- Rubin, J. E., & Terman, D. (2004). High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a computational model. *Journal of Computational Neuroscience*, 16(3), 211–235.
- Hahn, P. J., & McIntyre, C. C. (2010). Modeling shifts in the rate and pattern of subthalamopallidal network activity during deep brain stimulation. *Journal of Computational Neuroscience*, 28(3), 425–441.
- Kumaravelu, K., Brocker, D. T., & Grill, W. M. (2016). A biophysical model of the cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model of Parkinson's disease. *Journal of Computational Neuroscience*, 40(2), 207–229.
- Stimberg, M., Brette, R., & Goodman, D. F. (2019). Brian 2, an intuitive and efficient neural simulator. *eLife*, 8, e47314.

---

## License

MIT License — see LICENSE file.
