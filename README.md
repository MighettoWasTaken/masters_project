# Hodgkin-Huxley Network Simulation Framework

A fast, generalized Python framework for building complex brain network simulations, powered by a high-performance C++ backend.

## Overview

This project provides modular, reusable components for computational neuroscience simulations. The C++ core with Python bindings offers significant performance improvements over pure Python/NumPy implementations while maintaining a clean, readable API.

**Key features:**
- Multiple neuron models (Hodgkin-Huxley, Izhikevich - planned)
- Configurable ion channels and synaptic dynamics
- Selectable integration methods (Euler, RK4)
- Network simulations with synaptic connectivity

See [project.md](project.md) for project goals and [task.md](task.md) for the development roadmap.

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/hodgkin-huxley.git
cd hodgkin-huxley

# Install build dependencies
pip install scikit-build-core pybind11 numpy

# Install in development mode
pip install -e .
```

## Quick Start

```python
from hodgkin_huxley import HHNeuron, Network, IntegrationMethod

# Single neuron simulation
neuron = HHNeuron()
neuron.integration_method = IntegrationMethod.RK4  # or EULER
trace = neuron.simulate(duration=100, dt=0.01, I_ext=10)

# Network simulation
net = Network(3)
net.add_synapse(0, 1, weight=0.5)
net.add_synapse(1, 2, weight=0.5)
traces = net.simulate(duration=100, dt=0.01, I_ext=current_matrix)
```

## Current Status

- [x] Hodgkin-Huxley neuron (squid giant axon parameters)
- [x] Basic network with exponential synapses
- [x] Euler and RK4 integration methods
- [x] Python bindings via pybind11
- [ ] Izhikevich neuron model
- [ ] Additional ion channels (T-type Ca, L-type Ca, A-type K, M-current)
- [ ] Advanced synapse models (alpha-function, delays)
- [ ] Population/region abstractions

## Building

### Requirements

- CMake >= 3.15
- C++14 compatible compiler
- Python >= 3.8
- pybind11 >= 2.11

### Build commands

```bash
# Standard build
pip install .

# Development build with verbose output
pip install -v -e .

# Build C++ tests
cmake -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Running Tests

```bash
# Python tests
pytest tests/python -v

# C++ tests (after building with BUILD_TESTS=ON)
ctest --test-dir build
```

## Project Structure

```
hodgkin-huxley/
├── CMakeLists.txt          # Root CMake configuration
├── pyproject.toml          # Python package configuration
├── project.md              # Project goals and overview
├── task.md                 # Implementation roadmap
├── src/
│   ├── cpp/                # C++ source code
│   │   ├── include/        # Header files
│   │   └── src/            # Implementation files
│   ├── python/             # pybind11 bindings
│   └── hodgkin_huxley/     # Python package
└── tests/
    ├── cpp/                # C++ unit tests
    └── python/             # Python tests
```

## License

MIT License - see LICENSE file.
