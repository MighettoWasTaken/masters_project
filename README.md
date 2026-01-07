# Hodgkin-Huxley Neuron Simulation

A fast C++ implementation of the Hodgkin-Huxley neuron model with Python bindings.

## Installation

### From pip (once published)

```bash
pip install hodgkin-huxley
```

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

### From conda (once published)

```bash
conda install -c conda-forge hodgkin-huxley
```

## Quick Start

```python
from hodgkin_huxley import HHNeuron, Network

# Single neuron simulation
neuron = HHNeuron()
trace = neuron.simulate(duration=100, dt=0.01, I_ext=10)

# Network simulation
net = Network(3)
net.add_synapse(0, 1, weight=0.5)
net.add_synapse(1, 2, weight=0.5)
traces = net.simulate(duration=100, dt=0.01, I_ext=current_matrix)
```

## Building

### Requirements

- CMake >= 3.15
- C++17 compatible compiler
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
├── src/
│   ├── cpp/                # C++ source code
│   │   ├── include/        # Header files
│   │   └── src/            # Implementation files
│   ├── python/             # pybind11 bindings
│   └── hodgkin_huxley/     # Python package
├── tests/
│   ├── cpp/                # C++ unit tests
│   └── python/             # Python tests
├── examples/               # Example scripts
└── conda/                  # Conda recipe
```

## License

MIT License - see LICENSE file.
