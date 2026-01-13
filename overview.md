# Hodgkin-Huxley Network Simulation Framework

## Project Goal

Create a fast, generalized Python framework for building complex brain network simulations, with a high-performance C++ backend.

## Motivation

Computational neuroscience models, such as basal ganglia-thalamo-cortical networks used in Parkinson's disease research, are computationally expensive and often implemented as monolithic, single-purpose scripts. This project aims to provide:

1. **Speed** - C++ core with Python bindings for significant performance gains over pure Python/NumPy implementations
2. **Modularity** - Reusable neuron models, ion channels, and synapse types that can be composed into arbitrary network architectures
3. **Simplicity** - Clean Python API that makes building complex models straightforward and readable

## Approach

### Benchmark-Driven Development

The project uses an existing rat Parkinson's disease model (`simulate_network_model_CL_fast.py`) as both a feature specification and performance benchmark.

**Development phases:**
1. Implement features required to recreate the benchmark model
2. Validate outputs match the original implementation
3. Measure and optimize performance
4. Generalize the API for broader use cases

### Validation Criteria

| Metric | Target |
|--------|--------|
| Output accuracy | Comparable spike times and spectral properties |
| Computation speed | Significant improvement over NumPy/Numba baseline |
| Code complexity | Fewer lines, clearer structure than benchmark |

## Architecture

```
Python API (user-facing)
        │
        ▼
   pybind11 bindings
        │
        ▼
  C++ core library
   ├── Neuron models (HH, Izhikevich, etc.)
   ├── Ion channels (modular, composable)
   ├── Synapse models (exponential, alpha, double-exp)
   ├── Network (populations, connectivity)
   └── Solvers (Euler, RK4)
```

## Current Status

- [x] Basic Hodgkin-Huxley neuron (squid giant axon)
- [x] Simple network with exponential synapses
- [x] Euler and RK4 integration methods
- [ ] Izhikevich neuron model
- [ ] Additional ion channels (T-type Ca, L-type Ca, A-type K, M-current, AHP)
- [ ] Calcium dynamics
- [ ] Advanced synapse models (alpha-function, double-exponential, delays)
- [ ] Population/region abstractions
- [ ] DBS stimulation module

See `task.md` for detailed implementation requirements.
