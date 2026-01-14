# Overview

project: Hodgkin-Huxley Network Simulation Framework

author: Edward Mighetto
github: https://github.com/MighettoWasTaken/masters_project

## Summary

Computational neuroscience models (e.g., basal ganglia-thalamo-cortical networks used in Parkinson's disease research) can be computationally expensive and are often implemented as monolithic, single-purpose scripts, making them hard to scale, reuse, and validate.

This project builds a fast, generalized Python framework for complex brain network simulations with a high-performance C++ backend and Python bindings, aiming for speed, modularity, and a clean user-facing API.

## Outputs / Deliverables

- A Python package (`hodgpy`) exposing a clean API for neurons, synapses, networks, and solvers
- A C++ core library with pybind11 bindings for performance-critical simulation components
- Reproducible benchmark(s) based on an existing rat Parkinson’s disease model (`simulate_network_model_CL_fast.py`)

## Evaluation Plan

- Baselines: Python/NumPy, Python/Numba, MATLAB (if available)
- Metrics and targets:

| Metric | Measurement | Target |
|--------|-------------|--------|
| Computation speed | Wall-clock time for benchmark simulation | >5x faster than NumPy, >2x faster than Numba |
| Memory usage | Peak RAM during simulation | Comparable or lower than Python baseline |
| Scalability | Time vs. network size (100 to 10,000 neurons) | Linear or sub-linear scaling |
| Accuracy | Spike timing difference, power spectral density | <1% error vs. reference implementation |
| Code complexity | Lines of code for benchmark recreation | <50% of monolithic baseline |
| API usability | Time to implement new model from scratch | Subjective evaluation + user testing |

## System Design / Architecture

### Main Components

- Python API (user-facing)
- pybind11 bindings
- C++ core library:
  - Neuron models (HH, Izhikevich, etc.)
  - Ion channels (modular, composable)
  - Synapse models (exponential, alpha, double-exp)
  - Network (populations, connectivity)
  - Solvers (Euler, RK4)

### Interfaces

- Python package import + API usage (neurons, networks, synapses, solvers)
- Packaging/distribution:
  - Install: `uv pip install hodgpy` or `pip install hodgpy`
  - Dev workflow: `uv venv` then `uv pip install -e ".[dev]"`

### Dependencies / Tech Stack

- C++ core + pybind11 bindings
- Python packaging with `uv` (development workflow)

## Timeline for main Components

- Phase 1: Implement features required to recreate the benchmark model
- Phase 2: Validate outputs match the original implementation
- Phase 3: Measure and optimize performance
- Phase 4: Generalize the API for broader use cases

## Current Status

- Implemented: basic Hodgkin-Huxley neuron, simple network with exponential synapses, Euler and RK4 integration
- Planned: Izhikevich neuron model, additional ion channels, calcium dynamics, advanced synapse models, population abstractions, DBS module
- See `task.md` for the current implementation plan.

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). “A quantitative description of membrane current and its application to conduction and excitation in nerve.” *The Journal of Physiology*, 117(4), 500-544.
- Izhikevich, E. M. (2003). “Simple model of spiking neurons.” *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.
- Rubin, J. E., & Terman, D. (2004). “High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a computational model.” *Journal of Computational Neuroscience*, 16(3), 211-235.
- Hahn, P. J., & McIntyre, C. C. (2010). “Modeling shifts in the rate and pattern of subthalamopallidal network activity during deep brain stimulation.” *Journal of Computational Neuroscience*, 28(3), 425-441.
- Stimberg, M., Brette, R., & Goodman, D. F. (2019). “Brian 2, an intuitive and efficient neural simulator.” *eLife*, 8, e47314.
- K. Kumaravelu, D. T. Brocker, and W. M. Grill, “A biophysical model of the cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease,” *Journal of Computational Neuroscience*, vol. 40, no. 2, pp. 207–229, 2016.
