# Overview

project: Hodgkin-Huxley Network Simulation Framework

author: Edward Mighetto
github: https://github.com/MighettoWasTaken/masters_project

## Summary

Computational neuroscience models (e.g., basal ganglia-thalamo-cortical networks used in Parkinson's disease research) can be computationally expensive and are often implemented as monolithic, single-purpose scripts, making them hard to scale, reuse, and validate.

This project builds a fast, generalized Python framework for complex brain network simulations with a high-performance C++ backend and Python bindings, aiming for speed, modularity, and a clean user-facing API. The benchmark target is the rat CTX-BG-TH Parkinson's disease model (Kumaravelu 2016 / Hahn & McIntyre 2010); the framework has reproduced this model at >10× the speed of the pure-Python reference with equivalent biological accuracy.

## Outputs / Deliverables

- A Python package (`hodgkin_huxley`) exposing a clean API for neurons, synapses, networks, stimulators, and analysis tools
- A C++ core library with pybind11 bindings for performance-critical simulation components
- Reproducible benchmark based on the rat Parkinson's disease CTX-BG-TH model
- Web documentation with API reference and tutorials (planned, see `docs/roadmap.md`)

## Evaluation Metrics

| Metric | Measurement | Status |
|--------|-------------|--------|
| Computation speed | Wall-clock time for CTX-BG-TH benchmark | **>10× faster than NumPy** (achieved) |
| Memory usage | Peak RAM during simulation | **Lower than Python baseline** (achieved) |
| Scalability | Time vs. network size | Sub-linear scaling (Eigen SIMD pools) |
| Accuracy | Firing rates, GPi beta-band power | **<1% error vs. reference** (achieved) |
| Code complexity | Non-blank, non-comment lines for benchmark | **<50% of monolithic baseline** (achieved) |

## Architecture

### Main Components

- **Python API** (`src/hodgkin_huxley/`): user-facing layer — thin wrappers around C++ classes plus pure-Python analysis tools
- **pybind11 bindings** (`src/python/bindings.cpp`): zero-copy numpy interface, GIL-released hot loop
- **C++ core** (`src/cpp/`):
  - Neuron models: Hodgkin-Huxley, Izhikevich (5 presets), Composable (arbitrary channel/gate combinations)
  - Ion channels: parameterized Boltzmann gates, 6 tau forms, alpha-beta kinetics; generalized intracellular dynamics (`IntracellularDynamics` + `Modulation` — calcium, dopamine, cAMP, or any user-defined substance via SymPy ODE)
  - Synapse models: unified `SynapseSpec` covering 7 forms — EXP_DECAY, ALPHA_FUNC, DOUBLE_EXP, TANH_GATE, BOLTZMANN_GATE, ALPHA_BETA, CUSTOM_EXPR; AMPA/NMDA/GABA-A receptor presets; arbitrary kinetics via SymPy/VM
  - Network: `RegionalNetwork` (population-level API), `Network` (internal, SoA synapses, Eigen pools)
  - Stimulators: `PulseStimulator`, `DBSStimulator`, `StimPlan` compact descriptor
  - Analysis: multitaper spectral analysis (Chronux mtspectrumpt), beta-band power, recording configs

### Canonical Workflow

```python
import hodgkin_huxley as hh

# 1. Define models
ctx = hh.NeuronModelSpec.izhikevich(hh.IzhikevichType.REGULAR_SPIKING)
stn = hh.NeuronModelSpec.stn()

# 2. Build network
net = hh.RegionalNetwork()
net.add_population("CTX", 10, ctx)
net.add_population("STN", 10, stn)
net.connect("CTX", "STN", hh.ConnectivityPattern.ONE_TO_ONE,
            synapse=hh.SynapseSpec.ampa(),
            weight=hh.WeightDistribution.constant(0.5))

# 3. Simulate
result = net.simulate(2000.0, dt=0.01,
                      I_ext={"CTX": 10.0, "STN": 0.0},
                      recording=hh.RecordingConfig.all_neuron_metrics())

# 4. Analyse
print(result["CTX"].firing_rate.mean())
hh.analyze_beta_power(result["STN"])
```

### Performance Architecture

- **Eigen SIMD pools** (`HHPool`, `IzPool`, `ComposablePool`): batched neuron update with vectorized gate/channel computation; zero heap allocation in hot loop
- **Structure-of-Arrays synapses**: flat arrays for SIMD-friendly conductance updates; type-separated groups eliminate branches in inner loop
- **StimPlan descriptors**: compact representation of DBS/pulse stimulation avoids (n_neurons × n_steps) matrix allocation
- **Fast polynomial exp**: `~8`-digit approximation via 7th-degree Taylor (default on, toggleable)

### Install

```
pip install -e .          # dev install (requires CMake 3.15+ and C++14 compiler)
pytest tests/python/      # run test suite
```

---

## Development Roadmap

See `docs/roadmap.md` for the full multi-phase plan. Summary:

| Phase | Focus | Tasks |
|-------|-------|-------|
| 1 (Complete) | Core framework + benchmark | task1–11 |
| 2 (Complete) | API cleanup, equation system, intracellular | task12 (done), task13 (done), task14 (done) |
| 3 (Next) | Plasticity | task15 |
| 4 | Performance scaling | task16, task17, task19 |
| 5 | Ecosystem / docs | task18, task20 |

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). "A quantitative description of membrane current and its application to conduction and excitation in nerve." *The Journal of Physiology*, 117(4), 500–544.
- Izhikevich, E. M. (2003). "Simple model of spiking neurons." *IEEE Transactions on Neural Networks*, 14(6), 1569–1572.
- Rubin, J. E., & Terman, D. (2004). "High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a computational model." *Journal of Computational Neuroscience*, 16(3), 211–235.
- Hahn, P. J., & McIntyre, C. C. (2010). "Modeling shifts in the rate and pattern of subthalamopallidal network activity during deep brain stimulation." *Journal of Computational Neuroscience*, 28(3), 425–441.
- Kumaravelu, K., Brocker, D. T., & Grill, W. M. (2016). "A biophysical model of the cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model of Parkinson's disease." *Journal of Computational Neuroscience*, 40(2), 207–229.
- Stimberg, M., Brette, R., & Goodman, D. F. (2019). "Brian 2, an intuitive and efficient neural simulator." *eLife*, 8, e47314.
