# Task 18: Web Documentation

## Priority: 3

## Overview

Create comprehensive, web-accessible documentation hosted on GitHub Pages. The README covers installation and a quick example, but there is no reference for the full API, no tutorial explaining how to build a custom neuron model, and no architecture guide for contributors. This task produces a documentation site with auto-generated API reference, narrative tutorials, and a contributor guide.

---

## 18.1 Toolchain

**MkDocs** with the **Material** theme. Chosen over Sphinx for:
- Markdown-native (consistent with existing task files)
- Material theme is modern and well-supported
- `mkdocstrings-python` handles pybind11 class docstrings

```
pip install mkdocs mkdocs-material mkdocstrings[python]
```

**Deployment:** GitHub Pages via `mkdocs gh-deploy`. A GitHub Actions workflow deploys automatically on push to `main`.

---

## 18.2 Site Structure

```
docs-site/
├── mkdocs.yml
└── docs/
    ├── index.md                   (landing page / quick start)
    ├── installation.md
    ├── tutorials/
    │   ├── first-simulation.md    (single HH neuron → network → recording)
    │   ├── custom-neurons.md      (composable system: TH model from scratch)
    │   ├── bg-thalamus-model.md   (CTX-BG-TH benchmark walkthrough)
    │   ├── dbs-stimulation.md     (DBS setup, beta power analysis)
    │   └── plasticity.md          (STDP on a simple two-neuron circuit)
    ├── user-guide/
    │   ├── neuron-models.md       (HH, Izhikevich, composable; when to use each)
    │   ├── synapse-types.md       (exp, alpha, double-exp, kinetic; receptor presets)
    │   ├── networks.md            (RegionalNetwork, populations, connectivity patterns)
    │   ├── stimulation.md         (I_ext, PulseStimulator, DBSStimulator, NoiseInjector)
    │   ├── recording.md           (RecordingConfig, MetricsResult, population access)
    │   ├── spectral-analysis.md   (mtspectrumpt, beta band power)
    │   └── performance.md         (fast_math, StimPlan, CUDA, OpenMP)
    ├── api-reference/
    │   ├── network.md             (auto-generated from docstrings)
    │   ├── neuron-models.md
    │   ├── synapses.md
    │   ├── stimulation.md
    │   ├── recording.md
    │   └── spectral.md
    └── contributing/
        ├── architecture.md        (links to docs/implementation_details.md)
        ├── adding-neuron-model.md
        ├── adding-synapse-type.md
        └── building.md
```

---

## 18.3 Docstring Requirements

pybind11 class and method docstrings are the source for API reference generation. Before running `mkdocstrings`, all public classes and methods in `bindings.cpp` require docstrings covering:

- One-line summary
- Parameters (name, type, description)
- Returns
- Raises
- Example (where non-trivial)

Priority order for docstring coverage:
1. `RegionalNetwork` (all methods)
2. `NeuronModelSpec` (all presets and factories)
3. `SynapseSpec`, `WeightDistribution`, `ConnectivityPattern`
4. `RecordingConfig`, `MetricsResult`, `PopulationMetricsResult`
5. `DBSStimulator`, `PulseStimulator`, `NoiseInjector`
6. Composable system types (`GateSpec`, `ChannelSpec`, `Tau`, `RateFunc`, etc.)

---

## 18.4 GitHub Actions Workflow

```yaml
# .github/workflows/docs.yml
name: Deploy docs
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install mkdocs mkdocs-material mkdocstrings[python]
      - run: mkdocs gh-deploy --force
```

---

## 18.5 Content Priorities

The most important pages, in order:

1. **index.md** — must show the complete canonical workflow (from task 17) in < 30 lines of code
2. **tutorials/custom-neurons.md** — the thalamic neuron from scratch; this is the most common advanced use case
3. **user-guide/networks.md** — population setup, connectivity patterns, weight distributions
4. **api-reference/** — auto-generated; requires docstrings complete first (see 18.3)
5. **tutorials/bg-thalamus-model.md** — showcase the benchmark as a worked example

---

## 18.6 Implementation Checklist

### Setup
- [ ] Add `mkdocs.yml` to project root with Material theme configuration
- [ ] Add `docs-site/` directory structure
- [ ] Add `.github/workflows/docs.yml` GitHub Actions deployment workflow
- [ ] Add `mkdocs-material`, `mkdocstrings[python]` to `pyproject.toml` optional deps `[docs]`

### Docstrings
- [ ] Write docstrings for all `RegionalNetwork` methods in `bindings.cpp`
- [ ] Write docstrings for `NeuronModelSpec` presets and factories
- [ ] Write docstrings for `SynapseSpec`, `WeightDistribution`, `RecordingConfig`
- [ ] Write docstrings for `DBSStimulator`, `PulseStimulator`, `NoiseInjector`

### Tutorial Content
- [ ] Write `tutorials/first-simulation.md`
- [ ] Write `tutorials/custom-neurons.md` (thalamic model walkthrough)
- [ ] Write `tutorials/bg-thalamus-model.md` (CTX-BG-TH benchmark)
- [ ] Write `tutorials/dbs-stimulation.md`

### User Guide
- [ ] Write `user-guide/neuron-models.md`
- [ ] Write `user-guide/networks.md`
- [ ] Write `user-guide/recording.md`
- [ ] Write `user-guide/performance.md`

### Contributing
- [ ] Write `contributing/adding-neuron-model.md` (links to `docs/template.md`)
- [ ] Write `contributing/building.md` (CMake, conda env, test runner)
