# Task 21: `CompartmentSpec` + `MorphologySpec` — Data Structures

**Depends on:** task14 (IntracellularSpec, GateSpec, ChannelSpec must be defined)  
**Unlocks:** task22 (HinesSolver needs MorphologySpec), task24 (Python builder needs these types)

---

## What to implement

This task adds the two foundational data structs for multi-compartment models — `CompartmentSpec` and `MorphologySpec` — and hooks them into `NeuronModelSpec`. No simulation logic changes; this is pure data layer work. All existing point-neuron code is unaffected: `NeuronModelSpec::has_morphology()` returns false when `morphology` is empty.

### `src/cpp/include/hodgkin_huxley/model/compartment_spec.hpp` — new file

```cpp
#pragma once
#include "hodgkin_huxley/model/gate_spec.hpp"
#include "hodgkin_huxley/model/channel_spec.hpp"
#include <string>
#include <vector>

namespace hodgkin_huxley {

struct CompartmentSpec {
    std::string name;             // "soma", "dend[0]", "axon", etc.
    double length_um   = 100.0;   // compartment length (µm)
    double diameter_um =   1.0;   // compartment diameter (µm)
    double Ra          = 100.0;   // axial resistivity (Ω·cm)
    double Cm          =   1.0;   // specific membrane capacitance (µF/cm²)

    std::vector<GateSpec>          gates;
    std::vector<ChannelSpec>       channels;
    std::vector<IntracellularSpec> intracellular;
};

struct MorphologySpec {
    std::vector<CompartmentSpec> compartments;
    // parent_idx[i] = index of parent compartment; -1 for root (soma).
    // Invariant: parent_idx[i] < i for all i > 0 (topological order).
    std::vector<int> parent_idx;

    // Axial coupling conductances (µS) — filled by compute_coupling().
    // g_axial[i] = pi * d_i^2 / (4 * Ra_i * L_i)  [converted to µS from geometry].
    // g_axial[0] = 0 (root has no parent).
    std::vector<double> g_axial;

    int n_comps() const { return static_cast<int>(compartments.size()); }

    // Validate topology: parent_idx.size() == compartments.size(),
    // parent_idx[0] == -1, parent_idx[i] < i for i > 0.
    // Throws std::invalid_argument on failure.
    void validate() const;

    // Compute g_axial from compartment geometry.  Must be called after
    // validate() and before passing to HinesSolver or MCPool.
    void compute_coupling();
};

} // namespace hodgkin_huxley
```

### `src/cpp/include/hodgkin_huxley/model/neuron_spec.hpp` — add morphology field

Find the `NeuronModelSpec` struct and add after the existing `intracellular` field:

```cpp
MorphologySpec morphology;  // empty = point neuron (existing behaviour)

bool has_morphology() const { return !morphology.compartments.empty(); }
```

No existing fields change. CPU pools check `has_morphology()` at construction and throw if true (until task23 lands). `MCPool` (task23) will consume the morphology.

### `src/cpp/src/morphology_spec.cpp` — new file

Implements `validate()` and `compute_coupling()`:

```cpp
void MorphologySpec::validate() const {
    const int C = static_cast<int>(compartments.size());
    if (static_cast<int>(parent_idx.size()) != C)
        throw std::invalid_argument("MorphologySpec: parent_idx size mismatch");
    if (C > 0 && parent_idx[0] != -1)
        throw std::invalid_argument("MorphologySpec: root compartment must have parent_idx = -1");
    for (int i = 1; i < C; ++i)
        if (parent_idx[i] < 0 || parent_idx[i] >= i)
            throw std::invalid_argument(
                "MorphologySpec: parent_idx[" + std::to_string(i) +
                "] must satisfy 0 <= parent_idx < i");
}

void MorphologySpec::compute_coupling() {
    const int C = n_comps();
    g_axial.assign(C, 0.0);
    for (int i = 1; i < C; ++i) {
        const auto& comp = compartments[i];
        // Convert: length µm → cm (×1e-4), diameter µm → cm (×1e-4)
        const double L_cm = comp.length_um * 1e-4;
        const double d_cm = comp.diameter_um * 1e-4;
        // g = pi * d^2 / (4 * Ra * L)  [Ω^-1 = S → multiply by 1e6 for µS]
        g_axial[i] = M_PI * d_cm * d_cm / (4.0 * comp.Ra * L_cm) * 1e6;
    }
}
```

### `src/python/bindings.cpp` — bind new types

Add before the existing `NeuronModelSpec` binding block:

```cpp
py::bind_vector<std::vector<CompartmentSpec>>(m, "CompartmentSpecVector");

py::class_<CompartmentSpec>(m, "CompartmentSpec")
    .def(py::init<>())
    .def(py::init([](const std::string& name, double length_um, double diameter_um) {
        CompartmentSpec s;
        s.name = name; s.length_um = length_um; s.diameter_um = diameter_um;
        return s;
    }), py::arg("name"), py::arg("length_um") = 100.0, py::arg("diameter_um") = 1.0)
    .def_readwrite("name",          &CompartmentSpec::name)
    .def_readwrite("length_um",     &CompartmentSpec::length_um)
    .def_readwrite("diameter_um",   &CompartmentSpec::diameter_um)
    .def_readwrite("Ra",            &CompartmentSpec::Ra)
    .def_readwrite("Cm",            &CompartmentSpec::Cm)
    .def_readwrite("gates",         &CompartmentSpec::gates)
    .def_readwrite("channels",      &CompartmentSpec::channels)
    .def_readwrite("intracellular", &CompartmentSpec::intracellular)
    .def("__repr__", [](const CompartmentSpec& c) {
        return "<CompartmentSpec '" + c.name + "' " +
               std::to_string(c.length_um) + "µm × ø" +
               std::to_string(c.diameter_um) + "µm>";
    });

py::class_<MorphologySpec>(m, "MorphologySpec")
    .def(py::init<>())
    .def(py::init([](std::vector<CompartmentSpec> comps, std::vector<int> parents) {
        MorphologySpec m;
        m.compartments = std::move(comps);
        m.parent_idx   = std::move(parents);
        m.validate();
        m.compute_coupling();
        return m;
    }), py::arg("compartments"), py::arg("parent_idx"))
    .def_readwrite("compartments", &MorphologySpec::compartments)
    .def_readwrite("parent_idx",   &MorphologySpec::parent_idx)
    .def_readonly ("g_axial",      &MorphologySpec::g_axial)
    .def("n_comps",           &MorphologySpec::n_comps)
    .def("validate",          &MorphologySpec::validate)
    .def("compute_coupling",  &MorphologySpec::compute_coupling)
    .def("__repr__", [](const MorphologySpec& m) {
        return "<MorphologySpec " + std::to_string(m.n_comps()) + " compartments>";
    });
```

Add `has_morphology()` and `morphology` to the existing `NeuronModelSpec` binding:
```cpp
.def_readwrite("morphology",    &NeuronModelSpec::morphology)
.def("has_morphology",          &NeuronModelSpec::has_morphology)
```

### `src/hodgkin_huxley/__init__.py` — exports

Add to the `from ._core import (...)` block:
```python
CompartmentSpec,
MorphologySpec,
```

Add both names to `__all__`.

### Guard in existing CPU pools

In `ComposablePool::add()` (or constructor), add:
```cpp
if (spec_.has_morphology())
    throw std::runtime_error(
        "ComposablePool does not support multi-compartment morphologies. "
        "Use MCPool (task23) instead.");
```

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/model/compartment_spec.hpp` | New — `CompartmentSpec`, `MorphologySpec` |
| `src/cpp/src/morphology_spec.cpp` | New — `validate()`, `compute_coupling()` |
| `src/cpp/include/hodgkin_huxley/model/neuron_spec.hpp` | Add `morphology` field + `has_morphology()` |
| `src/cpp/CMakeLists.txt` | Add `src/morphology_spec.cpp` to library sources |
| `src/python/bindings.cpp` | Bind `CompartmentSpec`, `MorphologySpec`, new `NeuronModelSpec` fields |
| `src/hodgkin_huxley/__init__.py` | Export `CompartmentSpec`, `MorphologySpec` |

---

## Baseline tests (before PR to testing branch)

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] Construct a 3-compartment morphology (soma + 2 dendrites, star topology): `parent_idx = [-1, 0, 0]`; verify `n_comps() == 3`
- [ ] `validate()` passes for valid topology; raises `ValueError` for `parent_idx[1] = 2` (forward reference) and for `parent_idx[0] = 1` (non-root root)
- [ ] `compute_coupling()` fills `g_axial[0] == 0`; `g_axial[1]` matches hand-calculated `pi * d^2 / (4 * Ra * L)` in µS
- [ ] `NeuronModelSpec.stn().has_morphology() == False` (existing presets unaffected)
- [ ] `NeuronModelSpec` with morphology assigned: `has_morphology() == True`
- [ ] `ComposablePool` construction with a morphology-bearing spec raises `RuntimeError`

---

## Contract for downstream tasks

- task22's `HinesSolver` is constructed from a `const MorphologySpec&` — assumes `validate()` and `compute_coupling()` already called.
- `g_axial[i]` is in µS; `L_cm` and `A_cm2` are derived locally in the pool from `compartments[i]` geometry.
- `parent_idx[0] == -1`; all other indices satisfy `0 <= parent_idx[i] < i` — topological order is guaranteed.
