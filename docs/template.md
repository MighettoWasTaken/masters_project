# Development Template

This document defines the programming conventions, patterns, and step-by-step procedures that govern all development in this project. Every new feature must follow these conventions to maintain a uniform structure across the codebase.

---

## 1. Core Principle: Data Over Code

All neuron and synapse models are described as **parameter structs**, not class hierarchies. A model is a collection of named numeric parameters. Behaviour (computation) is implemented once in a pool or update function that reads from the struct.

**Do this:**
```cpp
// Model is pure data
struct MyGateSpec { double v_half; double k; double tau_const; };
NeuronModelSpec spec;
spec.gates.push_back(build_gate(my_gate));  // pool handles the math
```

**Not this:**
```cpp
// Model encodes behaviour in a class — can't be vectorized
class MyNeuron : public NeuronBase {
    double compute_gate(double V) { return 1.0 / (1.0 + exp(-(V - v_half) / k)); }
};
```

The distinction matters because pools vectorize computation across N neurons using Eigen. A model spec is replicated across N state arrays; a class hierarchy cannot be.

---

## 2. Adding a New Neuron Model

### Step 1: Define a NeuronModelSpec (no C++ required for most cases)

If your model uses only existing gate update forms and tau forms, express it entirely through `NeuronModelSpec`:

```cpp
// In ion_channels.hpp, add a static factory:
static NeuronModelSpec my_cell() {
    NeuronModelSpec spec;
    spec.name = "MyCell";
    spec.C_m = 1.0;

    // Gate: voltage-dependent, INF_TAU form
    GateSpec m_gate;
    m_gate.name = "m";
    m_gate.update_form = GateSpec::UpdateForm::INF_TAU;
    m_gate.dependency  = GateDependency::VOLTAGE;
    m_gate.inf  = BoltzmannParams{-40.0, 8.0};        // v_half, k
    m_gate.tau  = TauParams::constant(0.5);            // see task16 for named constructors
    spec.gates.push_back(m_gate);

    // Channel: Na, uses m^3
    ChannelSpec na;
    na.name  = "Na";
    na.g     = 50.0;
    na.E_rev = 60.0;
    na.gates = {{0, 3}};   // gate index 0, power 3
    spec.channels.push_back(na);

    // Leak
    ChannelSpec leak;
    leak.name  = "Leak";
    leak.g     = 0.1;
    leak.E_rev = -65.0;
    // no gates = pure leak
    spec.channels.push_back(leak);

    return spec;
}
```

### Step 2: Bind the factory in bindings.cpp

```cpp
// In the NeuronModelSpec pybind section:
py::class_<NeuronModelSpec>(m, "NeuronModelSpec")
    // ...existing bindings...
    .def_static("my_cell", &NeuronModelSpec::my_cell,
                "My custom cell model (brief description)");
```

### Step 3: Export in `__init__.py`

`NeuronModelSpec` is already exported. No changes needed if the factory is a static method on it.

### Step 4: Write tests

Add a test file or extend an existing one in `tests/python/`:
```python
def test_my_cell_resting():
    """Cell should rest near -65 mV with zero drive."""
    spec = hh.NeuronModelSpec.my_cell()
    net = hh.RegionalNetwork()
    net.add_population("test", 1, spec)
    result = net.simulate(100.0, 0.01,
                          I_ext={"test": 0.0},
                          recording=hh.RecordingConfig.voltage_only())
    assert abs(result["test"].V[0, -1] - (-65.0)) < 5.0
```

### Step 5 (if new math): Extend the pool

If the model requires a gate update form not in `GateSpec::UpdateForm` or a tau form not in `TauParams::Form`:

1. Add the new variant to the enum
2. Add a branch in `ComposablePool::step()` handling the new form (keep it a single vectorized Eigen expression where possible)
3. Add validation in `ion_channels.hpp` to reject specs with unknown forms at construction time

---

## 3. Adding a New Synapse Type

### Step 1: Extend `SynType` and `SynArrays` in `network.hpp`

```cpp
// In Network's private SynType enum:
enum class SynType : uint8_t { SYN_EXP = 0, SYN_ALPHA = 1, SYN_DEXP = 2, SYN_KINETIC = 3, SYN_MYTYPE = 4 };

// In SynArrays, add type-specific state fields:
std::vector<double> mytype_param1;
std::vector<double> mytype_state1;

// In SynArrays::push_type_defaults():
mytype_param1.push_back(0.0);
mytype_state1.push_back(0.0);
```

### Step 2: Add construction method in `network.hpp` / `network.cpp`

```cpp
// In Network:
size_t add_my_synapse(size_t pre, size_t post, double weight,
                      double E_syn, double param1, double delay = 0.0);

// In network.cpp:
size_t Network::add_my_synapse(...) {
    // Push common fields
    sa_.pre.push_back(pre); sa_.post.push_back(post);
    sa_.weight.push_back(weight); sa_.E_syn.push_back(E_syn);
    sa_.g.push_back(0.0); sa_.type.push_back(SynType::SYN_MYTYPE);
    // Push type defaults (fills all other type-specific fields with 0)
    sa_.push_type_defaults();
    // Override this type's fields
    sa_.mytype_param1.back() = param1;
    // ... delay setup ...
    groups_built_ = false;  // invalidate
    // Create polymorphic object (for API access)
    synapses_.push_back(std::make_unique<MySynapse>(...));
    return synapses_.size() - 1;
}
```

### Step 3: Add to `build_synapse_groups()` and `update_synapses_grouped()`

```cpp
// In build_synapse_groups():
case SynType::SYN_MYTYPE: syn_groups_.mytype.push_back(i); break;

// In update_synapses_grouped(), add a new type-specific loop:
for (size_t i : syn_groups_.mytype) {
    // update sa_.g[i] and sa_.mytype_state1[i]
    // This loop must be branch-free within the loop body
}
```

### Step 4: Bind in bindings.cpp

```cpp
network_class
    .def("add_my_synapse", &Network::add_my_synapse,
         py::arg("pre"), py::arg("post"), py::arg("weight"),
         py::arg("E_syn"), py::arg("param1"), py::arg("delay") = 0.0);
```

### Step 5: Add to SynapseSpec if population-level use is needed

Add a static factory to `SynapseSpec` in `regional_network.hpp`:
```cpp
static SynapseSpec my_type(double E_syn, double param1);
```

---

## 4. IO Conventions

### 4.1 Arrays Crossing the Python/C++ Boundary

All arrays passed between Python and C++ are `float64`, C-contiguous numpy arrays. The pybind11 signature must enforce this:

```cpp
void my_function(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr)
```

`forcecast` converts non-contiguous or non-float64 arrays automatically (with a copy). Use `c_style` to ensure row-major layout.

### 4.2 Pre-Allocated Buffers (Zero-Copy Pattern)

For large output arrays (recording buffers), Python pre-allocates the buffer and passes a raw pointer to C++:

```python
# Python: allocate
buf = np.zeros((n_neurons, n_steps), dtype=np.float64)
net._simulate_into_buffers(..., V_buf=buf)
# buf is now filled in-place
```

```cpp
// C++: accept pointer, fill in-place
void Network::simulate_into_buffers(..., double* V_buf, ...) {
    V_buf[neuron * n_rec + step] = V;
}
```

This avoids a copy of potentially hundreds of MB. The convention is: if the output is large (>1 MB), pre-allocate in Python and fill in C++.

### 4.3 Scalar and Small Outputs

Small results (single values, short vectors) are returned by value:

```cpp
// Return by value — pybind11 converts to Python list/float
std::vector<double> Network::get_potentials() const;
double Network::get_kin_S(size_t idx) const;
```

### 4.4 I_ext Interface (Python Layer)

Accepted forms for `I_ext` in `RegionalNetwork.simulate()`:

| Form | Type | Meaning |
|------|------|---------|
| Per-population scalar | `{"CTX": 10.0, "STN": 0.0}` | Constant current for all neurons in population |
| Per-population stimulator | `{"CTX": DBSStimulator(...)}` | Time-varying, evaluated on-the-fly |
| Per-population 1D array | `{"CTX": np.array([10.0, 12.0, ...])}` | Per-neuron constant (length = pop size) |
| Dense 2D array | `np.zeros((N, n_steps))` | Legacy path; avoid unless necessary |

The Python `simulate()` method converts these to either a `_StimPlan` (for scalar/stimulator inputs, zero-allocation path) or a dense matrix (fallback).

---

## 5. Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| C++ classes | PascalCase | `ComposablePool`, `SynArrays` |
| C++ methods/fields | snake_case | `step_rk4()`, `exp_decay_` |
| C++ private members | trailing underscore | `n_`, `fast_math_`, `V_cache_` |
| C++ enums | UPPER_SNAKE | `SynType::SYN_EXP`, `GateDependency::VOLTAGE` |
| C++ spec structs | PascalCase + Spec/Params | `TauParams`, `NeuronModelSpec` |
| Python-exposed names | snake_case (match C++) | `add_population()`, `num_neurons()` |
| Python spec types | PascalCase | `SynapseSpec`, `WeightDistribution` |
| Test files | `test_<feature>.py` | `test_composable_neuron.py` |
| Task files | `task<N>.md` | `task12.md` |

---

## 6. Testing Conventions

### File Organization

One test file per major feature area (see `tests/python/`). Tests within a file are grouped by behaviour, not implementation.

### Test Structure

```python
# Test naming: test_<what>_<expected_behaviour>
def test_hh_neuron_fires_under_injection():
    """HH neuron should spike when I_ext exceeds threshold."""
    ...

def test_hh_neuron_rests_at_zero_drive():
    """HH neuron should maintain stable resting potential without drive."""
    ...
```

### What to Test

For each new feature, write tests at three levels:

1. **Unit tests** — isolated component behaviour (gate value at steady state, synapse conductance decay constant, etc.)
2. **Integration tests** — feature works inside a network simulation (synapse transmits spike, population fires at expected rate)
3. **Regression tests** — numerical values match reference for key biological parameters (e.g., STN preset fires at ~18 Hz under tonic drive)

### Numerical Tolerances

- Gate steady states: `abs(actual - expected) < 1e-6`
- Firing rates: `abs(actual - expected) / expected < 0.05` (5% tolerance)
- Spike times: `abs(actual - expected) < 1.0` ms
- Voltage traces: only test qualitative features (spiking, resting) unless validating against known reference

### Running Tests

```
pytest tests/python/                    # all tests
pytest tests/python/test_networks.py   # one file
pytest tests/python/ -k "hh"           # tests matching pattern
```

---

## 7. Python Binding Conventions

### Binding a New Class

```cpp
py::class_<MyClass>(m, "MyClass", "One-line docstring.")
    .def(py::init<double, int>(), py::arg("param1"), py::arg("param2"),
         "Constructor docstring.\n\nArgs:\n    param1: ...\n    param2: ...")
    .def("method", &MyClass::method, py::arg("x"), py::arg("y") = 0.0,
         "Method docstring.")
    .def_property("value",
        [](const MyClass& self) { return self.value(); },
        [](MyClass& self, double v) { self.set_value(v); },
        "Property docstring.");
```

### Hot-Loop Methods (GIL Release)

Any method that runs a long C++ simulation must release the GIL so other Python threads can run:

```cpp
.def("simulate", [](Network& self, ...) {
    py::gil_scoped_release release;
    return self.simulate_internal(...);
});
```

### Buffer Methods (Zero-Copy)

```cpp
.def("_simulate_into_buffers",
    [](Network& self, ...,
       py::array_t<double, py::array::c_style> V_buf) {
        py::gil_scoped_release release;
        self.simulate_into_buffers(..., V_buf.mutable_data(), ...);
    });
```

### Enums

```cpp
py::enum_<MyEnum>(m, "MyEnumName")
    .value("VALUE_A", MyEnum::VALUE_A, "Description of VALUE_A.")
    .value("VALUE_B", MyEnum::VALUE_B, "Description of VALUE_B.")
    .export_values();
```

---

## 8. Documentation Conventions

### C++ Docstrings (for pybind11)

All public methods and classes exposed to Python require a docstring in the binding that includes:
- One-line summary
- Args section (if any parameters)
- Returns section (if non-void)
- Example (for non-trivial methods)

These feed directly into the auto-generated API reference (task 18).

### Task Files

Each planned feature gets a `task<N>.md` in the project root following this structure:

```markdown
# Task N: Title
## Priority: 1/2/3
## Overview
What and why.
## N.1 Architecture / Design
Structs, algorithms, data flow.
## N.2 C++ Design
Code snippets for key types and methods.
## N.3 Python API
Usage examples.
## Implementation Checklist
- [ ] item
```

Completed tasks move to `completed/`. Failed or abandoned tasks move to `failed/`.

---

## 9. Build System Conventions

### Adding a New Source File

1. Add `src/cpp/src/my_file.cpp` (implementation) and `src/cpp/include/hodgkin_huxley/my_file.hpp` (header)
2. Register in `src/cpp/CMakeLists.txt`:
   ```cmake
   target_sources(hodgkin_huxley_core PRIVATE src/my_file.cpp)
   ```

### Optional Compile-Time Features

Use CMake options and preprocessor definitions for optional features (CUDA, OpenMP):
```cmake
option(USE_MYFEATURE "Enable my feature" OFF)
if(USE_MYFEATURE)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_MYFEATURE)
endif()
```

Wrap feature-specific C++ code:
```cpp
#ifdef HH_USE_MYFEATURE
// ...
#endif
```

### Header-Only Dependencies

Prefer FetchContent for header-only deps (like Eigen):
```cmake
FetchContent_Declare(eigen ...)
FetchContent_MakeAvailable(eigen)
target_link_libraries(hodgkin_huxley_core PUBLIC Eigen3::Eigen)
```

---

## 10. Common Pitfalls

### Eigen Lazy Evaluation (CRITICAL)

`auto x = (array >= threshold)` creates a **lazy expression** referencing `array`. If `array` is modified before `x` is used, `x` reads the modified values.

**Always** materialize boolean masks before modifying the source:
```cpp
auto fired = (v_ >= 30.0).eval();   // .eval() materializes immediately
v_ = fired.select(c_, v_);          // safe: fired is not lazy
u_ = fired.select(u_ + d_, u_);    // safe
```

### SoA Invalidation

After adding or removing synapses, `groups_built_` must be set to `false` so `build_synapse_groups()` rebuilds the type-separated index lists on the next simulation call. Always set this flag at the end of any method that modifies `sa_`.

### Cached Decay Factors

`exp_decay[]` and `dexp_rise_decay[]` / `dexp_fall_decay[]` are cached values of `exp(-dt/tau)`. They are recomputed in `update_decay_factors(dt)` only when `dt` changes. If a synapse's tau parameter is modified after simulation has started, call `sa_.cached_dt = -1.0` to force recomputation on the next step.

### Thread Safety

The `Network` object is **not thread-safe**. Do not simulate the same network from multiple threads. For parallelism, use separate `Network` instances per thread (after task 15, the hot loop itself uses OpenMP internally but the object remains single-writer).
