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

### Python-only path (no C++ required — preferred for most cases)

Arbitrary kinetics can be expressed as SymPy expressions and compiled to the `CUSTOM_EXPR` VM path with no C++ changes:

```python
import sympy as sp
import hodgkin_huxley as hh

# One-variable custom ODE
dS_dt = sp.Float(2.0) * (1 + sp.tanh(hh.V_pre / 4)) * (1 - hh.S) - hh.S / 13.0
syn = hh.SynapseModel("my_gaba", dS_dt=dS_dt, g=0.1, E_syn=-80.0)

# Two-variable custom ODE (uses A as auxiliary state)
dS_dt2 = (hh.A - hh.S) / 5.0
dA_dt2 = -hh.A / 5.0
syn2 = hh.SynapseModel("my_alpha", dS_dt=dS_dt2, dA_dt=dA_dt2,
                        spike_A=1.0, g=0.1, E_syn=0.0)

net.connect("pre", "post", "all_to_all", synapse=syn, weight=0.5)
```

`SynapseModel.to_spec()` pattern-matches against known ODE forms (EXP_DECAY, ALPHA_FUNC, DOUBLE_EXP) and selects the fast C++ path automatically. Only expressions that fail matching fall through to the VM.

### C++ path (only for new dedicated update forms requiring novel exact-integration)

### Step 1: Add an `UpdateForm` variant to `model/synapse_spec.hpp`

```cpp
enum class UpdateForm {
    EXP_DECAY, ALPHA_FUNC, DOUBLE_EXP,
    TANH_GATE, BOLTZMANN_GATE, ALPHA_BETA,
    CUSTOM_EXPR,
    MY_FORM,   // ← add here
};
```

### Step 2: Add a static factory to `SynapseSpec` in `synapse_spec.cpp`

```cpp
// Declaration in synapse_spec.hpp:
static SynapseSpec my_form(double tau, double g, double E_syn);

// Implementation in synapse_spec.cpp:
SynapseSpec SynapseSpec::my_form(double tau, double g, double E_syn) {
    SynapseSpec s;
    s.name = "my_form";
    s.update_form = UpdateForm::MY_FORM;
    s.tau_S = tau;   // reuse existing unified fields where possible
    s.g = g;
    s.E_syn = E_syn;
    return s;
}
```

Use the existing unified `SynArrays` fields (`S`, `A`, `delta_S`, `tau_S`, `tau_A`, `decay_S`, `decay_A`, `norm`) wherever possible. Add new SoA fields only if the existing set genuinely cannot express the form.

### Step 3: Add to `SynapseGroups` and update loops in `network.hpp` / `network.cpp`

```cpp
// In SynapseGroups (network.hpp):
std::vector<size_t> my_form;

// In build_synapse_groups():
case UpdateForm::MY_FORM: syn_groups_.my_form.push_back(i); break;

// In update_synapses_grouped() — branch-free inner loop:
for (size_t i : syn_groups_.my_form) {
    // exact update using sa_.S[i], sa_.tau_S[i], etc.
    // no branching within the loop body
}
```

### Step 4: Bind the new `UpdateForm` value in `bindings.cpp`

```cpp
py::enum_<SynapseSpec::UpdateForm>(m, "SynapseUpdateForm")
    // ... existing values ...
    .value("MY_FORM", SynapseSpec::UpdateForm::MY_FORM, "My custom form.");
```

### Step 5: Add a `SynapseModel` named constructor in `_equations/__init__.py`

```python
@classmethod
def my_form(cls, name: str = "my_form", *, tau: float, g: float = 0.1,
            E_syn: float = 0.0) -> "SynapseModel":
    obj = cls.__new__(cls)
    s = _SynapseSpec()
    s.name = name
    s.update_form = _SynapseUpdateForm.MY_FORM
    s.tau_S = tau
    s.g = g
    s.E_syn = E_syn
    obj._name = name
    obj._spec = s
    return obj
```

### Step 6: Test

Add tests in `tests/python/test_synapses.py` at three levels:
1. Factory creates spec with correct fields
2. Synapse transmits a spike correctly inside a `RegionalNetwork` simulation
3. Conductance decays at the expected rate

---

## 4. Adding Intracellular Dynamics

### Python-only path (no C++ required — always use this)

Intracellular substances are expressed via `IntracellularDynamics` and `Modulation` builders using SymPy expressions. No C++ changes are needed.

```python
import sympy as sp
import hodgkin_huxley as hh

Ca = hh.Ca
I_source = hh.I_source

# Standard calcium ODE — pattern-matched to DRIVEN_DECAY_NERNST (no VM overhead)
ca_dyn = hh.IntracellularDynamics(
    "Ca",
    ode=sp.Float(5.182e-6) * (-I_source - sp.Float(386.0) * Ca),
    source_channels=["Ca_L"],
    nernst=(hh.R * hh.T / (2 * hh.F)) * sp.log(sp.Float(2000.0) / Ca),
    initial=5e-5,
)
net.add_intracellular(ca_dyn, populations=["STN"])

# Dopamine with channel-g modulation
Da = hh.Da
da_dyn = hh.IntracellularDynamics(
    "DA",
    ode=-sp.Float(0.1) * Da,           # DECAY form
    initial=0.0,
    modulations=[
        hh.Modulation.channel_g("K_AHP", sp.Float(1.0) / (sp.Float(1.0) + Da)),
    ],
)
net.add_intracellular(da_dyn, populations=["STN"])
```

### Pattern matching

`IntracellularDynamics.to_spec()` attempts pattern matching before VM compilation:

| ODE expression | Form selected | Cost |
|---|---|---|
| `-k*X` | `DECAY` | zero VM |
| `ε*(-I_source - k*X)` | `DRIVEN_DECAY` | zero VM |
| As above + standard Nernst `(RT/zF)*log(X_o/X)` | `DRIVEN_DECAY_NERNST` | zero VM |
| Anything else | `CUSTOM_EXPR` (VM bytecode) | small VM eval |

### Modulation targets

| Target | Classmethod | Effect |
|---|---|---|
| `CHANNEL_G` | `Modulation.channel_g(name, expr)` | `g_eff = g * expr(X)` |
| `CHANNEL_EREV` | `Modulation.channel_erev(name, expr)` | `E_rev = expr(X)` |
| `GATE_INF_SHIFT` | `Modulation.gate_inf_shift(name, scale)` | `x_inf(V + scale*X)` |
| `GATE_INF_SCALE` | `Modulation.gate_inf_scale(name, expr)` | `x_inf *= expr(X)` |
| `GATE_TAU_SCALE` | `Modulation.gate_tau_scale(name, expr)` | `tau *= expr(X)` |
| `GATE_INF_EXPR` | `Modulation.gate_inf_expr(name, expr)` | `x_inf = expr(V, X)` |
| `SYNAPSE_G` | `Modulation.synapse_g(expr)` | `I_syn *= expr(X)` per postsynaptic neuron |

### Tests

Add tests in `tests/python/test_intracellular.py`:
```python
def test_ca_decay_rate():
    """DECAY form: exponential decay constant should match k_decay within 1%."""
    ...

def test_modulation_channel_g():
    """CHANNEL_G mod: two identical networks, one with mod vs manual g scaling."""
    ...
```

---

## 5. IO Conventions

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

## 6. Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| C++ classes | PascalCase | `ComposablePool`, `SynArrays` |
| C++ methods/fields | snake_case | `step_rk4()`, `exp_decay_` |
| C++ private members | trailing underscore | `n_`, `fast_math_`, `V_cache_` |
| C++ enums | UPPER_SNAKE | `SynapseSpec::UpdateForm::EXP_DECAY`, `GateDependency::VOLTAGE` |
| C++ spec structs | PascalCase + Spec/Params | `TauParams`, `NeuronModelSpec` |
| Python-exposed names | snake_case (match C++) | `add_population()`, `num_neurons()` |
| Python spec types | PascalCase | `SynapseSpec`, `WeightDistribution` |
| Test files | `test_<feature>.py` | `test_composable_neuron.py` |
| Task files | `task<N>.md` | `task12.md` |

---

## 7. Testing Conventions

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

## 8. Python Binding Conventions

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

## 9. Documentation Conventions

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

## 10. Build System Conventions

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

## 11. Common Pitfalls

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

`decay_S[]` and `decay_A[]` are cached values of `exp(-dt/tau_S)` and `exp(-dt/tau_A)`. They are recomputed in `update_decay_factors(dt)` only when `dt` changes. If a synapse's tau parameter is modified after simulation has started, set `sa_.cached_dt = -1.0` to force recomputation on the next step.

### Thread Safety

The `Network` object is **not thread-safe**. Do not simulate the same network from multiple threads. For parallelism, use separate `Network` instances per thread (after task 15, the hot loop itself uses OpenMP internally but the object remains single-writer).
