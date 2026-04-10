# Task 13: Unified Equation System (SymPy)

## Priority: 2 — Prerequisite for task14, task15, and equation-related API cleanup (13.9)

## Overview

Every location in the library where a user defines mathematical relationships — gate kinetics, time constants, rate functions, synapse gating ODEs, intracellular dynamics, plasticity update rules — currently uses a different interface: `BoltzmannParams`, `TauParams` with a raw `params[8]` array, `RateFuncParams`, `KineticSynapseSpec` form enums, hardcoded ODE structures in `CalciumSpec`. Each of these is opaque, non-composable, and limited to a fixed set of built-in forms.

This task replaces all of these with a single interface: **SymPy expressions**. The user writes equations using SymPy symbols. The library compiles them to efficient C++ — Eigen-vectorized for CPU, scalar `__device__` functions for CUDA — with pre-compiled fast paths for all standard forms recognized by pattern matching.

The result: one equation language for the entire library, arbitrary expressiveness, and no performance penalty for standard use. This task also performs the final equation-type API cleanup (section 13.9), retiring the legacy spec types to `hodgkin_huxley.legacy`.

---

## 13.1 Equation Definition Points (Complete Inventory)

Every place in the codebase where the user currently writes a mathematical relationship:

| Location | Field(s) | Current Type | Symbols needed |
|----------|----------|--------------|----------------|
| `GateSpec.inf` | steady-state x_inf | `BoltzmannParams` | `V` or substance symbol |
| `GateSpec.tau` | time constant tau_x | `TauParams` (6 forms, `params[8]`) | `V` |
| `GateSpec.alpha/beta` | rate functions | `RateFuncParams` (4 forms) | `V` |
| `GateSpec.derived_*` | derived gate expression | raw scalar arithmetic fields | gate name symbols |
| `ChannelSpec.ahp_k1` | AHP gating | hardcoded `Ca/(Ca+k1)` | `Ca` |
| `CalciumSpec` | d[Ca]/dt ODE | 4 fixed numeric fields | `Ca`, `I_sources` |
| `KineticSynapseSpec` | dS/dt ODE | 3 form enums + parameter fields | `V_pre`, `S` |
| `KineticSynapseSpec` | synaptic current I | 2 form enums + parameter fields | `S`, `V_post` |
| `IntracellularSpec.dynamics` | d[X]/dt ODE (task14) | `IntracellularDynamicsSpec` fields | `X`, `I_sources` |
| `IntracellularSpec.modulations` | modulation function (task14) | linear `scale` field only | `X` |
| `STDPParams` update rules (task15) | weight change equation | hardcoded Δw = A * trace form | `x_pre`, `x_post`, `w` |
| `STPParams` dynamics (task15) | u, x ODEs | hardcoded Tsodyks-Markram | `u`, `x` |

All of these are replaced by SymPy expressions using pre-defined library symbols.

---

## 13.2 Pre-Defined Symbols

A set of named SymPy symbols is pre-defined in the `hodgkin_huxley` namespace so users never import from SymPy directly unless they need custom symbols:

```python
import hodgkin_huxley as hh

# Neuron state
hh.V       # membrane potential
hh.Ca      # calcium concentration (or any intracellular substance by symbol name)

# Kinetic synapse state
hh.V_pre   # pre-synaptic membrane potential
hh.V_post  # post-synaptic membrane potential
hh.S       # kinetic synapse gating variable

# Plasticity state
hh.x_pre   # pre-synaptic STDP trace
hh.x_post  # post-synaptic STDP trace
hh.w       # synaptic weight

# Gate variable access (for derived gates)
# hh.gate("name") returns a symbol that refers to that gate's current value
```

Users can also import SymPy directly for full expressiveness:
```python
from sympy import symbols, exp, tanh, log
my_V = symbols("V")  # equivalent to hh.V
```

---

## 13.3 Usage at Every Equation Point

### Gate kinetics

```python
V = hh.V

# Before: BoltzmannParams(-40, 8) + TauParams(TauForm.CONSTANT, [0.5, ...])
# After:
gate = hh.GateSpec(
    name="m",
    inf = 1 / (1 + exp(-(V + 40) / 8)),   # any SymPy expr in V
    tau = 0.5                               # constant — auto-converted to SymPy Float
)

# Alpha-beta form (Striatum gates)
gate = hh.GateSpec(
    name="m",
    alpha = 0.32 * (V + 54) / (1 - exp(-(V + 54) / 4)),
    beta  = 0.28 * (V + 27) / (exp((V + 27) / 5) - 1)
)

# Calcium-dependent gate (dependency inferred from symbols present in expr)
Ca = hh.Ca
gate = hh.GateSpec(
    name="R_ca",
    inf = 1 / (1 + exp(-(Ca - 0.17) / 0.08)),
    tau = 2.0
)

# Derived gate: 0.75*(1-H)
H = hh.gate("H")
gate = hh.GateSpec(name="n_K", expr=0.75 * (1 - H))
```

### Intracellular substance dynamics (task14)

```python
Ca, I_src = hh.Ca, hh.I_source  # I_source = sum of named source channels
dCa_dt = 1e-4 * (-I_src - 15.0 * Ca)
E_Ca_expr = 8314 * 298 / (2 * 96485) * log(2000 / Ca)  # Nernst

substance = hh.IntracellularSpec(
    name="calcium",
    ode=dCa_dt,
    source_channels=["L_Ca", "T_Ca"],
    nernst=E_Ca_expr,
    initial=0.005
)
```

### Kinetic synapse (dS/dt + current)

```python
V_pre, S = hh.V_pre, hh.S

# TANH_GATE equivalent
rate_open = 0.5 * (1 + tanh((V_pre + 20) / 16))
kin_spec = hh.KineticSynapseSpec(
    dS_dt   = rate_open * (1 - S) - S / 13.0,
    current = hh.g * S * (hh.V_post - hh.E_syn),
    g=0.1, E_syn=-80.0
)

# MG-block NMDA current
kin_spec = hh.KineticSynapseSpec(
    dS_dt   = ...,
    current = hh.g * S / (1 + 1.0 * exp(-0.062 * hh.V_post) / 3.57)
              * (hh.V_post - hh.E_syn)
)
```

### Plasticity update rules (task15)

```python
x_pre, x_post, w = hh.x_pre, hh.x_post, hh.w

stdp = hh.PlasticitySpec(
    dx_pre_dt  = -x_pre / 20.0,
    dx_post_dt = -x_post / 20.0,
    on_pre_spike  = w + 0.005 * x_post,
    on_post_spike = w - 0.006 * x_pre,
    w_min=0.0, w_max=1.0
)
```

---

## 13.4 Preset Helpers (Backwards-Compatible Shorthand)

`Tau`, `RateFunc`, and `Boltzmann` helpers are retained but now **return tagged SymPy expressions** rather than `TauParams`/`BoltzmannParams` structs. A tagged expression carries metadata identifying it as a known pattern, allowing the fast-path optimizer (section 13.6) to skip JIT compilation.

```python
Boltzmann(v_half=-40, k=8)
Tau.constant(0.5)
Tau.boltzmann(a=1.0, b=3.0, v_half=-53, k=0.7)
Tau.double_exp_sum(a=24.5, c1=-50, k1=15, c2=-50, k2=-16)
Tau.offset_double_exp(a=400, b=500, c1=-40, k1=15, c2=-20, k2=-20)
Tau.scaled_exp(a=0.15, b=28, v0=-25, k=10.5)
Tau.compound_ab(f1=RateFunc.exp_decay(0.128, -46, 18),
                f2=RateFunc.sigmoid(4, -23, 5))
RateFunc.linear_over_exp(A=0.32, B=54, C=4)
RateFunc.linear_over_expm1(A=0.28, B=27, C=5)
```

Users can mix helpers and raw SymPy freely — the system treats them identically.

---

## 13.5 Backend Code Generation

The SymPy expression tree is compiled to backend-specific code by two custom printer classes, both subclassing SymPy's `CodePrinter`, implemented in `hodgkin_huxley/_codegen.py`.

### EigenPrinter (CPU / Eigen)

Generates a C++ lambda body operating on `Eigen::ArrayXd`. Key differences from standard C codegen: `exp(x)` → `(x).exp()`, symbol `V` → `V_` (the pool's ArrayXd). All arithmetic operators and literals are identical to scalar C.

```cpp
// Generated from: 1 / (1 + exp(-(V + 40) / 8))
auto inf_fn = [](const Eigen::ArrayXd& V_) -> Eigen::ArrayXd {
    return 1.0 / (1.0 + (-(V_ + 40.0) / 8.0).exp());
};
```

When inlined by the compiler the full Eigen expression template chain is visible. SIMD vectorization is identical to handwritten Eigen.

**EigenPrinter mapping:**

| SymPy node | Eigen C++ |
|------------|-----------|
| `exp(x)` | `(x).exp()` |
| `log(x)` | `(x).log()` |
| `tanh(x)` | `(x).tanh()` |
| `Abs(x)` | `(x).abs()` |
| `x**n` (integer n) | `x.pow(n)` |
| symbol `V` | `V_` |
| symbol `Ca` | `Ca_[substance_idx]` |
| `+`, `-`, `*`, `/`, literals | identical |

### CUDAPrinter (GPU / CUDA device)

Generates a scalar `__device__` function. Nearly identical to SymPy's standard `ccode()` with a `__device__` qualifier. No Eigen — GPU parallelism is across threads, not SIMD lanes.

```cuda
__device__ double inf_fn(double V) {
    return 1.0 / (1.0 + exp(-(V + 40.0) / 8.0));
}
```

---

## 13.6 Optimization: Pattern-Matching Fast Path

Before JIT-compiling any expression, the system attempts to match it against a catalog of known patterns using SymPy's `match()`:

```python
from sympy import Wild, exp
v_half_w, k_w = Wild("v_half"), Wild("k")
boltzmann_pattern = 1 / (1 + exp(-(hh.V - v_half_w) / k_w))

result = user_expr.match(boltzmann_pattern)
if result:
    return FastPathBoltzmann(v_half=float(result[v_half_w]), k=float(result[k_w]))
```

The catalog covers all 6 `TauParams` forms and all 4 `RateFuncParams` forms. A match routes to the pre-compiled C++ implementation with the extracted parameters — zero JIT cost. Expressions returned by `Tau.*` / `RateFunc.*` / `Boltzmann()` are tagged, short-circuiting the match step entirely.

---

## 13.7 JIT Compilation Model

For expressions that don't match any known pattern:

1. **Hash** the SymPy expression tree → look up disk cache (`~/.cache/hodgkin_huxley/`)
2. **Cache hit**: load compiled shared library via `ctypes`, return immediately
3. **Cache miss**:
   - Run `EigenPrinter` → write `.cpp` to temp dir
   - Compile: `g++ -O3 -march=native -fPIC -shared -I<eigen_path> -o <hash>.so <hash>.cpp`
   - Load via `ctypes`, register function pointer in pool; save to disk cache
4. **CUDA path**: same sequence but emit `.cu`, compile with `nvcc`

Compilation time: ~0.5–2 s per unique expression; occurs only once across all sessions. If compilation fails, raise `HHEquationError` with the generated source and compiler stderr for debugging.

---

## 13.8 Implementation Checklist

### Expression Infrastructure
- [ ] Define pre-defined symbols (`hh.V`, `hh.Ca`, `hh.V_pre`, `hh.V_post`, `hh.S`, `hh.x_pre`, `hh.x_post`, `hh.w`, `hh.I_source`, `hh.gate(name)`) in `__init__.py`
- [ ] Implement `_codegen.py`: `EigenPrinter` and `CUDAPrinter` subclassing `sympy.printing.CodePrinter`
- [ ] Implement pattern-matching catalog for all 6 tau forms and 4 rate function forms
- [ ] Implement JIT compilation pipeline with disk cache (`~/.cache/hodgkin_huxley/`)

### Spec API Changes
- [ ] Update `GateSpec`: replace `BoltzmannParams inf`, `TauParams tau`, `RateFuncParams alpha/beta` with SymPy expression fields; infer update form and dependency from which expressions and symbols are present
- [ ] Update `ChannelSpec`: replace `ahp_k1` with SymPy gating expression field
- [ ] Update `KineticSynapseSpec`: replace form enums with `dS_dt` and `current` SymPy expression fields
- [ ] Update `IntracellularSpec` (task14): ODE and modulation fields accept SymPy expressions
- [ ] Update `PlasticitySpec` (task15): trace ODEs and on-spike update rules as SymPy expressions

### Preset Helpers
- [ ] Update `Boltzmann()` to return tagged SymPy expression (not `BoltzmannParams`)
- [ ] Update all `Tau.*` factory functions to return tagged SymPy expressions
- [ ] Update all `RateFunc.*` factory functions to return tagged SymPy expressions
- [ ] Update all five `NeuronModelSpec` presets to use SymPy expressions internally

### Pool Integration
- [ ] `ComposablePool::step()`: accept compiled Eigen lambda; call via inlined template (not raw function pointer)
- [ ] `Network::update_synapses_grouped()`: accept compiled lambda for kinetic synapse ODE
- [ ] Wire JIT compilation to fire on first `simulate()` call when custom expressions are present

### Tests
- [ ] Verify `EigenPrinter` output for all 6 tau forms matches pre-compiled numerical output
- [ ] Verify `CUDAPrinter` output for same forms
- [ ] Verify pattern-matching correctly identifies all standard forms (no false JIT triggered)
- [ ] Verify JIT-compiled custom expression produces correct numerical output vs reference
- [ ] Test disk cache: second call with same expression skips compilation
- [ ] Verify all five presets produce identical output before and after migration to SymPy

---

## 13.9 Equation Type API Cleanup

This section finalises the user-facing API changes that become possible once SymPy expressions are in place. It is the second part of the API streamlining work begun in task12.

### Types Removed from User API

The following types are moved to `hodgkin_huxley.legacy` and emit `DeprecationWarning` on import. They may persist internally as the compiled representation, but the user never writes them — the pattern-matching fast path (section 13.6) translates SymPy expressions into them transparently.

| Retired Type | Replaced By |
|---|---|
| `BoltzmannParams` | `Boltzmann(v_half, k)` helper or raw SymPy |
| `TauParams` + `TauForm` enum | `Tau.*` helper or raw SymPy |
| `RateFuncParams` + `RateFuncForm` enum | `RateFunc.*` helper or raw SymPy |
| `GateUpdateForm` enum | Inferred from which expression fields are present |
| `GateDependency` enum | Inferred from which symbols appear in the expression |
| `KineticUpdateForm` enum | Replaced by `dS_dt` SymPy field |
| `KineticCurrentForm` enum | Replaced by `current` SymPy field |

### Before and After: GateSpec

```python
# Before (opaque, positional, requires 7 type names)
gate = GateSpec()
gate.update_form = GateUpdateForm.INF_TAU
gate.dependency = GateDependency.VOLTAGE
gate.inf = BoltzmannParams(); gate.inf.v_half = -40.0; gate.inf.k = 8.0
gate.tau = TauParams(); gate.tau.form = TauForm.DOUBLE_EXP_SUM
gate.tau.set_param(0, 24.5); gate.tau.set_param(1, -50.0)  # ... 3 more calls

# After
V = hh.V
gate = hh.GateSpec(
    name="m",
    inf = 1 / (1 + exp(-(V + 40) / 8)),
    tau = Tau.double_exp_sum(a=24.5, c1=-50, k1=15, c2=-50, k2=-16)
)
# update_form (INF_TAU) and dependency (VOLTAGE) inferred automatically
```

### Before and After: KineticSynapseSpec

```python
# Before (12 individual field assignments across 3 enum-gated sub-types)
spec = KineticSynapseSpec()
spec.update_form = KineticUpdateForm.TANH_GATE
spec.tanh_amp = 0.5; spec.tanh_vh = -20.0; spec.tanh_k = 16.0; spec.tau_decay = 13.0
spec.current_form = KineticCurrentForm.LINEAR
spec.g = 0.1; spec.E_syn = -80.0; spec.power = 1

# After
V_pre, S = hh.V_pre, hh.S
rate_open = 0.5 * (1 + tanh((V_pre + 20) / 16))
spec = hh.KineticSynapseSpec(
    dS_dt   = rate_open * (1 - S) - S / 13.0,
    current = hh.g * S * (hh.V_post - hh.E_syn),
    g=0.1, E_syn=-80.0
)
```

### Final Export Structure

**`hodgkin_huxley` top-level (after task12 + task13):**
`RegionalNetwork`, `NeuronModelSpec`, `SynapseSpec`, `WeightDistribution`, `ConnectivityPattern`, `GateSpec`, `ChannelSpec`, `KineticSynapseSpec`, `IntracellularSpec`, `PlasticitySpec`, `RecordingConfig`, `IzhikevichType`, `IntegrationMethod`, `DBSStimulator`, `PulseStimulator`, `NoiseInjector`, `Boltzmann`, `Tau`, `RateFunc`, `analyze_beta_power`, `V`, `Ca`, `V_pre`, `V_post`, `S`, `x_pre`, `x_post`, `w`, `gate`

**`hodgkin_huxley.legacy` (after task12 + task13):**
`HHNeuron`, `IzhikevichNeuron`, `Network`, `HHParameters`, `HHState`, `IzhikevichParameters`, `IzhikevichState`, `NetworkNeuronType`, `BoltzmannParams`, `TauParams`, `TauForm`, `RateFuncParams`, `RateFuncForm`, `GateUpdateForm`, `GateDependency`, `KineticUpdateForm`, `KineticCurrentForm`

### Checklist for 13.9
- [ ] Move `BoltzmannParams`, `TauParams`, `TauForm`, `RateFuncParams`, `RateFuncForm`, `GateUpdateForm`, `GateDependency`, `KineticUpdateForm`, `KineticCurrentForm` to `hodgkin_huxley.legacy` with `DeprecationWarning`
- [ ] Update `GateSpec` Python-side constructor to accept SymPy expression fields and auto-infer `update_form` / `dependency`
- [ ] Update `KineticSynapseSpec` Python-side constructor to accept `dS_dt` and `current` expression fields
- [ ] Update `__init__.py` to final export structure (symbols exported at top level)
- [ ] Update all `examples/` and `benchmarks/` to use SymPy equation syntax
- [ ] Test that importing legacy equation types raises `DeprecationWarning`
- [ ] Test canonical model-building workflow end-to-end with new equation API
