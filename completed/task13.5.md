# Task 13.5: Implementation Record — Equation System & Unified Synapse Architecture

## Status: Completed

This document records what was actually built across tasks 12–13. It serves as an accurate reference alongside `task13.md`, which was the original plan. The implementation diverged from that plan in several significant ways, all documented below.

---

## Overview

Three interlocking systems were built:

1. **Unified Synapse Architecture** — A single `SynapseSpec` struct replacing two disconnected synapse subsystems, with two-variable state, branch-free SoA hot loops, and a non-virtual view class.
2. **Bytecode VM** — A stack-based virtual machine for evaluating arbitrary SymPy expressions at runtime, used as the `CUSTOM_EXPR` path for synapses and gate kinetics that fall outside pattern-matched forms.
3. **SymPy Expression System** — Full equation language over pre-defined symbols, with structural pattern matching to fast C++ paths, JIT compilation for gates, and builder classes (`NeuronModel`, `SynapseModel`) that accept SymPy expressions directly.

These three systems are tightly coupled: the SymPy compiler emits either a pattern-matched C++ path, a JIT-compiled `.so`, or a `VmExpr` bytecode blob depending on the expression. The VM is what allows arbitrary synapse kinetics without a full JIT pipeline for every use case.

---

## Part 1: Unified Synapse Architecture

### Background

Before this work the library had two entirely separate synapse systems:

- **Legacy spike-driven** (`regional_network.hpp`): Three subclasses (`ExponentialSynapse`, `AlphaSynapse`, `DoubleExponentialSynapse`) encoding update logic in virtual methods. A `SynapseSpec` struct in `regional_network.hpp` that held whichever subset of parameters each type needed.
- **Kinetic** (`synapse_spec.hpp`): `KineticSynapseSpec` with one state variable `S`, three `UpdateForm` values (`TANH_GATE`, `BOLTZMANN_GATE`, `ALPHA_BETA`), and a `CUSTOM_EXPR` VM path. No spike-jump mechanism.

Neither system covered the other's function space. Legacy forms had no voltage-gated path; kinetic forms had no spike-driven path and no auxiliary variable for alpha-function kinetics.

### What Was Built

#### `SynapseSpec` (unified struct — `model/synapse_spec.hpp`)

A single flat struct with an `UpdateForm` discriminant covering all seven forms:

```cpp
enum class UpdateForm {
    EXP_DECAY,       // S += delta_S on spike; S *= exp(-dt/tau_S)
    ALPHA_FUNC,      // dS/dt=(A-S)/tau, dA/dt=-A/tau; A += delta_A on spike (Euler)
    DOUBLE_EXP,      // S*=exp(-dt/tau_S), A*=exp(-dt/tau_A); g=norm*(S-A)
    TANH_GATE,       // voltage-gated: tanh open-rate with exact exp integration
    BOLTZMANN_GATE,  // voltage-gated: Boltzmann s_inf + TauParams
    ALPHA_BETA,      // voltage-gated: alpha/beta rate functions
    CUSTOM_EXPR,     // VM bytecode: dS_dt_vm (+ optional dA_dt_vm for 2-var ODEs)
};
```

All fields are present in the struct; unused ones are ignored by the update loop. Two state variables (`S` primary, `A` auxiliary) with per-synapse on-spike increments (`delta_S`, `delta_A`):

- `EXP_DECAY`: uses only `S`, `tau_S`, `delta_S`
- `ALPHA_FUNC`: uses `S` and `A` as a coupled 2-var ODE, `delta_A` spike jump
- `DOUBLE_EXP`: uses `S` and `A` as independent decays, `norm_factor` for peak normalization
- `TANH_GATE` / `BOLTZMANN_GATE` / `ALPHA_BETA`: voltage-gated, `S` only (existing kinetic forms, unchanged)
- `CUSTOM_EXPR`: `dS_dt_vm` always present; `dA_dt_vm` optional for 2-variable custom forms

The `CurrentForm` discriminant (`LINEAR` / `MG_BLOCK` / `CUSTOM_EXPR`) remains from the kinetic system.

Static factory methods on `SynapseSpec` cover all standard forms with validated parameters:
- `exponential(tau_S, g, E_syn)` — validates `tau_S > 0`
- `alpha_function(tau, g, E_syn)` — normalized so peak conductance = weight
- `double_exponential(tau_rise, tau_decay, g, E_syn)` — validates `tau_rise < tau_decay`, computes `norm_factor`
- `gaba_kinetic()` — Kumaravelu 2016 tanh model
- `nmda_kinetic()` — NMDA with Mg²⁺ block
- `gaba_b()` — slow GABA-B with cooperative gating (power=4)
- `ampa()`, `nmda()`, `gaba_a()` — receptor presets via `double_exponential`

Factory implementations live in `src/cpp/src/synapse_spec.cpp` (new file, added to `CMakeLists.txt`).

#### `SynArrays` (unified SoA — `network.hpp`)

The old type-specific field groups (`exp_tau`, `alpha_x`, `dexp_g_rise`, etc.) and the `SynType` enum were removed. All synapses now share one unified state layout:

```cpp
struct SynArrays {
    // Topology
    std::vector<size_t> pre, post;
    std::vector<double> weight, E_syn, g;
    std::vector<double> V_pre_prev;
    std::vector<double> delay;
    std::vector<std::vector<bool>> spike_buf;
    std::vector<size_t> buf_head;
    std::vector<bool>   delay_init;

    // Unified state
    std::vector<double> S;          // primary gating variable
    std::vector<double> A;          // auxiliary (0 if unused)
    std::vector<double> delta_S;    // on-spike increment for S
    std::vector<double> delta_A;    // on-spike increment for A
    std::vector<double> tau_S;
    std::vector<double> tau_A;
    std::vector<double> inv_tau_A;  // cached 1/tau_A
    std::vector<double> norm;       // DOUBLE_EXP peak normalization
    std::vector<double> decay_S;    // cached exp(-dt/tau_S)
    std::vector<double> decay_A;    // cached exp(-dt/tau_A)

    std::vector<size_t> spec_idx;   // index into Network::synapse_specs_
};
```

`synapse_specs_` (renamed from `kinetic_specs_`) stores one `SynapseSpec` per unique named spec; all synapses referencing the same spec share one copy via `spec_idx`.

#### `SynapseGroups` (branch-free dispatch — `network.hpp`)

Four index lists replace the old type-specific grouped arrays:

```cpp
struct SynapseGroups {
    std::vector<size_t> exp_decay;
    std::vector<size_t> alpha_func;
    std::vector<size_t> double_exp;
    std::vector<size_t> voltage_gated;  // TANH_GATE, BOLTZMANN_GATE, ALPHA_BETA, CUSTOM_EXPR
};
```

`build_synapse_groups()` dispatches on `UpdateForm`. Each group has its own tight inner loop in `update_synapses_grouped()` with no per-iteration branching:

- **EXP_DECAY**: spike check → `S += delta_S` → `S *= decay_S`
- **ALPHA_FUNC**: spike check → `A += delta_A` → Euler step `dS=(A-S)/tau`, `dA=-A/tau`
- **DOUBLE_EXP**: spike check → `S += delta_S`, `A += delta_A` → `S *= decay_S`, `A *= decay_A` → `g = norm*(S-A)`
- **voltage_gated**: existing TANH/BOLTZMANN/ALPHA_BETA/CUSTOM_EXPR loop (extended with 2-var CUSTOM_EXPR branch)

#### `SynapseBase` (non-virtual view — `synapse_base.hpp` / `synapse_base.cpp`)

All type-specific subclasses (`ExponentialSynapse`, `AlphaSynapse`, `DoubleExponentialSynapse`) were deleted. `SynapseBase` is now a lightweight non-virtual view:

```cpp
class SynapseBase {
public:
    SynapseBase() = default;
    SynapseBase(size_t idx, const Network* net) : idx_(idx), net_(net) {}

    [[nodiscard]] double conductance()        const;
    [[nodiscard]] double reversal_potential() const;
    [[nodiscard]] double weight()             const;
    [[nodiscard]] size_t pre_idx()            const;
    [[nodiscard]] size_t post_idx()           const;
    [[nodiscard]] double delay()              const;
    [[nodiscard]] std::string type_name()     const;
    [[nodiscard]] const SynapseSpec& spec()   const;

private:
    size_t         idx_ = 0;
    const Network* net_ = nullptr;
};
```

All methods read from `net_->syn_arrays()`. The `Network` exposes `syn_arrays()` and `synapse_spec()` as public `const` accessors. `synapses_` is now `std::vector<SynapseBase>` (no heap allocation per synapse, no virtual dispatch).

#### `Network::add_synapse()` (unified adder — `network.cpp`)

A single `add_synapse(pre, post, weight, spec, delay)` replaces all type-specific adders (`add_alpha_synapse`, `add_double_exp_synapse`, `add_kinetic_synapse`). Those become deprecated wrappers calling `add_synapse()` with the appropriate preset. The new adder populates all unified SoA arrays from `spec` fields in one place.

`sort_synapses_by_pre()` was updated to permute all new SoA arrays.

#### `RegionalNetwork` cleanup (`regional_network.hpp` / `regional_network.cpp`)

The old `SynapseSpec` struct (defined in `regional_network.hpp`, separate from `model/synapse_spec.hpp`) was removed. All routing through `connect()` and `add_connection_from_spec()` now uses the unified `SynapseSpec` from `model/synapse_spec.hpp`.

#### Python bindings (`bindings.cpp`)

- `KineticSynapseSpec` binding renamed to `SynapseSpec`
- `KineticUpdateForm` → `SynapseUpdateForm`; `KineticCurrentForm` → `SynapseCurrentForm`
- New `UpdateForm` values bound: `EXP_DECAY`, `ALPHA_FUNC`, `DOUBLE_EXP`
- New SoA fields bound: `tau_S`, `tau_A`, `delta_S`, `delta_A`, `A_init`, `norm_factor`, `dA_dt_vm`
- `PUSH_A` added to `VmOp` binding
- `SynapseBase` updated to reflect the simplified view class (no subclasses)

#### Backward Compatibility

| Old API | New behaviour |
|---|---|
| `SynapseSpec.exponential(E_syn=x, tau=y)` | `SynapseSpec.exponential(tau_S, g, E_syn)` positional |
| `SynapseSpec.alpha(E_syn, tau)` renamed | `SynapseSpec.alpha_function(tau, g, E_syn)` |
| `KineticSynapseSpec` | Python alias for `SynapseSpec` |
| `KineticUpdateForm` | Python alias for `SynapseUpdateForm` |
| `ExponentialSynapse`, `AlphaSynapse`, `DoubleExponentialSynapse` | Removed from C++; `legacy.py` provides Python shims delegating to `SynapseSpec` factories |

---

## Part 2: Bytecode VM

### Purpose

The VM provides a zero-overhead escape hatch for expressions that fall outside pattern-matched standard forms. It enables arbitrary SymPy expressions for synapse kinetics without spawning a compiler process per expression. For gates, the JIT pipeline handles most cases; for synapses, the VM handles `CUSTOM_EXPR` forms (including novel 2-variable ODEs).

### `VmExpr` / `VmOp` (`model/gate_spec.hpp`)

A flat bytecode representation: a list of `VmInstruction` (opcode + integer operand) and a `constants` vector for `PUSH_CONST`:

```cpp
enum class VmOp {
    PUSH_DEP, PUSH_CONST, PUSH_S, PUSH_A,
    ADD, MUL, NEG, RCP,
    POW_INT, POW_HALF, POW_GEN,
    EXP, LOG, TANH, SIN, COS, SQRT, ABS,
};
```

`PUSH_A` was added specifically for 2-variable synapse ODEs — it was not in the original `gate_spec.hpp` opcode set.

### `vm_eval_scalar_3arg` (`model/kinetics.hpp`)

Added alongside the existing `vm_eval_scalar_2arg`. Handles programs with `dep`, `S`, and `A` arguments — used in the `CUSTOM_EXPR` voltage-gated synapse loop when `dA_dt_vm` is non-empty:

```cpp
inline double vm_eval_scalar_3arg(const VmExpr& prog, double dep, double S, double A);
```

The 2-var CUSTOM_EXPR branch in `update_synapses_grouped()`:
```cpp
if (!spec.dA_dt_vm.empty()) {
    double dS = vm_eval_scalar_3arg(spec.dS_dt_vm, Vpre, S, A_k);
    double dA = vm_eval_scalar_3arg(spec.dA_dt_vm, Vpre, S, A_k);
    S  = std::max(0.0, std::min(1.0, S  + dt * dS));
    A_k = A_k + dt * dA;
}
```

### `compile_to_vm_bytecode()` (`_codegen.py`)

Recursive SymPy AST → postfix bytecode compiler:

```python
def compile_to_vm_bytecode(expr, dep_sym, extra_syms=None) -> VmExpr
```

- `dep_sym`: the primary dependent symbol (maps to `PUSH_DEP`)
- `extra_syms`: dict mapping SymPy symbols to specific `VmOp` values (e.g., `{S: VmOp.PUSH_S, A: VmOp.PUSH_A}`)

The internal `_emit_vm(expr, constants, instructions)` does a depth-first walk, emitting postfix instructions. Handles all standard SymPy node types: `Add`, `Mul`, `Pow`, `exp`, `log`, `tanh`, `sin`, `cos`, `sqrt`, `Abs`, `Integer`, `Float`, `Symbol`. Floating-point constants are deduplicated into the `constants` list by value (within `1e-12` tolerance).

### When the VM Is Used

| Case | Path |
|---|---|
| Gate kinetics — standard form recognized | Pattern-matched C++ path (no VM) |
| Gate kinetics — non-standard | VM bytecode via `compile_to_vm_bytecode()` |
| Synapse `CUSTOM_EXPR` dS_dt | `compile_to_vm_bytecode(dS_dt, V_pre, {S: PUSH_S, A: PUSH_A})` |
| Synapse `CUSTOM_EXPR` dA_dt | `compile_to_vm_bytecode(dA_dt, V_pre, {S: PUSH_S, A: PUSH_A})` |
| Synapse `CUSTOM_EXPR` current | `compile_to_vm_bytecode(current, V_post, {S: PUSH_S, A: PUSH_A})` |

---

## Part 3: SymPy Expression System

### Pre-Defined Symbols (`_codegen.py`)

All symbols available directly from `hodgkin_huxley` without importing SymPy:

```python
V        # membrane potential
Ca       # calcium concentration
V_pre    # pre-synaptic potential
V_post   # post-synaptic potential
S        # synapse gating variable (primary)
A        # synapse auxiliary variable (2-var ODEs) — added for task13.5
x        # generic gate variable
x_pre    # pre-synaptic STDP trace
x_post   # post-synaptic STDP trace
w        # synaptic weight
I_source # current source symbol
g_sym    # conductance symbol
E_syn_sym # reversal potential symbol
```

### `EigenPrinter` (`_codegen.py`)

Converts SymPy expressions to C++ strings. Two modes:

**Vectorized** (for Eigen `ArrayXd` pools):
- Symbols → `Symbol_` suffixed names (e.g., `V` → `V_`)
- `exp(x)` → `(x).exp()`, `tanh(x)` → `(x).tanh()`, `log(x)` → `(x).log()`
- `sqrt(x)` → `(x).sqrt()`, `abs(x)` → `(x).abs()`
- `Pow(x, n)` where `n` is integer → `(x).pow(n)`
- Produces an inlineable lambda over Eigen arrays; SIMD preserved

**Scalar** (for `__device__` functions and scalar C contexts):
- Standard `std::exp`, `std::tanh`, etc.
- Reserved for task17 CUDA codegen (`__device__` functions)

### `TaggedExpr` (`_codegen.py`)

A wrapper around a SymPy expression that carries pre-matched C++ parameters, short-circuiting the pattern matching step:

```python
@dataclass
class TaggedExpr:
    expr: sympy.Expr
    tag: str          # e.g., "boltzmann", "tau_constant"
    params: dict      # matched C++ parameter values
```

The `Boltzmann`, `Tau.*`, and `RateFunc.*` helper functions return `TaggedExpr` instances. When `to_spec()` encounters a `TaggedExpr`, it reads `params` directly without re-running pattern matching.

### Pattern Matching Catalog (`_codegen.py`)

Structural coefficient extraction via `sympy.Poly` (not `Wild`-based matching — handles float constants that SymPy eagerly evaluates). Three catalogs:

**Boltzmann** (1 form):
```
1 / (1 + exp((v_half - V) / k))
```
Extracts `v_half`, `k` via `Poly` coefficient comparison.

**Tau** (6 forms):
| Tag | Expression |
|-----|-----------|
| `CONSTANT` | scalar numeric |
| `BOLTZMANN` | `tau0 + tau1 / (1 + exp((v_half - V) / k))` |
| `DOUBLE_EXP_SUM` | `a*exp(b*V) + c*exp(d*V)` |
| `SCALED_EXP` | `a*exp(b*V)` |
| `COMPOUND_AB` | `1 / (alpha(V) + beta(V))` where alpha, beta are rate funcs |
| `OFFSET_DOUBLE_EXP` | `tau0 + a*exp(b*V) + c*exp(d*V)` |

**RateFunc** (4 forms):
| Tag | Expression |
|-----|-----------|
| `LINEAR_OVER_EXP` | `(a + b*V) / (exp(c + d*V) - 1)` (HH-style) |
| `EXP_DECAY` | `a*exp(b*V)` |
| `LINEAR_OVER_EXPM1` | variant of HH denominator |
| `SIGMOID` | `a / (1 + exp(b*V + c))` |

When a match is found, the corresponding C++ struct (`BoltzmannParams`, `TauParams`, `RateFuncParams`) is populated directly with no runtime VM evaluation.

### Expression Fallback Path (`_codegen.py`)

All expressions that fail pattern matching — for both gate kinetics and synapse ODEs — compile to VM bytecode via `compile_to_vm_bytecode()`. No `g++` subprocess or Eigen headers are required at runtime; the VM interpreter runs entirely in the pre-compiled C++ core.

**Post-task13.5 change**: A `JITCache` / `jit_compile` pipeline (g++ subprocess, Eigen header discovery, `.so` loading via `ctypes`) was originally present but never wired into the gate fallback path — gates already used the VM exclusively. The dead code was removed to ensure cross-platform compatibility (no dependency on a system compiler or Eigen headers at runtime). `EigenPrinter` is retained for task17 CUDA codegen.

---

### Supported SymPy Expression Forms by Entry Point

#### VM-supported node types (applies to all VM paths below)

Any expression composed entirely of:

| SymPy construct | VM opcode(s) |
|---|---|
| Numeric literal (`Integer`, `Float`, `Rational`, any `.is_number`) | `PUSH_CONST` |
| Primary dependent symbol (`V`, `Ca`, `S`, `A`, `X`, ...) | `PUSH_DEP` |
| Extra symbols passed via `extra_syms` (`S`, `A`, substance names) | `PUSH_S`, `PUSH_A`, `PUSH_X(n)` |
| `a + b + ...` | `PUSH` each term, repeated `ADD` |
| `a * b * ...` | `PUSH` each factor, repeated `MUL` |
| `-x` (negation, i.e. `Mul(-1, x)`) | `NEG` |
| `1 / x` (i.e. `Pow(x, -1)`) | `RCP` |
| `x ** n` for integer `n` | `POW_INT(n)` |
| `sqrt(x)` (i.e. `Pow(x, Rational(1,2))`) | `POW_HALF` |
| `x ** y` for general float `y` | `POW_GEN` |
| `exp(x)` | `EXP` |
| `log(x)` | `LOG` |
| `tanh(x)` | `TANH` |
| `sin(x)` | `SIN` |
| `cos(x)` | `COS` |
| `Abs(x)` | `ABS` |

Unsupported nodes (`erf`, `asin`, `atan`, etc.) raise `HHEquationError` at build time.

---

#### `NeuronModel.add_gate()` / `GateSpec()`

All four arguments accept `Boltzmann` / `Tau.*` / `RateFunc.*` shorthand (return `TaggedExpr`, zero pattern-matching cost), raw SymPy expressions (pattern-matched first, VM fallback), or pre-built C++ structs.

**`inf` — steady-state activation** — primary symbol `V` (or `Ca` if `dependency="calcium"`)

| Pattern | Route |
|---|---|
| `1 / (1 + exp((v_half - V) / k))` (Boltzmann) | `BoltzmannParams` — C++ fast path |
| Anything else | VM (primary: `V` or `Ca`) |

**`tau` — time constant** — primary symbol `V`

| Pattern | Route |
|---|---|
| Scalar numeric | `TauParams(CONSTANT, tau0)` |
| `tau0 + tau1 / (1 + exp((v_half - V) / k))` | `TauParams(BOLTZMANN, ...)` |
| `a*exp(b*V) + c*exp(d*V)` | `TauParams(DOUBLE_EXP_SUM, ...)` |
| `tau0 + a*exp(b*V) + c*exp(d*V)` | `TauParams(OFFSET_DOUBLE_EXP, ...)` |
| `a*exp(b*V)` (single exponential) | `TauParams(SCALED_EXP, ...)` |
| `1 / (alpha(V) + beta(V))` where each is a recognized rate func | `TauParams(COMPOUND_AB, ...)` |
| Anything else | VM (primary: `V`) |

**`alpha` / `beta` — rate functions** — primary symbol `V`

| Pattern | Route |
|---|---|
| `(a + b*V) / (exp(c + d*V) - 1)` (HH-style) | `RateFuncParams(LINEAR_OVER_EXP, ...)` |
| `a*exp(b*V)` | `RateFuncParams(EXP_DECAY, ...)` |
| `(a + b*V) / (1 - exp(c + d*V))` | `RateFuncParams(LINEAR_OVER_EXPM1, ...)` |
| `a / (1 + exp(b*V + c))` | `RateFuncParams(SIGMOID, ...)` |
| Anything else | VM (primary: `V`) |

**`expr` — full custom ODE** `dx/dt = f(V, x)` — always VM

| Symbol | Meaning |
|---|---|
| `V` (or `Ca` for calcium-dependent) | `PUSH_DEP` |
| `x` | `PUSH_S` (the gate variable itself) |

---

#### `SynapseModel()` general constructor

`dS_dt`, `dA_dt`, `current` are pattern-matched first; fall through to VM if unrecognized.

**`dS_dt` / `dA_dt` — ODE pattern matching** (priority order)

| Pattern | Route |
|---|---|
| `dS_dt = c*S` (c < 0, no `dA_dt`) | `EXP_DECAY`, `tau_S = -1/c` |
| `dS_dt = (A-S)*k`, `dA_dt = -A*k` (same `k`) | `ALPHA_FUNC` |
| `dS_dt = -S*k1`, `dA_dt = -A*k2` (k1 ≠ k2, both > 0) | `DOUBLE_EXP` |
| Anything else | `CUSTOM_EXPR` VM |

**`dS_dt` / `dA_dt` when VM** — available symbols:

| Symbol | Meaning |
|---|---|
| `V_pre` | `PUSH_DEP` |
| `S` | `PUSH_S` |
| `A` | `PUSH_A` |

**`current`** — available symbols:

| Symbol | Meaning |
|---|---|
| `V_post` | `PUSH_DEP` |
| `S` | `PUSH_S` |
| `A` | `PUSH_A` |

---

#### `IntracellularDynamics(ode=..., nernst=...)`

**`ode` — dX/dt** — pattern-matched first, VM fallback

| Pattern | Route |
|---|---|
| `-k * X` (k > 0, X only) | `DECAY` — C++ fast path |
| `eps * (-I_source - k * X)` (ε, k > 0) | `DRIVEN_DECAY` — C++ fast path |
| Same + standard Nernst form | `DRIVEN_DECAY_NERNST` — C++ fast path |
| Anything else | `CUSTOM_EXPR` VM |

**`ode` when VM** — available symbols:

| Symbol | Meaning |
|---|---|
| `I_source` | `PUSH_DEP` (sum of source channel currents) |
| `X` (e.g. `sympy.Symbol("Ca")`) | `PUSH_S` |
| Other substance names | `PUSH_X(n)` via `extra_syms` |

**`nernst` — reversal potential as f(X)** — primary symbol `X`

| Pattern | Route |
|---|---|
| `(R*T / (z*F)) * log(X_o / X)` with numeric constants | Standard Nernst params — C++ fast path |
| Anything else | VM (primary: `X`) |

---

#### `Modulation(..., expr=...)`

`expr` is always compiled to VM. Primary symbol is the substance's own `sympy.Symbol` (e.g. `Ca`), mapped to `PUSH_DEP`. No other variables are available — modulation functions are scalar functions of the substance concentration only.

---

#### `compile_gate_product_vm()` (channel conductance gating)

Used internally when `add_channel(gates=...)` receives a SymPy expression instead of a `{gate_name: power}` dict. Supports a restricted subset:

| Construct | Support |
|---|---|
| `gate("m")` | Single gate reference |
| `gate("m") ** n` (integer `n`) | Gate raised to integer power |
| `gate("m") ** 3 * gate("h")` | Product of gate terms |
| `scalar * gate(...)` | Numeric coefficient |
| Sum of above | Sum of gate products |

`V`, `Ca`, and general SymPy functions are **not** available here — only gate symbols, their integer powers, products, sums, and scalar coefficients.

### `NeuronModel` Builder (`_equations/__init__.py`)

Accepts SymPy expressions at every gate definition point:

```python
model = NeuronModel("custom_hh")
model.add_gate("m", inf=1/(1 + exp((V + 40)/8)), tau=Tau.constant(0.5))
model.add_gate("h", inf=1/(1 + exp(-(V + 40)/8)), tau=Tau.boltzmann(-40, 8, 0.1, 5.0))
model.add_channel("Na", g=120, E=50, gates={"m": 3, "h": 1})
spec = model.to_spec()  # returns NeuronModelSpec
```

`to_spec()` workflow per gate:
1. Check if `inf` is `TaggedExpr` → use `params` directly
2. Try pattern matching catalog
3. Fall back to `compile_to_vm_bytecode()` (VM path — cross-platform, no compiler required)

**Divergence from task13.md**: task13 planned `GateSpec(inf=sympy_expr)` direct assignment. The `NeuronModel` builder was implemented instead — it is more ergonomic for multi-gate, multi-channel models and naturally groups gates with their parent channel.

### `SynapseModel` Builder (`_equations/__init__.py`)

Accepts SymPy expressions for `dS_dt`, `dA_dt`, and `current`:

```python
# Named constructors (no pattern matching needed)
SynapseModel.exponential(name="exp", *, tau, g=1.0, E_syn=0.0)
SynapseModel.alpha_function(name="alpha", *, tau, g=1.0, E_syn=0.0)
SynapseModel.double_exponential(name="dexp", *, tau_rise, tau_decay, g=1.0, E_syn=0.0)
SynapseModel.tanh_gate(name, *, amp, v_half, k, tau_decay, g, E_syn)
SynapseModel.boltzmann_gate(name, *, v_half, k, tau, g, E_syn)
SynapseModel.alpha_beta(name, *, alpha, beta, g, E_syn)

# General constructor with pattern matching
SynapseModel("custom", dS_dt=expr, dA_dt=expr2, current=expr3,
             spike_S=0.0, spike_A=0.0, g=0.1, E_syn=0.0,
             S_init=0.0, A_init=0.0)
```

Pattern matching in `to_spec()` (priority order):
1. `dS_dt = c*S` (c<0, no dA_dt) → `EXP_DECAY`, `tau_S = -1/c`
2. `dS_dt = (A-S)*k` + `dA_dt = -A*k` (same k) → `ALPHA_FUNC`, `tau_A = 1/k`
3. `dS_dt = -S*k1` + `dA_dt = -A*k2` (k1 ≠ k2, both negative) → `DOUBLE_EXP`
4. Otherwise → `CUSTOM_EXPR`, compile `dS_dt_vm` (and `dA_dt_vm` if `dA_dt` given)

`KineticSynapseModel` is a deprecated alias for `SynapseModel`.

### Integration with `_network/__init__.py`

Both `connect()` and `add_connection()` accept `SynapseModel` instances and call `.to_spec()` internally before passing to C++:

```python
if isinstance(synapse, SynapseModel):
    synapse = synapse.to_spec()
```

---

## Divergences from `task13.md`

| task13.md plan | What was actually built |
|---|---|
| Full `CUDAPrinter` implementation | Stub only — CUDA codegen deferred to task17 |
| JIT for both gate kinetics and synapse ODEs | VM bytecode exclusively for all custom expressions — no `g++` subprocess, no Eigen headers required at runtime; cross-platform |
| `GateSpec(inf=sympy_expr)` direct field assignment | `NeuronModel` builder with `add_gate()`/`add_channel()` API |
| Task13 only (no synapse overhaul) | Major synapse architecture overhaul added (unified SynapseSpec, SoA redesign, SynapseGroups) — this was not in task13 at all |
| `KineticSynapseSpec` extended with new forms | `KineticSynapseSpec` removed; new `SynapseSpec` covers all 7 forms |
| 13.9 equation-type API cleanup as a sub-step | Merged into the synapse overhaul work; legacy types moved to `hodgkin_huxley.legacy` with deprecation warnings and Python shims |

---

## Files Changed

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/model/gate_spec.hpp` | Added `PUSH_A` opcode; added `VmExpr dA_dt_vm` to synapse path |
| `src/cpp/include/hodgkin_huxley/model/synapse_spec.hpp` | Full rewrite: unified `SynapseSpec` with 7 UpdateForms, 2-var state |
| `src/cpp/include/hodgkin_huxley/model/kinetics.hpp` | Added `vm_eval_scalar_3arg` |
| `src/cpp/include/hodgkin_huxley/synapse_base.hpp` | Replaced virtual class hierarchy with non-virtual view |
| `src/cpp/src/synapse_base.cpp` | New file: implements all `SynapseBase` view methods |
| `src/cpp/src/synapse_spec.cpp` | New file: static factory implementations |
| `src/cpp/include/hodgkin_huxley/network.hpp` | Unified `SynArrays`, new `SynapseGroups`, added `syn_arrays()` / `synapse_spec()` accessors |
| `src/cpp/src/network.cpp` | Four update sites: `update_decay_factors`, `build_synapse_groups`, `update_synapses_grouped`, `add_synapse` |
| `src/cpp/include/hodgkin_huxley/regional_network.hpp` | Removed old `SynapseSpec` struct |
| `src/cpp/src/regional_network.cpp` | Updated routing to unified `SynapseSpec` |
| `src/python/bindings.cpp` | Renamed bindings, added new fields and UpdateForm values, bound `SynapseBase` view |
| `src/cpp/CMakeLists.txt` | Added `synapse_base.cpp`, `synapse_spec.cpp` to library sources |
| `src/hodgkin_huxley/_codegen.py` | `A` symbol, `PUSH_A` VM opcode, `vm_eval_scalar_3arg` call path, `compile_to_vm_bytecode` extra_syms; JIT pipeline (`JITCache`, `jit_compile`, `_find_eigen_include`) removed post-task13.5 — VM used exclusively |
| `src/hodgkin_huxley/_equations/__init__.py` | `SynapseModel` builder, pattern matching, `KineticSynapseModel` alias |
| `src/hodgkin_huxley/_network/__init__.py` | `SynapseModel` → `SynapseSpec` conversion in `connect()` and `add_connection()` |
| `src/hodgkin_huxley/__init__.py` | Export `SynapseModel`, `A`, `SynapseSpec`, `SynapseUpdateForm`, `SynapseCurrentForm` |
| `src/hodgkin_huxley/_core.pyi` | Updated stubs for all changed types |
| `src/hodgkin_huxley/legacy.py` | Python shims for removed C++ synapse subclasses; deprecation wrappers |
