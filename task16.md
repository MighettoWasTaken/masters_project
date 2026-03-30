# Task 16: Better Equation Support

## Priority: 2

## Overview

The current system for defining gate kinetics uses raw parameter arrays (`double params[8]`) and enum codes that carry no semantic meaning at the call site. A user writing `TauParams(TauForm.DOUBLE_EXP_SUM, [24.5, -50, 15, -50, -16, 0, 0, 0])` has no indication of what each positional value represents without consulting documentation. This is a source of bugs, confusion, and maintenance burden.

The solution has two layers:
1. **Named parameter structs** (C++ and Python): replace positional arrays with descriptively named fields per form variant
2. **Builder DSL** (Python): fluent, keyword-argument-based factory functions that read like equations

An optional third layer — symbolic expression support via SymPy — is described as a future extension.

---

## 16.1 Current State (Problem)

```python
# Current: opaque positional array — what does each number mean?
tau = TauParams(TauForm.DOUBLE_EXP_SUM, [24.5, -50, 15, -50, -16, 0, 0, 0])

# Current: alpha/beta also positional
alpha = RateFuncParams(RateFuncForm.LINEAR_OVER_EXP, 0.32, 54.0, 4.0)
```

Existing `TauForm` variants and their parameter meanings are buried in `ion_channels.hpp` inline comments. There is no Python-level documentation of which `params[i]` index corresponds to which mathematical symbol.

---

## 16.2 Named Parameter Structs (C++)

Replace `double params[8]` in `TauParams` with form-specific named fields using a `std::variant` or tagged union.

```cpp
struct TauParams {
    enum class Form {
        CONSTANT,          // tau = c
        BOLTZMANN,         // tau = a + b / (1 + exp((V - v_half) / k))
        DOUBLE_EXP_SUM,    // tau = a / (exp((V-c1)/k1) + exp((V-c2)/k2))
        OFFSET_DOUBLE_EXP, // tau = a + b / (exp((V-c1)/k1) + exp((V-c2)/k2))
        SCALED_EXP,        // tau = a * (b + exp((V - v0) / k))
        COMPOUND_AB        // tau = 1 / (f1(V) + f2(V))
    };

    Form form;

    // Form-specific parameters (named, not positional)
    struct Constant        { double c; };
    struct Boltzmann       { double a, b, v_half, k; };
    struct DoubleExpSum    { double a, c1, k1, c2, k2; };
    struct OffsetDoubleExp { double a, b, c1, k1, c2, k2; };
    struct ScaledExp       { double a, b, v0, k; };
    struct CompoundAB      { RateFuncParams f1, f2; };

    std::variant<Constant, Boltzmann, DoubleExpSum,
                 OffsetDoubleExp, ScaledExp, CompoundAB> data;

    // Named factory functions
    static TauParams constant(double c);
    static TauParams boltzmann(double a, double b, double v_half, double k);
    static TauParams double_exp_sum(double a, double c1, double k1, double c2, double k2);
    static TauParams offset_double_exp(double a, double b, double c1, double k1, double c2, double k2);
    static TauParams scaled_exp(double a, double b, double v0, double k);
    static TauParams compound_ab(RateFuncParams f1, RateFuncParams f2);
};
```

Apply the same pattern to `RateFuncParams`:
```cpp
struct RateFuncParams {
    enum class Form { LINEAR_OVER_EXP, EXP_DECAY, LINEAR_OVER_EXPM1, SIGMOID };
    Form form;

    struct LinearOverExp  { double A, B, C; };
    struct ExpDecay       { double A, B, C; };
    struct LinearOverExpm1{ double A, B, C; };
    struct Sigmoid        { double A, B, C; };

    std::variant<LinearOverExp, ExpDecay, LinearOverExpm1, Sigmoid> data;

    static RateFuncParams linear_over_exp(double A, double B, double C);
    static RateFuncParams exp_decay(double A, double B, double C);
    static RateFuncParams linear_over_expm1(double A, double B, double C);
    static RateFuncParams sigmoid(double A, double B, double C);
};
```

**Backwards compatibility:** The old raw-array constructors are retained with `[[deprecated]]` for one release, then removed.

---

## 16.3 Python Builder DSL

The Python layer exposes the named factory functions as a clean, equation-like DSL under the `Tau` and `RateFunc` namespaces:

```python
# Before (opaque)
tau = TauParams(TauForm.BOLTZMANN, [1.0, 3.0, -53.0, 0.7, 0, 0, 0, 0])

# After (self-documenting)
tau = Tau.boltzmann(a=1.0, b=3.0, v_half=-53.0, k=0.7)
# Reads as: tau = a + b / (1 + exp((V - v_half) / k))
```

Full `Tau` namespace:
```python
from hodgkin_huxley import Tau, RateFunc, Boltzmann

Tau.constant(c=0.5)
Tau.boltzmann(a=1.0, b=3.0, v_half=-53.0, k=0.7)
Tau.double_exp_sum(a=24.5, c1=-50.0, k1=15.0, c2=-50.0, k2=-16.0)
Tau.offset_double_exp(a=400.0, b=500.0, c1=-40.0, k1=15.0, c2=-20.0, k2=-20.0)
Tau.scaled_exp(a=0.15, b=28.0, v0=-25.0, k=10.5)
Tau.compound_ab(
    f1=RateFunc.exp_decay(A=0.128, B=-46.0, C=18.0),
    f2=RateFunc.sigmoid(A=4.0, B=-23.0, C=5.0)
)
```

`BoltzmannParams` also gets keyword-argument support:
```python
Boltzmann(v_half=-41.0, k=-4.0)   # explicit kwargs (was positional)
```

---

## 16.4 Documentation Strings

All structs and factory functions receive docstrings that include:
- The mathematical form they represent
- Symbol definitions
- A usage example

Example:
```python
Tau.double_exp_sum(a, c1, k1, c2, k2)
"""
Time constant form: tau(V) = a / (exp((V - c1) / k1) + exp((V - c2) / k2))

Parameters
----------
a   : numerator scale
c1  : first exponential centre voltage (mV)
k1  : first exponential slope (mV)
c2  : second exponential centre voltage (mV)
k2  : second exponential slope (mV)

Example: STN tau_h — a=24.5, c1=-50, k1=15, c2=-50, k2=-16
"""
```

---

## 16.5 Optional: SymPy Integration

As a longer-term extension (separate sub-task), allow users to define gate kinetics as SymPy expressions that are compiled to C++ lambdas or LLVM IR via `sympy.lambdify`:

```python
from sympy import symbols, exp
V = symbols("V")
tau_expr = 24.5 / (exp((V + 50) / 15) + exp(-(V + 50) / 16))
gate = GateSpec(name="H", inf=Boltzmann(v_half=-45.5, k=-6.4),
                tau=Tau.from_sympy(tau_expr, V))
```

This is out of scope for the initial implementation but should be kept in mind when designing `TauParams` — the `std::variant` structure should have a reserved `CUSTOM_LAMBDA` variant for future use.

---

## 16.6 Implementation Checklist

### C++ Core
- [ ] Refactor `TauParams` to use named nested structs and `std::variant` (or tagged union for C++14 compat)
- [ ] Refactor `RateFuncParams` similarly
- [ ] Add static factory methods (`TauParams::constant()`, `TauParams::boltzmann()`, etc.)
- [ ] Mark old array-based constructors `[[deprecated]]`
- [ ] Update all five preset factories (`thalamic()`, `stn()`, `gpe()`, `gpi()`, `striatum()`) to use named constructors
- [ ] Ensure `compute_tau_scalar()` and `ComposablePool` vectorised tau evaluation handle the new `variant` layout

### Python Bindings
- [ ] Expose `Tau` namespace with all factory functions + keyword arguments
- [ ] Expose `RateFunc` namespace similarly
- [ ] Update `BoltzmannParams` to support keyword arguments
- [ ] Add docstrings with equations and examples to all factories
- [ ] Mark old positional constructors deprecated in Python with `DeprecationWarning`

### Tests
- [ ] Verify each named factory produces identical numerical output to the old positional form
- [ ] Test all five presets still produce correct gate values after refactor
- [ ] Smoke-test that old positional constructors still work (with deprecation warning) until removed
