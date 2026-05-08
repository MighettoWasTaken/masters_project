# Task 17.4: CUDAPrinter — SymPy → `__device__` Codegen

**Role:** Codegen engineer  
**Status:** Not started  
**Depends on:** 17.1 (parallel track — no runtime dependency)  
**Unlocks:** 17.8 (CudaComposablePool CUSTOM_EXPR uses generated device functions)

---

## What to implement

The stub `CUDAPrinter` class already exists in `src/hodgkin_huxley/_codegen.py`. Flesh it out to produce valid CUDA `__device__` scalar C code from a SymPy expression.

### `src/hodgkin_huxley/_codegen.py` — `CUDAPrinter`

Current stub (locate by class name):

```python
class CUDAPrinter:
    """Stub — generates __device__ scalar C code from a SymPy expression."""
    pass
```

Replace with:

```python
class CUDAPrinter:
    """
    Converts a SymPy scalar expression to a CUDA __device__ function body.

    Differences from EigenPrinter:
    - No Eigen types — all variables are plain C doubles.
    - exp/log/sqrt map to CUDA math intrinsics (expf/logf/sqrtf for float,
      exp/log/sqrt for double — we use double throughout).
    - Output is a complete __device__ inline function string ready to paste
      into a .cu file.
    """

    # Maps SymPy symbol names to the C variable names used inside the kernel.
    # Caller supplies this when building the printer.
    def __init__(self, symbol_map: dict[str, str] | None = None):
        self._sym_map = symbol_map or {}

    def doprint(self, expr) -> str:
        """Return C scalar expression string (no semicolon)."""
        from sympy.printing.c import C99CodePrinter
        printer = C99CodePrinter()
        code = printer.doprint(expr)
        for sym, cname in self._sym_map.items():
            # whole-word replace: avoid partial matches
            import re
            code = re.sub(r'\b' + re.escape(sym) + r'\b', cname, code)
        return code

    def print_device_fn(self, fn_name: str, params: list[str],
                        expr, return_type: str = "double") -> str:
        """
        Return a complete CUDA __device__ inline function.

        params: list of "double v", "double s", etc.
        expr:   SymPy expression for the return value.
        """
        body = self.doprint(expr)
        param_str = ", ".join(params)
        return (
            f"__device__ __forceinline__ {return_type} {fn_name}"
            f"({param_str}) {{\n"
            f"    return {body};\n"
            f"}}"
        )
```

### Helper: `compile_gate_cuda(gate_spec, fn_prefix) -> str`

Add a module-level function that generates the pair of `__device__` functions (`_inf` and `_tau`) for a gate, using `CUDAPrinter`. This mirrors `compile_gate_product_vm` but produces CUDA source instead of VM bytecode.

```python
def compile_gate_cuda(gate_spec, fn_prefix: str) -> str:
    """
    Generate CUDA __device__ functions for a gate's inf and tau expressions.

    Returns a string containing two __device__ functions:
      {fn_prefix}_inf(double V) -> double
      {fn_prefix}_tau(double V) -> double

    Raises HHEquationError if the expressions contain unsupported SymPy nodes.
    """
    from sympy import symbols
    V_sym = symbols("V")
    printer = CUDAPrinter({"V": "V"})

    inf_code = printer.print_device_fn(
        f"{fn_prefix}_inf", ["double V"], gate_spec.inf_expr
    )
    tau_code = printer.print_device_fn(
        f"{fn_prefix}_tau", ["double V"], gate_spec.tau_expr
    )
    return inf_code + "\n" + tau_code
```

### Exports

Add `CUDAPrinter` to `__init__.py` exports (it's already imported — just verify it's present in `__all__`).

Add `compile_gate_cuda` to `_codegen.py` exports and to `__init__.py` `from ._codegen import (...)` and `__all__`.

---

## Key files

| File | Change |
|---|---|
| `src/hodgkin_huxley/_codegen.py` | Implement `CUDAPrinter`, add `compile_gate_cuda` |
| `src/hodgkin_huxley/__init__.py` | Add `compile_gate_cuda` to imports and `__all__` |

---

## Contract for downstream tasks

- Task 17.8 calls `compile_gate_cuda(gate_spec, fn_prefix)` to generate `__device__` functions for each non-pattern-matched gate in a `NeuronModelSpec`.
- Output must be valid C99/CUDA scalar code — no Eigen, no Python, no `std::` types.
- `CUDAPrinter.print_device_fn()` is the primitive; task 17.8 calls it directly for synapse and intracellular ODE expressions.
