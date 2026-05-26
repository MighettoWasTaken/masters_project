"""
hodgkin_huxley._codegen
=======================

SymPy expression compilation infrastructure for the hodgkin-huxley library.

Provides:
  - Pre-defined SymPy symbols (V, Ca, V_pre, V_post, S, ...)
  - EigenPrinter  — generates Eigen C++ from SymPy (vectorized, SIMD)
  - CUDAPrinter   — generates scalar CUDA device code from SymPy / VM expressions
  - Pattern matching catalog (10 forms: 6 tau + 4 rate + Boltzmann)
  - TaggedExpr    — SymPy expr with pre-matched parameter metadata
  - compile_to_vm_bytecode — cross-platform VM bytecode for all custom expressions
  - HHEquationError — raised when expression compilation fails

Insertion points for future tasks:
  - task17 extension: wire generated CUDA code into the runtime pool builders
  - task13 extension: add more patterns to PATTERN_CATALOG as new forms are added
"""

from __future__ import annotations

import math as _math
import sys
from dataclasses import dataclass, field
from typing import Any

# On Windows, gmpy2's DLL may fail to load (entry-point not found) in some
# conda environments. Block it before importing sympy so that mpmath falls back
# to its pure-Python arithmetic backend, which is numerically equivalent for
# the pattern matching and code generation we do here.
if sys.platform == "win32":
    sys.modules.setdefault("gmpy2", None)  # type: ignore[assignment]

import sympy
from sympy import symbols, Symbol, Wild, exp, tanh, log, sqrt, Abs
from sympy.printing.codeprinter import CodePrinter

# =============================================================================
# Pre-defined symbols
# =============================================================================

V, Ca = symbols("V Ca")
V_pre, V_post, S = symbols("V_pre V_post S")
A = symbols("A")  # auxiliary state variable for 2-variable synapse ODEs
x = symbols("x")  # gate state variable for arbitrary dx/dt = F(x, V) ODEs
x_pre, x_post, w = symbols("x_pre x_post w")
I_source = symbols("I_source")
g_sym = symbols("g")
E_syn_sym = symbols("E_syn")

# Pre-defined intracellular substance symbols (task14)
DA   = symbols("DA")      # dopamine
cAMP = symbols("cAMP")    # cyclic AMP
IP3  = symbols("IP3")     # inositol trisphosphate
NO   = symbols("NO")      # nitric oxide
X_ic = symbols("X_ic")   # generic intracellular substance

# Cache for substance() helper — avoids duplicate Symbol objects
_substance_cache: dict[str, Symbol] = {
    "DA": DA, "cAMP": cAMP, "IP3": IP3, "NO": NO, "X_ic": X_ic,
    "Ca": Ca,
}


def substance(name: str) -> Symbol:
    """Return a SymPy Symbol representing a named intracellular substance."""
    if name not in _substance_cache:
        _substance_cache[name] = symbols(name)
    return _substance_cache[name]


def gate(name: str) -> Symbol:
    """Return a SymPy Symbol representing a named gate variable."""
    return symbols(f"gate_{name}")


# =============================================================================
# Symbol name mapping for code generation
# =============================================================================

_SYMBOL_MAP: dict[str, str] = {
    "V":        "V_",
    "Ca":       "Ca_",
    "V_pre":    "V_pre_",
    "V_post":   "V_post_",
    "S":        "S_",
    "x_pre":    "x_pre_",
    "x_post":   "x_post_",
    "w":        "w_",
    "I_source": "I_source_",
    "g":        "g_",
    "E_syn":    "E_syn_",
}


# =============================================================================
# EigenPrinter
# =============================================================================

class EigenPrinter(CodePrinter):
    """
    Prints SymPy expressions as Eigen C++ code operating on ArrayXd.

    Parameters
    ----------
    use_fast_exp : bool
        When True, `exp(x)` is emitted as `hh_fast_exp(x)` which references
        the fast_exp helper included in the JIT preamble. When False (default),
        uses `(x).exp()` — Eigen's element-wise exp method.
    scalar_mode : bool
        When True, emits standard C scalar math (`std::exp`, `std::tanh`, etc.)
        suitable for scalar2 (two-argument) JIT functions. No Eigen types.
    """

    printmethod = "_hh_print"

    _default_settings: dict = {
        "order": None,
        "full_prec": "auto",
        "error_on_reserved": False,
        "reserved_word_suffix": "_",
        "human": True,
        "inline": True,
        "allow_unknown_functions": True,
        "contract": True,
    }

    def __init__(self, use_fast_exp: bool = False, scalar_mode: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._use_fast_exp = use_fast_exp
        self._scalar_mode = scalar_mode

    # ---- Arithmetic operators -----------------------------------------------

    def _print_Add(self, expr, order=None):
        parts = [self._print(a) for a in expr.args]
        return "(" + " + ".join(parts) + ")"

    def _print_Mul(self, expr):
        parts = [self._print(a) for a in expr.args]
        return "(" + " * ".join(parts) + ")"

    def _print_Pow(self, expr, rational=False):
        base, exp_val = expr.args
        base_s = self._print(base)
        if exp_val == sympy.Integer(2):
            if self._scalar_mode:
                return f"(({base_s}) * ({base_s}))"
            return f"({base_s}).square()"
        if exp_val == sympy.Rational(1, 2):
            if self._scalar_mode:
                return f"std::sqrt({base_s})"
            return f"({base_s}).sqrt()"
        if exp_val == sympy.Integer(-1):
            return f"(1.0 / ({base_s}))"
        if isinstance(exp_val, sympy.Integer):
            if self._scalar_mode:
                n = int(exp_val)
                if n > 0:
                    return "(" + " * ".join([f"({base_s})"] * n) + ")"
                return f"std::pow({base_s}, {n})"
            return f"({base_s}).pow({int(exp_val)})"
        exp_s = self._print(exp_val)
        if self._scalar_mode:
            return f"std::pow({base_s}, {exp_s})"
        return f"({base_s}).pow({exp_s})"

    # ---- Functions ----------------------------------------------------------

    def _print_exp(self, expr):
        arg = self._print(expr.args[0])
        if self._scalar_mode:
            if self._use_fast_exp:
                return f"hh_fast_exp({arg})"
            return f"std::exp({arg})"
        if self._use_fast_exp:
            return f"hh_fast_exp({arg})"
        return f"({arg}).exp()"

    def _print_log(self, expr):
        arg = self._print(expr.args[0])
        if self._scalar_mode:
            return f"std::log({arg})"
        return f"({arg}).log()"

    def _print_tanh(self, expr):
        arg = self._print(expr.args[0])
        if self._scalar_mode:
            return f"std::tanh({arg})"
        return f"({arg}).tanh()"

    def _print_sin(self, expr):
        arg = self._print(expr.args[0])
        if self._scalar_mode:
            return f"std::sin({arg})"
        return f"({arg}).sin()"

    def _print_cos(self, expr):
        arg = self._print(expr.args[0])
        if self._scalar_mode:
            return f"std::cos({arg})"
        return f"({arg}).cos()"

    def _print_Abs(self, expr):
        arg = self._print(expr.args[0])
        if self._scalar_mode:
            return f"std::abs({arg})"
        return f"({arg}).abs()"

    def _print_sqrt(self, expr):
        arg = self._print(expr.args[0])
        if self._scalar_mode:
            return f"std::sqrt({arg})"
        return f"({arg}).sqrt()"

    # ---- Symbols and literals -----------------------------------------------

    def _print_Symbol(self, expr):
        name = str(expr)
        if name in _SYMBOL_MAP:
            return _SYMBOL_MAP[name]
        if name.startswith("gate_"):
            return name + "_"
        return name

    def _print_Integer(self, expr):
        return f"{int(expr)}.0"

    def _print_Float(self, expr):
        return repr(float(expr))

    def _print_Rational(self, expr):
        return repr(float(expr))

    def _print_Number(self, expr):
        return repr(float(expr))

    def _print_NumberSymbol(self, expr):
        return repr(float(expr))

    def _print_NegativeOne(self, expr):
        return "-1.0"

    def _print_One(self, expr):
        return "1.0"

    def _print_Zero(self, expr):
        return "0.0"

    def _print_Half(self, expr):
        return "0.5"

    # ---- Division (SymPy represents a/b as a * b**-1) ----------------------

    def _format_code(self, lines):
        # Required by CodePrinter; we generate inline expressions so just join.
        return lines

    def doprint(self, expr, assign_to=None):
        # Emit a single inline C++ expression string.  Using _print() directly
        # avoids the CodePrinter statement-formatting machinery (_format_code,
        # assign_to handling, etc.) which is intended for full statements.
        return self._print(expr)


# =============================================================================
# CUDAPrinter
# =============================================================================

class CUDAPrinter(CodePrinter):
    """
    Print SymPy scalar expressions as CUDA-ready C code.

    Unlike :class:`EigenPrinter`, this printer emits plain scalar math suitable
    for ``__device__`` helper functions inside CUDA kernels. Symbol names may be
    remapped at construction time so callers can target existing kernel locals.
    """

    _default_settings: dict = {
        "order": None,
        "full_prec": "auto",
        "error_on_reserved": False,
        "reserved_word_suffix": "_",
        "human": True,
        "inline": True,
        "allow_unknown_functions": True,
        "contract": True,
    }

    def __init__(self, symbol_map: dict[str, str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._sym_map = dict(symbol_map or {})

    def _format_code(self, lines):
        return lines

    def doprint(self, expr, assign_to=None) -> str:  # type: ignore[override]
        from sympy.printing.c import C99CodePrinter
        import re

        code = C99CodePrinter().doprint(expr)
        for sym, cname in self._sym_map.items():
            code = re.sub(r"\b" + re.escape(sym) + r"\b", cname, code)
        return code

    def print_device_fn(self, fn_name: str, params: list[str], expr,
                        return_type: str = "double") -> str:
        body = self.doprint(expr)
        param_str = ", ".join(params)
        return (
            f"__device__ __forceinline__ {return_type} {fn_name}"
            f"({param_str}) {{\n"
            f"    return {body};\n"
            "}"
        )


# =============================================================================
# Tagged expression
# =============================================================================

@dataclass
class TaggedExpr:
    """
    A SymPy expression annotated with pre-matched parameter metadata.

    Returned by Boltzmann(), Tau.*(), RateFunc.*() helpers. The `params`
    field holds a BoltzmannParams | TauParams | RateFuncParams (C++ structs
    from _core). Passing a TaggedExpr to GateSpec short-circuits pattern
    matching, avoiding both SymPy matching overhead and JIT compilation.
    """
    expr: Any       # sympy.Expr — the actual SymPy expression
    params: Any     # BoltzmannParams | TauParams | RateFuncParams
    form: Any = None  # TauParams.Form or RateFuncParams.Form, if applicable


# =============================================================================
# Pattern matching catalog
# =============================================================================

def _is_number(expr) -> bool:
    """Return True if expr is a SymPy numeric literal (no free symbols)."""
    return not expr.free_symbols and expr.is_number


# ---------------------------------------------------------------------------
# Low-level structural helpers
# ---------------------------------------------------------------------------

def _try_linear(expr, V_sym):
    """
    Return (slope, intercept) as floats if ``expr == slope*V_sym + intercept``.
    Returns None if not degree-1 linear or if V_sym not in expr.
    """
    try:
        if V_sym not in expr.free_symbols:
            return None
        poly = sympy.Poly(expr, V_sym)
        if poly.degree() != 1:
            return None
        coeffs = poly.all_coeffs()  # [slope, intercept]
        return float(coeffs[0]), float(coeffs[1])
    except Exception:
        return None


def _try_exp_linear(expr, V_sym):
    """
    Return (a, b) floats if ``expr`` can be interpreted as ``exp(a*V + b)``.

    Handles two canonical forms SymPy produces:
    1. ``exp(a*V + b)``       — single exp node, arg is linear in V
    2. ``K * exp(a*V + b')``  — numeric K > 0 pre-multiplied; absorbs log(K)
                                 into the intercept, returning (a, b' + log K)

    Form 2 arises when SymPy eagerly evaluates ``exp(float_const + slope*V)``
    into ``exp(float_const) * exp(slope*V)`` at expression construction time.
    """
    if isinstance(expr, sympy.exp):
        return _try_linear(expr.args[0], V_sym)

    if isinstance(expr, sympy.Mul):
        exp_factors = [a for a in expr.args if isinstance(a, sympy.exp)]
        num_factors = [a for a in expr.args if a.is_number and a.is_real]
        other = [
            a for a in expr.args
            if not (isinstance(a, sympy.exp) or (a.is_number and a.is_real))
        ]
        if len(exp_factors) == 1 and not other and num_factors:
            try:
                K = float(sympy.Mul(*num_factors))
                if K > 0:
                    result = _try_exp_linear(exp_factors[0], V_sym)
                    if result is not None:
                        a, b = result
                        return a, b + _math.log(K)
            except Exception:
                pass

    return None


def _extract_exp_factor(expr):
    """
    If ``expr`` is ``K * exp(g_arg)`` (K a real numeric, possibly K=1),
    return ``(K_float, g_arg_expr)``.  Otherwise return ``(None, None)``.
    """
    if isinstance(expr, sympy.exp):
        return 1.0, expr.args[0]
    if isinstance(expr, sympy.Mul):
        exp_fs = [a for a in expr.args if isinstance(a, sympy.exp)]
        num_fs = [a for a in expr.args if a.is_number and a.is_real]
        other  = [a for a in expr.args
                  if not (isinstance(a, sympy.exp) or (a.is_number and a.is_real))]
        if len(exp_fs) == 1 and not other and num_fs:
            try:
                return float(sympy.Mul(*num_fs)), exp_fs[0].args[0]
            except Exception:
                pass
    return None, None


def _structural_sigmoid_form(expr, V_sym):
    """
    Return ``(A, v_half, k)`` if ``expr == A / (1 + exp(a*V + b))``, else None.

    Handles four SymPy representations:
    1. ``A / (1 + exp(f))``         — num is A (number), denom has lone exp
    2. ``A*exp(g) / (exp(g) + c)``  — SymPy multiplied through by exp(-f)
       where g = -f and c = exp(const_part_of_f)

    Invariant: ``a = -1/k``, ``b = v_half/k``.
    """
    try:
        num, denom = expr.as_numer_denom()
    except Exception:
        return None

    # ------------------------------------------------------------------
    # Case 1: num is a plain number  =>  standard  A/(1+exp(f))
    # ------------------------------------------------------------------
    if num.is_number and num.is_real and V_sym not in num.free_symbols:
        exp_term = denom - sympy.Integer(1)
        result = _try_exp_linear(exp_term, V_sym)
        if result is None:
            try:
                result = _try_exp_linear(sympy.expand(exp_term), V_sym)
            except Exception:
                pass
        if result is not None:
            a, b = result
            if abs(a) > 1e-20:
                try:
                    return float(num), -b / a, -1.0 / a
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Case 2: SymPy normalized  =>  A*exp(g) / (exp(g) + c)
    # num = A * exp(g),  denom = exp(g) + c  (c is a positive number)
    # Original expression: A*c / (c + exp(-g)) = A/(1 + exp(-g)/c)
    #                     = A/(1 + exp(-g - log(c)))
    # ------------------------------------------------------------------
    K_num, g_arg = _extract_exp_factor(num)
    if K_num is not None and g_arg is not None and abs(K_num) > 1e-30:
        # Reconstruct exp(g) from g_arg (K_num absorbed into A)
        exp_g_sympy = sympy.exp(g_arg)
        c_expr = sympy.simplify(denom - exp_g_sympy)
        if c_expr.is_number and not c_expr.free_symbols:
            try:
                c_val = float(c_expr)
                if c_val > 0:
                    g_lin = _try_linear(g_arg, V_sym)
                    if g_lin is not None:
                        g_slope, g_intercept = g_lin
                        if abs(g_slope) > 1e-20:
                            # effective exp arg = log(c) - g_arg
                            # = log(c) - g_slope*V - g_intercept
                            # a_eff = -g_slope, b_eff = log(c) - g_intercept
                            a_eff = -g_slope
                            b_eff = _math.log(c_val) - g_intercept
                            A_val = K_num  # the numeric factor from num = K_num * exp(g)
                            return A_val, -b_eff / a_eff, -1.0 / a_eff
            except Exception:
                pass

    return None


def _structural_boltzmann(expr, V_sym):
    """Return ``(v_half, k)`` if ``expr == 1/(1+exp(linear))``, else None."""
    sf = _structural_sigmoid_form(expr, V_sym)
    if sf is None:
        return None
    A, v_half, k = sf
    # Accept any amplitude — the C++ BoltzmannParams has no amplitude field;
    # amplitude is handled by GateSpec.scale.  We only use this matcher for
    # pure Boltzmann (A=1) detection; sigmoid (A≠1) is handled separately.
    if abs(A - 1.0) > 1e-9:
        return None
    return v_half, k


def _structural_tau_boltzmann(expr, V_sym):
    """
    Return ``(base, amp, v_half, k)`` if ``expr == base + amp/(1+exp(linear))``.
    """
    if not isinstance(expr, sympy.Add):
        return None
    constant_total = sympy.Integer(0)
    boltzmann_term = None
    for term in expr.args:
        if not term.free_symbols:
            constant_total += term
        else:
            try:
                t_num, t_denom = term.as_numer_denom()
                if not t_num.free_symbols:  # amp is a number
                    bm = _structural_boltzmann(sympy.Integer(1) / t_denom, V_sym)
                    if bm is not None:
                        if boltzmann_term is not None:
                            return None  # more than one boltzmann term
                        boltzmann_term = (float(t_num),) + bm
                        continue
            except Exception:
                pass
            return None  # unexpected non-constant, non-boltzmann term
    if boltzmann_term is None:
        return None
    amp, v_half, k = boltzmann_term
    return float(constant_total), amp, v_half, k


def _structural_tau_double_exp_sum(expr, V_sym):
    """
    Return ``(base, amp, v1, s1, v2, s2)`` if
    ``expr == base + amp / (exp((V+v1)/s1) + exp(-(V+v2)/s2))``, else None.

    Maps to C++ TauParams::Form::DOUBLE_EXP_SUM with layout:
      params[0]=base, params[1]=amp, params[2]=v1, params[3]=s1,
      params[4]=0 (unused), params[5]=v2, params[6]=s2

    Works directly with SymPy's ``Mul``/``Pow`` representation rather than
    ``as_numer_denom()``, which transforms float-constant exp expressions.
    """
    # Separate constant base from the fractional term
    if isinstance(expr, sympy.Add):
        const_terms = [t for t in expr.args if not t.free_symbols]
        var_terms   = [t for t in expr.args if t.free_symbols]
        if len(var_terms) != 1:
            return None
        base = float(sum(float(t) for t in const_terms)) if const_terms else 0.0
        var_term = var_terms[0]
    else:
        base = 0.0
        var_term = expr

    # var_term must be Mul with: numeric amp × Pow(Add(e1, e2), -1)
    if not isinstance(var_term, sympy.Mul):
        return None

    pow_inv = None
    num_parts = []
    for factor in var_term.args:
        if (isinstance(factor, sympy.Pow) and
                factor.args[1] == sympy.Integer(-1) and
                isinstance(factor.args[0], sympy.Add)):
            if pow_inv is not None:
                return None  # multiple inverse-Add factors
            pow_inv = factor
        else:
            num_parts.append(factor)

    if pow_inv is None:
        return None

    denom_sum = pow_inv.args[0]
    if len(denom_sum.args) != 2:
        return None

    try:
        amp = float(sympy.Mul(*num_parts)) if num_parts else 1.0
    except Exception:
        return None
    if abs(amp) < 1e-30:
        return None

    # Each summand must be exp(linear in V_sym)
    r1 = _try_exp_linear(denom_sum.args[0], V_sym)
    r2 = _try_exp_linear(denom_sum.args[1], V_sym)
    if r1 is None or r2 is None:
        return None

    slope1, intercept1 = r1
    slope2, intercept2 = r2

    # Identify positive-slope as exp((V+v1)/s1), negative as exp(-(V+v2)/s2)
    if slope1 > 0 and slope2 < 0:
        s1 = 1.0 / slope1;   v1 = intercept1 / slope1
        s2 = -1.0 / slope2;  v2 = intercept2 / slope2
    elif slope2 > 0 and slope1 < 0:
        s1 = 1.0 / slope2;   v1 = intercept2 / slope2
        s2 = -1.0 / slope1;  v2 = intercept1 / slope1
    else:
        return None

    return base, amp, v1, s1, v2, s2


def _structural_rate_sigmoid(expr, V_sym):
    """
    Return ``(A, B, C)`` if ``expr == A/(1+exp((V+B)/C))`` with A ≠ 1, else None.
    """
    sf = _structural_sigmoid_form(expr, V_sym)
    if sf is None:
        return None
    A, v_half, k = sf
    # sigmoid parameterisation: exp arg = (V+B)/C  =>  a=1/C, b=B/C
    # v_half = -b/a = -B, k = -1/a = -C
    C = -k
    B = -v_half
    return A, B, C


def _structural_rate_exp_decay(expr, V_sym):
    """Return ``(A, B, C)`` if ``expr == A * exp((V+B)/C)``, else None."""
    # Handles float-evaluated form: A*exp(B/C) * exp(V/C) = K * exp(V/C)
    if isinstance(expr, sympy.Mul):
        exp_factors = [a for a in expr.args if isinstance(a, sympy.exp)]
        num_factors = [a for a in expr.args if a.is_number and a.is_real]
        other = [
            a for a in expr.args
            if not (isinstance(a, sympy.exp) or (a.is_number and a.is_real))
        ]
        if len(exp_factors) == 1 and not other:
            A = float(sympy.Mul(*num_factors)) if num_factors else 1.0
            inner = exp_factors[0].args[0]
            lin = _try_linear(inner, V_sym)
            if lin is not None:
                slope, intercept = lin
                if abs(slope) > 1e-20:
                    C = 1.0 / slope
                    B = intercept * C  # intercept = B/C
                    return A, B, C
    if isinstance(expr, sympy.exp):
        lin = _try_linear(expr.args[0], V_sym)
        if lin is not None:
            slope, intercept = lin
            if abs(slope) > 1e-20:
                C = 1.0 / slope
                return 1.0, intercept * C, C
    return None


def _structural_rate_linear_over_exp(expr, V_sym):
    """
    Return ``(A, B, C)`` if ``expr == A*(V+B) / (exp((V+B)/C) - 1)``.

    Recovers B from the numerator even when SymPy has split
    ``exp((V+B)/C)`` into ``K * exp(V/C)`` due to float constants.

    Note: after SymPy's float canonicalisation the numerator A*(V+B) remains
    as a plain polynomial (no exp factor is introduced in this fraction's
    numerator), so only the standard linear-numerator form is needed here.
    """
    try:
        num, denom = expr.as_numer_denom()
    except Exception:
        return None

    # num = A*(V+B)  (linear polynomial, no exp factor)
    num_lin = _try_linear(num, V_sym)
    if num_lin is None:
        return None
    A_val, AB_val = num_lin
    if abs(A_val) < 1e-20:
        return None
    A, B = A_val, AB_val / A_val
    # denom = exp(...) - 1  =>  denom + 1 = exp(...)
    exp_part = denom + sympy.Integer(1)
    exp_lin = _try_exp_linear(exp_part, V_sym)
    if exp_lin is None:
        try:
            exp_lin = _try_exp_linear(sympy.expand(exp_part), V_sym)
        except Exception:
            pass
    if exp_lin is None:
        return None
    slope_exp, _ = exp_lin
    if abs(slope_exp) < 1e-20:
        return None
    return A, B, 1.0 / slope_exp


def _structural_rate_linear_over_expm1(expr, V_sym):
    """
    Return ``(A, B, C)`` if ``expr == A*(V+B) / (1 - exp(-(V+B)/C))``.
    """
    try:
        num, denom = expr.as_numer_denom()
    except Exception:
        return None

    # Standard form: num = A*(V+B) (linear, no exp factor)
    num_lin = _try_linear(num, V_sym)
    if num_lin is not None:
        A_val, AB_val = num_lin
        if abs(A_val) > 1e-20:
            A, B = A_val, AB_val / A_val
            # denom = 1 - exp(-(V+B)/C)  =>  1 - denom = exp(-(V+B)/C)
            exp_part = sympy.Integer(1) - denom
            exp_lin = _try_exp_linear(exp_part, V_sym)
            if exp_lin is None:
                try:
                    exp_lin = _try_exp_linear(sympy.expand(exp_part), V_sym)
                except Exception:
                    pass
            if exp_lin is not None:
                slope_exp, _ = exp_lin
                # slope_exp = -1/C  (negative exp has negative slope)
                if abs(slope_exp) > 1e-20 and slope_exp < 0:
                    return A, B, -1.0 / slope_exp

    # SymPy-normalized form: num = A*(V+B)*exp(g), denom = exp(g) - K
    # where g = slope*V (slope = 1/C > 0), K = exp(-B/C)
    # Original: A*(V+B)/(1 - K*exp(-slope*V)) × exp(slope*V)/exp(slope*V)
    if isinstance(num, sympy.Mul):
        exp_fs = [a for a in num.args if isinstance(a, sympy.exp)]
        other_fs = [a for a in num.args if not isinstance(a, sympy.exp)]
        if len(exp_fs) == 1:
            exp_g = exp_fs[0]
            poly_part = sympy.Mul(*other_fs) if other_fs else sympy.Integer(1)
            g_lin = _try_linear(exp_g.args[0], V_sym)
            if g_lin is not None:
                g_slope, g_intercept = g_lin
                if g_slope > 1e-20:  # must be positive (exp was pulled from denom)
                    poly_lin = _try_linear(poly_part, V_sym)
                    if poly_lin is not None:
                        A_val, AB_val = poly_lin
                        if abs(A_val) > 1e-20:
                            A, B = A_val, AB_val / A_val
                            # Check denom = exp(g) - K  =>  K = exp(g) - denom
                            c_expr = sympy.simplify(denom - exp_g)
                            if c_expr.is_number and not c_expr.free_symbols:
                                # C = 1/g_slope (g_slope = 1/C, positive)
                                C = 1.0 / g_slope
                                return A, B, C

    return None


def try_pattern_match(expr, dep_sym=None):
    """
    Attempt to match ``expr`` against all known standard forms.

    Returns a tuple ``(params, form_enum)`` where ``params`` is a
    BoltzmannParams | TauParams | RateFuncParams instance, or ``(None, None)``
    if no pattern matches.

    Uses structural coefficient extraction via sympy.Poly rather than Wild-based
    pattern matching, so it correctly handles expressions with Python float
    constants (which SymPy eagerly evaluates, changing structural form).
    """
    from hodgkin_huxley._core import (
        BoltzmannParams as _BoltzmannParams,
        TauParams as _TauParams,
        RateFuncParams as _RateFuncParams,
        TauForm,
        RateFuncForm,
    )

    # Use V as the voltage symbol for matching (the primary free symbol)
    V_sym = dep_sym if dep_sym is not None else V

    # Normalise: rewrite trig (tanh, cosh) in terms of exp
    try:
        norm = expr.rewrite(exp)
    except Exception:
        norm = expr

    # ------------------------------------------------------------------
    # 1. Boltzmann: 1 / (1 + exp(a*V + b))
    # ------------------------------------------------------------------
    bm = _structural_boltzmann(norm, V_sym)
    if bm is not None:
        v_half, k = bm
        bp = _BoltzmannParams()
        bp.v_half = v_half
        bp.k = k
        return bp, None

    # ------------------------------------------------------------------
    # 2. TauParams — CONSTANT
    # ------------------------------------------------------------------
    if _is_number(norm):
        tp = _TauParams()
        tp.form = TauForm.CONSTANT
        tp.set_param(0, float(norm))
        return tp, TauForm.CONSTANT

    # ------------------------------------------------------------------
    # 3. TauParams — BOLTZMANN: base + amp / (1 + exp(linear))
    # ------------------------------------------------------------------
    tb = _structural_tau_boltzmann(norm, V_sym)
    if tb is not None:
        base, amp, v_half, k = tb
        tp = _TauParams()
        tp.form = TauForm.BOLTZMANN
        tp.set_param(0, base); tp.set_param(1, amp)
        tp.set_param(2, v_half); tp.set_param(3, k)
        return tp, TauForm.BOLTZMANN

    # ------------------------------------------------------------------
    # 3.5. TauParams — DOUBLE_EXP_SUM: base + amp/(exp((V+v1)/s1)+exp(-(V+v2)/s2))
    # ------------------------------------------------------------------
    des = _structural_tau_double_exp_sum(norm, V_sym)
    if des is not None:
        base, amp, v1, s1, v2, s2 = des
        tp = _TauParams()
        tp.form = TauForm.DOUBLE_EXP_SUM
        tp.set_param(0, base); tp.set_param(1, amp)
        tp.set_param(2, v1);   tp.set_param(3, s1)
        tp.set_param(4, 0.0)   # unused
        tp.set_param(5, v2);   tp.set_param(6, s2)
        return tp, TauForm.DOUBLE_EXP_SUM

    # ------------------------------------------------------------------
    # 4. RateFuncParams — LINEAR_OVER_EXP: A*(V+B)/(exp((V+B)/C)-1)
    # ------------------------------------------------------------------
    loe = _structural_rate_linear_over_exp(norm, V_sym)
    if loe is not None:
        A, B, C = loe
        rp = _RateFuncParams()
        rp.form = RateFuncForm.LINEAR_OVER_EXP
        rp.A = A; rp.B = B; rp.C = C
        return rp, RateFuncForm.LINEAR_OVER_EXP

    # ------------------------------------------------------------------
    # 5. RateFuncParams — EXP_DECAY: A*exp((V+B)/C)
    # ------------------------------------------------------------------
    ed = _structural_rate_exp_decay(norm, V_sym)
    if ed is not None:
        A, B, C = ed
        rp = _RateFuncParams()
        rp.form = RateFuncForm.EXP_DECAY
        rp.A = A; rp.B = B; rp.C = C
        return rp, RateFuncForm.EXP_DECAY

    # ------------------------------------------------------------------
    # 6. RateFuncParams — LINEAR_OVER_EXPM1: A*(V+B)/(1-exp(-(V+B)/C))
    # ------------------------------------------------------------------
    loem1 = _structural_rate_linear_over_expm1(norm, V_sym)
    if loem1 is not None:
        A, B, C = loem1
        rp = _RateFuncParams()
        rp.form = RateFuncForm.LINEAR_OVER_EXPM1
        rp.A = A; rp.B = B; rp.C = C
        return rp, RateFuncForm.LINEAR_OVER_EXPM1

    # ------------------------------------------------------------------
    # 7. RateFuncParams — SIGMOID: A/(1+exp((V+B)/C))
    # ------------------------------------------------------------------
    sg = _structural_rate_sigmoid(norm, V_sym)
    if sg is not None:
        A, B, C = sg
        rp = _RateFuncParams()
        rp.form = RateFuncForm.SIGMOID
        rp.A = A; rp.B = B; rp.C = C
        return rp, RateFuncForm.SIGMOID

    # ------------------------------------------------------------------
    # 8. TauParams — COMPOUND_AB: 1/(alpha + beta) where each is a rate form
    # ------------------------------------------------------------------
    try:
        recip = sympy.simplify(1 / norm)
        if recip.is_Add and len(recip.args) == 2:
            a_expr, b_expr = recip.args
            r_a = try_pattern_match(a_expr, dep_sym)
            r_b = try_pattern_match(b_expr, dep_sym)
            if (r_a[0] is not None and isinstance(r_a[0], _RateFuncParams) and
                    r_b[0] is not None and isinstance(r_b[0], _RateFuncParams)):
                tp = _TauParams()
                tp.form = TauForm.COMPOUND_AB
                tp.set_param(0, r_a[0].A); tp.set_param(1, r_a[0].B); tp.set_param(2, r_a[0].C)
                tp.set_param(3, r_b[0].A); tp.set_param(4, r_b[0].B); tp.set_param(5, r_b[0].C)
                return tp, TauForm.COMPOUND_AB
    except Exception:
        pass

    return None, None


# =============================================================================
# HHEquationError
# =============================================================================

class HHEquationError(Exception):
    """
    Raised when compilation of a SymPy expression fails.

    Attributes
    ----------
    source : str
        The expression source string (or generated code) that triggered the error.
    stderr : str
        Error message or diagnostic output.
    """
    def __init__(self, source: str, stderr: str):
        self.source = source
        self.stderr = stderr
        super().__init__(
            f"Expression compilation failed.\n"
            f"--- error ---\n{stderr}\n"
            f"--- source ---\n{source}"
        )


def _float_code(value) -> str:
    """Format a Python / C++ numeric value as CUDA scalar source."""
    val = float(value)
    if not _math.isfinite(val):
        raise HHEquationError(source=str(value), stderr="non-finite constant in CUDA codegen")
    return repr(val)


def _expr_attr(obj, name: str):
    """Return a raw SymPy expression if *obj* exposes one, else None."""
    raw = getattr(obj, name, None)
    if isinstance(raw, TaggedExpr):
        return raw.expr
    if isinstance(raw, sympy.Basic):
        return raw
    return None


def _vm_nonempty(vm) -> bool:
    return bool(vm is not None and hasattr(vm, "empty") and not vm.empty())


def _dep_param_name(gate_spec) -> str:
    dep = getattr(getattr(gate_spec, "dependency", None), "name", "")
    return "dep" if dep == "INTRACELLULAR" else "V"


def _sanitize_cuda_ident(name: str, fallback: str) -> str:
    """Return a CUDA-safe identifier based on *name*."""
    import re

    ident = re.sub(r"\W+", "_", str(name)).strip("_")
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = f"{fallback}_{ident}"
    return ident


def _vm_aux_param_names(
    prog,
    *,
    gate_names: list[str] | None = None,
    x_names: list[str] | None = None,
    reserved: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ordered gate/substance parameter names referenced by a VM program."""
    from hodgkin_huxley._core import VmOp

    reserved = set(reserved or ())
    gate_params: list[str] = []
    x_params: list[str] = []

    def add_unique(dest: list[str], value: str) -> None:
        if value in reserved:
            value = f"{value}_arg"
        if value not in reserved and value not in dest:
            dest.append(value)
            reserved.add(value)

    for ins in getattr(prog, "instructions", ()):
        if ins.op == VmOp.PUSH_GATE:
            raw = (
                gate_names[ins.operand]
                if gate_names is not None and 0 <= ins.operand < len(gate_names)
                else f"gate_{ins.operand}"
            )
            add_unique(gate_params, _sanitize_cuda_ident(raw, f"gate_{ins.operand}"))
        elif ins.op == VmOp.PUSH_X:
            raw = (
                x_names[ins.operand]
                if x_names is not None and 0 <= ins.operand < len(x_names)
                else f"X_{ins.operand}"
            )
            add_unique(x_params, _sanitize_cuda_ident(raw, f"X_{ins.operand}"))

    return gate_params, x_params


def _vmexpr_to_cuda_expr(prog, *, dep_name="V", s_name="x", a_name="A",
                         gate_names: list[str] | None = None,
                         x_names: list[str] | None = None) -> str:
    """Rebuild a scalar CUDA expression string from a VmExpr program."""
    from hodgkin_huxley._core import VmOp

    stack: list[str] = []

    def pop1() -> str:
        if not stack:
            raise HHEquationError(source=str(prog), stderr="VM stack underflow during CUDA emission")
        return stack.pop()

    def pop2() -> tuple[str, str]:
        rhs = pop1()
        lhs = pop1()
        return lhs, rhs

    for ins in prog.instructions:
        if ins.op == VmOp.PUSH_DEP:
            stack.append(dep_name)
        elif ins.op == VmOp.PUSH_CONST:
            stack.append(_float_code(prog.constants[ins.operand]))
        elif ins.op == VmOp.PUSH_S:
            stack.append(s_name)
        elif ins.op == VmOp.PUSH_A:
            stack.append(a_name)
        elif ins.op == VmOp.PUSH_GATE:
            if gate_names is not None and 0 <= ins.operand < len(gate_names):
                stack.append(gate_names[ins.operand])
            else:
                stack.append(f"gate_{ins.operand}")
        elif ins.op == VmOp.PUSH_X:
            if x_names is not None and 0 <= ins.operand < len(x_names):
                stack.append(x_names[ins.operand])
            else:
                stack.append(f"X_{ins.operand}")
        elif ins.op == VmOp.ADD:
            lhs, rhs = pop2()
            stack.append(f"(({lhs}) + ({rhs}))")
        elif ins.op == VmOp.MUL:
            lhs, rhs = pop2()
            stack.append(f"(({lhs}) * ({rhs}))")
        elif ins.op == VmOp.NEG:
            stack.append(f"(-({pop1()}))")
        elif ins.op == VmOp.RCP:
            stack.append(f"(1.0 / ({pop1()}))")
        elif ins.op == VmOp.POW_INT:
            stack.append(f"pow({pop1()}, {_float_code(ins.operand)})")
        elif ins.op == VmOp.POW_HALF:
            stack.append(f"sqrt({pop1()})")
        elif ins.op == VmOp.POW_GEN:
            lhs, rhs = pop2()
            stack.append(f"pow(({lhs}), ({rhs}))")
        elif ins.op == VmOp.EXP:
            stack.append(f"exp({pop1()})")
        elif ins.op == VmOp.LOG:
            stack.append(f"log({pop1()})")
        elif ins.op == VmOp.TANH:
            stack.append(f"tanh({pop1()})")
        elif ins.op == VmOp.SIN:
            stack.append(f"sin({pop1()})")
        elif ins.op == VmOp.COS:
            stack.append(f"cos({pop1()})")
        elif ins.op == VmOp.SQRT:
            stack.append(f"sqrt({pop1()})")
        elif ins.op == VmOp.ABS:
            stack.append(f"fabs({pop1()})")
        else:
            raise HHEquationError(source=str(prog), stderr=f"unsupported VM opcode {ins.op!r}")

    if len(stack) != 1:
        raise HHEquationError(source=str(prog), stderr="VM program did not reduce to one CUDA expression")
    return stack[0]


def _boltzmann_cuda_expr(params, dep_name="V") -> str:
    return (
        f"(1.0 / (1.0 + exp(-(({dep_name}) - {_float_code(params.v_half)})"
        f" / {_float_code(params.k)})))"
    )


def _tau_cuda_expr(params, dep_name="V") -> str:
    from hodgkin_huxley._core import TauForm

    p = [params.get_param(i) for i in range(8)]
    if params.form == TauForm.CONSTANT:
        return _float_code(p[0])
    if params.form == TauForm.BOLTZMANN:
        return (
            f"({_float_code(p[0])} + {_float_code(p[1])} / "
            f"(1.0 + exp(-(({dep_name}) - {_float_code(p[2])}) / {_float_code(p[3])})))"
        )
    if params.form == TauForm.DOUBLE_EXP_SUM:
        return (
            f"({_float_code(p[0])} + {_float_code(p[1])} / "
            f"(exp((({dep_name}) + {_float_code(p[2])}) / {_float_code(p[3])}) + "
            f"exp(-(({dep_name}) + {_float_code(p[5])}) / {_float_code(p[6])})))"
        )
    if params.form == TauForm.OFFSET_DOUBLE_EXP:
        return (
            f"({_float_code(p[0])}"
            f" + {_float_code(p[1])} * exp(-pow((({dep_name}) + {_float_code(p[2])}) / {_float_code(p[3])}, 2.0))"
            f" + {_float_code(p[4])} * exp(-pow((({dep_name}) + {_float_code(p[5])}) / {_float_code(p[6])}, 2.0)))"
        )
    if params.form == TauForm.SCALED_EXP:
        return (
            f"({_float_code(p[0])} / cosh((({dep_name}) - {_float_code(p[1])})"
            f" / {_float_code(2.0 * p[2])}))"
        )
    if params.form == TauForm.COMPOUND_AB:
        return (
            f"(1.0 / ("
            f"{_float_code(p[0])} * exp((({dep_name}) + {_float_code(p[1])}) / {_float_code(p[2])}) + "
            f"{_float_code(p[3])} * exp((({dep_name}) + {_float_code(p[4])}) / {_float_code(p[5])})"
            f"))"
        )
    raise HHEquationError(source=str(params), stderr="unsupported TauForm in CUDA codegen")


def _rate_cuda_expr(params, dep_name="V") -> str:
    from hodgkin_huxley._core import RateFuncForm

    x = f"(({dep_name}) + {_float_code(params.B)})"
    xc = f"({x} / {_float_code(params.C)})"
    if params.form == RateFuncForm.LINEAR_OVER_EXP:
        return (
            f"((fabs({xc}) < 1e-6)"
            f" ? ({_float_code(params.A)} * {_float_code(params.C)} * (1.0 + 0.5 * ({xc})))"
            f" : ({_float_code(params.A)} * {x} / (exp({xc}) - 1.0)))"
        )
    if params.form == RateFuncForm.EXP_DECAY:
        return f"({_float_code(params.A)} * exp({xc}))"
    if params.form == RateFuncForm.LINEAR_OVER_EXPM1:
        return (
            f"((fabs({xc}) < 1e-6)"
            f" ? ({_float_code(params.A)} * {_float_code(params.C)} * (1.0 + 0.5 * ({xc})))"
            f" : ({_float_code(params.A)} * {x} / (1.0 - exp(-({xc})))))"
        )
    if params.form == RateFuncForm.SIGMOID:
        return f"({_float_code(params.A)} / (1.0 + exp({xc})))"
    raise HHEquationError(source=str(params), stderr="unsupported RateFuncForm in CUDA codegen")


def _emit_cuda_vm_fn(
    fn_name: str,
    prog,
    *,
    base_params: list[str],
    dep_name: str,
    s_name: str = "x",
    a_name: str = "A",
    gate_names: list[str] | None = None,
    x_names: list[str] | None = None,
) -> str:
    """Emit a complete CUDA helper function from a VM program."""
    reserved = {
        param.rsplit(" ", 1)[-1]
        for param in base_params
    }
    gate_params, x_params = _vm_aux_param_names(
        prog,
        gate_names=gate_names,
        x_names=x_names,
        reserved=reserved,
    )
    expr_code = _vmexpr_to_cuda_expr(
        prog,
        dep_name=dep_name,
        s_name=s_name,
        a_name=a_name,
        gate_names=gate_params or None,
        x_names=x_params or None,
    )
    all_params = list(base_params)
    all_params.extend(f"double {name}" for name in gate_params)
    all_params.extend(f"double {name}" for name in x_params)
    return (
        f"__device__ __forceinline__ double {fn_name}"
        f"({', '.join(all_params)}) {{\n"
        f"    return {expr_code};\n"
        "}"
    )


def compile_gate_cuda(
    gate_spec,
    fn_prefix: str,
    *,
    gate_names: list[str] | None = None,
    x_names: list[str] | None = None,
) -> str:
    """
    Generate CUDA ``__device__`` helpers for one gate specification.

    The live project stores either:
      1. raw SymPy expressions on higher-level helper objects, or
      2. matched parameter structs / VM bytecode on the compiled core GateSpec.

    This helper supports both forms and emits only the functions that are
    actually defined for the given gate.
    """
    if gate_names is not None:
        gate_names = [
            _sanitize_cuda_ident(name, f"gate_{idx}")
            for idx, name in enumerate(gate_names)
        ]
    if x_names is not None:
        x_names = [
            _sanitize_cuda_ident(name, f"X_{idx}")
            for idx, name in enumerate(x_names)
        ]

    printer = CUDAPrinter({"V": "V", "Ca": "dep", "x": "x", "S": "x", "A": "A"})
    dep_name = _dep_param_name(gate_spec)
    update_name = getattr(getattr(gate_spec, "update_form", None), "name", "")
    pieces: list[str] = []

    def add_expr_fn(suffix: str, params: list[str], expr) -> None:
        pieces.append(printer.print_device_fn(f"{fn_prefix}_{suffix}", params, expr))

    def add_code_fn(suffix: str, params: list[str], expr_code: str) -> None:
        pieces.append(
            f"__device__ __forceinline__ double {fn_prefix}_{suffix}"
            f"({', '.join(params)}) {{\n"
            f"    return {expr_code};\n"
            "}"
        )

    inf_expr = _expr_attr(gate_spec, "inf_expr")
    tau_expr = _expr_attr(gate_spec, "tau_expr")
    alpha_expr = _expr_attr(gate_spec, "alpha_expr")
    beta_expr = _expr_attr(gate_spec, "beta_expr")
    dxdt_expr = _expr_attr(gate_spec, "dxdt_expr")

    if inf_expr is not None:
        add_expr_fn("inf", [f"double {dep_name}"], inf_expr)
    elif _vm_nonempty(getattr(gate_spec, "inf_vm", None)):
        pieces.append(
            _emit_cuda_vm_fn(
                f"{fn_prefix}_inf",
                gate_spec.inf_vm,
                base_params=[f"double {dep_name}"],
                dep_name=dep_name,
                gate_names=gate_names,
                x_names=x_names,
            )
        )
    elif hasattr(gate_spec, "inf") and update_name in {"INF_TAU", "INSTANT"}:
        add_code_fn("inf", [f"double {dep_name}"], _boltzmann_cuda_expr(gate_spec.inf, dep_name))

    if tau_expr is not None:
        add_expr_fn("tau", ["double V"], tau_expr)
    elif _vm_nonempty(getattr(gate_spec, "tau_vm", None)):
        pieces.append(
            _emit_cuda_vm_fn(
                f"{fn_prefix}_tau",
                gate_spec.tau_vm,
                base_params=["double V"],
                dep_name="V",
                gate_names=gate_names,
                x_names=x_names,
            )
        )
    elif hasattr(gate_spec, "tau") and update_name == "INF_TAU":
        add_code_fn("tau", ["double V"], _tau_cuda_expr(gate_spec.tau, "V"))

    if alpha_expr is not None:
        add_expr_fn("alpha", ["double V"], alpha_expr)
    elif _vm_nonempty(getattr(gate_spec, "alpha_vm", None)):
        pieces.append(
            _emit_cuda_vm_fn(
                f"{fn_prefix}_alpha",
                gate_spec.alpha_vm,
                base_params=["double V"],
                dep_name="V",
                gate_names=gate_names,
                x_names=x_names,
            )
        )
    elif hasattr(gate_spec, "alpha") and update_name == "ALPHA_BETA":
        add_code_fn("alpha", ["double V"], _rate_cuda_expr(gate_spec.alpha, "V"))

    if beta_expr is not None:
        add_expr_fn("beta", ["double V"], beta_expr)
    elif _vm_nonempty(getattr(gate_spec, "beta_vm", None)):
        pieces.append(
            _emit_cuda_vm_fn(
                f"{fn_prefix}_beta",
                gate_spec.beta_vm,
                base_params=["double V"],
                dep_name="V",
                gate_names=gate_names,
                x_names=x_names,
            )
        )
    elif hasattr(gate_spec, "beta") and update_name == "ALPHA_BETA":
        add_code_fn("beta", ["double V"], _rate_cuda_expr(gate_spec.beta, "V"))

    if dxdt_expr is not None:
        add_expr_fn("dxdt", [f"double {dep_name}", "double x"], dxdt_expr)
    elif _vm_nonempty(getattr(gate_spec, "dxdt_vm", None)):
        pieces.append(
            _emit_cuda_vm_fn(
                f"{fn_prefix}_dxdt",
                gate_spec.dxdt_vm,
                base_params=[f"double {dep_name}", "double x"],
                dep_name=dep_name,
                s_name="x",
                gate_names=gate_names,
                x_names=x_names,
            )
        )

    if not pieces:
        raise HHEquationError(source=repr(gate_spec), stderr="gate has no CUDA-emittable expressions")

    return "\n\n".join(pieces)


def compile_intracellular_cuda(
    intr_spec,
    fn_prefix: str,
    *,
    gate_names: list[str] | None = None,
    x_names: list[str] | None = None,
) -> str:
    """
    Generate CUDA ``__device__`` helpers for intracellular VM expressions.

    This is the explicit 17.5 -> 17.9 bridge for the same CUSTOM_EXPR ODEs,
    Nernst overrides, and modulation programs that the CUDA composable runtime
    evaluates through the VM today.
    """
    if gate_names is not None:
        gate_names = [
            _sanitize_cuda_ident(name, f"gate_{idx}")
            for idx, name in enumerate(gate_names)
        ]
    if x_names is not None:
        x_names = [
            _sanitize_cuda_ident(name, f"X_{idx}")
            for idx, name in enumerate(x_names)
        ]

    pieces: list[str] = []

    if _vm_nonempty(getattr(intr_spec, "ode_vm", None)):
        pieces.append(
            _emit_cuda_vm_fn(
                f"{fn_prefix}_ode",
                intr_spec.ode_vm,
                base_params=["double I_source", "double x"],
                dep_name="I_source",
                s_name="x",
                gate_names=gate_names,
                x_names=x_names,
            )
        )

    if _vm_nonempty(getattr(intr_spec, "nernst_vm", None)):
        pieces.append(
            _emit_cuda_vm_fn(
                f"{fn_prefix}_nernst",
                intr_spec.nernst_vm,
                base_params=["double x"],
                dep_name="x",
                s_name="x",
                gate_names=gate_names,
                x_names=x_names,
            )
        )

    for mod_idx, mod in enumerate(getattr(intr_spec, "modulations", ())):
        if _vm_nonempty(getattr(mod, "mod_vm", None)):
            pieces.append(
                _emit_cuda_vm_fn(
                    f"{fn_prefix}_mod_{mod_idx}",
                    mod.mod_vm,
                    base_params=["double dep"],
                    dep_name="dep",
                    s_name="dep",
                    gate_names=gate_names,
                    x_names=x_names,
                )
            )

    if not pieces:
        raise HHEquationError(
            source=repr(intr_spec),
            stderr="intracellular spec has no CUDA-emittable VM expressions",
        )

    return "\n\n".join(pieces)


def compile_model_cuda(model_spec, fn_prefix: str = "model") -> str:
    """
    Emit CUDA helper code for all VM-backed gate and intracellular pieces in a model.

    The generated source is a readable inspection tool today and matches the
    real custom-gate / intracellular programs that task 17.9 runs on CUDA.
    """
    model_prefix = _sanitize_cuda_ident(fn_prefix, "model")
    gate_names = [getattr(g, "name", "") or f"gate_{idx}" for idx, g in enumerate(model_spec.gates)]
    x_names = [getattr(ic, "name", "") or f"X_{idx}" for idx, ic in enumerate(model_spec.intracellular)]

    pieces: list[str] = []
    for idx, gate_spec in enumerate(model_spec.gates):
        pieces.append(
            compile_gate_cuda(
                gate_spec,
                f"{model_prefix}_gate_{_sanitize_cuda_ident(gate_names[idx], f'gate_{idx}')}",
                gate_names=gate_names,
                x_names=x_names,
            )
        )
    for idx, intr_spec in enumerate(model_spec.intracellular):
        try:
            pieces.append(
                compile_intracellular_cuda(
                    intr_spec,
                    f"{model_prefix}_x_{_sanitize_cuda_ident(x_names[idx], f'X_{idx}')}",
                    gate_names=gate_names,
                    x_names=x_names,
                )
            )
        except HHEquationError:
            continue

    if not pieces:
        raise HHEquationError(
            source=repr(model_spec),
            stderr="model has no CUDA-emittable gate or intracellular expressions",
        )

    return "\n\n".join(pieces)


# =============================================================================
# Bytecode VM compiler — converts SymPy AST to VmExpr (no C++ compilation)
# =============================================================================

def _emit_vm(expr, dep_sym, vm, extra_syms=None) -> None:
    """Emit VmInstructions for ``expr`` into ``vm`` (postfix/RPN order).

    Parameters
    ----------
    expr       : sympy expression
    dep_sym    : the primary dependent symbol → PUSH_DEP
    vm         : VmExpr being built
    extra_syms : dict mapping additional sympy.Symbol → VmOp or (VmOp, operand) tuple
                 e.g. {S: VmOp.PUSH_S} or {DA: (VmOp.PUSH_X, 1)}
    """
    from hodgkin_huxley._core import VmOp
    if extra_syms and isinstance(expr, sympy.Symbol) and expr in extra_syms:
        val = extra_syms[expr]
        if isinstance(val, tuple):
            vm.add_instruction(val[0], val[1])
        else:
            vm.add_instruction(val)
        return
    if expr == dep_sym:
        vm.add_instruction(VmOp.PUSH_DEP)
        return
    if expr.is_number:
        idx = vm.add_constant(float(expr))
        vm.add_instruction(VmOp.PUSH_CONST, idx)
        return
    if isinstance(expr, sympy.Add):
        args = list(expr.args)
        _emit_vm(args[0], dep_sym, vm, extra_syms)
        for a in args[1:]:
            _emit_vm(a, dep_sym, vm, extra_syms)
            vm.add_instruction(VmOp.ADD)
        return
    if isinstance(expr, sympy.Mul):
        if sympy.Integer(-1) in expr.args:
            rest = sympy.Mul(*[a for a in expr.args if a != sympy.Integer(-1)])
            _emit_vm(rest, dep_sym, vm, extra_syms)
            vm.add_instruction(VmOp.NEG)
            return
        args = list(expr.args)
        _emit_vm(args[0], dep_sym, vm, extra_syms)
        for a in args[1:]:
            _emit_vm(a, dep_sym, vm, extra_syms)
            vm.add_instruction(VmOp.MUL)
        return
    if isinstance(expr, sympy.Pow):
        base, exp_val = expr.args
        if exp_val == sympy.Rational(1, 2):
            _emit_vm(base, dep_sym, vm, extra_syms)
            vm.add_instruction(VmOp.POW_HALF)
            return
        if exp_val == sympy.Integer(-1):
            _emit_vm(base, dep_sym, vm, extra_syms)
            vm.add_instruction(VmOp.RCP)
            return
        if isinstance(exp_val, sympy.Integer):
            _emit_vm(base, dep_sym, vm, extra_syms)
            vm.add_instruction(VmOp.POW_INT, int(exp_val))
            return
        _emit_vm(base, dep_sym, vm, extra_syms)
        _emit_vm(exp_val, dep_sym, vm, extra_syms)
        vm.add_instruction(VmOp.POW_GEN)
        return
    _UNARY = {
        sympy.exp:  VmOp.EXP,
        sympy.log:  VmOp.LOG,
        sympy.tanh: VmOp.TANH,
        sympy.sin:  VmOp.SIN,
        sympy.cos:  VmOp.COS,
        sympy.Abs:  VmOp.ABS,
    }
    # Note: sympy.sqrt(x) produces Pow(x, Rational(1,2)) which is handled above
    # by the POW_HALF case; sympy.sqrt is a function, not a class, so omitted here.
    for cls, op in _UNARY.items():
        if isinstance(expr, cls):
            _emit_vm(expr.args[0], dep_sym, vm, extra_syms)
            vm.add_instruction(op)
            return
    raise HHEquationError(
        source=str(expr),
        stderr=f"VM compiler: unsupported SymPy node {type(expr).__name__!r}: {expr}",
    )


def compile_to_vm_bytecode(expr, dep_sym=None, extra_syms=None):
    """Convert a SymPy expression to a VmExpr for the pre-compiled C++ VM.

    Parameters
    ----------
    expr       : sympy expression
    dep_sym    : which symbol is the primary dependent variable (default: auto-detect V or Ca)
    extra_syms : dict mapping additional sympy.Symbol → VmOp, e.g. ``{S: VmOp.PUSH_S}``
                 for kinetic synapse expressions that depend on both V and S.

    Returns
    -------
    VmExpr  : bytecode program ready to be stored on a GateSpec or KineticSynapseSpec
    """
    from hodgkin_huxley._core import VmExpr as _VmExpr
    if dep_sym is None:
        free = expr.free_symbols
        dep_sym = Ca if (Ca in free and V not in free) else V
    vm = _VmExpr()
    _emit_vm(expr, dep_sym, vm, extra_syms)
    return vm


# =============================================================================
# Gate-product fast-path extractor
# =============================================================================

def _try_extract_gate_pairs(expr, gate_name_to_idx: dict):
    """Try to express *expr* as the legacy ``[(gate_idx, power)]`` list.

    Returns a list of ``(int, int)`` pairs if *expr* is a pure product of
    integer-power gate symbols (the common HH case: ``m³·h``, ``n⁴``, etc.).
    Returns ``None`` if the expression is more complex and needs the VM.
    """
    import sympy

    def _as_pair(e):
        """Return (idx, power) if *e* is ``gate_sym`` or ``gate_sym**n``, else None."""
        if isinstance(e, sympy.Symbol):
            idx = gate_name_to_idx.get(str(e))
            if idx is not None:
                return (idx, 1)
        if isinstance(e, sympy.Pow):
            base, exp = e.args
            if (isinstance(base, sympy.Symbol)
                    and isinstance(exp, sympy.Integer)
                    and int(exp) > 0):
                idx = gate_name_to_idx.get(str(base))
                if idx is not None:
                    return (idx, int(exp))
        return None

    # Single gate or gate**n
    pair = _as_pair(expr)
    if pair is not None:
        return [pair]

    # Product of gates (and/or powers)
    if isinstance(expr, sympy.Mul):
        pairs = []
        for arg in expr.args:
            pair = _as_pair(arg)
            if pair is None:
                return None  # numeric constant or non-gate term → need VM
            pairs.append(pair)
        return pairs

    return None


# =============================================================================
# Gate-product VM compiler — compile SymPy expressions over gate symbols
# =============================================================================

def _emit_gate_vm(expr, gate_name_to_idx: dict, vm) -> None:
    """Emit VmInstructions for a gate product expression (postfix/RPN order).

    Handles all arithmetic nodes (Add, Mul, Pow) and gate symbols.
    Gate symbols are ``Symbol("gate_<name>")`` as produced by :func:`gate`.
    """
    import sympy
    from hodgkin_huxley._core import VmOp
    # Gate symbol
    if isinstance(expr, sympy.Symbol):
        name = str(expr)
        if name in gate_name_to_idx:
            vm.add_instruction(VmOp.PUSH_GATE, gate_name_to_idx[name])
            return
        raise HHEquationError(
            source=name,
            stderr=(
                f"Gate VM compiler: unknown gate symbol {name!r}. "
                f"Known gates: {list(gate_name_to_idx)}"
            ),
        )
    # Numeric constant
    if expr.is_number:
        vm.add_instruction(VmOp.PUSH_CONST, vm.add_constant(float(expr)))
        return
    # Addition
    if isinstance(expr, sympy.Add):
        args = list(expr.args)
        _emit_gate_vm(args[0], gate_name_to_idx, vm)
        for a in args[1:]:
            _emit_gate_vm(a, gate_name_to_idx, vm)
            vm.add_instruction(VmOp.ADD)
        return
    # Multiplication (detect negation pattern -1 * x → NEG)
    if isinstance(expr, sympy.Mul):
        if sympy.Integer(-1) in expr.args:
            rest = sympy.Mul(*[a for a in expr.args if a != sympy.Integer(-1)])
            _emit_gate_vm(rest, gate_name_to_idx, vm)
            vm.add_instruction(VmOp.NEG)
            return
        args = list(expr.args)
        _emit_gate_vm(args[0], gate_name_to_idx, vm)
        for a in args[1:]:
            _emit_gate_vm(a, gate_name_to_idx, vm)
            vm.add_instruction(VmOp.MUL)
        return
    # Powers
    if isinstance(expr, sympy.Pow):
        base, exp_val = expr.args
        if exp_val == sympy.Rational(1, 2):
            _emit_gate_vm(base, gate_name_to_idx, vm)
            vm.add_instruction(VmOp.POW_HALF)
            return
        if exp_val == sympy.Integer(-1):
            _emit_gate_vm(base, gate_name_to_idx, vm)
            vm.add_instruction(VmOp.RCP)
            return
        if isinstance(exp_val, sympy.Integer):
            _emit_gate_vm(base, gate_name_to_idx, vm)
            vm.add_instruction(VmOp.POW_INT, int(exp_val))
            return
        _emit_gate_vm(base, gate_name_to_idx, vm)
        _emit_gate_vm(exp_val, gate_name_to_idx, vm)
        vm.add_instruction(VmOp.POW_GEN)
        return
    raise HHEquationError(
        source=str(expr),
        stderr=(
            f"Gate VM compiler: unsupported SymPy node "
            f"{type(expr).__name__!r}: {expr}"
        ),
    )


def compile_gate_product_vm(expr, gate_name_to_idx: dict):
    """Convert a SymPy gate product expression to a :class:`VmExpr`.

    Parameters
    ----------
    expr : sympy expression
        An expression over gate symbols (``gate("m")``, ``gate("h")``, etc.)
        and numeric constants only.  All free symbols must appear as keys in
        *gate_name_to_idx*.
    gate_name_to_idx : dict[str, int]
        Maps symbol name (e.g. ``"gate_m"``) to gate index in the model spec.

    Returns
    -------
    VmExpr
        Bytecode program stored on :attr:`ChannelSpec.gate_product_vm`.
    """
    from hodgkin_huxley._core import VmExpr as _VmExpr
    vm = _VmExpr()
    _emit_gate_vm(expr, gate_name_to_idx, vm)
    return vm
