"""
hodgkin_huxley._codegen
=======================

SymPy expression compilation infrastructure for the hodgkin-huxley library.

Provides:
  - Pre-defined SymPy symbols (V, Ca, V_pre, V_post, S, ...)
  - EigenPrinter  — generates Eigen C++ from SymPy (vectorized, SIMD)
  - CUDAPrinter   — stub placeholder (task17)
  - Pattern matching catalog (10 forms: 6 tau + 4 rate + Boltzmann)
  - TaggedExpr    — SymPy expr with pre-matched parameter metadata
  - JITCache / jit_compile — disk-cached g++ compilation pipeline
  - HHEquationError — raised when JIT compilation fails

Insertion points for future tasks:
  - task17: replace CUDAPrinter stub with full nvcc pipeline
  - task13 extension: add more patterns to PATTERN_CATALOG as new forms are added
"""

from __future__ import annotations

import ctypes
import hashlib
import math as _math
import os
import pathlib
import subprocess
import sys
import tempfile
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
# CUDAPrinter (stub — task17)
# =============================================================================

class CUDAPrinter(CodePrinter):
    """
    Placeholder CUDA printer.

    # TODO task17: full CUDA JIT pipeline not implemented.
    Generates a minimal __device__ function stub using sympy.ccode() for the
    expression body. No Eigen — GPU parallelism is across threads, not SIMD.
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

    def doprint(self, expr, assign_to=None) -> str:  # type: ignore[override]
        # TODO task17: full CUDA JIT pipeline not implemented.
        from sympy.printing.c import ccode
        # Use standard scalar C code
        body = ccode(expr)
        return (
            "// TODO task17: full CUDA JIT pipeline not implemented\n"
            "__device__ static inline double fn(double V) {\n"
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
# JIT preambles
# =============================================================================

# Inline fast_exp for Eigen ArrayXd — mirrors kinetics.hpp::fast_exp exactly.
# Uses range reduction by 32 + degree-7 Horner polynomial + 5 squarings.
_FAST_EXP_VEC_IMPL = """\
static inline Eigen::ArrayXd hh_fast_exp(const Eigen::ArrayXd& src) {
    Eigen::ArrayXd r = src * (1.0 / 32.0);
    Eigen::ArrayXd y = r * (1.0 / 5040.0) + (1.0 / 720.0);
    y = y * r + (1.0 / 120.0);
    y = y * r + (1.0 / 24.0);
    y = y * r + (1.0 / 6.0);
    y = y * r + 0.5;
    y = y * r + 1.0;
    y = y * r + 1.0;
    y *= y; y *= y; y *= y; y *= y; y *= y;  // 5 squarings: (exp(x/32))^32 = exp(x)
    return y;
}
"""

# Scalar version for scalar2 JIT functions
_FAST_EXP_SCALAR_IMPL = """\
static inline double hh_fast_exp(double x) {
    double r = x * (1.0 / 32.0);
    double y = r * (1.0 / 5040.0) + (1.0 / 720.0);
    y = y * r + (1.0 / 120.0);
    y = y * r + (1.0 / 24.0);
    y = y * r + (1.0 / 6.0);
    y = y * r + 0.5;
    y = y * r + 1.0;
    y = y * r + 1.0;
    y *= y; y *= y; y *= y; y *= y; y *= y;
    return y;
}
"""

_VEC_PREAMBLE_TEMPLATE = """\
#include <Eigen/Core>
#include <cstdint>
#include <algorithm>
using Eigen::ArrayXd;

{fast_exp_impl}
"""

_SCALAR2_PREAMBLE_TEMPLATE = """\
#include <cmath>
#include <algorithm>

{fast_exp_impl}
"""


def _generate_vec_src(expr, hash_str: str, use_fast_exp: bool) -> str:
    """Generate a complete C++ source for a vectorized gate function."""
    fast_exp_impl = _FAST_EXP_VEC_IMPL if use_fast_exp else ""
    preamble = _VEC_PREAMBLE_TEMPLATE.format(fast_exp_impl=fast_exp_impl)
    printer = EigenPrinter(use_fast_exp=use_fast_exp, scalar_mode=False)
    body = printer.doprint(expr)
    return (
        preamble +
        f"extern \"C\" void fn_{hash_str}(const double* V_in, double* out, int n) {{\n"
        f"    Eigen::Map<const ArrayXd> V_(V_in, n);\n"
        f"    Eigen::Map<ArrayXd> out_(out, n);\n"
        f"    out_ = {body};\n"
        f"}}\n"
    )


def _generate_scalar2_src(expr, hash_str: str, use_fast_exp: bool,
                           arg0_name: str = "V_pre_",
                           arg1_name: str = "S_") -> str:
    """Generate a scalar2 C++ source: double fn(double arg0, double arg1)."""
    fast_exp_impl = _FAST_EXP_SCALAR_IMPL if use_fast_exp else ""
    preamble = _SCALAR2_PREAMBLE_TEMPLATE.format(fast_exp_impl=fast_exp_impl)
    printer = EigenPrinter(use_fast_exp=use_fast_exp, scalar_mode=True)
    body = printer.doprint(expr)
    return (
        preamble +
        f"extern \"C\" double fn_{hash_str}(double {arg0_name}, double {arg1_name}) {{\n"
        f"    return {body};\n"
        f"}}\n"
    )


# =============================================================================
# Eigen header discovery
# =============================================================================

def _find_eigen_include() -> str:
    """
    Return the path to Eigen headers, or raise HHEquationError if not found.
    """
    # 1. Environment variable
    env_path = os.environ.get("EIGEN3_INCLUDE_DIR")
    if env_path and pathlib.Path(env_path, "Eigen", "Core").exists():
        return env_path

    # 2. pkg-config
    try:
        result = subprocess.run(
            ["pkg-config", "--cflags", "eigen3"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for flag in result.stdout.split():
                if flag.startswith("-I"):
                    path = flag[2:]
                    if pathlib.Path(path, "Eigen", "Core").exists():
                        return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Common install locations
    candidates = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
        "/usr/include",
        "/usr/local/include",
    ]
    for path in candidates:
        if pathlib.Path(path, "Eigen", "Core").exists():
            return path

    # 4. Python package (scikit-build or similar may bundle Eigen)
    try:
        import importlib.resources as ilr
        pkg_path = pathlib.Path(str(ilr.files("hodgkin_huxley")) if hasattr(ilr, "files") else "")
        for parent in [pkg_path, pkg_path.parent, pkg_path.parent.parent]:
            for sub in ["eigen3", "Eigen3", "eigen"]:
                candidate = parent / sub
                if (candidate / "Eigen" / "Core").exists():
                    return str(candidate)
    except Exception:
        pass

    raise HHEquationError(
        source="",
        stderr=(
            "Could not locate Eigen3 headers. Set EIGEN3_INCLUDE_DIR environment variable "
            "to the directory containing Eigen/Core, or install Eigen3 via your package "
            "manager (e.g., `apt install libeigen3-dev` or `brew install eigen`)."
        )
    )


# =============================================================================
# HHEquationError
# =============================================================================

class HHEquationError(Exception):
    """
    Raised when JIT compilation of a SymPy expression fails.

    Attributes
    ----------
    source : str
        The generated C++ source that was passed to the compiler.
    stderr : str
        Compiler stderr output (or other error message).
    """
    def __init__(self, source: str, stderr: str):
        self.source = source
        self.stderr = stderr
        super().__init__(
            f"JIT compilation failed.\n"
            f"--- compiler stderr ---\n{stderr}\n"
            f"--- generated source ---\n{source}"
        )


# =============================================================================
# JIT cache
# =============================================================================

_DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "hodgkin_huxley"


class JITCache:
    """
    Disk-backed cache for JIT-compiled SymPy expressions.

    Expressions are hashed by their SymPy `srepr` string + fn_type. Compiled
    shared libraries are stored as ``<hash>.so`` under `cache_dir`. Function
    pointers are kept in-process in ``_loaded`` so that reloads are avoided
    across multiple simulate() calls within the same process.
    """

    def __init__(self, cache_dir: pathlib.Path | str | None = None):
        self.cache_dir = pathlib.Path(cache_dir or _DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, int] = {}  # hash_str → raw function pointer (int)

    def compile(
        self,
        expr,
        fn_type: str = "vec",
        use_fast_exp: bool = True,
    ) -> int:
        """
        Compile a SymPy expression to a shared library and return the raw
        function pointer as a Python ``int``.

        Parameters
        ----------
        expr : sympy.Expr
            The expression to compile.
        fn_type : str
            ``"vec"``     — void(*)(const double* in, double* out, int n)
            ``"scalar2"`` — double(*)(double arg0, double arg1)
        use_fast_exp : bool
            When True, ``exp(x)`` uses the fast_exp approximation from
            ``kinetics.hpp`` (range-reduction + degree-7 Taylor).
        """
        hash_str = self._hash(expr, fn_type)
        if hash_str in self._loaded:
            return self._loaded[hash_str]

        so_path = self.cache_dir / f"{hash_str}.so"
        if not so_path.exists():
            self._compile_to_disk(expr, fn_type, use_fast_exp, hash_str, so_path)

        ptr = self._load_fn(so_path, hash_str)
        self._loaded[hash_str] = ptr
        return ptr

    @staticmethod
    def _hash(expr, fn_type: str) -> str:
        key = sympy.srepr(expr) + "|" + fn_type
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _compile_to_disk(self, expr, fn_type: str, use_fast_exp: bool,
                         hash_str: str, so_path: pathlib.Path) -> None:
        if fn_type == "vec":
            src = _generate_vec_src(expr, hash_str, use_fast_exp)
        elif fn_type == "scalar2":
            src = _generate_scalar2_src(expr, hash_str, use_fast_exp)
        else:
            raise ValueError(f"Unknown fn_type: {fn_type!r}. Expected 'vec' or 'scalar2'.")

        cpp_path = self.cache_dir / f"{hash_str}.cpp"
        cpp_path.write_text(src)

        eigen_include = _find_eigen_include()

        cmd = [
            "g++", "-O3", "-fPIC", "-shared", "-std=c++17",
            f"-I{eigen_include}",
            "-o", str(so_path),
            str(cpp_path),
        ]
        if sys.platform != "win32":
            cmd.insert(1, "-march=native")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise HHEquationError(source=src, stderr=result.stderr)

    def _load_fn(self, so_path: pathlib.Path, hash_str: str) -> int:
        lib = ctypes.CDLL(str(so_path))
        fn_name = f"fn_{hash_str}"
        fn = getattr(lib, fn_name)
        # Cast to void* to extract the raw address as a Python int
        ptr = ctypes.cast(fn, ctypes.c_void_p).value
        if ptr is None:
            raise HHEquationError(
                source="",
                stderr=f"Could not extract function pointer for symbol '{fn_name}' from {so_path}"
            )
        return ptr


# Module-level singleton cache — shared across all compile() calls
_GLOBAL_JIT_CACHE = JITCache()


def jit_compile(
    expr,
    fn_type: str = "vec",
    use_fast_exp: bool = True,
    cache: JITCache | None = None,
) -> int:
    """
    Compile ``expr`` to a shared library and return its function pointer.

    Convenience wrapper around the global JITCache. See ``JITCache.compile``
    for parameter documentation.
    """
    c = cache or _GLOBAL_JIT_CACHE
    return c.compile(expr, fn_type=fn_type, use_fast_exp=use_fast_exp)


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
    extra_syms : dict mapping additional sympy.Symbol → VmOp (e.g. {S: VmOp.PUSH_S})
    """
    from hodgkin_huxley._core import VmOp
    if extra_syms and isinstance(expr, sympy.Symbol) and expr in extra_syms:
        vm.add_instruction(extra_syms[expr])
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
