"""
hodgkin_huxley._equations — SymPy equation helpers, gate/neuron builders, synapse builders.

Public exports: Boltzmann, Tau, RateFunc, GateSpec, NeuronModel,
                SynapseModel, KineticSynapseModel (deprecated alias), KineticSynapseSpec
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as _sympy

from .._core import (
    BoltzmannParams as _BoltzmannParams,
    CalciumSpec,
    ChannelSpec,
    GateDependency as _GateDependency,
    IntracellularSpec as _IntracellularSpec,
    IntracellularUpdateForm as _IntracellularUpdateForm,
    IntracellularModulation as _IntracellularModulation,
    IntracellularModulationTarget as _IcModTarget,
    GateSpec as _CoreGateSpec,
    GateUpdateForm as _GateUpdateForm,
    KineticCurrentForm as _KineticCurrentForm,   # alias for SynapseCurrentForm
    KineticSynapseSpec as _KineticSynapseSpec,   # alias for SynapseSpec
    KineticUpdateForm as _KineticUpdateForm,     # alias for SynapseUpdateForm
    NeuronModelSpec,
    RateFuncForm as _RateFuncForm,
    RateFuncParams as _RateFuncParams,
    SynapseCurrentForm as _SynapseCurrentForm,
    SynapseSpec as _SynapseSpec,
    SynapseUpdateForm as _SynapseUpdateForm,
    TauForm as _TauForm,
    TauParams as _TauParams,
    VmOp as _VmOp,
)
from .._codegen import (
    Ca,
    DA,
    cAMP,
    IP3,
    NO,
    X_ic,
    I_source as _I_source_sym,
    TaggedExpr,
    V,
    compile_to_vm_bytecode,
    gate,
    substance,
    try_pattern_match,
    x as _x_sym,
)


# =============================================================================
# Equation helper classes
# =============================================================================


class Boltzmann:
    """Helper to create BoltzmannParams / TaggedExpr for gate steady-state kinetics."""

    def __init__(self, v_half: float, k: float):
        self.v_half = v_half
        self.k = k

    def to_params(self) -> "_BoltzmannParams":
        p = _BoltzmannParams()
        p.v_half = self.v_half
        p.k = self.k
        return p

    def to_tagged(self) -> "TaggedExpr":
        """Return a TaggedExpr carrying the SymPy form and pre-matched params."""
        expr = 1 / (1 + _sympy.exp(-(V - self.v_half) / self.k))
        return TaggedExpr(expr=expr, params=self.to_params(), form=None)


class Tau:
    """Helper to create TauParams wrapped in TaggedExpr (carries SymPy expression + params)."""

    @staticmethod
    def constant(value: float) -> "TaggedExpr":
        t = _TauParams()
        t.form = _TauForm.CONSTANT
        t.set_param(0, value)
        return TaggedExpr(expr=_sympy.Float(value), params=t, form=_TauForm.CONSTANT)

    @staticmethod
    def boltzmann(base: float, amp: float, v_half: float, k: float) -> "TaggedExpr":
        t = _TauParams()
        t.form = _TauForm.BOLTZMANN
        t.set_param(0, base)
        t.set_param(1, amp)
        t.set_param(2, v_half)
        t.set_param(3, k)
        expr = base + amp / (1 + _sympy.exp(-(V - v_half) / k))
        return TaggedExpr(expr=expr, params=t, form=_TauForm.BOLTZMANN)

    @staticmethod
    def scaled_exp(scale: float, v_half: float, k: float) -> "TaggedExpr":
        t = _TauParams()
        t.form = _TauForm.SCALED_EXP
        t.set_param(0, scale)
        t.set_param(1, v_half)
        t.set_param(2, k)
        expr = scale / _sympy.cosh((V - v_half) / (2 * k))
        return TaggedExpr(expr=expr, params=t, form=_TauForm.SCALED_EXP)

    @staticmethod
    def double_exp_sum(base: float, amp: float, v1: float, s1: float,
                       v2: float, s2: float) -> "TaggedExpr":
        t = _TauParams()
        t.form = _TauForm.DOUBLE_EXP_SUM
        t.set_param(0, base)
        t.set_param(1, amp)
        t.set_param(2, v1)
        t.set_param(3, s1)
        t.set_param(5, v2)
        t.set_param(6, s2)
        expr = base + amp / (
            _sympy.exp((V + v1) / s1) + _sympy.exp(-(V + v2) / s2)
        )
        return TaggedExpr(expr=expr, params=t, form=_TauForm.DOUBLE_EXP_SUM)


class RateFunc:
    """Helper to create RateFuncParams wrapped in TaggedExpr."""

    @staticmethod
    def linear_over_exp(A: float, B: float, C: float) -> "TaggedExpr":
        r = _RateFuncParams()
        r.form = _RateFuncForm.LINEAR_OVER_EXP
        r.A = A; r.B = B; r.C = C
        expr = A * (V + B) / (_sympy.exp((V + B) / C) - 1)
        return TaggedExpr(expr=expr, params=r, form=_RateFuncForm.LINEAR_OVER_EXP)

    @staticmethod
    def exp_decay(A: float, B: float, C: float) -> "TaggedExpr":
        r = _RateFuncParams()
        r.form = _RateFuncForm.EXP_DECAY
        r.A = A; r.B = B; r.C = C
        expr = A * _sympy.exp((V + B) / C)
        return TaggedExpr(expr=expr, params=r, form=_RateFuncForm.EXP_DECAY)

    @staticmethod
    def linear_over_expm1(A: float, B: float, C: float) -> "TaggedExpr":
        r = _RateFuncParams()
        r.form = _RateFuncForm.LINEAR_OVER_EXPM1
        r.A = A; r.B = B; r.C = C
        expr = A * (V + B) / (1 - _sympy.exp(-(V + B) / C))
        return TaggedExpr(expr=expr, params=r, form=_RateFuncForm.LINEAR_OVER_EXPM1)

    @staticmethod
    def sigmoid(A: float, B: float, C: float) -> "TaggedExpr":
        r = _RateFuncParams()
        r.form = _RateFuncForm.SIGMOID
        r.A = A; r.B = B; r.C = C
        expr = A / (1 + _sympy.exp((V + B) / C))
        return TaggedExpr(expr=expr, params=r, form=_RateFuncForm.SIGMOID)


# =============================================================================
# Gate / NeuronModel builder helpers
# =============================================================================


def _normalize_gate_input(raw):
    """
    Normalize a gate parameter input to (params, expr) tuple.

    Returns
    -------
    (params, expr) where params is a struct (_BoltzmannParams | _TauParams |
    _RateFuncParams) or None, and expr is a SymPy expression or None.
    """
    if raw is None:
        return None, None
    if isinstance(raw, Boltzmann):
        return raw.to_params(), 1 / (1 + _sympy.exp(-(V - raw.v_half) / raw.k))
    if isinstance(raw, TaggedExpr):
        return raw.params, raw.expr
    if isinstance(raw, (_BoltzmannParams, _TauParams, _RateFuncParams)):
        return raw, None  # struct-only, no SymPy expression available
    if isinstance(raw, (int, float)):
        # Scalar tau shorthand: constant TauParams
        tp = _TauParams()
        tp.form = _TauForm.CONSTANT
        tp.set_param(0, float(raw))
        return tp, _sympy.Float(raw)
    if isinstance(raw, _sympy.Basic):
        # Use Ca as the matching symbol when the expression depends on Ca (not V)
        dep_sym = Ca if (Ca in raw.free_symbols and V not in raw.free_symbols) else V
        params, _ = try_pattern_match(raw, dep_sym=dep_sym)
        return params, raw  # params may be None if no match
    return raw, None


def _has_calcium_dep(*exprs) -> bool:
    """Return True if any SymPy expression in exprs contains the Ca symbol."""
    for e in exprs:
        if isinstance(e, _sympy.Basic) and Ca in e.free_symbols:
            return True
        if isinstance(e, TaggedExpr) and isinstance(e.expr, _sympy.Basic):
            if Ca in e.expr.free_symbols:
                return True
    return False


_GATE_SPEC_UNSET = object()  # sentinel for GateSpec factory


def GateSpec(  # noqa: N802  (intentional CapWords to match C++ type / old API)
    name: str = "",
    *,
    inf=_GATE_SPEC_UNSET,
    tau=_GATE_SPEC_UNSET,
    alpha=_GATE_SPEC_UNSET,
    beta=_GATE_SPEC_UNSET,
    scale: float = 1.0,
    initial_value: float = 0.0,
    dependency: str = "auto",
    update_form: str = "",
    derived_source_gate: int = -1,
    derived_a: float = 1.0,
    derived_b: float = 0.0,
    derived_c: float = 1.0,
) -> "_CoreGateSpec":
    """
    Create a gate specification.

    When called with no expression arguments (``GateSpec()``) returns a blank
    C++ ``_core.GateSpec`` exactly as before, preserving backward compatibility
    with code that sets attributes directly::

        g = GateSpec()
        g.name = "m"
        g.update_form = GateUpdateForm.INF_TAU
        g.inf = Boltzmann(-40, 8).to_params()

    When called with ``inf``, ``tau``, ``alpha``, or ``beta`` keyword arguments
    (which may be ``Boltzmann``, ``TaggedExpr``, SymPy expressions, ``float``,
    or raw C++ struct instances), compiles and returns a configured
    ``_core.GateSpec``::

        g = GateSpec("m", inf=Boltzmann(-40, 8), tau=Tau.constant(0.5))
        g = GateSpec("m", inf=1/(1+sympy.exp(-(V+40)/8)), tau=some_novel_expr)

    Parameters
    ----------
    name : str
        Gate name.
    inf : Boltzmann | TaggedExpr | sympy.Expr | _BoltzmannParams, optional
    tau : TaggedExpr | sympy.Expr | _TauParams | float, optional
    alpha, beta : TaggedExpr | sympy.Expr | _RateFuncParams, optional
    scale : float
    initial_value : float
    dependency : str
        ``"voltage"`` (default), ``"calcium"``, or ``"auto"`` (auto-detect
        from SymPy free symbols).
    update_form : str
        ``"inf_tau"``, ``"alpha_beta"``, ``"instant"``, ``"derived"``, or
        ``"custom_expr"``.  Auto-inferred from provided params when omitted.
    """
    # No expression args → return blank C++ GateSpec for manual configuration
    _no_expr = (
        inf is _GATE_SPEC_UNSET
        and tau is _GATE_SPEC_UNSET
        and alpha is _GATE_SPEC_UNSET
        and beta is _GATE_SPEC_UNSET
    )
    if _no_expr and not name:
        return _CoreGateSpec()

    # Expression args provided: build and compile into a C++ GateSpec
    inf   = None if inf   is _GATE_SPEC_UNSET else inf
    tau   = None if tau   is _GATE_SPEC_UNSET else tau
    alpha = None if alpha is _GATE_SPEC_UNSET else alpha
    beta  = None if beta  is _GATE_SPEC_UNSET else beta

    # Normalize all inputs
    inf_p, inf_e    = _normalize_gate_input(inf)
    tau_p, tau_e    = _normalize_gate_input(tau)
    alpha_p, alpha_e = _normalize_gate_input(alpha)
    beta_p, beta_e  = _normalize_gate_input(beta)

    needs_vm = (
        (inf_e is not None and inf_p is None)
        or (tau_e is not None and tau_p is None)
        or (alpha_e is not None and alpha_p is None)
        or (beta_e is not None and beta_p is None)
    )

    g = _CoreGateSpec()
    g.name = name
    g.scale = scale
    g.initial_value = initial_value
    g.derived_source_gate = derived_source_gate
    g.derived_a = derived_a
    g.derived_b = derived_b
    g.derived_c = derived_c

    dep = dependency.lower()
    if dep == "auto":
        has_ca = _has_calcium_dep(inf, tau, alpha, beta)
        g.dependency = _GateDependency.INTRACELLULAR if has_ca else _GateDependency.VOLTAGE
    else:
        g.dependency = (
            _GateDependency.INTRACELLULAR if dep in ("calcium", "intracellular")
            else _GateDependency.VOLTAGE
        )

    if needs_vm:
        g.update_form = _GateUpdateForm.CUSTOM_EXPR
        _dep = Ca if _has_calcium_dep(inf, tau, alpha, beta) else V
        for _e, vm_attr in [
            (inf_e,   "inf_vm"),
            (tau_e,   "tau_vm"),
            (alpha_e, "alpha_vm"),
            (beta_e,  "beta_vm"),
        ]:
            if _e is not None:
                dep_sym = V if vm_attr == "tau_vm" else _dep
                setattr(g, vm_attr, compile_to_vm_bytecode(_e, dep_sym=dep_sym))
    else:
        uf_str = update_form.lower() if update_form else ""
        if not uf_str:
            if alpha_p is not None or beta_p is not None:
                uf_str = "alpha_beta"
            elif inf_p is not None and tau_p is not None:
                uf_str = "inf_tau"
            elif inf_p is not None:
                uf_str = "instant"
            elif derived_source_gate >= 0:
                uf_str = "derived"
            else:
                uf_str = "inf_tau"
        uf_map = {
            "inf_tau":     _GateUpdateForm.INF_TAU,
            "alpha_beta":  _GateUpdateForm.ALPHA_BETA,
            "instant":     _GateUpdateForm.INSTANT,
            "derived":     _GateUpdateForm.DERIVED,
            "custom_expr": _GateUpdateForm.CUSTOM_EXPR,
        }
        g.update_form = uf_map.get(uf_str, _GateUpdateForm.INF_TAU)
        if inf_p is not None:
            g.inf = inf_p
        if tau_p is not None:
            g.tau = tau_p
        if alpha_p is not None:
            g.alpha = alpha_p
        if beta_p is not None:
            g.beta = beta_p

    return g


class GateRef:
    """
    Return value of :meth:`NeuronModel.add_gate`.

    Acts as an ``int`` (gate index) for the legacy ``gates=[(idx, power)]``
    API via :meth:`__index__`, and as a SymPy symbol for the ``gating=``
    expression API via arithmetic operators.

    Examples
    --------
    >>> m = NeuronModel("hh")
    >>> m_ref = m.add_gate("m", inf=Boltzmann(-40, 9), tau=Tau.constant(1.0))
    >>> h_ref = m.add_gate("h", inf=Boltzmann(-65, -7), tau=Tau.constant(3.0))
    >>> m.add_channel("Na", g=120.0, E_rev=50.0, gating=m_ref**3 * h_ref)
    """

    def __init__(self, idx: int, name: str):
        self._idx = idx
        self._sym = gate(name)

    def __index__(self) -> int:
        return self._idx

    def __int__(self) -> int:
        return self._idx

    @property
    def sym(self):
        """The underlying SymPy symbol (``Symbol("gate_<name>"``)."""
        return self._sym

    def _sympy_(self):
        """SymPy coercion hook — tells sympify() to treat this as the gate symbol."""
        return self._sym

    def __pow__(self, exp):
        return self._sym ** exp

    def __mul__(self, other):
        return self._sym * (other._sym if isinstance(other, GateRef) else other)

    def __rmul__(self, other):
        return (other._sym if isinstance(other, GateRef) else other) * self._sym

    def __add__(self, other):
        return self._sym + (other._sym if isinstance(other, GateRef) else other)

    def __radd__(self, other):
        return (other._sym if isinstance(other, GateRef) else other) + self._sym

    def __repr__(self) -> str:
        return f"GateRef(idx={self._idx}, sym={self._sym})"


class NeuronModel:
    """
    Ergonomic builder for NeuronModelSpec.

    Examples
    --------
    >>> model = NeuronModel("custom", C_m=1.0, V_init=-65.0)
    >>> model.add_gate("m", update_form="instant", inf=Boltzmann(-37, 7))
    >>> model.add_channel("Leak", g=0.3, E_rev=-54.3)
    >>> spec = model.to_spec()
    """

    def __init__(self, name: str = "custom", C_m: float = 1.0, V_init: float = -65.0):
        self._spec = NeuronModelSpec()
        self._spec.name = name
        self._spec.C_m = C_m
        self._spec.V_init = V_init

    def add_gate(
        self,
        name: str = "",
        update_form: str = "inf_tau",
        dependency: str = "voltage",
        scale: float = 1.0,
        initial_value: float = 0.0,
        inf=None,
        tau=None,
        alpha=None,
        beta=None,
        derived_source_gate: "int | GateRef" = -1,
        derived_a: float = 1.0,
        derived_b: float = 0.0,
        derived_c: float = 1.0,
        gs: "_GateSpec | None" = None,
        expr=None,
    ) -> "GateRef":
        """Add a gate and return its index.

        Accepts a ``GateSpec`` object via ``gs=`` for the SymPy-aware path, or
        the legacy keyword arguments (``inf``, ``tau``, ``alpha``, ``beta``)
        which may now be ``Boltzmann`` / ``TaggedExpr`` / ``sympy.Expr`` /
        raw C++ struct instances.
        """
        if gs is not None:
            # Accept either a pre-built _CoreGateSpec or the result of GateSpec(...)
            # (both return _CoreGateSpec since GateSpec is now a factory function)
            self._spec.gates.append(gs)
            return len(self._spec.gates) - 1

        # Normalize all inputs to (struct_or_None, sympy_expr_or_None)
        inf_p, inf_e    = _normalize_gate_input(inf)
        tau_p, tau_e    = _normalize_gate_input(tau)
        alpha_p, alpha_e = _normalize_gate_input(alpha)
        beta_p, beta_e  = _normalize_gate_input(beta)

        needs_vm = (
            expr is not None
            or (inf_e is not None and inf_p is None)
            or (tau_e is not None and tau_p is None)
            or (alpha_e is not None and alpha_p is None)
            or (beta_e is not None and beta_p is None)
        )

        g = _CoreGateSpec()
        g.name = name
        g.scale = scale
        g.initial_value = initial_value
        g.derived_source_gate = derived_source_gate
        g.derived_a = derived_a
        g.derived_b = derived_b
        g.derived_c = derived_c

        _dep_str = dependency.lower()
        if _dep_str == "auto":
            has_ca = _has_calcium_dep(inf, tau, alpha, beta)
            g.dependency = _GateDependency.INTRACELLULAR if has_ca else _GateDependency.VOLTAGE
        else:
            g.dependency = (
                _GateDependency.INTRACELLULAR if _dep_str in ("calcium", "intracellular")
                else _GateDependency.VOLTAGE
            )

        if needs_vm:
            g.update_form = _GateUpdateForm.CUSTOM_EXPR
            _dep_sym = Ca if g.dependency == _GateDependency.INTRACELLULAR else V
            if expr is not None:
                # Arbitrary ODE: dx/dt = F(x, V).  PUSH_DEP = V (or Ca), PUSH_S = x.
                _extra = {_x_sym: _VmOp.PUSH_S}
                g.dxdt_vm = compile_to_vm_bytecode(expr, dep_sym=_dep_sym, extra_syms=_extra)
            for _e, vm_attr in [
                (inf_e,   "inf_vm"),
                (tau_e,   "tau_vm"),
                (alpha_e, "alpha_vm"),
                (beta_e,  "beta_vm"),
            ]:
                if _e is not None:
                    _ds = V if vm_attr == "tau_vm" else _dep_sym
                    setattr(g, vm_attr, compile_to_vm_bytecode(_e, dep_sym=_ds))
        else:
            # Auto-infer update_form when the caller left it at the default
            # ("inf_tau") but provided alpha/beta, mirroring GateSpec factory.
            uf_str = update_form.lower()
            if uf_str == "inf_tau" and (alpha_p is not None or beta_p is not None):
                uf_str = "alpha_beta"
            elif uf_str == "inf_tau" and inf_p is not None and tau_p is None:
                uf_str = "instant"

            g.update_form = {
                "inf_tau":     _GateUpdateForm.INF_TAU,
                "alpha_beta":  _GateUpdateForm.ALPHA_BETA,
                "instant":     _GateUpdateForm.INSTANT,
                "derived":     _GateUpdateForm.DERIVED,
                "custom_expr": _GateUpdateForm.CUSTOM_EXPR,
            }.get(uf_str, _GateUpdateForm.INF_TAU)
            if inf_p is not None:
                g.inf = inf_p  # type: ignore[assignment]
            if tau_p is not None:
                g.tau = tau_p  # type: ignore[assignment]
            if alpha_p is not None:
                g.alpha = alpha_p  # type: ignore[assignment]
            if beta_p is not None:
                g.beta = beta_p  # type: ignore[assignment]

        self._spec.gates.append(g)
        idx = len(self._spec.gates) - 1
        return GateRef(idx, g.name or str(idx))

    def add_channel(
        self,
        name: str,
        g: float,
        E_rev: float,
        gates: "list[tuple[int, int]] | None" = None,
        gating=None,
        nernst_substance_idx: int = -1,
        is_ahp: bool = False,
        ahp_k1: float = 0.0,
        ahp_substance_idx: int = 0,
        use_calcium_nernst: bool = False,  # deprecated — use nernst_substance_idx=0
    ) -> int:
        """Add a channel and return its index.

        Parameters
        ----------
        gating : SymPy expression or GateRef, optional
            Gating expression over :func:`gate` symbols (e.g. ``m_ref**3 * h_ref``).
            When provided, replaces the ``gates=[(idx, power)]`` integer list.
            The expression is compiled to a :class:`VmExpr` via
            :func:`compile_gate_product_vm` and stored on the channel spec.
        gates : list of (int, int), optional
            Legacy ``(gate_index, power)`` list.  Ignored when *gating* is given.
        nernst_substance_idx : int, optional
            Index of the intracellular substance whose Nernst reversal overrides
            E_rev.  -1 = disabled (default).  0 = first substance (calcium by
            convention when set via ``set_calcium()``).
        ahp_substance_idx : int, optional
            Index of the intracellular substance that drives the AHP current.
            Only used when ``is_ahp=True``.  Default 0 (calcium).
        use_calcium_nernst : bool, optional
            Deprecated.  Equivalent to ``nernst_substance_idx=0``.
        """
        # Resolve deprecated flag
        if use_calcium_nernst and nernst_substance_idx < 0:
            nernst_substance_idx = 0
        ch = ChannelSpec()
        ch.name = name
        ch.g = g
        ch.E_rev = E_rev
        ch.gates = gates or []
        ch.nernst_substance_idx = nernst_substance_idx
        ch.is_ahp = is_ahp
        ch.ahp_k1 = ahp_k1
        ch.ahp_substance_idx = ahp_substance_idx
        if gating is not None:
            from .._codegen import (
                _try_extract_gate_pairs,
                compile_gate_product_vm as _compile_gate_product_vm,
            )
            expr = gating._sym if isinstance(gating, GateRef) else gating
            name_to_idx = {
                f"gate_{gs.name}": i
                for i, gs in enumerate(self._spec.gates)
            }
            pairs = _try_extract_gate_pairs(expr, name_to_idx)
            if pairs is not None:
                ch.gates = pairs          # fast path: in-place pool multiplication
            else:
                ch.gate_product_vm = _compile_gate_product_vm(expr, name_to_idx)
        self._spec.channels.append(ch)
        return len(self._spec.channels) - 1

    def add_leak(self, g: float, E_rev: float) -> int:
        """Add a leak channel (no gates)."""
        return self.add_channel("Leak", g, E_rev)

    def set_calcium(
        self,
        epsilon: float = 1e-4,
        K_Ca: float = 15.0,
        Ca_init: float = 0.1,
        use_nernst: bool = False,
        Ca_o: float = 2000.0,
        source_channels: "list[int] | None" = None,
    ) -> None:
        """Configure calcium dynamics (populates intracellular[0])."""
        ic = _IntracellularSpec()
        ic.name = "Ca"
        ic.initial = Ca_init
        ic.epsilon = epsilon
        ic.k_decay = K_Ca
        ic.source_channels = source_channels or []
        if use_nernst:
            ic.update_form = _IntracellularUpdateForm.DRIVEN_DECAY_NERNST
            ic.nernst_enabled = True
            ic.nernst_Ca_o = Ca_o
        else:
            ic.update_form = _IntracellularUpdateForm.DRIVEN_DECAY
        # Place at index 0 (calcium by convention)
        if len(self._spec.intracellular) == 0:
            self._spec.intracellular.append(ic)
        else:
            self._spec.intracellular[0] = ic

    def to_spec(self) -> NeuronModelSpec:
        """Build and return the NeuronModelSpec."""
        return self._spec

    def __repr__(self) -> str:
        return (
            f"<NeuronModel '{self._spec.name}' "
            f"gates={len(self._spec.gates)} "
            f"channels={len(self._spec.channels)}>"
        )


# =============================================================================
# Unified synapse builder
# =============================================================================

import math as _math


def _try_match_exp_decay(dS_dt, S_sym):
    """Return tau_S if dS_dt = -k*S (k>0), else None."""
    import sympy as _sp
    expr = _sp.sympify(dS_dt)
    if expr.free_symbols - {S_sym}:
        return None
    coeffs = _sp.Poly(expr, S_sym).all_coeffs()
    if len(coeffs) != 2 or coeffs[1] != 0:
        return None
    k = float(-coeffs[0])
    if k > 0:
        return 1.0 / k
    return None


def _try_match_alpha_func(dS_dt, dA_dt, S_sym, A_sym):
    """Return tau_A if both match the 2-var alpha ODE."""
    import sympy as _sp
    dS = _sp.sympify(dS_dt)
    dA = _sp.sympify(dA_dt)
    if dS.free_symbols - {S_sym, A_sym} or dA.free_symbols - {A_sym}:
        return None
    # dA = -k*A
    dA_c = _sp.Poly(dA, A_sym).all_coeffs()
    if len(dA_c) != 2 or dA_c[1] != 0:
        return None
    k = float(-dA_c[0])
    if k <= 0:
        return None
    # dS = k*(A - S) = k*A - k*S
    dS_exp = k * A_sym - k * S_sym
    if _sp.simplify(dS - dS_exp) == 0:
        return 1.0 / k
    return None


def _try_match_double_exp(dS_dt, dA_dt, S_sym, A_sym):
    """Return (tau_S, tau_A) for double-exp if both match -k_i*var pattern."""
    import sympy as _sp
    dS = _sp.sympify(dS_dt)
    dA = _sp.sympify(dA_dt)
    if dS.free_symbols - {S_sym} or dA.free_symbols - {A_sym}:
        return None
    dS_c = _sp.Poly(dS, S_sym).all_coeffs()
    dA_c = _sp.Poly(dA, A_sym).all_coeffs()
    if len(dS_c) != 2 or dS_c[1] != 0:
        return None
    if len(dA_c) != 2 or dA_c[1] != 0:
        return None
    k1 = float(-dS_c[0])
    k2 = float(-dA_c[0])
    if k1 > 0 and k2 > 0 and abs(k1 - k2) > 1e-12:
        return 1.0 / k1, 1.0 / k2
    return None


class SynapseModel:
    """
    Unified builder for SynapseSpec — covers all synapse forms.

    Named constructors:
        SynapseModel.exponential(name, *, tau, g, E_syn)
        SynapseModel.alpha_function(name, *, tau, g, E_syn)
        SynapseModel.double_exponential(name, *, tau_rise, tau_decay, g, E_syn)
        SynapseModel.tanh_gate(name, *, amp, v_half, k, tau_decay, g, E_syn)
        SynapseModel.boltzmann_gate(name, *, v_half, k, tau, g, E_syn)
        SynapseModel.alpha_beta(name, *, alpha, beta, g, E_syn)

    General constructor with pattern matching:
        SynapseModel("name", dS_dt=expr, current=expr, dA_dt=expr)
    """

    def __init__(
        self,
        name: str = "",
        *,
        dS_dt=None,
        current=None,
        dA_dt=None,
        spike_S: float = 0.0,
        spike_A: float = 0.0,
        g: float = 0.1,
        E_syn: float = 0.0,
        S_init: float = 0.0,
        A_init: float = 0.0,
    ):
        import sympy as _sp
        from .._codegen import S as _S_sym, A as _A_sym
        self._name = name
        self._spec: "_SynapseSpec | None" = None

        if dS_dt is None:
            return  # named constructors fill _spec directly

        # --- Pattern matching ---
        # 1. EXP_DECAY: dS_dt = -k*S, no dA_dt
        if dA_dt is None:
            tau_s = _try_match_exp_decay(dS_dt, _S_sym)
            if tau_s is not None:
                s = _SynapseSpec.exponential(tau_s, g, E_syn)
                s.name = name
                s.delta_S = spike_S if spike_S != 0.0 else 1.0
                s.S_init = S_init
                self._spec = s
                return

        # 2. ALPHA_FUNC: dS_dt=(A-S)*k, dA_dt=-A*k
        if dA_dt is not None:
            tau_a = _try_match_alpha_func(dS_dt, dA_dt, _S_sym, _A_sym)
            if tau_a is not None:
                s = _SynapseSpec.alpha_function(tau_a, g, E_syn)
                s.name = name
                s.delta_A = spike_A if spike_A != 0.0 else s.delta_A
                s.S_init = S_init
                s.A_init = A_init
                self._spec = s
                return

            # 3. DOUBLE_EXP: dS_dt=-k1*S, dA_dt=-k2*A (k1≠k2)
            result = _try_match_double_exp(dS_dt, dA_dt, _S_sym, _A_sym)
            if result is not None:
                tau_slow, tau_fast = max(result), min(result)
                s = _SynapseSpec.double_exponential(tau_fast, tau_slow, g, E_syn)
                s.name = name
                s.delta_S = spike_S if spike_S != 0.0 else s.delta_S
                s.delta_A = spike_A if spike_A != 0.0 else s.delta_A
                s.S_init = S_init
                s.A_init = A_init
                self._spec = s
                return

        # 4. CUSTOM_EXPR — compile to VM
        from hodgkin_huxley._core import VmOp as _VmOp
        _extra = {_S_sym: _VmOp.PUSH_S, _A_sym: _VmOp.PUSH_A}

        spec = _SynapseSpec()
        spec.name = name
        spec.update_form = _SynapseUpdateForm.CUSTOM_EXPR
        spec.current_form = _SynapseCurrentForm.LINEAR
        spec.g = g
        spec.E_syn = E_syn
        spec.S_init = S_init
        spec.A_init = A_init
        spec.delta_S = spike_S
        spec.delta_A = spike_A

        spec.dS_dt_vm = compile_to_vm_bytecode(
            dS_dt, dep_sym=_sp.Symbol("V_pre"), extra_syms=_extra
        )
        if dA_dt is not None:
            spec.dA_dt_vm = compile_to_vm_bytecode(
                dA_dt, dep_sym=_sp.Symbol("V_pre"), extra_syms=_extra
            )
        if current is not None:
            spec.current_vm = compile_to_vm_bytecode(
                current, dep_sym=_sp.Symbol("V_post"), extra_syms=_extra
            )
            spec.current_form = _SynapseCurrentForm.CUSTOM_EXPR
        self._spec = spec

    @classmethod
    def exponential(cls, name: str = "exponential", *, tau: float,
                    g: float = 1.0, E_syn: float = 0.0) -> "SynapseModel":
        obj = cls.__new__(cls)
        s = _SynapseSpec.exponential(tau, g, E_syn)
        s.name = name
        obj._name = name
        obj._spec = s
        return obj

    @classmethod
    def alpha_function(cls, name: str = "alpha_function", *, tau: float,
                       g: float = 1.0, E_syn: float = 0.0) -> "SynapseModel":
        obj = cls.__new__(cls)
        s = _SynapseSpec.alpha_function(tau, g, E_syn)
        s.name = name
        obj._name = name
        obj._spec = s
        return obj

    @classmethod
    def double_exponential(cls, name: str = "double_exponential", *,
                           tau_rise: float, tau_decay: float,
                           g: float = 1.0, E_syn: float = 0.0) -> "SynapseModel":
        obj = cls.__new__(cls)
        s = _SynapseSpec.double_exponential(tau_rise, tau_decay, g, E_syn)
        s.name = name
        obj._name = name
        obj._spec = s
        return obj

    @classmethod
    def tanh_gate(cls, name: str = "tanh_gate", *,
                  amp: float = 2.0, v_half: float = 0.0, k: float = 4.0,
                  tau_decay: float = 13.0, g: float = 0.1,
                  E_syn: float = -80.0) -> "SynapseModel":
        obj = cls.__new__(cls)
        s = _SynapseSpec()
        s.name = name
        s.update_form = _SynapseUpdateForm.TANH_GATE
        s.current_form = _SynapseCurrentForm.LINEAR
        s.tanh_amp = amp
        s.tanh_vh = v_half
        s.tanh_k = k
        s.tau_decay = tau_decay
        s.g = g
        s.E_syn = E_syn
        obj._name = name
        obj._spec = s
        return obj

    @classmethod
    def boltzmann_gate(cls, name: str = "boltzmann_gate", *,
                       v_half: float, k: float, tau: float,
                       g: float = 0.1, E_syn: float = 0.0) -> "SynapseModel":
        obj = cls.__new__(cls)
        s = _SynapseSpec()
        s.name = name
        s.update_form = _SynapseUpdateForm.BOLTZMANN_GATE
        s.current_form = _SynapseCurrentForm.LINEAR
        s_inf = _BoltzmannParams()
        s_inf.v_half = v_half
        s_inf.k = k
        s.s_inf = s_inf
        t = _TauParams()
        t.form = _TauForm.CONSTANT
        t.set_param(0, tau)
        s.tau = t
        s.g = g
        s.E_syn = E_syn
        obj._name = name
        obj._spec = s
        return obj

    @classmethod
    def alpha_beta(cls, name: str = "alpha_beta", *,
                   alpha, beta, g: float = 0.1,
                   E_syn: float = 0.0) -> "SynapseModel":
        obj = cls.__new__(cls)
        s = _SynapseSpec()
        s.name = name
        s.update_form = _SynapseUpdateForm.ALPHA_BETA
        s.current_form = _SynapseCurrentForm.LINEAR
        alpha_p, _ = _normalize_gate_input(alpha)
        beta_p, _  = _normalize_gate_input(beta)
        s.alpha = alpha_p
        s.beta = beta_p
        s.g = g
        s.E_syn = E_syn
        obj._name = name
        obj._spec = s
        return obj

    def to_spec(self) -> "_SynapseSpec":
        """Return the built SynapseSpec."""
        if self._spec is None:
            raise RuntimeError("SynapseModel has no spec — call a named constructor or pass dS_dt")
        return self._spec

    def __repr__(self) -> str:
        return f"<SynapseModel '{self._name}'>"


# =============================================================================
# Kinetic synapse builder (deprecated alias for SynapseModel voltage-gated forms)
# =============================================================================


class KineticSynapseModel:
    """
    Fluent builder for KineticSynapseSpec.

    Examples
    --------
    >>> spec = (KineticSynapseModel("GABA_kin")
    ...         .tanh_gate(amp=2.0, v_half=0.0, k=4.0, tau_decay=13.0)
    ...         .linear_current(g=0.1, E_syn=-80.0)
    ...         .to_spec())
    """

    def __init__(self, name: str):
        self._spec = _SynapseSpec()
        self._spec.name = name

    def tanh_gate(
        self,
        amp: float = 2.0,
        v_half: float = 0.0,
        k: float = 4.0,
        tau_decay: float = 13.0,
    ) -> "KineticSynapseModel":
        self._spec.update_form = _SynapseUpdateForm.TANH_GATE
        self._spec.tanh_amp = amp
        self._spec.tanh_vh = v_half
        self._spec.tanh_k = k
        self._spec.tau_decay = tau_decay
        return self

    def boltzmann_gate(
        self,
        v_half: float,
        k: float,
        tau: float,
    ) -> "KineticSynapseModel":
        self._spec.update_form = _SynapseUpdateForm.BOLTZMANN_GATE
        s_inf = _BoltzmannParams()
        s_inf.v_half = v_half
        s_inf.k = k
        self._spec.s_inf = s_inf
        t = _TauParams()
        t.form = _TauForm.CONSTANT
        t.set_param(0, tau)
        self._spec.tau = t
        return self

    def custom_expr(
        self,
        dS_dt: "_sympy.Basic",
        current: "_sympy.Basic",
        g: float,
        E_syn: float,
    ) -> "KineticSynapseModel":
        """Configure kinetic synapse dynamics from SymPy expressions.

        Parameters
        ----------
        dS_dt : sympy.Expr
            ODE for gating variable S in terms of ``V_pre`` and ``S``.
            Compiled to VM bytecode; PUSH_DEP = V_pre, PUSH_S = S.
        current : sympy.Expr
            Conductance factor in terms of ``V_post`` and ``S``.
            The framework applies: I_syn += result * weight * (E_syn - V_post).
            Compiled to VM bytecode; PUSH_DEP = V_post, PUSH_S = S.
        g : float
            Peak conductance (stored on spec; used by the linear fallback path).
        E_syn : float
            Synaptic reversal potential in mV.
        """
        from hodgkin_huxley._core import VmOp as _VmOp
        self._spec.update_form = _SynapseUpdateForm.CUSTOM_EXPR
        self._spec.current_form = _SynapseCurrentForm.CUSTOM_EXPR
        self._spec.g = g
        self._spec.E_syn = E_syn
        # Map S symbol → PUSH_S opcode for both expressions
        _s_sym = {_sympy.Symbol("S"): _VmOp.PUSH_S}
        self._spec.dS_dt_vm = compile_to_vm_bytecode(
            dS_dt, dep_sym=_sympy.Symbol("V_pre"), extra_syms=_s_sym
        )
        self._spec.current_vm = compile_to_vm_bytecode(
            current, dep_sym=_sympy.Symbol("V_post"), extra_syms=_s_sym
        )
        return self

    def alpha_beta(
        self,
        alpha,
        beta,
    ) -> "KineticSynapseModel":
        """Set ALPHA_BETA gating. Accepts TaggedExpr or raw _RateFuncParams."""
        self._spec.update_form = _SynapseUpdateForm.ALPHA_BETA
        alpha_p, _ = _normalize_gate_input(alpha)
        beta_p, _  = _normalize_gate_input(beta)
        self._spec.alpha = alpha_p  # type: ignore[assignment]
        self._spec.beta = beta_p  # type: ignore[assignment]
        return self

    def linear_current(
        self,
        g: float,
        E_syn: float,
        power: int = 1,
    ) -> "KineticSynapseModel":
        self._spec.current_form = _SynapseCurrentForm.LINEAR
        self._spec.g = g
        self._spec.E_syn = E_syn
        self._spec.power = power
        return self

    def mg_block_current(
        self,
        g: float,
        E_syn: float,
        mg_conc: float = 1.0,
        mg_scale: float = 0.062,
        mg_denom: float = 3.57,
    ) -> "KineticSynapseModel":
        self._spec.current_form = _SynapseCurrentForm.MG_BLOCK
        self._spec.g = g
        self._spec.E_syn = E_syn
        self._spec.mg_conc = mg_conc
        self._spec.mg_scale = mg_scale
        self._spec.mg_denom = mg_denom
        return self

    def to_spec(self) -> "_SynapseSpec":
        """Return the built SynapseSpec."""
        return self._spec

    def __repr__(self) -> str:
        return f"<KineticSynapseModel '{self._spec.name}'>"


# =============================================================================
# IntracellularDynamics and Modulation builders (task14)
# =============================================================================


class Modulation:
    """Helper to build an IntracellularModulation targeting a channel or gate.

    Use the classmethod factories:
      Modulation.channel_g("CaL", expr)
      Modulation.gate_inf_shift("m", scale=0.5)
      Modulation.synapse_g(expr)
    """

    def __init__(self, target: str, target_name: str,
                 expr=None, shift_scale: float = 0.0):
        self._target = target
        self._target_name = target_name  # resolved to index at add_intracellular time
        self._expr = expr
        self._shift_scale = shift_scale

    @classmethod
    def channel_g(cls, name: str, expr) -> "Modulation":
        return cls("CHANNEL_G", name, expr=expr)

    @classmethod
    def channel_erev(cls, name: str, expr) -> "Modulation":
        return cls("CHANNEL_EREV", name, expr=expr)

    @classmethod
    def gate_inf_shift(cls, name: str, scale: float) -> "Modulation":
        return cls("GATE_INF_SHIFT", name, shift_scale=scale)

    @classmethod
    def gate_inf_scale(cls, name: str, expr) -> "Modulation":
        return cls("GATE_INF_SCALE", name, expr=expr)

    @classmethod
    def gate_tau_scale(cls, name: str, expr) -> "Modulation":
        return cls("GATE_TAU_SCALE", name, expr=expr)

    @classmethod
    def gate_inf_expr(cls, name: str, expr) -> "Modulation":
        return cls("GATE_INF_EXPR", name, expr=expr)

    @classmethod
    def synapse_g(cls, expr) -> "Modulation":
        return cls("SYNAPSE_G", "", expr=expr)

    def _to_spec(self, channel_names: "list[str]", gate_names: "list[str]",
                 substance_names: "list[str]", substance_idx: int,
                 substance_sym, dep_sym=None) -> "_IntracellularModulation":
        """Convert to C++ IntracellularModulation. Called by IntracellularDynamics.to_spec()."""
        import sympy as _sp
        _VmOp = __import__("hodgkin_huxley._core", fromlist=["VmOp"]).VmOp

        mod = _IntracellularModulation()
        mod.substance_idx = substance_idx

        target_map = {
            "CHANNEL_G":      _IcModTarget.CHANNEL_G,
            "CHANNEL_EREV":   _IcModTarget.CHANNEL_EREV,
            "GATE_INF_SHIFT": _IcModTarget.GATE_INF_SHIFT,
            "GATE_INF_SCALE": _IcModTarget.GATE_INF_SCALE,
            "GATE_TAU_SCALE": _IcModTarget.GATE_TAU_SCALE,
            "GATE_INF_EXPR":  _IcModTarget.GATE_INF_EXPR,
            "SYNAPSE_G":      _IcModTarget.SYNAPSE_G,
        }
        if self._target not in target_map:
            raise ValueError(f"Unknown modulation target: {self._target!r}")
        mod.target = target_map[self._target]

        # Resolve target name → index
        if self._target in ("CHANNEL_G", "CHANNEL_EREV"):
            if self._target_name not in channel_names:
                raise ValueError(
                    f"Modulation target channel {self._target_name!r} not in spec"
                )
            mod.target_idx = channel_names.index(self._target_name)
        elif self._target.startswith("GATE_"):
            if self._target_name not in gate_names:
                raise ValueError(
                    f"Modulation target gate {self._target_name!r} not in spec"
                )
            mod.target_idx = gate_names.index(self._target_name)
        # SYNAPSE_G: target_idx unused

        # Compile mod_vm (substance concentration is PUSH_DEP)
        if self._target == "GATE_INF_SHIFT":
            mod.shift_scale = self._shift_scale
            # mod_vm left empty for scalar shift path
        elif self._expr is not None:
            # dep_sym is the substance's SymPy symbol — compile with PUSH_DEP = substance
            _dsym = dep_sym if dep_sym is not None else substance_sym
            mod.mod_vm = compile_to_vm_bytecode(self._expr, dep_sym=_dsym)

        return mod

    def __repr__(self) -> str:
        return f"<Modulation {self._target} target={self._target_name!r}>"


def _match_ode_decay(ode, X_sym):
    """Return k if ode = -k*X (k>0), else None."""
    import sympy as _sp
    expr = _sp.expand(_sp.sympify(ode))
    if expr.free_symbols - {X_sym}:
        return None
    # Must be linear in X with no constant term
    c1 = expr.coeff(X_sym, 1)
    c0 = _sp.simplify(expr - c1 * X_sym)
    if c0 != 0:
        return None
    try:
        k = float(-c1)
    except Exception:
        return None
    return k if k > 0 else None


def _match_ode_driven_decay(ode, X_sym, I_sym):
    """Return (eps, k) if ode = eps*(-I_src - k*X), else None."""
    import sympy as _sp
    expr = _sp.expand(_sp.sympify(ode))
    if expr.free_symbols - {X_sym, I_sym}:
        return None
    # Extract coefficients: ode = c_X*X + c_I*I + c0
    c_X = expr.coeff(X_sym, 1)
    c_I = expr.coeff(I_sym, 1)
    c0 = _sp.simplify(expr - c_X * X_sym - c_I * I_sym)
    if c0 != 0:
        return None
    try:
        k_coeff = float(c_X)
        I_coeff = float(c_I)
    except Exception:
        return None
    if I_coeff == 0.0 or k_coeff == 0.0:
        return None
    eps = float(-I_coeff)
    k = float(-k_coeff / eps)
    if eps <= 0 or k <= 0:
        return None
    return eps, k


class IntracellularDynamics:
    """
    Builder for intracellular substance dynamics.

    Parameters
    ----------
    name : str
        Substance name (e.g. "Ca", "DA").
    ode : sympy.Expr
        ODE for dX/dt.  Standard symbols: X (self concentration),
        I_source (source channel current sum), other X_n (cross-substance).
    source_channels : list[str] | None
        Channel names whose currents feed I_source in the ODE.
    nernst : sympy.Expr | None
        Expression returning the Nernst reversal in mV given the substance X.
        Standard: ``(R*T / (z*F)) * log(X_o / X)``.
    initial : float
        Initial concentration.
    modulations : list[Modulation] | None
        Downstream modulations driven by this substance.
    """

    def __init__(self, name: str, *, ode, source_channels=None,
                 nernst=None, initial: float = 0.0,
                 modulations=None):
        self.name = name
        self.ode = ode
        self.source_channels = source_channels or []
        self.nernst = nernst
        self.initial = initial
        self.modulations = modulations or []

    def to_spec(self, channel_names: "list[str]", gate_names: "list[str]",
                substance_map: "dict[str, int] | None" = None) -> "_IntracellularSpec":
        """
        Pattern-match the ODE and Nernst expressions, then compile to IntracellularSpec.

        Parameters
        ----------
        channel_names : list[str]
            Channel names for resolving modulation targets.
        gate_names : list[str]
            Gate names for resolving modulation targets.
        substance_map : dict[str, int] | None
            Existing substance name → index mapping (for PUSH_X cross-references).
        """
        import sympy as _sp
        from .._codegen import _substance_cache, I_source as _I_src

        substance_map = substance_map or {}
        substance_idx = len(substance_map)  # this substance will be appended

        X_sym = _sp.Symbol(self.name)

        ic = _IntracellularSpec()
        ic.name = self.name
        ic.initial = self.initial
        ic.source_channels = [
            channel_names.index(ch) for ch in self.source_channels
            if ch in channel_names
        ]

        # --- Pattern match ODE ---
        k_decay = _match_ode_decay(self.ode, X_sym)
        if k_decay is not None:
            ic.update_form = _IntracellularUpdateForm.DECAY
            ic.k_decay = k_decay
        else:
            driven = _match_ode_driven_decay(self.ode, X_sym, _I_src)
            if driven is not None:
                eps, k = driven
                # Check if Nernst is also standard form
                if self.nernst is not None and _is_standard_nernst(self.nernst, X_sym):
                    ic.update_form = _IntracellularUpdateForm.DRIVEN_DECAY_NERNST
                    ic.nernst_enabled = True
                    _fill_nernst_params(ic, self.nernst, X_sym)
                else:
                    ic.update_form = _IntracellularUpdateForm.DRIVEN_DECAY
                ic.epsilon = eps
                ic.k_decay = k
                # Compile nernst_vm if non-standard
                if self.nernst is not None and not _is_standard_nernst(self.nernst, X_sym):
                    ic.nernst_vm = compile_to_vm_bytecode(self.nernst, dep_sym=X_sym)
                    ic.nernst_enabled = True
            else:
                ic.update_form = _IntracellularUpdateForm.CUSTOM_EXPR
                # Compile ode_vm: PUSH_DEP = I_source, PUSH_S = X (self), PUSH_X(n) = X_n
                from hodgkin_huxley._core import VmOp as _VmOp
                extra: dict = {X_sym: _VmOp.PUSH_S}
                # Map other substance symbols to PUSH_X with the right index
                ode_free = _sp.sympify(self.ode).free_symbols
                for sname, sidx in substance_map.items():
                    other_sym = _sp.Symbol(sname)
                    if other_sym in ode_free:
                        extra[other_sym] = (_VmOp.PUSH_X, sidx)
                ic.ode_vm = compile_to_vm_bytecode(
                    self.ode, dep_sym=_I_src, extra_syms=extra
                )
                if self.nernst is not None:
                    ic.nernst_vm = compile_to_vm_bytecode(self.nernst, dep_sym=X_sym)
                    ic.nernst_enabled = True

        # --- Compile modulations ---
        for mod in self.modulations:
            m_spec = mod._to_spec(
                channel_names, gate_names,
                list(substance_map.keys()), substance_idx, X_sym
            )
            ic.modulations.append(m_spec)

        return ic

    def __repr__(self) -> str:
        return f"<IntracellularDynamics '{self.name}'>"


class DopamineGating(IntracellularDynamics):
    """
    Convenience preset for dopamine-gated synaptic gain.

    Wraps ``IntracellularDynamics("DA", ode=-k_decay*DA, initial=da_level,
    modulations=[Modulation.synapse_g(1/(1+exp(-k*(DA-0.5))))])``.

    Parameters
    ----------
    da_level : float
        Initial / resting DA concentration (dimensionless, 0–1). Default 0.5.
    k : float
        Sigmoid steepness for DA → SYNAPSE_G gain curve. Default 10.0.
    k_decay : float
        First-order DA decay rate (1/ms). Default 0.01 (τ = 100 ms).
    """

    def __init__(self, da_level: float = 0.5, k: float = 10.0,
                 k_decay: float = 0.01):
        import sympy as _sp
        _DA = _sp.Symbol("DA")
        ode = -_sp.Float(k_decay) * _DA
        gain = 1 / (1 + _sp.exp(-_sp.Float(k) * (_DA - _sp.Float(0.5))))
        super().__init__("DA", ode=ode, initial=da_level,
                         modulations=[Modulation.synapse_g(gain)])
        self._da_level = da_level
        self._k = k
        self._k_decay = k_decay

    def __repr__(self) -> str:
        return (f"<DopamineGating da_level={self._da_level} "
                f"k={self._k} k_decay={self._k_decay}>")


class HomeostaticScaling(IntracellularDynamics):
    """
    Convenience preset for homeostatic synaptic scaling.

    Tracks a slow exponential average of post-synaptic drive and scales
    incoming synaptic conductance inversely.  The substance H evolves as::

        dH/dt = (I_source - target_rate - H) / tau

    and the SYNAPSE_G multiplier is ``1 / (1 + H)``.

    H converges to ``I_source - target_rate``, so set ``target_rate`` close
    to the mean I_ext that produces the desired firing rate.  If
    ``|I_source - target_rate|`` is large the gain expression may diverge;
    use ``target_rate`` accordingly.

    Note: the ODE does not match the DRIVEN_DECAY pattern (sign difference),
    so it compiles via CUSTOM_EXPR — this is expected and correct.

    Parameters
    ----------
    target_rate : float
        Target activity level in µA/cm² (same units as I_ext). Default 20.0.
    tau : float
        Integrator time constant in ms. Default 5000.0 (5 s).
    """

    def __init__(self, target_rate: float = 20.0, tau: float = 5000.0):
        import sympy as _sp
        _H = _sp.Symbol("H")
        ode = (_I_source_sym - _sp.Float(target_rate) - _H) / _sp.Float(tau)
        gain = 1 / (1 + _H)
        super().__init__("H", ode=ode, initial=0.0,
                         modulations=[Modulation.synapse_g(gain)])
        self._target_rate = target_rate
        self._tau = tau

    def __repr__(self) -> str:
        return (f"<HomeostaticScaling target_rate={self._target_rate} "
                f"tau={self._tau}>")


def _is_standard_nernst(expr, X_sym) -> bool:
    """Check if expr matches (R*T/(z*F))*log(X_o/X) pattern."""
    import sympy as _sp
    e = _sp.sympify(expr)
    free = e.free_symbols
    if X_sym not in free:
        return False
    # Must be of form C * log(something / X)
    if not isinstance(e, _sp.Mul):
        return False
    log_parts = [a for a in e.args if isinstance(a, _sp.log)]
    if not log_parts:
        return False
    inner = log_parts[0].args[0]
    # inner should be X_o / X
    if not isinstance(inner, _sp.Mul):
        return False
    has_X_inv = any(
        isinstance(a, _sp.Pow) and a.args[0] == X_sym and a.args[1] == -1
        for a in inner.args
    )
    return has_X_inv


def _fill_nernst_params(ic: "_IntracellularSpec", expr, X_sym) -> None:
    """Extract scalar Nernst params from standard (R*T/(z*F))*log(X_o/X) expression."""
    import sympy as _sp
    e = _sp.sympify(expr)
    # Collect multiplier and X_o
    for arg in e.args:
        if isinstance(arg, _sp.log):
            inner = arg.args[0]
            for a in _sp.Mul.make_args(inner):
                if not (isinstance(a, _sp.Pow) and a.args[0] == X_sym):
                    try:
                        ic.nernst_Ca_o = float(a)
                    except Exception:
                        pass
    # Overall scale = R*T/(z*F); we don't parse individual R,T,z,F here since
    # the defaults in IntracellularSpec already encode physiological values.
    # The nernst_enabled flag + defaults will be used.


if TYPE_CHECKING:
    KineticSynapseSpec = _SynapseSpec
else:
    class _KineticSynapseSpecMeta(type):
        """Metaclass that makes KineticSynapseSpec act as both a factory and a type."""

        def __instancecheck__(cls, instance) -> bool:
            return type.__instancecheck__(cls, instance) or isinstance(instance, _SynapseSpec)

        def __call__(cls, *args, **kwargs):
            return cls.__new__(cls, *args, **kwargs)

    class KineticSynapseSpec(metaclass=_KineticSynapseSpecMeta):
        """
        Deprecated alias for the unified SynapseSpec factory.

        Calling ``KineticSynapseSpec(name, dS_dt=..., current=...)`` with SymPy
        expressions compiles them and returns a ``SynapseSpec``
        with ``update_form = CUSTOM_EXPR``.  Without SymPy args, returns a blank
        ``SynapseSpec``.

        Preset factory methods are forwarded from the C++ class:

        * ``KineticSynapseSpec.gaba_kinetic()``
        * ``KineticSynapseSpec.nmda_kinetic()``
        * ``KineticSynapseSpec.gaba_b()``

        ``isinstance(spec, KineticSynapseSpec)`` returns ``True`` for any
        ``SynapseSpec`` instance.
        """

        gaba_kinetic = staticmethod(_SynapseSpec.gaba_kinetic)
        nmda_kinetic = staticmethod(_SynapseSpec.nmda_kinetic)
        gaba_b       = staticmethod(_SynapseSpec.gaba_b)

        def __new__(
            cls,
            name: str = "",
            *,
            dS_dt=None,
            current=None,
            g: float = 0.1,
            E_syn: float = -80.0,
            **kwargs,
        ):
            if dS_dt is not None and current is not None:
                spec = (
                    KineticSynapseModel(name)
                    .custom_expr(dS_dt, current, g, E_syn)
                    .to_spec()
                )
            else:
                spec = _SynapseSpec()
                spec.name = name
                spec.g = g
                spec.E_syn = E_syn
            for k_attr, v_attr in kwargs.items():
                setattr(spec, k_attr, v_attr)
            return spec


# =============================================================================
# Plasticity rule builders (task15)
# =============================================================================


class STDPRule:
    """
    Spike-Timing Dependent Plasticity rule specification.

    Pass as ``plasticity=`` to :meth:`RegionalNetwork.connect` or
    ``Network.add_synapse``.

    Parameters
    ----------
    A_plus : float   LTP learning rate. Default 0.005.
    A_minus : float  LTD learning rate. Default 0.006.
    tau_plus : float Pre-trace time constant (ms). Default 20.0.
    tau_minus : float Post-trace time constant (ms). Default 20.0.
    w_min : float    Minimum weight. Default 0.0.
    w_max : float    Maximum weight. Default 1.0.
    modulator_population : str | None
        Population name whose substance gates learning rate. None = no gating.
    modulator_substance : str | None
        Substance name within that population (e.g. ``"DA"``).
    modulator_scale : float
        Linear multiplier applied to the modulator concentration before
        scaling A_plus / A_minus. Default 1.0.
    """

    def __init__(self, *, A_plus: float = 0.005, A_minus: float = 0.006,
                 tau_plus: float = 20.0, tau_minus: float = 20.0,
                 w_min: float = 0.0, w_max: float = 1.0,
                 modulator_population: "str | None" = None,
                 modulator_substance: "str | None" = None,
                 modulator_scale: float = 1.0):
        self.A_plus          = A_plus
        self.A_minus         = A_minus
        self.tau_plus        = tau_plus
        self.tau_minus       = tau_minus
        self.w_min           = w_min
        self.w_max           = w_max
        self.modulator_population = modulator_population
        self.modulator_substance  = modulator_substance
        self.modulator_scale      = modulator_scale
        self._resolved_pop_start  = -1   # filled by RegionalNetwork.connect()
        self._resolved_subst_idx  = -1   # filled by RegionalNetwork.connect()

    def to_spec(self):
        """Return a C++ ``PlasticitySpec`` from current parameters."""
        from .._core import (PlasticitySpec as _PS, STDPParams as _STDP,
                             PlasticityType as _PT)
        ps = _PS()
        ps.type = _PT.STDP
        s = _STDP()
        s.A_plus              = self.A_plus
        s.A_minus             = self.A_minus
        s.tau_plus            = self.tau_plus
        s.tau_minus           = self.tau_minus
        s.w_min               = self.w_min
        s.w_max               = self.w_max
        s.modulator_pop_start    = self._resolved_pop_start
        s.modulator_substance_idx = self._resolved_subst_idx
        s.modulator_scale        = self.modulator_scale
        ps.stdp = s
        return ps

    def __repr__(self) -> str:
        return (f"<STDPRule A+={self.A_plus} A-={self.A_minus} "
                f"tau+={self.tau_plus} tau-={self.tau_minus} "
                f"[{self.w_min},{self.w_max}]>")


class STPRule:
    """
    Tsodyks-Markram Short-Term Plasticity (depression / facilitation) rule.

    Pass as ``plasticity=`` to :meth:`RegionalNetwork.connect` or
    ``Network.add_synapse``.

    Parameters
    ----------
    U : float      Baseline utilization (release probability). Default 0.5.
    tau_u : float  Recovery time constant of *u* (ms). Default 1000.0.
    tau_x : float  Recovery time constant of *x* (ms). Default 100.0.
    """

    def __init__(self, *, U: float = 0.5,
                 tau_u: float = 1000.0, tau_x: float = 100.0):
        self.U     = U
        self.tau_u = tau_u
        self.tau_x = tau_x

    def to_spec(self):
        """Return a C++ ``PlasticitySpec`` from current parameters."""
        from .._core import (PlasticitySpec as _PS, STPParams as _STP,
                             PlasticityType as _PT)
        ps = _PS()
        ps.type = _PT.STP
        s = _STP()
        s.U     = self.U
        s.tau_u = self.tau_u
        s.tau_x = self.tau_x
        ps.stp = s
        return ps

    def __repr__(self) -> str:
        return f"<STPRule U={self.U} tau_u={self.tau_u} tau_x={self.tau_x}>"
