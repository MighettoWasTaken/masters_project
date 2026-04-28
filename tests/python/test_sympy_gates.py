"""
tests/python/test_sympy_gates.py
=================================
Integration tests for the SymPy equation system as surfaced through the public
API (Boltzmann, Tau.*, RateFunc.*, GateSpec, NeuronModel.add_gate, etc.).

Test structure
--------------
- TestHelperReturnTypes   : Boltzmann/Tau/RateFunc return TaggedExpr with correct fields
- TestPresetEquivalence   : TaggedExpr fast-path + pattern-match fast-path give identical
                            NeuronModelSpec as the old struct-only path
- TestNeuronModelAddGate  : NeuronModel.add_gate() accepts every input variant
- TestGateSpecClass       : Python GateSpec.to_core() produces correct _core.GateSpec
- TestVMExprGate          : CUSTOM_EXPR path end-to-end and edge cases (no compiler needed)
- TestVMExprEdgeCases     : VM gate forms: Ca-dependent, alpha-beta, trig, multi-gate
- TestBackwardCompat      : legacy deprecation path still accessible
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import hodgkin_huxley as hh  # must precede sympy to block broken gmpy2 DLL on Windows
import sympy
from hodgkin_huxley import (
    V, Ca, V_pre, V_post, S,
    Boltzmann, Tau, RateFunc,
    GateSpec, NeuronModel,
    TaggedExpr, EigenPrinter,
)
from hodgkin_huxley._codegen import try_pattern_match
from hodgkin_huxley._core import (
    BoltzmannParams, TauParams, TauForm, RateFuncParams, RateFuncForm,
    GateSpec as _CoreGateSpec, GateUpdateForm, GateDependency,
    NeuronModelSpec,
)


# =============================================================================
# TestHelperReturnTypes
# =============================================================================

class TestHelperReturnTypes:
    """Boltzmann/Tau.*/RateFunc.* helpers return TaggedExpr with correct fields."""

    def test_boltzmann_to_tagged(self):
        b = Boltzmann(-40.0, 8.0)
        te = b.to_tagged()
        assert isinstance(te, TaggedExpr)
        assert isinstance(te.params, BoltzmannParams)
        assert abs(te.params.v_half - (-40.0)) < 1e-12
        assert abs(te.params.k - 8.0) < 1e-12
        assert te.expr is not None

    def test_boltzmann_to_params_unchanged(self):
        b = Boltzmann(-37.0, 7.0)
        p = b.to_params()
        assert isinstance(p, BoltzmannParams)
        assert abs(p.v_half - (-37.0)) < 1e-12

    def test_tau_constant_returns_tagged(self):
        te = Tau.constant(0.5)
        assert isinstance(te, TaggedExpr)
        assert isinstance(te.params, TauParams)
        assert te.form == TauForm.CONSTANT
        # Expr should be a SymPy numeric
        assert te.expr is not None
        assert float(te.expr) == pytest.approx(0.5)

    def test_tau_boltzmann_returns_tagged(self):
        te = Tau.boltzmann(0.2, 3.0, -53.0, -0.7)
        assert isinstance(te, TaggedExpr)
        assert te.form == TauForm.BOLTZMANN
        assert isinstance(te.expr, sympy.Basic)

    def test_tau_scaled_exp_returns_tagged(self):
        te = Tau.scaled_exp(1.0, -41.0, -4.0)
        assert isinstance(te, TaggedExpr)
        assert te.form == TauForm.SCALED_EXP
        assert isinstance(te.expr, sympy.Basic)

    def test_tau_double_exp_sum_returns_tagged(self):
        te = Tau.double_exp_sum(20.0, 100.0, 39.7, 9.32, 0.6, 7.4)
        assert isinstance(te, TaggedExpr)
        assert te.form == TauForm.DOUBLE_EXP_SUM
        assert isinstance(te.expr, sympy.Basic)

    def test_ratefunc_linear_over_exp(self):
        te = RateFunc.linear_over_exp(0.32, 54.0, 4.0)
        assert isinstance(te, TaggedExpr)
        assert isinstance(te.params, RateFuncParams)
        assert te.form == RateFuncForm.LINEAR_OVER_EXP
        assert abs(te.params.A - 0.32) < 1e-12

    def test_ratefunc_exp_decay(self):
        te = RateFunc.exp_decay(0.128, 50.0, -18.0)
        assert isinstance(te, TaggedExpr)
        assert te.form == RateFuncForm.EXP_DECAY

    def test_ratefunc_linear_over_expm1(self):
        te = RateFunc.linear_over_expm1(0.032, 52.0, 5.0)
        assert isinstance(te, TaggedExpr)
        assert te.form == RateFuncForm.LINEAR_OVER_EXPM1

    def test_ratefunc_sigmoid(self):
        te = RateFunc.sigmoid(4.0, 27.0, -5.0)
        assert isinstance(te, TaggedExpr)
        assert te.form == RateFuncForm.SIGMOID


# =============================================================================
# TestPresetEquivalence
# =============================================================================

class TestPresetEquivalence:
    """
    Spec built with Tau.*() helpers must match a spec built directly with
    private _TauParams structs (the fast path must not alter gate parameters).
    """

    def _spec_with_helpers(self):
        m = NeuronModel("test_helpers")
        m.add_gate("m", update_form="inf_tau",
                   inf=Boltzmann(-41.0, -4.0),
                   tau=Tau.scaled_exp(1.0, -41.0, -4.0))
        return m.to_spec()

    def _spec_with_structs(self):
        from hodgkin_huxley._core import (
            BoltzmannParams, TauParams, TauForm,
            GateSpec as _GS, GateUpdateForm, GateDependency, NeuronModelSpec,
        )
        spec = NeuronModelSpec()
        g = _GS()
        g.name = "m"
        g.update_form = GateUpdateForm.INF_TAU
        g.dependency = GateDependency.VOLTAGE
        g.scale = 1.0
        g.initial_value = 0.0
        bp = BoltzmannParams()
        bp.v_half = -41.0; bp.k = -4.0
        g.inf = bp
        tp = TauParams()
        tp.form = TauForm.SCALED_EXP
        tp.set_param(0, 1.0); tp.set_param(1, -41.0); tp.set_param(2, -4.0)
        g.tau = tp
        spec.gates.append(g)
        return spec

    def test_gate_count(self):
        s_helper = self._spec_with_helpers()
        s_struct  = self._spec_with_structs()
        assert len(s_helper.gates) == len(s_struct.gates)

    def test_inf_params(self):
        g_h = self._spec_with_helpers().gates[0]
        g_s = self._spec_with_structs().gates[0]
        assert abs(g_h.inf.v_half - g_s.inf.v_half) < 1e-12
        assert abs(g_h.inf.k     - g_s.inf.k)      < 1e-12

    def test_tau_params(self):
        g_h = self._spec_with_helpers().gates[0]
        g_s = self._spec_with_structs().gates[0]
        # Both should be SCALED_EXP with same parameters
        assert g_h.tau.form == g_s.tau.form
        for i in range(3):
            assert abs(g_h.tau.get_param(i) - g_s.tau.get_param(i)) < 1e-12

    def test_update_form(self):
        g_h = self._spec_with_helpers().gates[0]
        assert g_h.update_form == GateUpdateForm.INF_TAU

    def test_alpha_beta_equivalent(self):
        """RateFunc helpers produce same struct as direct construction."""
        from hodgkin_huxley._core import (
            RateFuncParams, RateFuncForm,
            GateSpec as _GS, GateUpdateForm, GateDependency, NeuronModelSpec,
        )
        m = NeuronModel("ab_test")
        m.add_gate("m", update_form="alpha_beta",
                   alpha=RateFunc.linear_over_expm1(0.32, 54.0, 4.0),
                   beta=RateFunc.exp_decay(0.28, 27.0, 5.0))
        spec_h = m.to_spec()
        g_h = spec_h.gates[0]

        assert g_h.update_form == GateUpdateForm.ALPHA_BETA
        assert abs(g_h.alpha.A - 0.32) < 1e-12
        assert abs(g_h.beta.A  - 0.28) < 1e-12


# =============================================================================
# TestNeuronModelAddGate
# =============================================================================

class TestNeuronModelAddGate:
    """NeuronModel.add_gate() accepts every documented input variant."""

    def test_boltzmann_object(self):
        m = NeuronModel()
        idx = m.add_gate("m", update_form="instant", inf=Boltzmann(-37.0, 7.0))
        g = m.to_spec().gates[idx]
        assert g.update_form == GateUpdateForm.INSTANT
        assert abs(g.inf.v_half - (-37.0)) < 1e-12

    def test_boltzmann_tagged(self):
        m = NeuronModel()
        idx = m.add_gate("m", update_form="instant", inf=Boltzmann(-37.0, 7.0).to_tagged())
        g = m.to_spec().gates[idx]
        assert abs(g.inf.v_half - (-37.0)) < 1e-12

    def test_raw_boltzmann_params(self):
        bp = BoltzmannParams()
        bp.v_half = -40.0; bp.k = 9.0
        m = NeuronModel()
        idx = m.add_gate("n", update_form="instant", inf=bp)
        g = m.to_spec().gates[idx]
        assert abs(g.inf.v_half - (-40.0)) < 1e-12

    def test_tau_constant_tagged(self):
        m = NeuronModel()
        idx = m.add_gate("n", tau=Tau.constant(1.0))
        g = m.to_spec().gates[idx]
        assert g.tau.form == TauForm.CONSTANT
        assert abs(g.tau.get_param(0) - 1.0) < 1e-12

    def test_tau_float_shorthand(self):
        m = NeuronModel()
        idx = m.add_gate("h", tau=2.5)
        g = m.to_spec().gates[idx]
        assert g.tau.form == TauForm.CONSTANT
        assert abs(g.tau.get_param(0) - 2.5) < 1e-12

    def test_gs_kwarg_delegates_to_gate_spec(self):
        # GateSpec() now returns _CoreGateSpec directly — pass it via gs=
        core_g = GateSpec("m", inf=Boltzmann(-40.0, 8.0), tau=Tau.constant(0.5),
                          update_form="inf_tau")
        m = NeuronModel()
        idx = m.add_gate(gs=core_g)
        g = m.to_spec().gates[idx]
        assert g.update_form == GateUpdateForm.INF_TAU
        assert abs(g.inf.v_half - (-40.0)) < 1e-12

    def test_sympy_boltzmann_pattern_matches(self):
        """A SymPy Boltzmann expr that pattern-matches should not trigger JIT."""
        expr = 1 / (1 + sympy.exp(-(V - (-40.0)) / 8.0))
        m = NeuronModel()
        idx = m.add_gate("m", update_form="instant", inf=expr)
        g = m.to_spec().gates[idx]
        # Pattern-matched: update_form stays INSTANT, inf is a BoltzmannParams
        assert g.update_form == GateUpdateForm.INSTANT
        assert abs(g.inf.v_half - (-40.0)) < 1e-12


# =============================================================================
# TestGateSpecClass
# =============================================================================

class TestGateSpecFactory:
    """GateSpec() factory produces correctly configured _core.GateSpec instances."""

    def test_no_arg_returns_blank_core_spec(self):
        g = GateSpec()
        assert isinstance(g, _CoreGateSpec)

    def test_basic_inf_tau(self):
        g = GateSpec("m", inf=Boltzmann(-40.0, 8.0), tau=Tau.constant(0.5))
        assert isinstance(g, _CoreGateSpec)
        assert g.update_form == GateUpdateForm.INF_TAU
        assert abs(g.inf.v_half - (-40.0)) < 1e-12

    def test_alpha_beta_inferred(self):
        g = GateSpec("m",
                     alpha=RateFunc.linear_over_expm1(0.32, 54.0, 4.0),
                     beta=RateFunc.exp_decay(0.28, 27.0, 5.0))
        assert g.update_form == GateUpdateForm.ALPHA_BETA

    def test_instant_inferred(self):
        g = GateSpec("m", inf=Boltzmann(-37.0, 7.0))
        assert g.update_form == GateUpdateForm.INSTANT

    def test_dependency_auto_voltage(self):
        g = GateSpec("m", inf=Boltzmann(-37.0, 7.0))
        assert g.dependency == GateDependency.VOLTAGE

    def test_dependency_auto_calcium(self):
        expr = 1 / (1 + sympy.exp(-(Ca - 0.1) / 0.02))
        g = GateSpec("c", inf=expr)
        assert g.dependency == GateDependency.CALCIUM

    def test_dependency_explicit_calcium(self):
        g = GateSpec("c", inf=Boltzmann(0.1, 0.02), dependency="calcium")
        assert g.dependency == GateDependency.CALCIUM

    def test_scale_and_initial_value(self):
        g = GateSpec("m", inf=Boltzmann(-40.0, 8.0), scale=3.0, initial_value=0.5)
        assert abs(g.scale - 3.0) < 1e-12
        assert abs(g.initial_value - 0.5) < 1e-12

    def test_sympy_boltzmann_pattern_matched(self):
        expr = 1 / (1 + sympy.exp(-(V - (-55.0)) / 5.0))
        g = GateSpec("m", inf=expr, tau=Tau.constant(2.0))
        # Pattern match → stays in fast path, no CUSTOM_EXPR
        assert g.update_form == GateUpdateForm.INF_TAU
        assert g.inf_vm.empty()  # no bytecode compiled (fast pre-compiled path)

    def test_appendable_to_spec_gates(self):
        g = GateSpec("m", inf=Boltzmann(-40.0, 8.0), tau=Tau.constant(1.0))
        spec = NeuronModelSpec()
        spec.gates.append(g)  # must not raise
        assert len(spec.gates) == 1


# =============================================================================
# TestVMExprGate  (unconditional — no compiler needed)
# =============================================================================

class TestVMExprGate:
    """CUSTOM_EXPR path for novel SymPy gates — no g++ or Eigen3 needed at runtime."""

    def test_custom_expr_gate_runs(self):
        """Novel SymPy expression uses CUSTOM_EXPR and produces finite voltages."""
        novel_inf = 1 / (1 + sympy.exp(-(V + 40) / 8))
        jit_inf   = novel_inf * sympy.Rational(999, 1000) + sympy.Rational(1, 1000)
        jit_tau   = sympy.Float(2.0) + sympy.sqrt(V**2 + 1) * sympy.Float(0.0)

        m = NeuronModel("custom_gate", C_m=1.0, V_init=-65.0)
        m.add_gate("m", inf=jit_inf, tau=jit_tau)
        m.add_channel("Leak", g=0.1, E_rev=-65.0)
        spec = m.to_spec()

        from hodgkin_huxley._core import GateUpdateForm
        g = spec.gates[0]
        assert g.update_form == GateUpdateForm.CUSTOM_EXPR
        assert not g.inf_vm.empty()
        assert not g.tau_vm.empty()

        rn = hh.RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        traces = rn.simulate(5.0, dt=0.1, I_ext={"pop": 5.0})
        assert "pop" in traces
        assert np.all(np.isfinite(traces["pop"]))

    def test_pattern_matched_sympy_vs_helper_equivalent(self):
        """SymPy Boltzmann (pattern-matched, INF_TAU path) gives identical traces to Boltzmann() helper."""
        v_half, k = -41.0, -4.0
        inf_sympy = 1 / (1 + sympy.exp(-(V - v_half) / k))

        def make_model(use_sympy):
            m = NeuronModel("hh_like", C_m=1.0, V_init=-65.0)
            inf_arg = inf_sympy if use_sympy else Boltzmann(v_half, k)
            m.add_gate("m", update_form="instant", inf=inf_arg)
            m.add_channel("Leak", g=0.1, E_rev=-65.0)
            return m.to_spec()

        rn_sym = hh.RegionalNetwork()
        rn_sym.add_population("pop", 1, model=make_model(use_sympy=True))
        traces_sym = rn_sym.simulate(20.0, dt=0.1, I_ext={"pop": 5.0})

        rn_ref = hh.RegionalNetwork()
        rn_ref.add_population("pop", 1, model=make_model(use_sympy=False))
        traces_ref = rn_ref.simulate(20.0, dt=0.1, I_ext={"pop": 5.0})

        np.testing.assert_allclose(
            traces_sym["pop"], traces_ref["pop"], rtol=1e-6,
            err_msg="SymPy pattern-matched path differs from Boltzmann() helper path",
        )


# =============================================================================
# TestVMExprEdgeCases  (unconditional — no compiler needed)
# =============================================================================

class TestVMExprEdgeCases:
    """Edge cases for the CUSTOM_EXPR gate path."""

    def test_vm_alpha_beta_form_runs(self):
        """CUSTOM_EXPR with alpha/beta rate functions produces finite traces."""
        # Gaussian-shaped rates — don't match any catalog form
        alpha_expr = sympy.Float(0.5) * sympy.exp(-((V + 55) / sympy.Float(10.0)) ** 2)
        beta_expr  = sympy.Float(0.5) * sympy.exp(-((V + 80) / sympy.Float(10.0)) ** 2)

        m = NeuronModel("ab_vm", C_m=1.0, V_init=-65.0)
        m.add_gate("m", alpha=alpha_expr, beta=beta_expr)
        m.add_channel("Leak", g=0.1, E_rev=-65.0)
        spec = m.to_spec()

        from hodgkin_huxley._core import GateUpdateForm
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert not spec.gates[0].alpha_vm.empty()
        assert not spec.gates[0].beta_vm.empty()

        rn = hh.RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        traces = rn.simulate(10.0, dt=0.1, I_ext={"pop": 5.0})
        assert np.all(np.isfinite(traces["pop"]))

    def test_vm_calcium_dependent_gate_runs(self):
        """CUSTOM_EXPR gate with Ca as dependency produces finite traces."""
        # Hill-like function — doesn't match any catalog form
        ca_inf = Ca ** 2 / (Ca ** 2 + sympy.Float(0.001))

        m = NeuronModel("ca_vm", C_m=1.0, V_init=-65.0)
        m.add_gate("q", inf=ca_inf, tau=sympy.Float(10.0), dependency="calcium")
        m.add_channel("Leak", g=0.1, E_rev=-65.0)
        spec = m.to_spec()

        from hodgkin_huxley._core import GateUpdateForm, GateDependency
        g = spec.gates[0]
        assert g.update_form == GateUpdateForm.CUSTOM_EXPR
        assert g.dependency == GateDependency.CALCIUM
        assert not g.inf_vm.empty()

        rn = hh.RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        traces = rn.simulate(10.0, dt=0.1, I_ext={"pop": 5.0})
        assert np.all(np.isfinite(traces["pop"]))

    def test_vm_trig_gate_runs(self):
        """CUSTOM_EXPR gate using sin/cos functions produces finite traces."""
        # Odd inf expression using sin — just tests the trig opcode path
        trig_inf = (sympy.sin(V / sympy.Float(20.0)) + sympy.Float(1.0)) / sympy.Float(2.0)

        m = NeuronModel("trig_vm", C_m=1.0, V_init=-65.0)
        m.add_gate("m", inf=trig_inf, tau=sympy.Float(2.0))
        m.add_channel("Leak", g=0.1, E_rev=-65.0)
        spec = m.to_spec()

        from hodgkin_huxley._core import GateUpdateForm, VmOp
        g = spec.gates[0]
        assert g.update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in g.inf_vm.instructions]
        assert VmOp.SIN in ops

        rn = hh.RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        traces = rn.simulate(5.0, dt=0.1, I_ext={"pop": 5.0})
        assert np.all(np.isfinite(traces["pop"]))

    def test_vm_two_gates_independent(self):
        """Two independent CUSTOM_EXPR gates on the same model both run correctly."""
        novel1 = (1 / (1 + sympy.exp(-(V + 40) / 8))) * sympy.Rational(999, 1000) + sympy.Rational(1, 1000)
        novel2 = sympy.Float(0.5) + sympy.Float(0.4) * sympy.tanh(V / sympy.Float(10.0))

        m = NeuronModel("two_vm", C_m=1.0, V_init=-65.0)
        gi1 = m.add_gate("m", inf=novel1, tau=sympy.Float(1.0))
        gi2 = m.add_gate("h", inf=novel2, tau=sympy.Float(3.0))
        m.add_channel("Na", g=1.0, E_rev=50.0, gating=gi1**3 * gi2)

        spec = m.to_spec()
        from hodgkin_huxley._core import GateUpdateForm
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert spec.gates[1].update_form == GateUpdateForm.CUSTOM_EXPR

        rn = hh.RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        traces = rn.simulate(5.0, dt=0.1, I_ext={"pop": 5.0})
        assert np.all(np.isfinite(traces["pop"]))

    def test_vm_log_gate_runs(self):
        """CUSTOM_EXPR gate using log produces finite traces for positive inputs."""
        # log(|V| + 1) / 5 — stays in (0,1) for typical membrane potentials
        log_inf = sympy.log(sympy.Abs(V) + sympy.Integer(1)) / sympy.Float(5.0)

        m = NeuronModel("log_vm", C_m=1.0, V_init=-65.0)
        m.add_gate("m", inf=log_inf, tau=sympy.Float(1.0))
        m.add_channel("Leak", g=0.1, E_rev=-65.0)
        spec = m.to_spec()

        from hodgkin_huxley._core import GateUpdateForm, VmOp
        g = spec.gates[0]
        assert g.update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in g.inf_vm.instructions]
        assert VmOp.LOG in ops

        rn = hh.RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        traces = rn.simulate(5.0, dt=0.1, I_ext={"pop": 5.0})
        assert np.all(np.isfinite(traces["pop"]))


# =============================================================================
# TestDoubleExpSumPatternMatch
# =============================================================================

class TestDoubleExpSumPatternMatch:
    """Verify that DOUBLE_EXP_SUM tau expressions used in benchmark specs
    pattern-match to INF_TAU (not CUSTOM_EXPR), so the pre-compiled C++ path is
    used for all STN and TH gates."""

    def test_stn_gates_all_pattern_matched(self):
        """All STN NeuronModel gates must use INF_TAU or ALPHA_BETA — not CUSTOM_EXPR."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../benchmarks'))
        from ctxbgth_model import _make_stn_spec
        spec, _ = _make_stn_spec()
        custom_expr_gates = [
            g.name for g in spec.gates
            if g.update_form == GateUpdateForm.CUSTOM_EXPR
        ]
        assert custom_expr_gates == [], (
            f"STN gates fell to CUSTOM_EXPR (should be pattern-matched): {custom_expr_gates}"
        )

    def test_thalamic_gates_all_pattern_matched(self):
        """All TH NeuronModel gates must use INF_TAU or ALPHA_BETA — not CUSTOM_EXPR."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../benchmarks'))
        from ctxbgth_model import _make_th_spec
        spec = _make_th_spec()
        custom_expr_gates = [
            g.name for g in spec.gates
            if g.update_form == GateUpdateForm.CUSTOM_EXPR
        ]
        assert custom_expr_gates == [], (
            f"TH gates fell to CUSTOM_EXPR (should be pattern-matched): {custom_expr_gates}"
        )


# =============================================================================
# TestBackwardCompat
# =============================================================================

class TestBackwardCompat:
    """Legacy equation struct types accessible via hodgkin_huxley.legacy."""

    def test_boltzmann_params_via_legacy(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from hodgkin_huxley import legacy
            bp = legacy.BoltzmannParams
            assert len(w) >= 1
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert bp is BoltzmannParams

    def test_tau_form_via_legacy(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from hodgkin_huxley import legacy
            tf = legacy.TauForm
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert tf is TauForm

    def test_ratefunc_form_via_legacy(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from hodgkin_huxley import legacy
            rf = legacy.RateFuncForm
        assert rf is RateFuncForm

    def test_gate_update_form_via_legacy(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from hodgkin_huxley import legacy
            guf = legacy.GateUpdateForm
        assert guf is GateUpdateForm

    def test_deprecated_names_not_in_all(self):
        """Legacy names must not appear in hodgkin_huxley.__all__."""
        import hodgkin_huxley as hh
        assert "BoltzmannParams" not in hh.__all__
        assert "TauParams"       not in hh.__all__
        assert "RateFuncParams"  not in hh.__all__
        assert "GateUpdateForm"  not in hh.__all__
        assert "TauForm"         not in hh.__all__

    def test_deprecated_names_emit_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import hodgkin_huxley as hh
            _ = hh.BoltzmannParams
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_old_api_boltzmann_to_params_still_works(self):
        """Boltzmann.to_params() must still return a _BoltzmannParams struct."""
        b = Boltzmann(-40.0, 8.0)
        p = b.to_params()
        assert isinstance(p, BoltzmannParams)
        assert abs(p.v_half - (-40.0)) < 1e-12

    def test_neuron_model_accepts_raw_boltzmann_params(self):
        """NeuronModel.add_gate() still accepts raw BoltzmannParams structs."""
        bp = BoltzmannParams()
        bp.v_half = -40.0; bp.k = 8.0
        m = NeuronModel()
        idx = m.add_gate("m", update_form="instant", inf=bp)
        g = m.to_spec().gates[idx]
        assert abs(g.inf.v_half - (-40.0)) < 1e-12
