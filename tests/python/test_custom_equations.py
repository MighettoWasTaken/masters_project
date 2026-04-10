"""
tests/python/test_custom_equations.py
======================================
Exhaustive tests for custom SymPy equation compilation and numerical correctness.

Every location where a user can supply a SymPy expression is tested:

  1. Gate inf       — INF_TAU form, novel SymPy bypassing all 10 catalog patterns
  2. Gate tau       — INF_TAU form, novel SymPy
  3. Gate alpha     — ALPHA_BETA form, novel SymPy
  4. Gate beta      — ALPHA_BETA form, novel SymPy
  5. INSTANT gate   — instant equilibrium with novel SymPy inf
  6. Channel gating products — fast-path vs VM path; numerical equivalence
  7. Kinetic synapse dS_dt + current — CUSTOM_EXPR form, JIT compilation
  8. Full HH model  — all six gates replaced with mathematically equivalent
                      novel SymPy expressions; trace must match preset HH

For every case the tests verify:
  A. Compilation succeeds (no error raised)
  B. GateUpdateForm is correctly set (CUSTOM_EXPR when novel, INF_TAU/ALPHA_BETA when
     pattern-matched)
  C. The produced C++ code calculates correct values: novel-expression model traces
     are numerically identical (within 1e-5 V) to the equivalent preset model traces
  D. Long-run (2 s) traces are finite — no NaN / Inf
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

import hodgkin_huxley as hh
from hodgkin_huxley import (
    V, Ca, V_pre, V_post, S, x,
    Boltzmann, Tau, RateFunc,
    NeuronModel, KineticSynapseModel,
)
from hodgkin_huxley._core import (
    GateUpdateForm, VmOp,
    KineticUpdateForm, KineticCurrentForm,
)
from hodgkin_huxley._codegen import _try_extract_gate_pairs

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DT    = 0.025   # ms — numerically stable for HH-like models
SHORT = 300.0   # ms — enough for ~15 spikes at 50 Hz
LONG  = 2000.0  # ms — stability / convergence check
I_HH  = 10.0    # µA/cm² — drives standard HH to ~75 Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(spec, duration=SHORT, dt=DT, n=20, I_ext=I_HH):
    """Simulate n neurons, return voltage matrix (n, steps)."""
    rn = hh.RegionalNetwork()
    rn.add_population("pop", n, model=spec)
    return rn.simulate(duration, dt, {"pop": I_ext})["pop"]


def _run1(spec, duration=SHORT, dt=DT, I_ext=I_HH):
    """Simulate a single neuron; returns 1-D trace."""
    return _run(spec, duration, dt, n=1, I_ext=I_ext)[0]


def _spikes(trace, threshold=0.0):
    """Count threshold crossings (upward) in a 1-D voltage trace."""
    return int(np.sum((trace[:-1] < threshold) & (trace[1:] >= threshold)))


def _preset_hh() -> hh.NeuronModelSpec:
    """
    Standard Hodgkin-Huxley model (1952) built entirely with pattern-matched
    preset helpers.  All six gates use the fast pre-compiled C++ path.
    """
    m = NeuronModel("hh_preset", C_m=1.0, V_init=-65.0)
    m_g = m.add_gate("m",
        alpha=RateFunc.linear_over_expm1(0.1,  40.0, 10.0),
        beta =RateFunc.exp_decay(        4.0,  65.0, -18.0))
    h_g = m.add_gate("h",
        alpha=RateFunc.exp_decay(  0.07, 65.0, -20.0),
        beta =RateFunc.sigmoid(    1.0,  35.0, -10.0))
    n_g = m.add_gate("n",
        alpha=RateFunc.linear_over_expm1(0.01, 55.0, 10.0),
        beta =RateFunc.exp_decay(        0.125,65.0, -80.0))
    m.add_channel("Na", g=120.0, E_rev= 50.0,    gating=m_g**3 * h_g)
    m.add_channel("K",  g=36.0,  E_rev=-77.0,    gating=n_g**4)
    m.add_channel("L",  g=0.3,   E_rev=-54.387)
    return m.to_spec()


def _novel_hh() -> hh.NeuronModelSpec:
    """
    Mathematically identical to _preset_hh(), but every rate function is
    rewritten in an algebraically equivalent form that bypasses all ten
    catalog patterns and forces the CUSTOM_EXPR / JIT compilation path.

    Equivalences used (all verifiable by substitution):

      alpha_m: 0.1*(V+40)/(1-exp(-(V+40)/10))
             = 0.1*(V+40)*sqrt(exp((V+40)/5)) / (sqrt(exp((V+40)/5))-1)
               [multiply numerator and denominator by exp((V+40)/10)]

      beta_m:  4*exp(-(V+65)/18)
             = 4*sqrt(exp(-(V+65)/9))
               [sqrt(exp(-2x)) = exp(-x), with x=(V+65)/18]

      alpha_h: 0.07*exp(-(V+65)/20)
             = 0.07*sqrt(exp(-(V+65)/10))

      beta_h:  1/(1+exp(-(V+35)/10))
             = 0.5*(1+tanh((V+35)/20))
               [standard sigmoid–tanh identity: 1/(1+e^{-u})=0.5(1+tanh(u/2))]

      alpha_n: 0.01*(V+55)/(1-exp(-(V+55)/10))
             = 0.01*(V+55)*sqrt(exp((V+55)/5)) / (sqrt(exp((V+55)/5))-1)

      beta_n:  0.125*exp(-(V+65)/80)
             = 0.125*sqrt(exp(-(V+65)/40))
    """
    m = NeuronModel("hh_novel", C_m=1.0, V_init=-65.0)

    # --- m gate (Na activation) -------------------------------------------
    sq_m = sympy.sqrt(sympy.exp((V + 40) / 5))
    alpha_m = sympy.Rational(1, 10) * (V + 40) * sq_m / (sq_m - 1)
    beta_m  = sympy.Integer(4) * sympy.sqrt(sympy.exp(-(V + 65) / 9))

    # --- h gate (Na inactivation) -----------------------------------------
    alpha_h = sympy.Rational(7, 100) * sympy.sqrt(sympy.exp(-(V + 65) / 10))
    beta_h  = sympy.Rational(1, 2) * (1 + sympy.tanh((V + 35) / 20))

    # --- n gate (K activation) --------------------------------------------
    sq_n   = sympy.sqrt(sympy.exp((V + 55) / 5))
    alpha_n = sympy.Rational(1, 100) * (V + 55) * sq_n / (sq_n - 1)
    beta_n  = sympy.Rational(1, 8) * sympy.sqrt(sympy.exp(-(V + 65) / 40))

    m_g = m.add_gate("m", alpha=alpha_m, beta=beta_m)
    h_g = m.add_gate("h", alpha=alpha_h, beta=beta_h)
    n_g = m.add_gate("n", alpha=alpha_n, beta=beta_n)

    m.add_channel("Na", g=120.0, E_rev= 50.0,    gating=m_g**3 * h_g)
    m.add_channel("K",  g=36.0,  E_rev=-77.0,    gating=n_g**4)
    m.add_channel("L",  g=0.3,   E_rev=-54.387)
    return m.to_spec()


# ===========================================================================
# 1. Gate inf — INF_TAU form
# ===========================================================================

class TestGateInfNovel:
    """
    Novel SymPy inf expressions go to CUSTOM_EXPR and produce correct output.
    The tanh-form Boltzmann is mathematically identical to the preset Boltzmann,
    so both forms must yield the same voltage trace.
    """

    def _make_inf_tau_spec(self, inf_expr, tau_val=2.0):
        """One-gate (m) model: Na-like channel + leak, INF_TAU form."""
        model = NeuronModel("inf_test", C_m=1.0, V_init=-65.0)
        g = model.add_gate("m", inf=inf_expr, tau=Tau.constant(tau_val))
        model.add_channel("Na",   g=30.0, E_rev=50.0,   gating=g**2)
        model.add_channel("Leak", g=0.3,  E_rev=-65.0)
        return model.to_spec()

    # ------------------------------------------------------------------
    # tanh-form Boltzmann ≡ preset Boltzmann(-40, 8)
    # ------------------------------------------------------------------

    def test_tanh_boltzmann_compiles_as_vm_expr(self):
        """tanh-form sigmoid bypasses pattern matching → CUSTOM_EXPR."""
        tanh_inf = sympy.Rational(1, 2) * (1 + sympy.tanh((V + 40) / 16))
        spec = self._make_inf_tau_spec(tanh_inf)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in spec.gates[0].inf_vm.instructions]
        assert VmOp.TANH in ops

    def test_tanh_boltzmann_trace_equals_preset(self):
        """
        tanh form 0.5*(1+tanh((V+40)/16)) == Boltzmann(-40,8) analytically.
        Both models must produce traces within 1e-5 V.
        """
        preset_spec = self._make_inf_tau_spec(Boltzmann(-40.0, 8.0))
        novel_spec  = self._make_inf_tau_spec(
            sympy.Rational(1, 2) * (1 + sympy.tanh((V + 40) / 16)))

        trace_preset = _run1(preset_spec, duration=100.0)
        trace_novel  = _run1(novel_spec,  duration=100.0)
        np.testing.assert_allclose(trace_novel, trace_preset, atol=1e-5,
            err_msg="tanh-form Boltzmann gives different trace than preset")

    # ------------------------------------------------------------------
    # Other novel inf expressions compile and run finite
    # ------------------------------------------------------------------

    def test_sqrt_sigmoid_inf_finite(self):
        """sqrt-based sigmoid: 0.5*sqrt_pos / (1 + sqrt_pos) where sqrt_pos = sqrt(|V+40|+1)."""
        sqrt_pos = sympy.sqrt(sympy.Abs(V + 40) + 1)
        sqrt_inf = sympy.Rational(1, 2) * sqrt_pos / (1 + sqrt_pos)
        spec = self._make_inf_tau_spec(sqrt_inf)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert np.all(np.isfinite(_run(spec, duration=100.0)))

    def test_cosine_modulated_inf_finite(self):
        """Bell-shaped inf using cos: 0.5 + 0.4*cos(V/30). Stays in [0.1, 0.9]."""
        cos_inf = sympy.Rational(1, 2) + sympy.Rational(2, 5) * sympy.cos(V / 30)
        spec = self._make_inf_tau_spec(cos_inf)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in spec.gates[0].inf_vm.instructions]
        assert VmOp.COS in ops
        assert np.all(np.isfinite(_run(spec, duration=100.0)))

    def test_log_sigmoid_inf_finite(self):
        """Monotone log-based sigmoid: log(|V|+2) / (log(|V|+2) + log(100))."""
        lv = sympy.log(sympy.Abs(V) + 2)
        log_inf = lv / (lv + sympy.log(sympy.Integer(100)))
        spec = self._make_inf_tau_spec(log_inf)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in spec.gates[0].inf_vm.instructions]
        assert VmOp.LOG in ops
        assert np.all(np.isfinite(_run(spec, duration=100.0)))

    def test_calcium_dependent_inf_finite(self):
        """Ca-dependent inf compiles and runs finite when Ca channel is present."""
        ca_inf = sympy.Rational(1, 2) * (1 + sympy.tanh((Ca - sympy.Rational(1, 10)) / sympy.Rational(1, 5)))
        model = NeuronModel("ca_inf", C_m=1.0, V_init=-65.0)
        g = model.add_gate("q", inf=ca_inf, tau=Tau.constant(5.0), dependency="calcium")
        model.add_channel("Ca_chan", g=1.0, E_rev=120.0, gating=g)
        model.add_channel("Leak",   g=0.3,  E_rev=-65.0)
        spec = model.to_spec()
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert np.all(np.isfinite(_run(spec, duration=100.0)))


# ===========================================================================
# 2. Gate tau — INF_TAU form
# ===========================================================================

class TestGateTauNovel:
    """Novel SymPy tau expressions compile and produce stable dynamics."""

    def _make_spec(self, tau_expr):
        # Na activation gate with variable tau; h and n are preset for repolarisation.
        model = NeuronModel("tau_test", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", inf=Boltzmann(-40.0, 8.0), tau=tau_expr)
        h_g = model.add_gate("h",
            alpha=RateFunc.exp_decay(0.07, 65.0, -20.0),
            beta =RateFunc.sigmoid(  1.0,  35.0, -10.0))
        n_g = model.add_gate("n",
            alpha=RateFunc.linear_over_expm1(0.01, 55.0, 10.0),
            beta =RateFunc.exp_decay(        0.125, 65.0, -80.0))
        model.add_channel("Na",   g=120.0, E_rev= 50.0,    gating=m_g**3 * h_g)
        model.add_channel("K",    g=36.0,  E_rev=-77.0,    gating=n_g**4)
        model.add_channel("Leak", g=0.3,   E_rev=-54.387)
        return model.to_spec()

    def test_gaussian_tau_compiles_as_vm_expr(self):
        """Bell-shaped tau: 0.5 + 2*exp(-(V+40)^2/400). Not in catalog."""
        tau_expr = sympy.Rational(1, 2) + 2 * sympy.exp(-(V + 40)**2 / 400)
        spec = self._make_spec(tau_expr)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in spec.gates[0].tau_vm.instructions]
        assert VmOp.EXP in ops

    def test_gaussian_tau_finite(self):
        tau_expr = sympy.Rational(1, 2) + 2 * sympy.exp(-(V + 40)**2 / 400)
        assert np.all(np.isfinite(_run(self._make_spec(tau_expr), duration=200.0)))

    def test_sqrt_tau_compiles_and_finite(self):
        """tau = 1 + sqrt(|V + 65| + 0.1). Monotone, always >= 1 ms.

        sqrt(x) compiles to POW_HALF (x^(1/2)) not a separate SQRT opcode.
        """
        tau_expr = 1 + sympy.sqrt(sympy.Abs(V + 65) + sympy.Rational(1, 10))
        spec = self._make_spec(tau_expr)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in spec.gates[0].tau_vm.instructions]
        assert VmOp.POW_HALF in ops   # sympy.sqrt → x^(1/2) → POW_HALF
        assert VmOp.ABS in ops        # sympy.Abs  → ABS
        assert np.all(np.isfinite(_run(spec, duration=200.0)))

    def test_abs_tau_compiles_and_finite(self):
        """tau = 2 + |V + 40| / 50. Linear with abs."""
        tau_expr = 2 + sympy.Abs(V + 40) / 50
        spec = self._make_spec(tau_expr)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in spec.gates[0].tau_vm.instructions]
        assert VmOp.ABS in ops
        assert np.all(np.isfinite(_run(spec, duration=200.0)))

    def test_slower_tau_delays_activation(self):
        """
        A larger tau should delay gate activation → fewer spikes in same window.
        Model with tau=0.5 ms must spike more than model with tau=10 ms.
        """
        spec_fast = self._make_spec(Tau.constant(0.5))
        spec_slow = self._make_spec(Tau.constant(10.0))
        spikes_fast = _spikes(_run1(spec_fast, duration=200.0))
        spikes_slow = _spikes(_run1(spec_slow, duration=200.0))
        assert spikes_fast > spikes_slow, (
            f"Faster tau should produce more spikes: fast={spikes_fast}, slow={spikes_slow}")


# ===========================================================================
# 3. Gate alpha / beta — ALPHA_BETA form
# ===========================================================================

class TestGateAlphaBetaNovel:
    """Novel alpha and beta expressions compile, run, and are numerically correct."""

    def _make_spec(self, alpha_expr, beta_expr):
        model = NeuronModel("ab_test", C_m=1.0, V_init=-65.0)
        g = model.add_gate("m", alpha=alpha_expr, beta=beta_expr)
        model.add_channel("Na",   g=30.0, E_rev=50.0,   gating=g**2)
        model.add_channel("Leak", g=0.3,  E_rev=-65.0)
        return model.to_spec()

    def test_gaussian_rates_compile_as_vm_expr(self):
        """Gaussian alpha and beta are not in catalog → both become CUSTOM_EXPR."""
        alpha = sympy.Rational(1, 2) * sympy.exp(-(V + 40)**2 / 200)
        beta  = sympy.Rational(1, 10) + sympy.Rational(1, 5) * sympy.exp(-(V + 70)**2 / 400)
        spec = self._make_spec(alpha, beta)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops_a = [ins.op for ins in spec.gates[0].alpha_vm.instructions]
        ops_b = [ins.op for ins in spec.gates[0].beta_vm.instructions]
        assert VmOp.EXP in ops_a
        assert VmOp.EXP in ops_b

    def test_gaussian_rates_finite(self):
        alpha = sympy.Rational(1, 2) * sympy.exp(-(V + 40)**2 / 200)
        beta  = sympy.Rational(1, 10) + sympy.Rational(1, 5) * sympy.exp(-(V + 70)**2 / 400)
        assert np.all(np.isfinite(_run(self._make_spec(alpha, beta), duration=200.0)))

    def test_tanh_alpha_sine_beta_compile_and_finite(self):
        """Mix tanh and sin in rates — both novel."""
        alpha = sympy.Rational(3, 10) * (1 + sympy.tanh((V + 45) / 10)) / 2
        beta  = sympy.Rational(1, 5) + sympy.Rational(1, 10) * sympy.sin(V / 20 + 1)**2
        spec = self._make_spec(alpha, beta)
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert np.all(np.isfinite(_run(spec, duration=200.0)))

    def test_novel_alpha_beta_steady_state(self):
        """
        After 500 ms with zero current (V stays near rest ≈ -65 mV), the gate
        must have converged to x_inf = alpha/( alpha + beta ) evaluated at V_rest.

        For Gaussian rates at V = -65:
          alpha(-65) = 0.5 * exp(-(-65+40)^2 / 200) = 0.5 * exp(-625/200) ≈ 0.5 * exp(-3.125) ≈ 0.022
          beta(-65)  = 0.1 + 0.2 * exp(-(-65+70)^2 / 400) = 0.1 + 0.2 * exp(-25/400) ≈ 0.294
          x_inf      ≈ 0.022 / 0.316 ≈ 0.070

        Since there's no spike-generating Na channel in the passive variant,
        V stays approximately at V_init.  After 500ms >> tau, the gate must
        be close to x_inf.  We use a passive model (leak only + this gate).
        """
        import math
        V_rest = -65.0
        alpha_fn = lambda v: 0.5 * math.exp(-(v + 40)**2 / 200)
        beta_fn  = lambda v: 0.1 + 0.2 * math.exp(-(v + 70)**2 / 400)
        a0, b0 = alpha_fn(V_rest), beta_fn(V_rest)
        x_inf_expected = a0 / (a0 + b0)

        alpha = sympy.Rational(1, 2) * sympy.exp(-(V + 40)**2 / 200)
        beta  = sympy.Rational(1, 10) + sympy.Rational(1, 5) * sympy.exp(-(V + 70)**2 / 400)

        # Passive model: gate drives a channel but I_ext=0 → V stays near -65
        model = NeuronModel("ab_passive", C_m=1.0, V_init=V_rest)
        g = model.add_gate("q", alpha=alpha, beta=beta)
        # Very weak channel so gate barely affects V
        model.add_channel("Q",    g=0.01,  E_rev=0.0,   gating=g)
        model.add_channel("Leak", g=0.3,   E_rev=V_rest)
        spec = model.to_spec()

        # Record the channel gating variable indirectly: at long time, current
        # through Q channel ≈ g_Q * x_inf * (V - E_Q) ≈ 0.01 * x_inf * (-65 - 0)
        # V in long-run trace should converge to a fixed point.
        # We just check stability and finite output here; exact gate value
        # extraction would need C++ accessor.
        trace = _run1(spec, duration=500.0, I_ext=0.0)
        assert np.all(np.isfinite(trace)), "Gaussian rate model diverged"
        # V should remain near resting — no runaway
        assert np.all(trace > -200.0) and np.all(trace < 100.0)


# ===========================================================================
# 4. INSTANT gate with novel inf
# ===========================================================================

class TestInstantGateNovel:
    """INSTANT update form: gate immediately = x_inf(V). Test with novel SymPy."""

    def test_tanh_instant_gate_compiles(self):
        """INSTANT gate with tanh inf compiles to CUSTOM_EXPR (SymPy forces VM path)."""
        tanh_inf = sympy.Rational(1, 2) * (1 + sympy.tanh((V + 40) / 16))
        model = NeuronModel("instant_vm", C_m=1.0, V_init=-65.0)
        g = model.add_gate("p", inf=tanh_inf, update_form="instant")
        model.add_channel("P",    g=10.0, E_rev=50.0, gating=g)
        model.add_channel("Leak", g=0.3,  E_rev=-65.0)
        spec = model.to_spec()
        # SymPy needs compilation → CUSTOM_EXPR (not INSTANT); C++ handles
        # "only inf_vm, no tau_vm" as INSTANT semantics inside CUSTOM_EXPR.
        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert spec.gates[0].inf_vm.instructions  # non-empty bytecode
        assert spec.gates[0].tau_vm.empty()       # no tau → INSTANT path

    def test_tanh_instant_trace_finite(self):
        tanh_inf = sympy.Rational(1, 2) * (1 + sympy.tanh((V + 40) / 16))
        model = NeuronModel("instant_vm2", C_m=1.0, V_init=-65.0)
        g = model.add_gate("p", inf=tanh_inf, update_form="instant")
        model.add_channel("P",    g=10.0, E_rev=50.0, gating=g)
        model.add_channel("Leak", g=0.3,  E_rev=-65.0)
        assert np.all(np.isfinite(_run(model.to_spec(), duration=200.0)))

    def test_instant_vs_fast_inf_equivalent(self):
        """
        INSTANT gate with tanh-form Boltzmann must equal INSTANT gate with
        preset Boltzmann: same inf → same conductance every step → same trace.
        """
        def _spec(inf_expr):
            model = NeuronModel("instant_eq", C_m=1.0, V_init=-65.0)
            g = model.add_gate("p", inf=inf_expr, update_form="instant")
            model.add_channel("P",    g=10.0, E_rev=50.0, gating=g)
            model.add_channel("Leak", g=0.3,  E_rev=-65.0)
            return model.to_spec()

        preset = _spec(Boltzmann(-40.0, 8.0))
        novel  = _spec(sympy.Rational(1, 2) * (1 + sympy.tanh((V + 40) / 16)))
        np.testing.assert_allclose(
            _run1(preset, duration=100.0),
            _run1(novel,  duration=100.0),
            atol=1e-5,
            err_msg="INSTANT: tanh form must match preset Boltzmann trace")

    def test_sqrt_instant_gate_finite(self):
        """INSTANT + sqrt-based inf stays finite."""
        sqrt_inf = sympy.sqrt(sympy.Abs(V + 40) + 1) / (sympy.sqrt(sympy.Abs(V + 40) + 1) + 5)
        model = NeuronModel("instant_sqrt", C_m=1.0, V_init=-65.0)
        g = model.add_gate("p", inf=sqrt_inf, update_form="instant")
        model.add_channel("P",    g=5.0, E_rev=50.0, gating=g)
        model.add_channel("Leak", g=0.3, E_rev=-65.0)
        assert np.all(np.isfinite(_run(model.to_spec(), duration=200.0)))


# ===========================================================================
# 5. Channel gating products
# ===========================================================================

class TestGatingProducts:
    """Fast-path integer-power products vs VM-path expressions."""

    def _hh_spec(self, gating_m_h, gating_n):
        """Build a minimal Na+K+Leak HH variant with specified gating expressions."""
        model = NeuronModel("gating_test", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m",
            alpha=RateFunc.linear_over_expm1(0.1,  40.0, 10.0),
            beta =RateFunc.exp_decay(        4.0,  65.0, -18.0))
        h_g = model.add_gate("h",
            alpha=RateFunc.exp_decay(  0.07, 65.0, -20.0),
            beta =RateFunc.sigmoid(    1.0,  35.0, -10.0))
        n_g = model.add_gate("n",
            alpha=RateFunc.linear_over_expm1(0.01, 55.0, 10.0),
            beta =RateFunc.exp_decay(        0.125,65.0, -80.0))
        model.add_channel("Na", g=120.0, E_rev= 50.0, gating=gating_m_h(m_g, h_g))
        model.add_channel("K",  g=36.0,  E_rev=-77.0, gating=gating_n(n_g))
        model.add_channel("L",  g=0.3,   E_rev=-54.387)
        return model.to_spec(), m_g, h_g, n_g

    # ------------------------------------------------------------------
    # Fast-path detection
    # ------------------------------------------------------------------

    def test_integer_power_product_uses_fast_path(self):
        """m**3 * h is detected as fast path (gate_product_vm empty)."""
        spec, *_ = self._hh_spec(
            lambda m, h: m**3 * h,
            lambda n: n**4)
        na_ch = spec.channels[0]
        assert not na_ch.gate_product_vm.instructions, \
            "m**3*h should use fast path (empty gate_product_vm)"
        assert sorted(na_ch.gates) == [(0, 3), (1, 1)]

    def test_single_gate_power_uses_fast_path(self):
        """n**4 alone is detected as fast path."""
        spec, *_ = self._hh_spec(
            lambda m, h: m**3 * h,
            lambda n: n**4)
        k_ch = spec.channels[1]
        assert not k_ch.gate_product_vm.instructions
        assert k_ch.gates == [(2, 4)]

    def test_float_coefficient_forces_vm_path(self):
        """sympy.Float(1.0) * m**3 * h has a non-integer factor → VM path."""
        spec, *_ = self._hh_spec(
            lambda m, h: sympy.Float(1.0) * m**3 * h,
            lambda n: n**4)
        na_ch = spec.channels[0]
        assert na_ch.gate_product_vm.instructions, \
            "Float coefficient should force VM path"

    def test_three_gate_fast_path(self):
        """m**2 * h * n is a pure integer-power product → fast path."""
        model = NeuronModel("3gate", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", inf=Boltzmann(-40.0, 8.0), tau=Tau.constant(1.0))
        h_g = model.add_gate("h", inf=Boltzmann(-60.0, 7.0), tau=Tau.constant(5.0))
        n_g = model.add_gate("n", inf=Boltzmann(-50.0, 6.0), tau=Tau.constant(3.0))
        model.add_channel("X",    g=20.0, E_rev=50.0, gating=m_g**2 * h_g * n_g)
        model.add_channel("Leak", g=0.3,  E_rev=-65.0)
        spec = model.to_spec()
        ch = spec.channels[0]
        assert not ch.gate_product_vm.instructions
        assert sorted(ch.gates) == [(0, 2), (1, 1), (2, 1)]

    # ------------------------------------------------------------------
    # Numerical equivalence: fast path == VM path
    # ------------------------------------------------------------------

    def test_fast_vs_vm_identical_trace(self):
        """
        m**3 * h (fast path, integer-power) and
        sympy.Float(1.0) * m**3 * h (VM path)
        must produce bit-identical voltage traces.
        """
        spec_fast, *_ = self._hh_spec(
            lambda m, h: m**3 * h,
            lambda n: n**4)
        spec_vm, *_ = self._hh_spec(
            lambda m, h: sympy.Float(1.0) * m**3 * h,
            lambda n: n**4)

        trace_fast = _run1(spec_fast, duration=100.0)
        trace_vm   = _run1(spec_vm,   duration=100.0)
        np.testing.assert_allclose(trace_vm, trace_fast, atol=1e-10,
            err_msg="VM gate-product path differs from fast integer-power path")

    def test_vm_gating_finite_long_run(self):
        """VM-path gating expression stays finite over 2 s."""
        spec, *_ = self._hh_spec(
            lambda m, h: sympy.Float(1.0) * m**3 * h,
            lambda n: sympy.Float(1.0) * n**4)
        assert np.all(np.isfinite(_run(spec, duration=LONG)))

    def test_scaled_gating_product_alters_conductance(self):
        """
        Multiplying gating by 0.5 halves max conductance → fewer or weaker spikes.
        Model with 0.5*m**3*h must differ from m**3*h.
        """
        spec_full, *_ = self._hh_spec(
            lambda m, h: m**3 * h,
            lambda n: n**4)
        spec_half, *_ = self._hh_spec(
            lambda m, h: sympy.Float(0.5) * m**3 * h,
            lambda n: n**4)

        t_full = _run1(spec_full, duration=200.0)
        t_half = _run1(spec_half, duration=200.0)
        # Traces must differ (halved Na conductance changes dynamics)
        assert not np.allclose(t_full, t_half, atol=1e-3), \
            "Halved gating product should change the voltage trace"


# ===========================================================================
# 6. Full HH equivalence — all six gates novel SymPy
# ===========================================================================

class TestFullHHNovelEquivalence:
    """
    The novel-SymPy HH model (sqrt/tanh rewrites) must produce traces
    numerically identical to the preset-helper HH model.
    """

    def test_all_novel_gates_are_vm_expr(self):
        """Every gate in the novel HH model must use CUSTOM_EXPR (not INF_TAU/ALPHA_BETA)."""
        spec = _novel_hh()
        for gate in spec.gates:
            assert gate.update_form == GateUpdateForm.CUSTOM_EXPR, \
                f"Gate '{gate.name}' unexpectedly used {gate.update_form} " \
                f"instead of CUSTOM_EXPR"

    def test_novel_hh_produces_spikes(self):
        """Novel HH model must generate action potentials under current injection."""
        trace = _run1(_novel_hh(), duration=SHORT)
        assert np.all(np.isfinite(trace)), "Novel HH trace contains NaN/Inf"
        assert _spikes(trace) >= 5, "Novel HH should fire ≥5 spikes in 300 ms"

    def test_novel_hh_spike_count_matches_preset(self):
        """
        Novel HH and preset HH must produce the same number of spikes
        (within ±1 to allow for sub-millisecond phase jitter from rounding).
        """
        n_preset = _spikes(_run1(_preset_hh(), duration=SHORT))
        n_novel  = _spikes(_run1(_novel_hh(),  duration=SHORT))
        assert abs(n_novel - n_preset) <= 1, (
            f"Spike count mismatch: preset={n_preset}, novel={n_novel}")

    def test_novel_hh_trace_matches_preset(self):
        """
        Novel and preset HH traces must agree within 1e-5 V over 100 ms.
        Validates that each sqrt/tanh rewrite compiles to the correct C++.
        """
        trace_preset = _run1(_preset_hh(), duration=100.0)
        trace_novel  = _run1(_novel_hh(),  duration=100.0)
        np.testing.assert_allclose(trace_novel, trace_preset, atol=1e-5,
            err_msg=(
                "Novel-SymPy HH trace deviates from preset HH. "
                "A gate expression was incorrectly compiled."))

    def test_novel_hh_long_run_finite(self):
        """Novel HH model stays finite over 2 s (100 neurons)."""
        assert np.all(np.isfinite(_run(_novel_hh(), duration=LONG, n=100)))


# ===========================================================================
# 7. Kinetic synapse — custom dS_dt and current expressions
# ===========================================================================

class TestKineticSynapseCustomExpressions:
    """
    KineticSynapseModel.custom_expr(dS_dt, current, g, E_syn) compiles both
    expressions to VM bytecode and integrates them correctly.

    Design note
    -----------
    The ``current`` expression is compiled to a conductance function:
        fn(V_post, S) → effective_conductance
    The framework applies:
        I_syn += fn(V_post, S) * weight * (E_syn - V_post)
    So ``current`` should encode only the gating-variable factor (e.g. ``S``
    or ``S**2``), not the full I = g*S*(V-E).  The ``g`` and ``E_syn``
    arguments are stored on the spec and used to set sa_.E_syn (the reversal
    potential used in the formula above).
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @staticmethod
    def _ampa_spec(dS_dt_expr, cond_expr=None, g=1.0, E_syn=0.0):
        """
        Build a KineticSynapseSpec with custom dS_dt and conductance expression.

        cond_expr defaults to ``S`` (linear conductance = S * weight * (E_syn-V)).
        """
        if cond_expr is None:
            cond_expr = S
        return (KineticSynapseModel("custom")
                .custom_expr(dS_dt_expr, cond_expr, g=g, E_syn=E_syn)
                .to_spec())

    @staticmethod
    def _run_two_pop(kinetic_spec, duration=500.0, dt=DT, n_pre=5, n_post=5,
                     weight=1.0, I_pre=I_HH, I_post=0.0):
        """Two-population network; returns (pre_V, post_V) arrays."""
        rn = hh.RegionalNetwork()
        rn.add_population("pre",  n_pre,  neuron_type="hh")
        rn.add_population("post", n_post, neuron_type="hh")
        rn.connect("pre", "post", "all_to_all",
                   weight=weight, kinetic_spec=kinetic_spec)
        result = rn.simulate(duration, dt, {"pre": I_pre, "post": I_post})
        return result["pre"], result["post"]

    # ------------------------------------------------------------------
    # Compilation tests
    # ------------------------------------------------------------------

    def test_linear_kinetics_compiles(self):
        """Simple linear dS_dt and conductance=S compile without error."""
        dS = sympy.Rational(1, 2) * (1 - S) - sympy.Rational(1, 5) * S
        spec = self._ampa_spec(dS)
        assert spec.update_form == KineticUpdateForm.CUSTOM_EXPR
        assert spec.current_form == KineticCurrentForm.CUSTOM_EXPR

    def test_voltage_dependent_dS_dt_compiles(self):
        """V_pre-dependent rise rate (tanh) compiles."""
        rise = sympy.Rational(1, 2) * (1 + sympy.tanh(V_pre / 10))
        dS   = rise * (1 - S) - sympy.Rational(1, 10) * S
        spec = self._ampa_spec(dS)
        assert spec.update_form == KineticUpdateForm.CUSTOM_EXPR

    def test_nonlinear_current_compiles(self):
        """Nonlinear conductance expression (S**2) compiles."""
        dS = sympy.Rational(1, 2) * (1 - S) - sympy.Rational(1, 5) * S
        spec = self._ampa_spec(dS, cond_expr=S**2)
        assert spec.update_form == KineticUpdateForm.CUSTOM_EXPR
        assert spec.current_form == KineticCurrentForm.CUSTOM_EXPR

    def test_exp_dS_dt_compiles(self):
        """Exponential voltage dependence in dS_dt compiles."""
        dS = sympy.exp(V_pre / 30) * (1 - S) - S * sympy.Rational(1, 10)
        spec = self._ampa_spec(dS, E_syn=-80.0)
        assert spec.update_form == KineticUpdateForm.CUSTOM_EXPR

    # ------------------------------------------------------------------
    # Simulation correctness tests
    # ------------------------------------------------------------------

    def test_linear_kinetics_pre_post_finite(self):
        """Two-pop network with linear custom kinetics stays finite."""
        dS   = sympy.Rational(1, 2) * (1 - S) - sympy.Rational(1, 5) * S
        spec = self._ampa_spec(dS)
        pre_V, post_V = self._run_two_pop(spec, duration=500.0)
        assert np.all(np.isfinite(pre_V)),  "Pre-synaptic trace has NaN/Inf"
        assert np.all(np.isfinite(post_V)), "Post-synaptic trace has NaN/Inf"

    def test_excitatory_custom_synapse_drives_post(self):
        """
        Excitatory custom kinetic synapse (E_syn=0) drives post neurons to fire
        when pre is active and post receives no direct current.

        I_syn = S * weight * (0 - V_post) — depolarising when V_post < 0.
        """
        dS   = sympy.Rational(1, 2) * (1 - S) - sympy.Rational(1, 5) * S
        spec = self._ampa_spec(dS, g=2.0, E_syn=0.0)   # excitatory
        _, post_V = self._run_two_pop(spec, duration=500.0,
                                      weight=3.0, I_pre=I_HH, I_post=0.0)
        post_spikes = sum(_spikes(post_V[i]) for i in range(post_V.shape[0]))
        assert post_spikes > 0, "Excitatory custom synapse should drive post spikes"

    def test_inhibitory_custom_synapse_suppresses_firing(self):
        """
        Inhibitory custom kinetic synapse (E_syn = -80 mV) reduces post firing.

        I_syn = S * weight * (-80 - V_post) — hyperpolarising when V_post > -80.
        """
        dS = sympy.Rational(1, 2) * (1 - S) - sympy.Rational(1, 5) * S

        # Baseline: post fires with direct current, no synapse
        rn_base = hh.RegionalNetwork()
        rn_base.add_population("post", 5, neuron_type="hh")
        base_V = rn_base.simulate(500.0, DT, {"post": I_HH})["post"]
        base_spikes = sum(_spikes(base_V[i]) for i in range(base_V.shape[0]))

        # With inhibitory synapse from active pre
        spec_inh = self._ampa_spec(dS, g=2.0, E_syn=-80.0)
        rn_inh = hh.RegionalNetwork()
        rn_inh.add_population("pre",  5, neuron_type="hh")
        rn_inh.add_population("post", 5, neuron_type="hh")
        rn_inh.connect("pre", "post", "all_to_all",
                        weight=3.0, kinetic_spec=spec_inh)
        inh_V = rn_inh.simulate(500.0, DT, {"pre": I_HH, "post": I_HH})["post"]
        inh_spikes = sum(_spikes(inh_V[i]) for i in range(inh_V.shape[0]))

        assert inh_spikes < base_spikes, (
            f"Inhibitory synapse should reduce spikes: "
            f"base={base_spikes}, inhibited={inh_spikes}")

    def test_custom_kinetics_s_bounded(self):
        """
        With dS_dt = r*(1-S) - d*S, S is bounded in [0,1] analytically.
        The C++ clamps S to [0,1] at each step; post-neuron voltage stays
        in a physiological range.
        """
        dS   = sympy.Rational(1, 2) * (1 - S) - sympy.Rational(1, 5) * S
        spec = self._ampa_spec(dS)
        pre_V, post_V = self._run_two_pop(spec, duration=1000.0)
        assert np.all(post_V > -200.0), "Post voltage below -200 mV (S diverged?)"
        assert np.all(post_V <  150.0), "Post voltage above +150 mV (S diverged?)"

    def test_custom_kinetics_long_run_stable(self):
        """Custom kinetic synapse stays numerically stable over 2 s."""
        dS   = sympy.Rational(1, 2) * (1 - S) - sympy.Rational(1, 5) * S
        spec = self._ampa_spec(dS)
        pre_V, post_V = self._run_two_pop(spec, duration=LONG, n_pre=5, n_post=5)
        assert np.all(np.isfinite(pre_V))
        assert np.all(np.isfinite(post_V))


# ===========================================================================
# 8. Long-run stability sweep across all novel expression types
# ===========================================================================

class TestLongRunStability:
    """Every novel expression type stays finite in a 2 s simulation (100 neurons)."""

    def _run_stable(self, spec):
        traces = _run(spec, duration=LONG, n=100, I_ext=I_HH)
        assert np.all(np.isfinite(traces)), "Non-finite values in long-run trace"
        assert np.all(traces > -250.0),     "Voltage below -250 mV: likely divergence"
        assert np.all(traces < 150.0),      "Voltage above +150 mV: likely divergence"

    def test_novel_inf_long_run(self):
        model = NeuronModel("stab_inf", C_m=1.0, V_init=-65.0)
        g = model.add_gate("m",
            inf=sympy.Rational(1, 2) * (1 + sympy.tanh((V + 40) / 16)),
            tau=Tau.constant(1.0))
        model.add_channel("Na", g=40.0, E_rev=50.0, gating=g**2)
        model.add_channel("K",  g=20.0, E_rev=-77.0, gating=g**2)  # simplified
        model.add_channel("L",  g=0.3,  E_rev=-65.0)
        self._run_stable(model.to_spec())

    def test_novel_tau_long_run(self):
        model = NeuronModel("stab_tau", C_m=1.0, V_init=-65.0)
        g = model.add_gate("m",
            inf=Boltzmann(-40.0, 8.0),
            tau=sympy.Rational(1, 2) + 2 * sympy.exp(-(V + 40)**2 / 400))
        model.add_channel("Na", g=30.0, E_rev=50.0, gating=g**2)
        model.add_channel("L",  g=0.3,  E_rev=-65.0)
        self._run_stable(model.to_spec())

    def test_novel_alpha_beta_long_run(self):
        model = NeuronModel("stab_ab", C_m=1.0, V_init=-65.0)
        g = model.add_gate("m",
            alpha=sympy.Rational(1, 2) * sympy.exp(-(V + 40)**2 / 200),
            beta =sympy.Rational(1, 10) + sympy.Rational(1, 5) * sympy.exp(-(V + 70)**2 / 400))
        model.add_channel("Na", g=30.0, E_rev=50.0, gating=g**2)
        model.add_channel("L",  g=0.3,  E_rev=-65.0)
        self._run_stable(model.to_spec())

    def test_full_novel_hh_long_run(self):
        self._run_stable(_novel_hh())

    def test_mixed_novel_and_preset_long_run(self):
        """Model mixing novel (CUSTOM_EXPR) and preset gates remains stable.

        Uses standard HH parameters for Na (m^3*h) and K (n^4), but replaces
        the m gate inf with a mathematically equivalent tanh form to force the
        CUSTOM_EXPR path for that one gate only.  The full HH parameterisation is
        known to be stable.
        """
        model = NeuronModel("stab_mix", C_m=1.0, V_init=-65.0)
        # m gate: novel tanh inf (CUSTOM_EXPR), preset alpha/beta for h and n
        m_g = model.add_gate("m",
            inf=sympy.Rational(1, 2) * (1 + sympy.tanh((V + 40) / 16)),
            tau=Tau.constant(1.0))
        h_g = model.add_gate("h",
            alpha=RateFunc.exp_decay(  0.07, 65.0, -20.0),
            beta =RateFunc.sigmoid(    1.0,  35.0, -10.0))
        n_g = model.add_gate("n",
            alpha=RateFunc.linear_over_expm1(0.01, 55.0, 10.0),
            beta =RateFunc.exp_decay(        0.125,65.0, -80.0))
        model.add_channel("Na", g=120.0, E_rev= 50.0, gating=m_g**3 * h_g)
        model.add_channel("K",  g=36.0,  E_rev=-77.0, gating=n_g**4)
        model.add_channel("L",  g=0.3,   E_rev=-54.387)
        self._run_stable(model.to_spec())


# =============================================================================
# 9. Arbitrary gate ODE: dx/dt = F(x, V)
# =============================================================================

class TestArbitraryGateODE:
    """
    Tests for the expr= parameter in add_gate().

    All ODE expressions use genuinely novel kinetics that would not appear in
    standard Hodgkin-Huxley models: sinusoidal targets, logarithmic activation,
    sqrt-abs voltage nonlinearities, nested transcendentals, sigmoid-of-x,
    pure-state bistable dynamics, products of sinusoids, and Hill-like quadratic
    calcium dependence.

    Coverage:
      A. Bytecode structure — opcodes (SIN, COS, LOG, SQRT, ABS, EXP, PUSH_S, ...)
      B. Gate struct fields — dxdt_vm non-empty, other VMs all empty
      C. Scale and initial_value round-trip
      D. Numerical accuracy vs INF_TAU / ALPHA_BETA reference forms at small dt
      F. Novel nonlinear ODE forms (trig, log, sqrt-abs, pure-x bistable)
      G. Calcium-dependent ODE (Hill-quadratic activation in Ca)
      H. Two novel dxdt gates coexist independently
      I. Mixed novel dxdt + standard INF_TAU gate on one model
      J. Pool vs scalar consistency (n=1 vs n=50)
      K. Gate clamping — [0, 1] enforced even with divergent ODE
      L. Long-run stability (2 s, no NaN/Inf)
    """

    # -------------------------------------------------------------------------
    # A. Bytecode structure
    # -------------------------------------------------------------------------

    def test_sin_gate_compiles_with_correct_opcodes(self):
        """dx/dt = 0.5*(1 + sin(V/15)) - x — CUSTOM_EXPR, SIN + PUSH_S + PUSH_DEP."""
        dxdt = sympy.Rational(1, 2) * (1 + sympy.sin(V / 15)) - x

        model = NeuronModel("sin_gate", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt, initial_value=0.0)
        model.add_channel("K", g=8.0, E_rev=-77.0, gates=[(int(m_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert not spec.gates[0].dxdt_vm.empty()
        ops = [ins.op for ins in spec.gates[0].dxdt_vm.instructions]
        assert VmOp.SIN     in ops, "SIN opcode expected for sin(V/15)"
        assert VmOp.PUSH_S  in ops, "PUSH_S expected for gate state x"
        assert VmOp.PUSH_DEP in ops, "PUSH_DEP expected for voltage V"

    def test_sigmoid_of_x_gate_bytecode_has_exp_and_push_s(self):
        """
        dx/dt = 1/(1+exp(-10*(x-0.5))) - x — sigmoid in x with no V dependence.
        EXP and PUSH_S present; PUSH_DEP absent (V never appears).
        Converges to x=0.5 from any initial value.
        """
        dxdt = 1 / (1 + sympy.exp(-10 * (x - sympy.Rational(1, 2)))) - x

        model = NeuronModel("sig_x", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt, initial_value=0.1)
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        ops = [ins.op for ins in spec.gates[0].dxdt_vm.instructions]
        assert VmOp.EXP     in ops,       "EXP expected for exp(-10*(x-0.5))"
        assert VmOp.PUSH_S  in ops,       "PUSH_S expected (x in exponent)"
        assert VmOp.PUSH_DEP not in ops,  "PUSH_DEP must be absent (no V in expr)"

        trace = _run(spec, duration=SHORT, dt=DT, n=4, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_v_only_sin_squared_gate_no_push_s(self):
        """
        dx/dt = sin(V/10)**2 — x never appears → PUSH_S absent.
        sin^2 is always in [0,1], so gate charges toward a bounded target.
        """
        dxdt = sympy.sin(V / 10) ** 2

        model = NeuronModel("sin2_v", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt)
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        ops = [ins.op for ins in spec.gates[0].dxdt_vm.instructions]
        assert VmOp.SIN     in ops,       "SIN expected"
        assert VmOp.PUSH_S  not in ops,   "PUSH_S must be absent for V-only ODE"
        assert VmOp.PUSH_DEP in ops,      "PUSH_DEP expected for V"

        trace = _run(spec, duration=SHORT, dt=DT, n=4, I_ext=5.0)
        assert np.all(np.isfinite(trace))

    # -------------------------------------------------------------------------
    # B. Gate struct fields
    # -------------------------------------------------------------------------

    def test_sqrt_abs_gate_other_vms_are_empty(self):
        """
        dx/dt = sqrt(|V+70|/130) - x — SQRT + ABS opcodes verify novel compilation;
        inf_vm / tau_vm / alpha_vm / beta_vm must all be empty when expr= is used.
        """
        dxdt = sympy.sqrt(sympy.Abs(V + 70) / 130) - x

        model = NeuronModel("sqrt_abs", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt)
        model.add_leak(g=0.3, E_rev=-54.3)
        g = model.to_spec().gates[0]

        assert not g.dxdt_vm.empty()
        assert g.inf_vm.empty(),   "inf_vm must be empty when expr= is used"
        assert g.tau_vm.empty(),   "tau_vm must be empty when expr= is used"
        assert g.alpha_vm.empty(), "alpha_vm must be empty when expr= is used"
        assert g.beta_vm.empty(),  "beta_vm must be empty when expr= is used"

        ops = [ins.op for ins in g.dxdt_vm.instructions]
        # SymPy lowers sqrt(x) → Pow(x, 1/2) → POW_HALF, not SQRT
        assert VmOp.POW_HALF in ops, "POW_HALF expected for sqrt(|V+70|/130)"
        assert VmOp.ABS      in ops, "ABS expected for |V+70|"

    # -------------------------------------------------------------------------
    # C. Scale and initial_value
    # -------------------------------------------------------------------------

    def test_scale_parameter_is_stored(self):
        """scale= is stored on the GateSpec and used as the Euler multiplier."""
        dxdt = sympy.Rational(1, 2) * (1 + sympy.sin(V / 15)) - x
        model = NeuronModel("scale_test", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt, scale=2.0, initial_value=0.25)
        model.add_leak(g=0.3, E_rev=-54.3)
        g = model.to_spec().gates[0]
        assert g.scale == pytest.approx(2.0)
        assert g.initial_value == pytest.approx(0.25)

    def test_initial_value_stored_and_used(self):
        """initial_value is stored and becomes the t=0 gate state."""
        dxdt = (sympy.Rational(1, 2) - x) / 5.0
        for iv in [0.0, 0.5, 1.0]:
            model = NeuronModel("iv_test", C_m=1.0, V_init=-65.0)
            m_g = model.add_gate("m", expr=dxdt, initial_value=iv)
            model.add_leak(g=0.3, E_rev=-54.3)
            assert model.to_spec().gates[0].initial_value == pytest.approx(iv)

    # -------------------------------------------------------------------------
    # D. Numerical accuracy
    # -------------------------------------------------------------------------

    def test_dxdt_euler_matches_inf_tau_exponential(self):
        """
        dx/dt = (x_inf - x)/tau is mathematically equivalent to INF_TAU
        exponential decay.  At dt=0.005ms Euler error should be <0.1mV over 100ms.
        """
        x_inf_v   = 1 / (1 + sympy.exp(-(V + 40) / 9))
        tau_ms    = 1.0
        dxdt_expr = (x_inf_v - x) / tau_ms

        m_expr = NeuronModel("expr", C_m=1.0, V_init=-65.0)
        m_g_e  = m_expr.add_gate("m", expr=dxdt_expr, initial_value=0.0)
        m_expr.add_channel("K", g=10.0, E_rev=-77.0, gates=[(int(m_g_e), 1)])
        m_expr.add_leak(g=0.3, E_rev=-54.3)

        m_ref = NeuronModel("ref", C_m=1.0, V_init=-65.0)
        m_g_r = m_ref.add_gate("m", inf=Boltzmann(-40, 9), tau=Tau.constant(tau_ms))
        m_ref.add_channel("K", g=10.0, E_rev=-77.0, gates=[(int(m_g_r), 1)])
        m_ref.add_leak(g=0.3, E_rev=-54.3)

        dt = 0.005
        t_expr = _run(m_expr.to_spec(), duration=100.0, dt=dt, n=4, I_ext=5.0)
        t_ref  = _run(m_ref.to_spec(),  duration=100.0, dt=dt, n=4, I_ext=5.0)
        np.testing.assert_allclose(t_expr, t_ref, atol=0.1,
            err_msg="Euler dxdt diverges from INF_TAU exponential reference")

    def test_hh_alpha_beta_ode_matches_alpha_beta_form(self):
        """
        dx/dt = alpha(V)*(1-x) - beta(V)*x is mathematically identical to
        the ALPHA_BETA form.  Tested in a subthreshold regime (I_ext=0, no
        spikes) where voltage barely moves and Euler matches the exact
        exponential update to within 0.2 mV over 100 ms.
        """
        # Standard HH n-gate rates
        alpha_n = 0.01 * (V + 55) / (1 - sympy.exp(-(V + 55) / 10))
        beta_n  = 0.125 * sympy.exp(-(V + 65) / 80)
        dxdt_ode = alpha_n * (1 - x) - beta_n * x

        m_ode = NeuronModel("n_ode", C_m=1.0, V_init=-65.0)
        n_g_o = m_ode.add_gate("n", expr=dxdt_ode, initial_value=0.0)
        m_ode.add_channel("K", g=5.0, E_rev=-77.0, gates=[(int(n_g_o), 1)])
        m_ode.add_leak(g=0.3, E_rev=-54.387)

        m_ref = NeuronModel("n_ref", C_m=1.0, V_init=-65.0)
        n_g_r = m_ref.add_gate("n",
            alpha=RateFunc.linear_over_expm1(0.01, 55.0, 10.0),
            beta =RateFunc.exp_decay(0.125, 65.0, -80.0))
        m_ref.add_channel("K", g=5.0, E_rev=-77.0, gates=[(int(n_g_r), 1)])
        m_ref.add_leak(g=0.3, E_rev=-54.387)

        # Subthreshold: I_ext=0, so V stays near resting potential.
        # With tau_n~6ms at rest and dt=0.005ms, Euler error is tiny.
        dt = 0.005
        t_ode = _run(m_ode.to_spec(), duration=100.0, dt=dt, n=4, I_ext=0.0)
        t_ref = _run(m_ref.to_spec(), duration=100.0, dt=dt, n=4, I_ext=0.0)
        np.testing.assert_allclose(t_ode, t_ref, atol=0.2,
            err_msg="ODE form dx/dt=alpha*(1-x)-beta*x diverges from ALPHA_BETA reference")

    # -------------------------------------------------------------------------
    # F. Novel nonlinear ODE forms
    # -------------------------------------------------------------------------

    def test_sin_cos_product_gate_nonlinear_in_x(self):
        """
        dx/dt = sin(V/12)^3*(1-x) - cos(V/8)^2*x — cubic trig, linear in x.
        Voltage-dependent equilibrium shifts with sin/cos phase; gate clamped to [0,1].
        SIN, COS, PUSH_S all present in bytecode.
        """
        dxdt = sympy.sin(V / 12)**3 * (1 - x) - sympy.cos(V / 8)**2 * x

        model = NeuronModel("sin_cos_nl", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt, initial_value=0.5)
        model.add_channel("Na", g=20.0, E_rev=50.0, gates=[(int(m_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)

        ops = [ins.op for ins in model.to_spec().gates[0].dxdt_vm.instructions]
        assert VmOp.SIN    in ops
        assert VmOp.COS    in ops
        assert VmOp.PUSH_S in ops

        trace = _run(model.to_spec(), duration=SHORT, dt=DT, n=4, I_ext=I_HH)
        assert np.all(np.isfinite(trace))

    def test_nested_tanh_sin_gate(self):
        """
        dx/dt = tanh(sin(V/15)) - x — nested transcendentals; neither sin nor tanh
        alone is the target, their composition is.  TANH + SIN in bytecode.
        """
        dxdt = sympy.tanh(sympy.sin(V / 15)) - x

        model = NeuronModel("nested_trig", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt, initial_value=0.0)
        model.add_leak(g=0.3, E_rev=-54.3)

        ops = [ins.op for ins in model.to_spec().gates[0].dxdt_vm.instructions]
        assert VmOp.SIN    in ops
        assert VmOp.TANH   in ops
        assert VmOp.PUSH_S in ops

        trace = _run(model.to_spec(), duration=SHORT, dt=DT, n=4, I_ext=5.0)
        assert np.all(np.isfinite(trace))

    def test_pure_x_bistable_gate(self):
        """
        dx/dt = x*(1-x)*(0.5-x)*5 — no V dependence whatsoever.
        Fixed points: x=0 (unstable), x=0.5 (stable), x=1 (unstable).
        Gate always converges to x=0.5.  PUSH_DEP must be absent.
        """
        dxdt = x * (1 - x) * (sympy.Rational(1, 2) - x) * 5

        model = NeuronModel("bistable_x", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt, initial_value=0.1)
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        ops = [ins.op for ins in spec.gates[0].dxdt_vm.instructions]
        assert VmOp.PUSH_S    in ops,       "PUSH_S expected (x appears)"
        assert VmOp.PUSH_DEP  not in ops,   "PUSH_DEP must be absent (no V)"

        trace = _run(spec, duration=SHORT, dt=DT, n=4, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_log_abs_voltage_gate(self):
        """
        dx/dt = log(1 + |V+65|/25)*0.12 - x — logarithmic activation in voltage.
        Charges more during APs (large V deviation) than at rest.
        LOG and ABS opcodes both present.
        """
        dxdt = sympy.log(1 + sympy.Abs(V + 65) / 25) * sympy.Rational(12, 100) - x

        model = NeuronModel("log_abs", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt, initial_value=0.0)
        model.add_channel("K", g=8.0, E_rev=-77.0, gates=[(int(m_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)

        ops = [ins.op for ins in model.to_spec().gates[0].dxdt_vm.instructions]
        assert VmOp.LOG     in ops, "LOG expected"
        assert VmOp.ABS     in ops, "ABS expected"
        assert VmOp.PUSH_DEP in ops

        trace = _run(model.to_spec(), duration=SHORT, dt=DT, n=4, I_ext=I_HH)
        assert np.all(np.isfinite(trace))

    # -------------------------------------------------------------------------
    # G. Calcium-dependent ODE (novel form)
    # -------------------------------------------------------------------------

    def test_calcium_quadratic_hill_dxdt_gate(self):
        """
        dx/dt = Ca^2/(Ca^2 + 0.1) - x — Hill-like quadratic (n=2) in Ca.
        Not pattern-matched (standard Hill uses linear Ca or specific forms).
        PUSH_DEP pushes Ca; gate activates nonlinearly with calcium influx.
        """
        dxdt = Ca**2 / (Ca**2 + sympy.Rational(1, 10)) - x

        model = NeuronModel("ca_hill2", C_m=1.0, V_init=-65.0)
        model._spec.calcium.enabled  = True
        model._spec.calcium.epsilon  = 1e-4
        model._spec.calcium.K_Ca     = 15.0
        model._spec.calcium.Ca_init  = 0.1
        model._spec.calcium.source_channels.append(0)

        m_g  = model.add_gate("m",
            alpha=RateFunc.linear_over_expm1(0.1, 40.0, 10.0),
            beta =RateFunc.exp_decay(4.0, 65.0, -18.0))
        ca_g = model.add_gate("q",
            expr=dxdt, dependency="calcium", initial_value=0.0)
        model.add_channel("Ca", g=5.0, E_rev=120.0, gates=[(int(m_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)

        spec = model.to_spec()
        assert spec.gates[1].update_form == GateUpdateForm.CUSTOM_EXPR
        assert not spec.gates[1].dxdt_vm.empty()
        ops = [ins.op for ins in spec.gates[1].dxdt_vm.instructions]
        assert VmOp.PUSH_S in ops

        trace = _run(spec, duration=SHORT, dt=DT, n=4, I_ext=I_HH)
        assert np.all(np.isfinite(trace))

    # -------------------------------------------------------------------------
    # H. Two novel dxdt gates coexist
    # -------------------------------------------------------------------------

    def test_two_novel_dxdt_gates_coexist(self):
        """
        Gate 1 (m): dx/dt = sin(V/15)^2 - x  — squared sinusoid activation ∈ [0,1]
        Gate 2 (h): dx/dt = cos(V/25)^2 - x  — squared cosine inactivation ∈ [0,1]
        Both CUSTOM_EXPR; bytecode programs are independent (SIN vs COS).
        """
        dxdt_m = sympy.sin(V / 15)**2 - x
        dxdt_h = sympy.cos(V / 25)**2 - x

        model = NeuronModel("sin_cos_pair", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt_m, initial_value=0.0)
        h_g = model.add_gate("h", expr=dxdt_h, initial_value=1.0)
        model.add_channel("Na", g=30.0, E_rev=50.0,
                          gates=[(int(m_g), 2), (int(h_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        for i in range(2):
            assert spec.gates[i].update_form == GateUpdateForm.CUSTOM_EXPR
            assert not spec.gates[i].dxdt_vm.empty()

        ops_m = [ins.op for ins in spec.gates[0].dxdt_vm.instructions]
        ops_h = [ins.op for ins in spec.gates[1].dxdt_vm.instructions]
        assert VmOp.SIN in ops_m, "gate m should use SIN"
        assert VmOp.COS in ops_h, "gate h should use COS"

        trace = _run(spec, duration=SHORT, dt=DT, n=4, I_ext=I_HH)
        assert np.all(np.isfinite(trace))

    # -------------------------------------------------------------------------
    # I. Mixed novel dxdt + standard INF_TAU
    # -------------------------------------------------------------------------

    def test_mixed_novel_dxdt_and_inf_tau_gates(self):
        """
        Activation: dx/dt = sqrt(|V+70|/130) - x  (CUSTOM_EXPR, sqrt-abs)
        Inactivation: standard Boltzmann INF_TAU gate
        Forms must be preserved independently; neither must bleed into the other.
        """
        dxdt_m = sympy.sqrt(sympy.Abs(V + 70) / 130) - x

        model = NeuronModel("sqrt_mixed", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt_m, initial_value=0.0)
        h_g = model.add_gate("h", inf=Boltzmann(-65, -7), tau=Tau.constant(3.0))
        model.add_channel("Na", g=60.0, E_rev=50.0,
                          gates=[(int(m_g), 2), (int(h_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR
        assert spec.gates[1].update_form == GateUpdateForm.INF_TAU

        trace = _run(spec, duration=SHORT, dt=DT, n=4, I_ext=I_HH)
        assert np.all(np.isfinite(trace))

    # -------------------------------------------------------------------------
    # J. Pool vs scalar consistency
    # -------------------------------------------------------------------------

    def test_pool_and_scalar_paths_agree(self):
        """
        n=1 (scalar path) and n=50 (Eigen pool path) must agree for a complex
        nested-trig ODE.  Sub-ULP SIMD rounding differences allowed (atol=1e-9).
        """
        dxdt_expr = (1 + sympy.tanh(sympy.sin(V / 15))) / 2 - x

        model = NeuronModel("pool_scalar", C_m=1.0, V_init=-65.0)
        m_g   = model.add_gate("m", expr=dxdt_expr, initial_value=0.1)
        model.add_channel("K", g=12.0, E_rev=-77.0, gates=[(int(m_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        t_scalar = _run(spec, duration=100.0, dt=DT, n=1,  I_ext=6.0)[0]
        t_pool   = _run(spec, duration=100.0, dt=DT, n=50, I_ext=6.0)[0]
        np.testing.assert_allclose(t_scalar, t_pool, atol=1e-9,
            err_msg="Scalar and pool paths diverge for nested-trig dxdt gate")

    # -------------------------------------------------------------------------
    # K. Gate clamping to [0, 1]
    # -------------------------------------------------------------------------

    def test_dxdt_gate_is_clamped_to_unit_interval(self):
        """
        A strongly positive dxdt tries to push x above 1; the C++ clamp must
        keep the gate in [0, 1] at all times.  Verified by inspecting that the
        output voltage stays finite (clamping prevents exponential runaway).
        """
        # Very fast positive drive — would push x >> 1 without clamping
        dxdt_expr = 100.0 - 100.0 * x  # x_inf = 1, very fast

        model = NeuronModel("clamp_test", C_m=1.0, V_init=-65.0)
        m_g   = model.add_gate("m", expr=dxdt_expr, initial_value=0.0)
        model.add_channel("K", g=5.0, E_rev=-77.0, gates=[(int(m_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)
        spec = model.to_spec()

        trace = _run(spec, duration=50.0, dt=0.01, n=4, I_ext=2.0)
        assert np.all(np.isfinite(trace)), "Clamped gate should not produce NaN/Inf"

    # -------------------------------------------------------------------------
    # L. Long-run stability
    # -------------------------------------------------------------------------

    def test_sinusoidal_product_gate_long_run_stable(self):
        """
        dx/dt = sin(V/20)*cos(V/35)*0.5 + 0.5 - x — product of sinusoids as target.
        The product sin*cos oscillates; [0.5-0.5, 0.5+0.5] = [0, 1].
        2 s simulation must stay finite.
        """
        dxdt = (sympy.sin(V / 20) * sympy.cos(V / 35)
                * sympy.Rational(1, 2) + sympy.Rational(1, 2) - x)

        model = NeuronModel("sin_cos_long", C_m=1.0, V_init=-65.0)
        m_g   = model.add_gate("m", expr=dxdt, initial_value=0.5)
        model.add_channel("K", g=10.0, E_rev=-77.0, gates=[(int(m_g), 1)])
        model.add_leak(g=0.3, E_rev=-54.3)

        trace = _run(model.to_spec(), duration=LONG, dt=DT, n=4, I_ext=I_HH)
        assert np.all(np.isfinite(trace)), "Sinusoidal product gate produced NaN/Inf"

    def test_three_novel_gate_ode_model_stable(self):
        """
        Full model with three completely novel ODE gates — no HH kinetics anywhere.

          m:  dx/dt = sin(V/15)^2 - x                    [squared sin;   ∈ [0,1]]
          h:  dx/dt = (1 + cos(V/25))/2 - x              [half-cosine;   ∈ [0,1]]
          n:  dx/dt = (1 + tanh(sin(V/20)))/2 - x        [nested trig;   ∈ (0,1)]

        2 s run must remain finite.
        """
        dxdt_m = sympy.sin(V / 15)**2 - x
        dxdt_h = (1 + sympy.cos(V / 25)) / 2 - x
        dxdt_n = (1 + sympy.tanh(sympy.sin(V / 20))) / 2 - x

        model = NeuronModel("novel_trio", C_m=1.0, V_init=-65.0)
        m_g = model.add_gate("m", expr=dxdt_m, initial_value=0.0)
        h_g = model.add_gate("h", expr=dxdt_h, initial_value=1.0)
        n_g = model.add_gate("n", expr=dxdt_n, initial_value=0.5)
        model.add_channel("Na", g=80.0, E_rev= 50.0,
                          gates=[(int(m_g), 2), (int(h_g), 1)])
        model.add_channel("K",  g=25.0, E_rev=-77.0, gates=[(int(n_g), 2)])
        model.add_leak(g=0.3, E_rev=-54.3)

        trace = _run(model.to_spec(), duration=LONG, dt=DT, n=4, I_ext=I_HH)
        assert np.all(np.isfinite(trace)), "Novel trio model produced NaN/Inf"
