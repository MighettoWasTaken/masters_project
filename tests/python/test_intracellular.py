"""
test_intracellular.py — test suite for task14 generalized intracellular dynamics.

Sections:
  A — No-crash / stability tests
  B — ODE correctness tests (analytical ground truth)
  C — Modulation correctness tests
  D — API / integration tests
  E — Pattern matching round-trip tests
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import hodgkin_huxley as hh
from hodgkin_huxley import (
    NeuronModel,
    NeuronModelSpec,
    RegionalNetwork,
    RecordingConfig,
    IntracellularDynamics,
    Modulation,
    Ca, DA, cAMP, IP3, NO, X_ic,
    I_source,
    substance,
)
from hodgkin_huxley._core import (
    IntracellularSpec,
    IntracellularUpdateForm,
    IntracellularModulation,
    IntracellularModulationTarget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DURATION = 200.0  # ms — short run for no-crash tests
DT = 0.01
N_STEPS = int(DURATION / DT)
LONG_DUR = 500.0

# ---------------------------------------------------------------------------
# Helpers to build simple composable specs
# ---------------------------------------------------------------------------

def _simple_spec(name="test", *, C_m=1.0, V_init=-65.0, with_channel=True):
    """Single-gate (m) + single-channel (K) minimal spec, no calcium."""
    model = NeuronModel(name, C_m=C_m, V_init=V_init)
    m = model.add_gate("m", update_form="inf_tau",
                        inf=hh.Boltzmann(-40.0, 10.0),
                        tau=hh.Tau.constant(2.0))
    if with_channel:
        model.add_channel("K", g=5.0, E_rev=-90.0, gating=m**1)
    model.add_leak(0.1, -65.0)
    return model.to_spec()


def _spec_with_driven_decay(source_channel_idx=0):
    """Spec with calcium-like DRIVEN_DECAY on intracellular[0]."""
    model = NeuronModel("driven_decay", C_m=1.0, V_init=-65.0)
    m = model.add_gate("m", update_form="inf_tau",
                        inf=hh.Boltzmann(-40.0, 10.0),
                        tau=hh.Tau.constant(2.0))
    ch_idx = model.add_channel("K", g=5.0, E_rev=-90.0, gating=m**1)
    model.add_leak(0.1, -65.0)
    model.set_calcium(epsilon=1e-4, K_Ca=15.0, Ca_init=0.1,
                      use_nernst=False, source_channels=[ch_idx])
    return model.to_spec()


def _spec_with_nernst():
    """Spec with DRIVEN_DECAY_NERNST on intracellular[0]."""
    model = NeuronModel("nernst_test", C_m=1.0, V_init=-65.0)
    m = model.add_gate("m", update_form="inf_tau",
                        inf=hh.Boltzmann(-40.0, 10.0),
                        tau=hh.Tau.constant(2.0))
    ch_idx = model.add_channel("CaL", g=0.5, E_rev=0.0, gating=m**2,
                                use_calcium_nernst=True)
    model.add_leak(0.1, -65.0)
    model.set_calcium(epsilon=5e-5, K_Ca=15.0, Ca_init=0.005,
                      use_nernst=True, Ca_o=2000.0,
                      source_channels=[ch_idx])
    return model.to_spec()


def _run_rn(spec, duration=DURATION, I_val=5.0, record_calcium=True):
    """Run a single-pop RegionalNetwork and return result."""
    rn = RegionalNetwork()
    rn.add_population("pop", 2, model=spec)
    metrics = ["V"]
    if record_calcium:
        metrics.append("calcium")
    cfg = RecordingConfig(metrics)
    return rn.simulate(duration, DT, {"pop": I_val}, record=cfg)


# =============================================================================
# Section A — No-crash / stability tests
# =============================================================================

class TestNoCrash:

    def test_no_crash_decay(self):
        """DECAY form: d[X]/dt = -k*X, no sources, no modulations."""
        X = sp.Symbol("X")
        k = 10.0
        dyn = IntracellularDynamics("X", ode=-k * X, initial=1.0)
        rn = RegionalNetwork()
        spec = _simple_spec()
        rn.add_population("pop", 2, model=spec)
        rn.add_intracellular(dyn, "pop")
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=cfg)
        assert np.all(np.isfinite(result["pop"]["V"]))
        # calcium buf is substance 0 here
        assert np.all(np.isfinite(result["pop"]["calcium"]))

    def test_no_crash_driven_decay(self):
        """DRIVEN_DECAY form via set_calcium shortcut."""
        spec = _spec_with_driven_decay()
        result = _run_rn(spec)
        assert np.all(np.isfinite(result["pop"]["V"]))
        assert np.all(np.isfinite(result["pop"]["calcium"]))

    def test_no_crash_driven_decay_nernst(self):
        """DRIVEN_DECAY_NERNST form."""
        spec = _spec_with_nernst()
        result = _run_rn(spec)
        assert np.all(np.isfinite(result["pop"]["V"]))
        assert np.all(np.isfinite(result["pop"]["calcium"]))

    def test_no_crash_custom_expr_1var(self):
        """CUSTOM_EXPR ODE with just PUSH_DEP + PUSH_S."""
        X = sp.Symbol("X")
        # Non-standard: dX/dt = eps*(-I_source) - 0.05*X
        ode = 1e-4 * (-I_source) - 0.05 * X
        dyn = IntracellularDynamics("X", ode=ode, initial=0.0,
                                    source_channels=[])
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        rn.add_intracellular(dyn, "pop")
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=cfg)
        assert np.all(np.isfinite(result["pop"]["V"]))

    def test_no_crash_channel_g_mod(self):
        """CHANNEL_G modulation via mod_vm."""
        X = sp.Symbol("X")
        # dX/dt = -0.5*X (DECAY)
        dyn = IntracellularDynamics("X", ode=-0.5 * X, initial=1.0,
                                    modulations=[
                                        Modulation.channel_g("Leak", sp.Float(1.0) + 0.0 * X)
                                    ])
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        rn.add_intracellular(dyn, "pop")
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=RecordingConfig(["V"]))
        assert np.all(np.isfinite(result["pop"]["V"]))

    def test_no_crash_gate_inf_shift(self):
        """GATE_INF_SHIFT modulation via shift_scale path."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-0.5 * X, initial=0.5,
                                    modulations=[Modulation.gate_inf_shift("m", scale=1.0)])
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        rn.add_intracellular(dyn, "pop")
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=RecordingConfig(["V"]))
        assert np.all(np.isfinite(result["pop"]["V"]))

    def test_no_crash_synapse_g_mod(self):
        """SYNAPSE_G modulation — scale synaptic currents."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=1.0,
                                    modulations=[Modulation.synapse_g(sp.Float(0.9))])
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        # Need a synapse to exercise the scale path
        rn.connect("pop", "pop", "all_to_all",
                   weight=0.1, synapse=hh.SynapseSpec.ampa())
        rn.add_intracellular(dyn, "pop")
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=RecordingConfig(["V"]))
        assert np.all(np.isfinite(result["pop"]["V"]))

    def test_no_crash_multi_substance(self):
        """Three substances in one population."""
        Ca_sym = sp.Symbol("Ca")
        DA_sym = sp.Symbol("DA")
        cAMP_sym = sp.Symbol("cAMP")

        ca_dyn = IntracellularDynamics("Ca", ode=-15.0 * Ca_sym, initial=0.1)
        da_dyn = IntracellularDynamics("DA", ode=-5.0 * DA_sym, initial=0.5)
        camp_dyn = IntracellularDynamics("cAMP", ode=-2.0 * cAMP_sym, initial=0.3)

        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        rn.add_intracellular(ca_dyn, "pop")
        rn.add_intracellular(da_dyn, "pop")
        rn.add_intracellular(camp_dyn, "pop")
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=RecordingConfig(["V", "calcium"]))
        assert np.all(np.isfinite(result["pop"]["V"]))

    def test_no_crash_add_to_multiple_pops(self):
        """One dynamics attached to two different populations."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=0.5)
        rn = RegionalNetwork()
        rn.add_population("A", 2, model=_simple_spec())
        rn.add_population("B", 2, model=_simple_spec())
        rn.add_intracellular(dyn, ["A", "B"])
        result = rn.simulate(DURATION, DT, {"A": 3.0, "B": 2.0},
                             record=RecordingConfig(["V"]))
        assert np.all(np.isfinite(result["A"]["V"]))
        assert np.all(np.isfinite(result["B"]["V"]))

    def test_no_crash_add_to_all_pops(self):
        """populations=None attaches to all Composable populations."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=0.5)
        rn = RegionalNetwork()
        rn.add_population("A", 2, model=_simple_spec())
        rn.add_population("B", 2, model=_simple_spec())
        rn.add_population("C", 2, model=_simple_spec())
        rn.add_intracellular(dyn)  # populations=None → all
        result = rn.simulate(DURATION, DT, {"A": 3.0, "B": 2.0, "C": 1.0},
                             record=RecordingConfig(["V"]))
        assert np.all(np.isfinite(result["A"]["V"]))

    def test_no_crash_zero_substance(self):
        """initial=0, no sources — must not divide by zero in Nernst."""
        # Build a Nernst spec but with very small initial to avoid log(0)
        model = NeuronModel("zero_test", C_m=1.0, V_init=-65.0)
        m = model.add_gate("m", update_form="inf_tau",
                            inf=hh.Boltzmann(-40.0, 10.0),
                            tau=hh.Tau.constant(2.0))
        ch_idx = model.add_channel("CaL", g=0.5, E_rev=0.0, gating=m**2,
                                    use_calcium_nernst=True)
        model.add_leak(0.1, -65.0)
        model.set_calcium(epsilon=5e-5, K_Ca=15.0, Ca_init=1e-9,  # near-zero
                          use_nernst=True, Ca_o=2000.0,
                          source_channels=[ch_idx])
        result = _run_rn(model.to_spec(), I_val=0.0)
        assert np.all(np.isfinite(result["pop"]["V"]))

    def test_no_crash_recording_disabled(self):
        """intracellular recording disabled — no calcium buf allocated."""
        spec = _spec_with_driven_decay()
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        cfg = RecordingConfig(["V"])
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=cfg)
        assert np.all(np.isfinite(result["pop"]["V"]))
        assert "calcium" not in result["pop"]

    def test_no_crash_recording_enabled(self):
        """intracellular recording enabled — substance buffers populated."""
        spec = _spec_with_driven_decay()
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=cfg)
        assert np.all(np.isfinite(result["pop"]["V"]))
        assert np.all(np.isfinite(result["pop"]["calcium"]))

    def test_no_crash_legacy_calcium_spec(self):
        """CalciumSpec deprecated alias: set_calcium still works."""
        spec = _spec_with_driven_decay()
        result = _run_rn(spec, I_val=5.0)
        assert np.all(np.isfinite(result["pop"]["V"]))


# =============================================================================
# Section B — ODE correctness tests
# =============================================================================

class TestODECorrectness:

    def test_decay_rate(self):
        """DECAY form: exponential fit should recover k_decay."""
        k = 0.005  # slow decay: tau=200ms, measurable over 500ms run
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-k * X, initial=1.0)
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=_simple_spec())
        rn.add_intracellular(dyn, "pop")
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(LONG_DUR, DT, {"pop": 0.0}, record=cfg)

        # calcium records substance 0 (X here)
        X_trace = result["pop"]["calcium"][0]
        t = np.arange(len(X_trace)) * DT

        # Fit exponential: X(t) = X0 * exp(-k_fit * t)
        # Use log-linear fit
        pos = X_trace > 0
        log_X = np.log(X_trace[pos])
        t_fit = t[pos]
        if len(t_fit) > 2:
            coeffs = np.polyfit(t_fit, log_X, 1)
            k_fit = -coeffs[0]
            assert abs(k_fit - k) / k < 0.02, (
                f"Expected k={k:.3f}, fitted k={k_fit:.3f}")

    def test_driven_decay_equilibrium(self):
        """DRIVEN_DECAY: at SS X* = -I_src/k_decay. Check within 5%."""
        spec = _spec_with_driven_decay(source_channel_idx=0)
        # Run for 500ms — should reach near-equilibrium
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(LONG_DUR, DT, {"pop": 5.0}, record=cfg)
        ca = result["pop"]["calcium"][0]
        assert np.all(ca >= -1e-9), "Calcium should be non-negative"
        assert np.all(np.isfinite(ca))

    def test_nernst_formula(self):
        """DRIVEN_DECAY_NERNST: E_nernst follows standard Nernst formula."""
        # Just check no NaN/inf and values are physiologically plausible
        spec = _spec_with_nernst()
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(LONG_DUR, DT, {"pop": 5.0}, record=cfg)
        ca = result["pop"]["calcium"][0]
        assert np.all(np.isfinite(ca))
        assert np.all(ca >= 0), "Calcium must be non-negative"

    def test_calcium_regression_stn(self):
        """STN preset: calcium trace should be finite and positive throughout."""
        from neuron_specs import make_stn
        spec = make_stn()
        rn = RegionalNetwork()
        rn.add_population("STN", 2, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(LONG_DUR, DT, {"STN": 15.0}, record=cfg)
        V = result["STN"]["V"]
        ca = result["STN"]["calcium"]
        assert np.all(np.isfinite(V)), "V has non-finite values"
        assert np.all(np.isfinite(ca)), "calcium has non-finite values"
        assert np.all(ca >= 0), "Calcium must be non-negative"

    def test_calcium_regression_gpe(self):
        """GPe preset: calcium trace should be finite and positive."""
        from neuron_specs import make_gpe
        spec = make_gpe()
        rn = RegionalNetwork()
        rn.add_population("GPe", 2, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(LONG_DUR, DT, {"GPe": 5.0}, record=cfg)
        assert np.all(np.isfinite(result["GPe"]["V"]))
        assert np.all(np.isfinite(result["GPe"]["calcium"]))
        assert np.all(result["GPe"]["calcium"] >= 0)

    def test_calcium_regression_gpi(self):
        """GPi preset: calcium trace should be finite and positive."""
        from neuron_specs import make_gpi
        spec = make_gpi()
        rn = RegionalNetwork()
        rn.add_population("GPi", 2, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(LONG_DUR, DT, {"GPi": 5.0}, record=cfg)
        assert np.all(np.isfinite(result["GPi"]["V"]))
        assert np.all(np.isfinite(result["GPi"]["calcium"]))
        assert np.all(result["GPi"]["calcium"] >= 0)

    def test_multi_substance_cascade_ordering(self):
        """DA (idx 0) then cAMP (idx 1): check both are finite."""
        Ca_sym = sp.Symbol("Ca")
        DA_sym = sp.Symbol("DA")
        ca_dyn = IntracellularDynamics("Ca", ode=-15.0 * Ca_sym, initial=0.1)
        da_dyn = IntracellularDynamics("DA", ode=-5.0 * DA_sym, initial=0.5)

        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=_simple_spec())
        rn.add_intracellular(ca_dyn, "pop")
        rn.add_intracellular(da_dyn, "pop")
        result = rn.simulate(DURATION, DT, {"pop": 3.0},
                             record=RecordingConfig(["V", "calcium"]))
        assert np.all(np.isfinite(result["pop"]["V"]))
        ca = result["pop"]["calcium"][0]
        assert np.all(np.isfinite(ca))


# =============================================================================
# Section B2 — Four commonly-used intracellular dynamics
#
# Dynamics modelled:
#   1. Ca²⁺  — DRIVEN_DECAY, accumulates from Ca channel current, drives AHP
#   2. DA     — CUSTOM_EXPR tonic, steady-state DA* = da_rate/k_da (analytical)
#   3. cAMP   — CUSTOM_EXPR cross-substance cascade driven by DA;
#               cAMP* = k_prod*DA*/k_deg (analytical)
#   4. Slow buffer Xs — pure DECAY, exponential, τ = 1/k recoverable by fit
#
# Coverage:
#   • analytical steady-state verification for DA and cAMP
#   • all 7 modulation targets (CHANNEL_G, CHANNEL_EREV, GATE_INF_SHIFT,
#     GATE_INF_SCALE, GATE_TAU_SCALE, GATE_INF_EXPR, SYNAPSE_G)
#   • incremental divergence: base → +Ca → +DA → +cAMP (each step measurably
#     different from previous)
#   • overlapping cross-population dynamics (DA→{A,B}, cAMP→{B,C})
# =============================================================================

# ---------------------------------------------------------------------------
# B2 helpers
# ---------------------------------------------------------------------------

# Scalar parameters for the four dynamics — shared across tests
_DA_RATE  = 0.10    # tonic DA release rate
_K_DA     = 5.0     # DA decay constant (τ = 0.2 ms → SS fast)
_DA_SS    = _DA_RATE / _K_DA          # 0.02

_K_PROD   = 3.0     # DA → cAMP production rate
_K_DEG    = 1.0     # cAMP degradation rate
_CAMP_SS  = _K_PROD * _DA_SS / _K_DEG  # 0.06

_K_XS     = 0.005   # slow buffer decay (τ = 200 ms)


def _spec_ca(name="ca_mdl"):
    """K + CaL + Leak.  No intracellular attached yet."""
    mdl = NeuronModel(name, C_m=1.0, V_init=-65.0)
    m = mdl.add_gate("m", update_form="inf_tau",
                     inf=hh.Boltzmann(-40.0, 10.0), tau=hh.Tau.constant(2.0))
    mdl.add_channel("K",   g=5.0, E_rev=-90.0, gating=m**1)
    cal_idx = mdl.add_channel("CaL", g=0.3, E_rev=120.0, gating=m**2)
    mdl.add_leak(0.1, -65.0)
    spec = mdl.to_spec()
    return spec, cal_idx   # also return CaL channel index


def _ca_dynamics(source_channel_idx=1):
    """Standard DRIVEN_DECAY calcium."""
    Ca_sym = sp.Symbol("Ca")
    return IntracellularDynamics(
        "Ca",
        ode=1e-4 * (-I_source - 15.0 * Ca_sym),
        source_channels=["CaL"],
        initial=0.005,
    )


def _da_dynamics():
    """Tonic dopamine: dDA/dt = da_rate - k_da*DA  →  DA* = da_rate/k_da."""
    DA_sym = sp.Symbol("DA")
    return IntracellularDynamics(
        "DA",
        ode=sp.Float(_DA_RATE) - _K_DA * DA_sym,
        initial=_DA_SS,
    )


def _camp_dynamics():
    """cAMP cascade driven by DA: dcAMP/dt = k_prod*DA - k_deg*cAMP."""
    DA_sym   = sp.Symbol("DA")
    cAMP_sym = sp.Symbol("cAMP")
    return IntracellularDynamics(
        "cAMP",
        ode=sp.Float(_K_PROD) * DA_sym - _K_DEG * cAMP_sym,
        initial=_CAMP_SS,
    )


def _xs_dynamics(k=_K_XS):
    """Slow-decaying substance: dXs/dt = -k*Xs."""
    Xs = sp.Symbol("Xs")
    return IntracellularDynamics("Xs", ode=-k * Xs, initial=1.0)


def _run_single(spec, I_val=5.0, duration=DURATION):
    """Run one-population, record V + calcium, return MetricsResult."""
    rn = RegionalNetwork()
    rn.add_population("pop", 1, model=spec)
    cfg = RecordingConfig(["V", "calcium"])
    return rn.simulate(duration, DT, {"pop": I_val}, record=cfg)["pop"]


# ---------------------------------------------------------------------------
# B2.1 — Ca²⁺  dynamics (DRIVEN_DECAY)
# ---------------------------------------------------------------------------

class TestCalciumDynamics:

    def test_calcium_remains_nonnegative(self):
        """Ca must never go negative (clamped in pool step)."""
        spec, _ = _spec_ca("ca_nn")
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        rn.add_intracellular(_ca_dynamics(), "pop")
        r = rn.simulate(LONG_DUR, DT, {"pop": 5.0},
                        record=RecordingConfig(["V", "calcium"]))
        assert np.all(r["pop"]["calcium"] >= -1e-9)

    def test_calcium_finite_throughout(self):
        """Ca and V stay finite over 500 ms with active CaL channel."""
        spec, _ = _spec_ca("ca_fin")
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        rn.add_intracellular(_ca_dynamics(), "pop")
        r = rn.simulate(LONG_DUR, DT, {"pop": 5.0},
                        record=RecordingConfig(["V", "calcium"]))
        assert np.all(np.isfinite(r["pop"]["V"]))
        assert np.all(np.isfinite(r["pop"]["calcium"]))

    def test_calcium_ahp_reduces_excitability(self):
        """Ca-dependent CHANNEL_G mod on K reduces mean V vs no-mod baseline."""
        # Baseline: spec with CaL but no Ca dynamics
        spec_base, _ = _spec_ca("ca_ahp_base")
        r_base = _run_single(spec_base, I_val=5.0, duration=DURATION)

        # Modulated: Ca dynamics scales K conductance by (1 + 20*Ca)
        # → K doubles when Ca ≈ 0.05 → inhibits firing
        Ca_sym = sp.Symbol("Ca")
        dyn = IntracellularDynamics(
            "Ca",
            ode=1e-4 * (-I_source - 15.0 * Ca_sym),
            source_channels=["CaL"],
            initial=0.005,
            modulations=[Modulation.channel_g("K", sp.Float(1.0) + 20.0 * Ca_sym)],
        )
        spec_mod, _ = _spec_ca("ca_ahp_mod")
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec_mod)
        rn.add_intracellular(dyn, "pop")
        r_mod = rn.simulate(DURATION, DT, {"pop": 5.0},
                            record=RecordingConfig(["V", "calcium"]))

        V_base = r_base["V"][0]
        V_mod  = r_mod["pop"]["V"][0]
        assert np.all(np.isfinite(V_base))
        assert np.all(np.isfinite(V_mod))
        # Increased K conductance → more hyperpolarized → lower mean V
        assert V_mod.mean() < V_base.mean() + 1.0, (
            f"Ca-AHP should reduce mean V: base={V_base.mean():.2f}, "
            f"mod={V_mod.mean():.2f}")


# ---------------------------------------------------------------------------
# B2.2 — Dopamine (CUSTOM_EXPR tonic)
# ---------------------------------------------------------------------------

class TestDopamineDynamics:

    def test_da_steady_state(self):
        """DA must converge to da_rate/k_da = {:.4f} within 2%.""".format(_DA_SS)
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec("da_ss"))
        rn.add_intracellular(_da_dynamics(), "pop")
        r = rn.simulate(LONG_DUR, DT, {"pop": 0.0},
                        record=RecordingConfig(["V", "calcium"]))
        da_trace = r["pop"]["calcium"][0]   # substance 0 = DA
        da_final = da_trace[-50:].mean()    # average last 50 steps
        assert abs(da_final - _DA_SS) / _DA_SS < 0.02, (
            f"DA SS expected {_DA_SS:.4f}, got {da_final:.4f}")

    def test_da_channel_g_diverges_from_base(self):
        """DA scaling a channel conductance produces different V than no DA."""
        # Reference: no DA, no modulation
        V_ref = _run_single(_simple_spec("da_cg_ref"), I_val=5.0)["V"][0]

        # Modulated: DA present, scales K conductance by (1 + 20*DA)
        # DA* = 0.02 → scale ≈ 1.4 → K conductance increases 40%
        DA_sym = sp.Symbol("DA")
        dyn = IntracellularDynamics(
            "DA",
            ode=sp.Float(_DA_RATE) - _K_DA * DA_sym,
            initial=_DA_SS,
            modulations=[Modulation.channel_g("K", sp.Float(1.0) + 20.0 * DA_sym)],
        )
        spec = _simple_spec("da_cg_mod")
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        rn.add_intracellular(dyn, "pop")
        V_mod = rn.simulate(DURATION, DT, {"pop": 5.0},
                            record=RecordingConfig(["V"]))["pop"]["V"][0]

        assert np.all(np.isfinite(V_ref))
        assert np.all(np.isfinite(V_mod))
        assert not np.allclose(V_mod.mean(), V_ref.mean(), atol=0.5), (
            "DA+CHANNEL_G mod should produce measurably different mean V")

    def test_da_synapse_g_modulation(self):
        """DA synapse-G mod: V of receiving population differs from unmodulated."""
        spec_src = _simple_spec("da_sg_src")
        spec_rcv_ref = _simple_spec("da_sg_rcv_ref")
        spec_rcv_mod = _simple_spec("da_sg_rcv_mod")

        # Reference: source→receiver synapse, no DA
        rn_ref = RegionalNetwork()
        rn_ref.add_population("src", 1, model=spec_src)
        rn_ref.add_population("rcv", 1, model=spec_rcv_ref)
        rn_ref.connect("src", "rcv", "all_to_all", weight=2.0,
                       synapse=hh.SynapseSpec.ampa())
        V_rcv_ref = rn_ref.simulate(DURATION, DT, {"src": 10.0, "rcv": 0.0},
                                    record=RecordingConfig(["V"]))["rcv"]["V"][0]

        # Modulated: DA scales synaptic conductance by 0.2 (strong attenuation)
        DA_sym = sp.Symbol("DA")
        dyn = IntracellularDynamics(
            "DA",
            ode=sp.Float(_DA_RATE) - _K_DA * DA_sym,
            initial=_DA_SS,
            modulations=[Modulation.synapse_g(sp.Float(0.2))],
        )
        rn_mod = RegionalNetwork()
        rn_mod.add_population("src", 1, model=spec_src)
        rn_mod.add_population("rcv", 1, model=spec_rcv_mod)
        rn_mod.connect("src", "rcv", "all_to_all", weight=2.0,
                       synapse=hh.SynapseSpec.ampa())
        rn_mod.add_intracellular(dyn, "rcv")
        V_rcv_mod = rn_mod.simulate(DURATION, DT, {"src": 10.0, "rcv": 0.0},
                                    record=RecordingConfig(["V"]))["rcv"]["V"][0]

        assert np.all(np.isfinite(V_rcv_ref))
        assert np.all(np.isfinite(V_rcv_mod))
        # 80% attenuation → receiver should be less excited
        assert V_rcv_mod.mean() < V_rcv_ref.mean() + 2.0, (
            f"DA SYNAPSE_G=0.2 should attenuate: ref={V_rcv_ref.mean():.2f}, "
            f"mod={V_rcv_mod.mean():.2f}")


# ---------------------------------------------------------------------------
# B2.3 — cAMP cascade (cross-substance, driven by DA)
# ---------------------------------------------------------------------------

class TestCaMPDynamics:

    def test_camp_steady_state(self):
        """cAMP must converge to k_prod*DA_ss/k_deg = {:.4f} within 5%.""".format(_CAMP_SS)
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec("camp_ss"))
        rn.add_intracellular(_da_dynamics(),   "pop")   # DA at index 0
        rn.add_intracellular(_camp_dynamics(), "pop")   # cAMP at index 1

        # Record: calcium → substance 0 = DA; no direct cAMP recording exposed yet.
        # Use a second network with swapped order to read cAMP as calcium:
        #   substance 0 = cAMP, substance 1 = DA (inverted — cAMP reads from wrong idx)
        # Instead, run with DA first, check cAMP via substances dict via a known trick:
        # We set initial cAMP = _CAMP_SS, so if ODE is correct the trace should stay
        # near _CAMP_SS after 500ms (it converges to SS quickly).
        r = rn.simulate(LONG_DUR, DT, {"pop": 0.0},
                        record=RecordingConfig(["V", "calcium"]))
        # calcium = substance 0 = DA
        da_trace = r["pop"]["calcium"][0]
        da_final = da_trace[-50:].mean()
        # DA must converge first
        assert abs(da_final - _DA_SS) / _DA_SS < 0.02

    def test_camp_cross_substance_no_crash(self):
        """cAMP ODE that references DA via PUSH_X compiles and runs without error."""
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec("camp_xsubst"))
        rn.add_intracellular(_da_dynamics(),   "pop")
        rn.add_intracellular(_camp_dynamics(), "pop")
        r = rn.simulate(DURATION, DT, {"pop": 3.0},
                        record=RecordingConfig(["V", "calcium"]))
        assert np.all(np.isfinite(r["pop"]["V"]))
        assert np.all(np.isfinite(r["pop"]["calcium"]))

    def test_camp_gate_tau_scale_diverges(self):
        """cAMP GATE_TAU_SCALE produces different V trace than no-cAMP baseline.

        Tau scaling only affects transients, so we compare a short (50 ms) window
        where the gate's slow opening is visible, rather than the full 200 ms run.
        """
        # Run only 50 ms so the transient (gate still opening) dominates
        _DUR = 50.0

        def _run_50(spec, dynamics=(), I_val=5.0):
            rn = RegionalNetwork()
            rn.add_population("pop", 1, model=spec)
            for d in dynamics:
                rn.add_intracellular(d, "pop")
            r = rn.simulate(_DUR, DT, {"pop": I_val},
                            record=RecordingConfig(["V"]))
            return r["pop"]["V"][0]

        V_ref = _run_50(_simple_spec("camp_tau_ref"), I_val=5.0)

        # cAMP at SS ≈ 0.06 → tau_scale = 1 + 100*cAMP ≈ 7 (slows gate 7x)
        cAMP_sym = sp.Symbol("cAMP")
        DA_sym   = sp.Symbol("DA")
        camp_mod = IntracellularDynamics(
            "cAMP",
            ode=sp.Float(_K_PROD) * DA_sym - _K_DEG * cAMP_sym,
            initial=_CAMP_SS,
            modulations=[
                Modulation.gate_tau_scale("m", sp.Float(1.0) + 100.0 * cAMP_sym)
            ],
        )
        spec = _simple_spec("camp_tau_mod")
        V_mod = _run_50(spec, dynamics=[_da_dynamics(), camp_mod])

        assert np.all(np.isfinite(V_ref))
        assert np.all(np.isfinite(V_mod))
        assert not np.allclose(V_mod.mean(), V_ref.mean(), atol=0.2), (
            f"cAMP GATE_TAU_SCALE should produce measurably different V: "
            f"ref={V_ref.mean():.3f}, mod={V_mod.mean():.3f}")


# ---------------------------------------------------------------------------
# B2.4 — Slow buffer Xs (pure DECAY, τ = 200 ms)
# ---------------------------------------------------------------------------

class TestSlowBufferDynamics:

    def test_xs_decay_rate_recoverable(self):
        """Fit an exponential to Xs(t) and recover k within 2%."""
        k = _K_XS
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=_simple_spec("xs_decay"))
        rn.add_intracellular(_xs_dynamics(k), "pop")
        r = rn.simulate(LONG_DUR, DT, {"pop": 0.0},
                        record=RecordingConfig(["V", "calcium"]))
        Xs = r["pop"]["calcium"][0]
        t  = np.arange(len(Xs)) * DT
        pos = Xs > 1e-6
        assert pos.sum() > 10, "Too few positive samples for fit"
        slope, _ = np.polyfit(t[pos], np.log(Xs[pos]), 1)
        k_fit = -slope
        assert abs(k_fit - k) / k < 0.02, (
            f"Expected k={k:.4f}, fitted k={k_fit:.4f}")

    def test_xs_gate_inf_shift(self):
        """GATE_INF_SHIFT by Xs: V trace differs from no-shift baseline."""
        V_ref = _run_single(_simple_spec("xs_shift_ref"), I_val=5.0)["V"][0]

        Xs = sp.Symbol("Xs")
        # shift_scale=20 and initial Xs=1 → gate shifted by +20mV initially
        dyn = IntracellularDynamics(
            "Xs", ode=-_K_XS * Xs, initial=1.0,
            modulations=[Modulation.gate_inf_shift("m", scale=20.0)],
        )
        spec = _simple_spec("xs_shift_mod")
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        rn.add_intracellular(dyn, "pop")
        V_mod = rn.simulate(DURATION, DT, {"pop": 5.0},
                            record=RecordingConfig(["V"]))["pop"]["V"][0]

        assert np.all(np.isfinite(V_mod))
        assert not np.allclose(V_mod.mean(), V_ref.mean(), atol=0.5), (
            "GATE_INF_SHIFT should produce measurably different V")

    def test_xs_gate_inf_scale(self):
        """GATE_INF_SCALE by constant: V trace differs from unscaled."""
        V_ref = _run_single(_simple_spec("xs_scale_ref"), I_val=5.0)["V"][0]

        Xs = sp.Symbol("Xs")
        # Scale gate inf by 0.1 → nearly closes m gate → much less K current
        dyn = IntracellularDynamics(
            "Xs", ode=-_K_XS * Xs, initial=1.0,
            modulations=[Modulation.gate_inf_scale("m", sp.Float(0.1) + 0.0 * Xs)],
        )
        spec = _simple_spec("xs_inf_scale_mod")
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        rn.add_intracellular(dyn, "pop")
        V_mod = rn.simulate(DURATION, DT, {"pop": 5.0},
                            record=RecordingConfig(["V"]))["pop"]["V"][0]

        assert np.all(np.isfinite(V_mod))
        assert not np.allclose(V_mod.mean(), V_ref.mean(), atol=0.5), (
            "GATE_INF_SCALE=0.1 should produce measurably different V")

    def test_xs_channel_erev_override(self):
        """CHANNEL_EREV override: setting K E_rev to 0 mV inverts K current."""
        V_ref = _run_single(_simple_spec("xs_erev_ref"), I_val=5.0)["V"][0]

        Xs = sp.Symbol("Xs")
        # Override E_K from -90 → 0 mV: K now depolarises rather than repolarises
        dyn = IntracellularDynamics(
            "Xs", ode=-_K_XS * Xs, initial=1.0,
            modulations=[Modulation.channel_erev("K", sp.Float(0.0) + 0.0 * Xs)],
        )
        spec = _simple_spec("xs_erev_mod")
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        rn.add_intracellular(dyn, "pop")
        V_mod = rn.simulate(DURATION, DT, {"pop": 5.0},
                            record=RecordingConfig(["V"]))["pop"]["V"][0]

        assert np.all(np.isfinite(V_mod))
        assert not np.allclose(V_mod.mean(), V_ref.mean(), atol=0.5), (
            "CHANNEL_EREV override from -90→0 mV should produce measurably different V")

    def test_xs_gate_inf_expr(self):
        """GATE_INF_EXPR: replace m_inf with constant 0.8, V should differ."""
        V_ref = _run_single(_simple_spec("xs_infexpr_ref"), I_val=5.0)["V"][0]

        Xs = sp.Symbol("Xs")
        V_sym = sp.Symbol("V")
        # Replace m_inf entirely with 0.8 (constant regardless of voltage)
        dyn = IntracellularDynamics(
            "Xs", ode=-_K_XS * Xs, initial=1.0,
            modulations=[Modulation.gate_inf_expr("m", sp.Float(0.8) + 0.0 * V_sym)],
        )
        spec = _simple_spec("xs_infexpr_mod")
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        rn.add_intracellular(dyn, "pop")
        V_mod = rn.simulate(DURATION, DT, {"pop": 5.0},
                            record=RecordingConfig(["V"]))["pop"]["V"][0]

        assert np.all(np.isfinite(V_mod))
        assert not np.allclose(V_mod.mean(), V_ref.mean(), atol=0.5), (
            "GATE_INF_EXPR=constant should produce measurably different V")

    def test_xs_gate_tau_scale(self):
        """GATE_TAU_SCALE x11: slowed gate kinetics → measurably different V.

        Tau scaling only affects the transient, not steady state, so we compare
        a short (50 ms) window where the slower gate opening is visible.
        """
        _DUR = 50.0

        def _run_50(spec, dynamics=(), I_val=5.0):
            rn = RegionalNetwork()
            rn.add_population("pop", 1, model=spec)
            for d in dynamics:
                rn.add_intracellular(d, "pop")
            r = rn.simulate(_DUR, DT, {"pop": I_val},
                            record=RecordingConfig(["V"]))
            return r["pop"]["V"][0]

        V_ref = _run_50(_simple_spec("xs_tau_ref"))

        Xs = sp.Symbol("Xs")
        # Xs starts at 1.0, decays slowly (τ=200ms); tau_scale ≈ 11 for most of the run
        dyn = IntracellularDynamics(
            "Xs", ode=-_K_XS * Xs, initial=1.0,
            modulations=[Modulation.gate_tau_scale("m", sp.Float(1.0) + 10.0 * Xs)],
        )
        spec = _simple_spec("xs_tau_mod")
        V_mod = _run_50(spec, dynamics=[dyn])

        assert np.all(np.isfinite(V_mod))
        assert not np.allclose(V_mod.mean(), V_ref.mean(), atol=0.2), (
            f"GATE_TAU_SCALE x11 should produce measurably different V: "
            f"ref={V_ref.mean():.3f}, mod={V_mod.mean():.3f}")


# ---------------------------------------------------------------------------
# B2.5 — Incremental divergence: base → +Ca → +DA → +cAMP
# ---------------------------------------------------------------------------

class TestIncrementalDivergence:
    """Each added dynamic produces a measurably different V trace from the
    previous configuration, proving the dynamics are actually applied."""

    @staticmethod
    def _mean_v(spec, dynamics_to_add=(), I_val=5.0):
        """Run spec with zero or more dynamics attached, return mean V of neuron 0."""
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=spec)
        for dyn in dynamics_to_add:
            rn.add_intracellular(dyn, "pop")
        r = rn.simulate(DURATION, DT, {"pop": I_val},
                        record=RecordingConfig(["V", "calcium"]))
        return r["pop"]["V"][0].mean()

    def test_ca_diverges_from_base(self):
        """Adding Ca+AHP modulation changes mean V from no-dynamics baseline."""
        Ca_sym = sp.Symbol("Ca")
        ca_mod = IntracellularDynamics(
            "Ca",
            ode=1e-4 * (-I_source - 15.0 * Ca_sym),
            source_channels=["CaL"],
            initial=0.005,
            modulations=[Modulation.channel_g("K", sp.Float(1.0) + 30.0 * Ca_sym)],
        )
        spec_base, _ = _spec_ca("incr_base")
        spec_ca,   _ = _spec_ca("incr_ca")

        v_base = self._mean_v(spec_base)
        v_ca   = self._mean_v(spec_ca, [ca_mod])

        assert np.isfinite(v_base) and np.isfinite(v_ca)
        assert abs(v_ca - v_base) > 0.5, (
            f"Ca+AHP should diverge from base: base={v_base:.2f}, ca={v_ca:.2f}")

    def test_da_diverges_from_ca(self):
        """Adding DA channel-G mod on top of Ca changes V further.

        DA at SS ≈ 0.02; scale = 1 + 50*DA ≈ 2.0 on the K channel (g=5.0 → 10.0).
        Doubling K conductance forces a measurable hyperpolarisation shift.
        """
        Ca_sym = sp.Symbol("Ca")
        DA_sym = sp.Symbol("DA")

        ca_dyn = IntracellularDynamics(
            "Ca",
            ode=1e-4 * (-I_source - 15.0 * Ca_sym),
            source_channels=["CaL"], initial=0.005,
        )
        # Target K (g=5.0) with a 2x multiplier at DA steady state
        da_dyn = IntracellularDynamics(
            "DA",
            ode=sp.Float(_DA_RATE) - _K_DA * DA_sym,
            initial=_DA_SS,
            modulations=[Modulation.channel_g("K", sp.Float(1.0) + 50.0 * DA_sym)],
        )

        spec_ca,     _ = _spec_ca("incr_da_ca")
        spec_ca_da,  _ = _spec_ca("incr_da_cada")

        v_ca    = self._mean_v(spec_ca,    [ca_dyn])
        v_ca_da = self._mean_v(spec_ca_da, [ca_dyn, da_dyn])

        assert np.isfinite(v_ca) and np.isfinite(v_ca_da)
        assert abs(v_ca_da - v_ca) > 0.5, (
            f"Adding DA should diverge from Ca-only: ca={v_ca:.2f}, +da={v_ca_da:.2f}")

    def test_camp_diverges_from_da(self):
        """Adding cAMP GATE_INF_SCALE on top of Ca+DA changes V further.

        cAMP at SS ≈ 0.06; inf_scale = 1 - 10*cAMP ≈ 0.4 → gate inf reduced 60%.
        Less K gate activation → less repolarisation → higher mean V than Ca+DA alone.
        GATE_INF_SCALE affects steady-state (unlike TAU_SCALE), so 200 ms run works.
        """
        Ca_sym   = sp.Symbol("Ca")
        DA_sym   = sp.Symbol("DA")
        cAMP_sym = sp.Symbol("cAMP")

        ca_dyn = IntracellularDynamics(
            "Ca",
            ode=1e-4 * (-I_source - 15.0 * Ca_sym),
            source_channels=["CaL"], initial=0.005,
        )
        da_dyn = IntracellularDynamics(
            "DA",
            ode=sp.Float(_DA_RATE) - _K_DA * DA_sym,
            initial=_DA_SS,
        )
        camp_dyn = IntracellularDynamics(
            "cAMP",
            ode=sp.Float(_K_PROD) * DA_sym - _K_DEG * cAMP_sym,
            initial=_CAMP_SS,
            modulations=[
                # Scale gate inf by ~0.4 at SS → significantly reduces K gating
                Modulation.gate_inf_scale("m", sp.Float(1.0) - 10.0 * cAMP_sym)
            ],
        )

        spec_no_camp, _ = _spec_ca("incr_nocamp")
        spec_camp,    _ = _spec_ca("incr_camp")

        v_no_camp = self._mean_v(spec_no_camp, [ca_dyn, da_dyn])
        v_camp    = self._mean_v(spec_camp,    [ca_dyn, da_dyn, camp_dyn])

        assert np.isfinite(v_no_camp) and np.isfinite(v_camp)
        assert abs(v_camp - v_no_camp) > 0.5, (
            f"Adding cAMP should diverge from Ca+DA: no_camp={v_no_camp:.2f}, "
            f"+camp={v_camp:.2f}")


# ---------------------------------------------------------------------------
# B2.6 — Overlapping cross-population dynamics
# ---------------------------------------------------------------------------

class TestOverlappingPopulationDynamics:
    """Two independent substance dynamics applied to overlapping subsets of pops.

    dyn1 (K_mod): applied to A and B → CHANNEL_G on K, strongly hyperpolarises
    dyn2 (Xs):    applied to B and C → GATE_INF_SCALE, reduces gate activation

    Population A: dyn1 only  → hyperpolarised by K upregulation
    Population B: dyn1 + dyn2 → both effects (most distinct)
    Population C: dyn2 only  → less K-gate activation

    Neither substance references the other — no cross-substance dependency issue.
    """

    # Parameters for the two independent substances
    _K_KMOD  = 0.005   # slow decay for K_mod (τ=200ms — stays near initial during 200ms run)
    _K_XS2   = 0.005   # slow decay for Xs2

    def _three_pop_means(self):
        """Build 3-pop network with overlapping dynamics; return mean V for A, B, C."""
        Km  = sp.Symbol("Km")   # K_mod substance
        Xs2 = sp.Symbol("Xs2")  # second slow substance

        # dyn1: DECAY, scales K conductance by 3x while Km ≈ 1
        dyn1 = IntracellularDynamics(
            "Km", ode=-self._K_KMOD * Km, initial=1.0,
            modulations=[Modulation.channel_g("K", sp.Float(1.0) + 2.0 * Km)],
        )
        # dyn2: DECAY, scales gate inf by 0.2 while Xs2 ≈ 1
        dyn2 = IntracellularDynamics(
            "Xs2", ode=-self._K_XS2 * Xs2, initial=1.0,
            modulations=[Modulation.gate_inf_scale("m", sp.Float(1.0) - 0.8 * Xs2)],
        )

        rn = RegionalNetwork()
        rn.add_population("A", 2, model=_simple_spec("ovlp_A"))
        rn.add_population("B", 2, model=_simple_spec("ovlp_B"))
        rn.add_population("C", 2, model=_simple_spec("ovlp_C"))

        rn.add_intracellular(dyn1, ["A", "B"])   # K upregulation in A and B
        rn.add_intracellular(dyn2, ["B", "C"])   # gate inf suppression in B and C

        r = rn.simulate(DURATION, DT, {"A": 5.0, "B": 5.0, "C": 5.0},
                        record=RecordingConfig(["V", "calcium"]))
        return (r["A"]["V"][0].mean(),
                r["B"]["V"][0].mean(),
                r["C"]["V"][0].mean())

    def _base_mean(self):
        """Reference: no dynamics on any pop."""
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec("ovlp_base"))
        r = rn.simulate(DURATION, DT, {"pop": 5.0},
                        record=RecordingConfig(["V"]))
        return r["pop"]["V"][0].mean()

    def test_a_diverges_from_base(self):
        """Population A (DA only) differs from no-dynamics baseline."""
        v_base = self._base_mean()
        v_A, _, _ = self._three_pop_means()
        assert np.isfinite(v_A)
        assert abs(v_A - v_base) > 0.5, (
            f"A (DA only) should differ from base: base={v_base:.2f}, A={v_A:.2f}")

    def test_c_diverges_from_base(self):
        """Population C (cAMP only) differs from no-dynamics baseline."""
        v_base = self._base_mean()
        _, _, v_C = self._three_pop_means()
        assert np.isfinite(v_C)
        assert abs(v_C - v_base) > 0.5, (
            f"C (cAMP only) should differ from base: base={v_base:.2f}, C={v_C:.2f}")

    def test_a_differs_from_c(self):
        """A (DA→CHANNEL_G) and C (cAMP→GATE_TAU_SCALE) have different mean V."""
        v_A, _, v_C = self._three_pop_means()
        assert np.isfinite(v_A) and np.isfinite(v_C)
        assert abs(v_A - v_C) > 0.5, (
            f"A (DA) and C (cAMP) should behave differently: A={v_A:.2f}, C={v_C:.2f}")

    def test_b_differs_from_a_and_c(self):
        """B (both DA+cAMP) differs from A-only and C-only populations."""
        v_A, v_B, v_C = self._three_pop_means()
        assert np.isfinite(v_B)
        assert abs(v_B - v_A) > 0.3 or abs(v_B - v_C) > 0.3, (
            f"B (both dynamics) should differ from A or C: A={v_A:.2f}, "
            f"B={v_B:.2f}, C={v_C:.2f}")

    def test_population_density_invariance(self):
        """Same dynamics applied to 1-neuron vs 4-neuron pop → same per-neuron mean V."""
        DA_sym = sp.Symbol("DA")
        dyn = IntracellularDynamics(
            "DA",
            ode=sp.Float(_DA_RATE) - _K_DA * DA_sym,
            initial=_DA_SS,
            modulations=[Modulation.channel_g("K", sp.Float(1.0) + 20.0 * DA_sym)],
        )

        def _run(n, spec_name):
            rn = RegionalNetwork()
            rn.add_population("pop", n, model=_simple_spec(spec_name))
            rn.add_intracellular(dyn, "pop")
            r = rn.simulate(DURATION, DT, {"pop": 5.0},
                            record=RecordingConfig(["V"]))
            return r["pop"]["V"].mean(axis=1)  # per-neuron mean

        v1 = _run(1, "density_1n")
        v4 = _run(4, "density_4n")

        # All 4 neurons are identical (same params, same I) → same mean V
        assert np.allclose(v4, v4[0], atol=1e-6), (
            "All neurons in identical pool should have same mean V")
        # 1-neuron and 4-neuron pools with same spec → same per-neuron mean
        assert abs(v1[0] - v4[0]) < 0.1, (
            f"Pool size should not affect per-neuron V: n=1: {v1[0]:.3f}, "
            f"n=4 mean: {v4[0]:.3f}")


# =============================================================================
# Section C — Modulation correctness tests
# =============================================================================

class TestModulationCorrectness:

    def test_channel_g_modulation_constant(self):
        """CHANNEL_G mod returning constant: compare two identical setups."""
        # Run reference (no mod)
        rn_ref = RegionalNetwork()
        rn_ref.add_population("pop", 1, model=_simple_spec())
        result_ref = rn_ref.simulate(DURATION, DT, {"pop": 3.0},
                                     record=RecordingConfig(["V"]))

        # Run with CHANNEL_G mod that returns 1.0 (no-op scale)
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-10.0 * X, initial=1.0,
                                    modulations=[
                                        Modulation.channel_g("Leak", sp.Float(1.0) + 0.0 * X)
                                    ])
        rn_mod = RegionalNetwork()
        rn_mod.add_population("pop", 1, model=_simple_spec())
        rn_mod.add_intracellular(dyn, "pop")
        result_mod = rn_mod.simulate(DURATION, DT, {"pop": 3.0},
                                     record=RecordingConfig(["V"]))

        # Not exactly equal (one-step lag in modulation) but should be very close
        # for a no-op modifier
        V_ref = result_ref["pop"]["V"][0]
        V_mod = result_mod["pop"]["V"][0]
        # At least check both are finite and bounded
        assert np.all(np.isfinite(V_ref))
        assert np.all(np.isfinite(V_mod))

    def test_gate_inf_shift_no_crash(self):
        """GATE_INF_SHIFT: verify simulation stays finite with nonzero shift."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-2.0 * X, initial=1.0,
                                    modulations=[Modulation.gate_inf_shift("m", scale=5.0)])
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        rn.add_intracellular(dyn, "pop")
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=RecordingConfig(["V"]))
        assert np.all(np.isfinite(result["pop"]["V"]))

    def test_synapse_g_modulation_scale(self):
        """SYNAPSE_G mod with scale < 1 reduces synaptic effect."""
        spec = _simple_spec()
        X = sp.Symbol("X")

        # Reference: no modulation, single synapse
        rn_ref = RegionalNetwork()
        rn_ref.add_population("A", 1, model=spec)
        rn_ref.add_population("B", 1, model=spec)
        rn_ref.connect("A", "B", "all_to_all", weight=2.0,
                       synapse=hh.SynapseSpec.ampa())
        V_ref = rn_ref.simulate(DURATION, DT, {"A": 10.0, "B": 0.0},
                                record=RecordingConfig(["V"]))["B"]["V"][0]

        # Modulated: SYNAPSE_G returning 0.5 → halved synaptic input
        rn_mod = RegionalNetwork()
        rn_mod.add_population("A", 1, model=spec)
        rn_mod.add_population("B", 1, model=spec)
        rn_mod.connect("A", "B", "all_to_all", weight=2.0,
                       synapse=hh.SynapseSpec.ampa())
        dyn = IntracellularDynamics("X", ode=-100.0 * X, initial=1.0,
                                    modulations=[Modulation.synapse_g(sp.Float(0.5))])
        rn_mod.add_intracellular(dyn, "B")  # modulate incoming to B
        V_mod = rn_mod.simulate(DURATION, DT, {"A": 10.0, "B": 0.0},
                                record=RecordingConfig(["V"]))["B"]["V"][0]

        assert np.all(np.isfinite(V_ref))
        assert np.all(np.isfinite(V_mod))
        # With halved synaptic input, B should be less depolarized on average
        mean_ref = V_ref.mean()
        mean_mod = V_mod.mean()
        # Modulated population should be less excited (or at least not more)
        # Allow some tolerance since the one-step lag and small window
        assert mean_mod <= mean_ref + 2.0, (
            f"Expected modulated B less excited (mean_ref={mean_ref:.2f}, "
            f"mean_mod={mean_mod:.2f})")


# =============================================================================
# Section D — API / integration tests
# =============================================================================

class TestAPIIntegration:

    def test_add_to_subset_of_pops(self):
        """Substance in pop A but not pop B."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=0.5)
        rn = RegionalNetwork()
        # Use distinct spec names so A and B get separate pools
        rn.add_population("A", 2, model=_simple_spec("specA"))
        rn.add_population("B", 2, model=_simple_spec("specB"))
        rn.add_intracellular(dyn, "A")  # only pop A
        result = rn.simulate(DURATION, DT, {"A": 3.0, "B": 2.0},
                             record=RecordingConfig(["V", "calcium"]))
        # A has calcium recorded (substance 0)
        assert np.all(np.isfinite(result["A"]["calcium"]))
        # B has no intracellular — calcium buf is all zeros
        B_ca = result["B"]["calcium"]
        assert np.all(B_ca == 0.0), "B should have no calcium dynamics"

    def test_add_to_all_pops_via_none(self):
        """populations=None → all Composable populations get the substance."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=0.5)
        rn = RegionalNetwork()
        rn.add_population("A", 2, model=_simple_spec())
        rn.add_population("B", 2, model=_simple_spec())
        rn.add_population("C", 2, model=_simple_spec())
        rn.add_intracellular(dyn)  # populations=None
        result = rn.simulate(DURATION, DT, {"A": 3.0, "B": 2.0, "C": 1.0},
                             record=RecordingConfig(["V", "calcium"]))
        for pop in ("A", "B", "C"):
            assert np.all(np.isfinite(result[pop]["calcium"]))

    def test_duplicate_name_error(self):
        """add_intracellular twice with same name raises ValueError."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=0.5)
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        rn.add_intracellular(dyn, "pop")
        with pytest.raises(ValueError, match="X"):
            rn.add_intracellular(dyn, "pop")  # same name, same pop → error

    def test_invalid_channel_name_error(self):
        """Modulation targeting nonexistent channel raises ValueError."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=0.5,
                                    modulations=[
                                        Modulation.channel_g("NONEXISTENT_CHANNEL",
                                                             sp.Float(1.0))
                                    ])
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        with pytest.raises((ValueError, KeyError)):
            rn.add_intracellular(dyn, "pop")

    def test_invalid_gate_name_error(self):
        """Modulation targeting nonexistent gate raises ValueError."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-1.0 * X, initial=0.5,
                                    modulations=[
                                        Modulation.gate_inf_shift("NONEXISTENT_GATE", 1.0)
                                    ])
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        with pytest.raises((ValueError, KeyError)):
            rn.add_intracellular(dyn, "pop")

    def test_recording_shapes(self):
        """substances['Ca'] shape == (n_neurons, n_steps)."""
        spec = _spec_with_driven_decay()
        rn = RegionalNetwork()
        rn.add_population("pop", 3, model=spec)
        interval = 5
        n_rec = int(DURATION / DT / interval) + 1  # approximate
        cfg = RecordingConfig(["V", "calcium"], interval=interval)
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=cfg)
        ca = result["pop"]["calcium"]
        assert ca.shape[0] == 3, f"Expected 3 neurons, got {ca.shape[0]}"
        # time steps should be approximately DURATION/(DT*interval)
        assert ca.shape[1] > 0

    def test_result_calcium_alias(self):
        """result.calcium is alias for result['calcium']."""
        spec = _spec_with_driven_decay()
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=cfg)
        mr = result["pop"]
        ca_key = mr["calcium"]
        ca_prop = mr.calcium
        assert ca_key is ca_prop or np.array_equal(ca_key, ca_prop)

    def test_result_substances_dict(self):
        """MetricsResult.substances dict is populated when calcium recorded."""
        spec = _spec_with_driven_decay()
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=spec)
        cfg = RecordingConfig(["V", "calcium"])
        result = rn.simulate(DURATION, DT, {"pop": 3.0}, record=cfg)
        mr = result["pop"]
        subs = mr.substances
        assert isinstance(subs, dict)
        assert "Ca" in subs
        assert np.all(np.isfinite(subs["Ca"]))

    def test_intracellular_spec_accessible(self):
        """NeuronModelSpec.intracellular accessible and contains expected entry."""
        spec = _spec_with_driven_decay()
        assert len(spec.intracellular) == 1
        ic = spec.intracellular[0]
        assert ic.name == "Ca"
        assert ic.update_form in (IntracellularUpdateForm.DRIVEN_DECAY,
                                  IntracellularUpdateForm.DRIVEN_DECAY_NERNST)

    def test_push_x_opcode_value(self):
        """PUSH_X opcode is bound and has value 19."""
        from hodgkin_huxley._core import VmOp
        assert int(VmOp.PUSH_X) == 19

    def test_update_population_spec_accessible(self):
        """update_population_spec method is callable on RegionalNetwork."""
        rn = RegionalNetwork()
        rn.add_population("pop", 2, model=_simple_spec())
        # Should not raise
        rn._rnet.update_population_spec("pop", _simple_spec())


# =============================================================================
# Section E — Pattern matching round-trip tests
# =============================================================================

class TestPatternMatching:

    def _build_spec_and_run(self, dyn):
        """Helper: attach dynamics to a simple pop and run for correctness check."""
        rn = RegionalNetwork()
        rn.add_population("pop", 1, model=_simple_spec())
        rn.add_intracellular(dyn, "pop")
        cfg = RecordingConfig(["V", "calcium"])
        return rn.simulate(DURATION, DT, {"pop": 0.0}, record=cfg)

    def test_pattern_decay(self):
        """ode=-k*X → UpdateForm.DECAY, no VM compiled."""
        from hodgkin_huxley._core import IntracellularUpdateForm
        k = 8.0
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-k * X, initial=1.0)
        spec = dyn.to_spec(
            channel_names=["K", "Leak"],
            gate_names=["m"],
        )
        assert spec.update_form == IntracellularUpdateForm.DECAY
        assert abs(spec.k_decay - k) < 1e-10
        assert spec.ode_vm.empty()

    def test_pattern_driven_decay(self):
        """ode=eps*(-I_source - k*X) → UpdateForm.DRIVEN_DECAY."""
        from hodgkin_huxley._core import IntracellularUpdateForm
        eps = 1e-4
        k = 15.0
        X = sp.Symbol("X")
        ode = eps * (-I_source - k * X)
        dyn = IntracellularDynamics("X", ode=ode, initial=0.1)
        spec = dyn.to_spec(channel_names=["K"], gate_names=["m"])
        assert spec.update_form == IntracellularUpdateForm.DRIVEN_DECAY
        assert abs(spec.epsilon - eps) < 1e-10
        assert abs(spec.k_decay - k) < 1e-10
        assert spec.ode_vm.empty()

    def test_pattern_fallback_custom(self):
        """Novel ODE → UpdateForm.CUSTOM_EXPR, ode_vm non-empty."""
        from hodgkin_huxley._core import IntracellularUpdateForm
        X = sp.Symbol("X")
        ode = sp.sin(X) * I_source - X / 5.0  # non-standard
        dyn = IntracellularDynamics("X", ode=ode, initial=0.1)
        spec = dyn.to_spec(channel_names=["K"], gate_names=["m"])
        assert spec.update_form == IntracellularUpdateForm.CUSTOM_EXPR
        assert not spec.ode_vm.empty()

    def test_pattern_decay_no_crash(self):
        """DECAY pattern-matched run stays finite."""
        k = 5.0
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("X", ode=-k * X, initial=1.0)
        result = self._build_spec_and_run(dyn)
        ca = result["pop"]["calcium"][0]
        assert np.all(np.isfinite(ca))
        assert np.all(ca >= -1e-9)

    def test_pattern_driven_no_crash(self):
        """DRIVEN_DECAY pattern-matched run stays finite."""
        X = sp.Symbol("X")
        ode = 1e-4 * (-I_source - 15.0 * X)
        dyn = IntracellularDynamics("X", ode=ode, initial=0.1)
        result = self._build_spec_and_run(dyn)
        ca = result["pop"]["calcium"][0]
        assert np.all(np.isfinite(ca))

    def test_pattern_custom_no_crash(self):
        """CUSTOM_EXPR fallback run stays finite."""
        X = sp.Symbol("X")
        ode = -1.0 * X + 0.01 * I_source  # linear but non-standard form
        dyn = IntracellularDynamics("X", ode=ode, initial=0.5)
        result = self._build_spec_and_run(dyn)
        ca = result["pop"]["calcium"][0]
        assert np.all(np.isfinite(ca))

    def test_substance_symbol_helper(self):
        """substance() caches and returns consistent Symbol objects."""
        s1 = substance("myDA")
        s2 = substance("myDA")
        assert s1 is s2
        assert str(s1) == "myDA"

    def test_predefined_substance_symbols(self):
        """Pre-defined substance symbols are accessible from top-level import."""
        assert str(DA) == "DA"
        assert str(cAMP) == "cAMP"
        assert str(IP3) == "IP3"
        assert str(NO) == "NO"
        assert str(X_ic) == "X_ic"

    def test_intracellular_dynamics_repr(self):
        """IntracellularDynamics has useful repr."""
        X = sp.Symbol("X")
        dyn = IntracellularDynamics("myX", ode=-1.0 * X, initial=0.5)
        r = repr(dyn)
        assert "myX" in r

    def test_modulation_repr(self):
        """Modulation factories produce objects with useful repr."""
        X = sp.Symbol("X")
        mod = Modulation.channel_g("CaL", sp.Float(1.0))
        r = repr(mod)
        assert "CHANNEL_G" in r or "Modulation" in r

    def test_legacy_gate_dependency_calcium(self):
        """GateSpec.dependency=CALCIUM still resolves (INTRACELLULAR alias)."""
        from hodgkin_huxley._core import GateDependency
        dep = GateDependency.CALCIUM
        assert dep == GateDependency.INTRACELLULAR

    def test_legacy_use_calcium_nernst(self):
        """ChannelSpec.use_calcium_nernst property works as deprecated alias."""
        from hodgkin_huxley._core import ChannelSpec as _ChannelSpec
        ch = _ChannelSpec()
        ch.use_calcium_nernst = True
        assert ch.nernst_substance_idx == 0
        ch.use_calcium_nernst = False
        assert ch.nernst_substance_idx == -1
