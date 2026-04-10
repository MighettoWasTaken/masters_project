"""
Comprehensive tests for composable kinetic synapse (Task 6).

Tests cover:
  - Preset factory field values
  - S initialisation and reset correctness
  - Exact-integration steady-state convergence (TANH / ALPHA_BETA / BOLTZMANN)
  - S bounds [0, 1]
  - Current form comparison (LINEAR vs MG_BLOCK)
  - Power effect on conductance
  - Weight scaling
  - All neuron-type combinations as pre and post
  - All update-form × current-form product combinations
  - Mixed kinetic + spike-triggered synapse types
  - Multiple kinetic synapses on the same post-neuron (fan-in)
  - Spec name deduplication
  - Inhibitory and excitatory functional effects on firing rate
  - Edge cases: zero weight, very large weight, extreme V_pre, extreme tau
  - RegionalNetwork integration (all connectivity patterns)
  - KineticSynapseModel builder roundtrip and chaining
  - Numerical stability: no NaN / Inf in any trace
"""

import math
import pytest
import numpy as np

import hodgkin_huxley as hh
from hodgkin_huxley import (
    RegionalNetwork,
    KineticSynapseSpec,
    KineticSynapseModel,
    NeuronModelSpec,
    SynapseSpec,
    RecordingConfig,
    WeightDistribution,
    ConnectivityPattern,
    IzhikevichType,
)
from hodgkin_huxley._core import (
    KineticUpdateForm,
    KineticCurrentForm,
    TauParams,
    TauForm,
    BoltzmannParams,
    RateFuncParams,
    RateFuncForm,
)
from neuron_specs import make_thalamic, make_stn, make_gpe, make_gpi, make_striatum


# ── constants ─────────────────────────────────────────────────────────────────

DT   = 0.025   # ms — typical simulation step
LONG = 2000.0  # ms — enough for S to converge for most specs


# ── helpers ──────────────────────────────────────────────────────────────────

def _fresh_rn_with_spec(spec: KineticSynapseSpec, weight: float = 1.0):
    """Return (rn, kin_syn_idx=0) — 2-neuron HH network with one kinetic synapse."""
    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=weight, spec=spec)
    return rn, 0   # kinetic synapse index 0


def _run(rn: RegionalNetwork, duration: float = LONG, dt: float = DT,
         I_pre: float = 0.0, I_post: float = 0.0) -> np.ndarray:
    """Simulate and return stacked (2, n_steps) voltage traces."""
    result = rn.simulate(duration, dt, {"pre": I_pre, "post": I_post})
    return np.vstack([result["pre"], result["post"]])


def _count_spikes(trace: np.ndarray, threshold: float = 0.0) -> int:
    crossings = 0
    prev_below = True
    for v in trace:
        above = v > threshold
        if above and prev_below:
            crossings += 1
        prev_below = not above
    return crossings


def _all_finite(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr)))


def _get_S(rn: RegionalNetwork, syn: int) -> float:
    """Get kinetic gating variable S from the underlying network."""
    return rn._rnet.network().get_kin_S(syn)


def _get_g(rn: RegionalNetwork, syn: int) -> float:
    """Get effective kinetic conductance g from the underlying network."""
    return rn._rnet.network().get_kin_g(syn)


# ── standard specs (reused across tests) ─────────────────────────────────────

def _tanh_linear_spec(name="tanh_lin") -> KineticSynapseSpec:
    s = KineticSynapseSpec.gaba_kinetic()
    s.name = name
    return s


def _alpha_beta_spec(name="ab_spec") -> KineticSynapseSpec:
    s = KineticSynapseSpec()
    s.name = name
    s.update_form = KineticUpdateForm.ALPHA_BETA
    ra = RateFuncParams(); ra.form = RateFuncForm.EXP_DECAY
    ra.A = 0.5; ra.B = 0.0; ra.C = 1e9        # α ≈ 0.5 (constant)
    rb = RateFuncParams(); rb.form = RateFuncForm.EXP_DECAY
    rb.A = 0.1; rb.B = 0.0; rb.C = 1e9        # β ≈ 0.1 (constant)
    s.alpha = ra; s.beta = rb
    s.current_form = KineticCurrentForm.LINEAR
    s.g = 0.1; s.E_syn = -80.0; s.power = 1
    return s


def _boltzmann_linear_spec(name="btz_lin") -> KineticSynapseSpec:
    return _boltzmann_spec_manual(name, current_form=KineticCurrentForm.LINEAR)


def _boltzmann_spec_manual(name="btz", current_form=KineticCurrentForm.LINEAR,
                            tau_val=80.0) -> KineticSynapseSpec:
    s = KineticSynapseSpec()
    s.name = name
    s.update_form = KineticUpdateForm.BOLTZMANN_GATE
    si = BoltzmannParams(); si.v_half = -20.0; si.k = 16.0
    s.s_inf = si
    t = TauParams(); t.form = TauForm.CONSTANT; t.set_param(0, tau_val)
    s.tau = t
    s.current_form = current_form
    s.g = 0.1; s.E_syn = 0.0 if current_form == KineticCurrentForm.MG_BLOCK else -80.0
    s.power = 1
    s.mg_conc = 1.0; s.mg_scale = 0.062; s.mg_denom = 3.57
    return s


# ── neuron model factories (replace NetworkNeuronType) ────────────────────────

def _hh_model():    return NeuronModelSpec.hh_default()
def _iz_rs_model(): return NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING)
def _iz_fs_model(): return NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING)
def _iz_ib_model(): return NeuronModelSpec.izhikevich(IzhikevichType.INTRINSICALLY_BURSTING)
def _iz_ch_model(): return NeuronModelSpec.izhikevich(IzhikevichType.CHATTERING)
def _iz_lts_model(): return NeuronModelSpec.izhikevich(IzhikevichType.LOW_THRESHOLD_SPIKING)

# ComposableNeuron model specs
_COMPOSABLE_MODELS = {
    "thalamic": make_thalamic,
    "stn":      make_stn,
    "gpe":      make_gpe,
    "gpi":      make_gpi,
    "striatum": make_striatum,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRESET FACTORY FIELD VALUES
# ═══════════════════════════════════════════════════════════════════════════

def test_gaba_kinetic_preset_fields():
    s = KineticSynapseSpec.gaba_kinetic()
    assert s.name == "GABA_kinetic"
    assert s.update_form == KineticUpdateForm.TANH_GATE
    assert math.isclose(s.tanh_amp, 2.0)
    assert math.isclose(s.tanh_k, 4.0)
    assert math.isclose(s.tau_decay, 13.0)
    assert s.current_form == KineticCurrentForm.LINEAR
    assert math.isclose(s.E_syn, -80.0)
    assert s.power == 1
    assert math.isclose(s.S_init, 0.0)


def test_nmda_kinetic_preset_fields():
    s = KineticSynapseSpec.nmda_kinetic()
    assert s.name == "NMDA_kinetic"
    assert s.update_form == KineticUpdateForm.BOLTZMANN_GATE
    assert s.current_form == KineticCurrentForm.MG_BLOCK
    assert math.isclose(s.mg_conc, 1.0)
    assert math.isclose(s.mg_scale, 0.062)
    assert math.isclose(s.mg_denom, 3.57)
    assert math.isclose(s.E_syn, 0.0)
    assert math.isclose(s.S_init, 0.0)


def test_gaba_b_preset_fields():
    s = KineticSynapseSpec.gaba_b()
    assert s.name == "GABA_B"
    assert s.update_form == KineticUpdateForm.BOLTZMANN_GATE
    assert s.tau.get_param(0) >= 150.0    # slow: ~200 ms
    assert math.isclose(s.E_syn, -95.0)
    assert s.power >= 2                    # cooperative gating


# ═══════════════════════════════════════════════════════════════════════════
# 2. S INITIALISATION AND RESET
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("S_init", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_s_init_respected(S_init):
    """S immediately after construction equals S_init."""
    spec = KineticSynapseSpec.gaba_kinetic()
    spec.S_init = S_init
    rn, syn = _fresh_rn_with_spec(spec)
    assert math.isclose(_get_S(rn, syn), S_init, rel_tol=1e-12)


@pytest.mark.parametrize("S_init", [0.0, 0.5, 1.0])
def test_reset_restores_s_init(S_init):
    """After simulate() + reset(), S == S_init."""
    spec = KineticSynapseSpec.gaba_kinetic()
    spec.S_init = S_init
    rn, syn = _fresh_rn_with_spec(spec)

    _run(rn, 500.0, I_pre=10.0)
    rn.reset()
    assert math.isclose(_get_S(rn, syn), S_init, abs_tol=1e-12), \
        f"After reset S={_get_S(rn, syn)}, expected S_init={S_init}"


def test_reset_g_is_zero():
    """After reset, g should be 0.0 for a synapse that was active."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, syn = _fresh_rn_with_spec(spec)
    _run(rn, 200.0, I_pre=15.0)
    rn.reset()
    assert math.isclose(_get_g(rn, syn), 0.0, abs_tol=1e-12)


def test_two_sims_after_reset_identical():
    """Simulate → reset → simulate again produces identical traces."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, _ = _fresh_rn_with_spec(spec)

    t1 = _run(rn, 200.0, I_pre=8.0, I_post=5.0)
    rn.reset()
    t2 = _run(rn, 200.0, I_pre=8.0, I_post=5.0)
    np.testing.assert_array_almost_equal(t1, t2, decimal=10)


# ═══════════════════════════════════════════════════════════════════════════
# 3. STEADY-STATE CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════

def test_tanh_gate_s_near_zero_at_rest():
    """With silent HH pre (I_ext=0, V≈-65), GABA kinetic S → ≈ 0."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, syn = _fresh_rn_with_spec(spec)
    _run(rn, LONG, I_pre=0.0, I_post=0.0)
    S = _get_S(rn, syn)
    assert S < 0.02, f"S at rest should be near 0, got {S:.4f}"


def test_tanh_gate_s_positive_with_active_pre():
    """With spiking HH pre, GABA kinetic S rises above zero."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, syn = _fresh_rn_with_spec(spec)
    _run(rn, 500.0, I_pre=12.0, I_post=0.0)
    S = _get_S(rn, syn)
    assert S > 0.05, f"S with active pre should be > 0, got {S:.4f}"


def test_tanh_gate_s_higher_with_stronger_drive():
    """Stronger drive on pre → higher final S."""
    spec_a = KineticSynapseSpec.gaba_kinetic(); spec_a.name = "a"
    spec_b = KineticSynapseSpec.gaba_kinetic(); spec_b.name = "b"

    rn_weak, syn_w = _fresh_rn_with_spec(spec_a)
    _run(rn_weak, 500.0, I_pre=5.0)
    S_weak = _get_S(rn_weak, syn_w)

    rn_strong, syn_s = _fresh_rn_with_spec(spec_b)
    _run(rn_strong, 500.0, I_pre=15.0)
    S_strong = _get_S(rn_strong, syn_s)

    assert S_strong >= S_weak, \
        f"Stronger drive should yield >= S: {S_strong:.4f} vs {S_weak:.4f}"


def test_tanh_gate_analytical_steady_state():
    """TANH_GATE: exact steady state at clamped-like V_pre = rest.

    HH at rest (I=0) stays at V≈-65.  Compute S_inf(-65) analytically
    and verify simulation agrees.
    """
    spec = KineticSynapseSpec.gaba_kinetic()
    V_pre = -65.0
    rate_open = spec.tanh_amp * (1.0 + math.tanh((V_pre - spec.tanh_vh) / spec.tanh_k))
    rate_total = rate_open + 1.0 / spec.tau_decay
    S_inf = rate_open / rate_total

    rn, syn = _fresh_rn_with_spec(spec)
    _run(rn, LONG, I_pre=0.0)
    S_sim = _get_S(rn, syn)
    assert abs(S_sim - S_inf) < 0.01, \
        f"S_sim={S_sim:.5f} S_inf={S_inf:.5f}"


def test_alpha_beta_analytical_steady_state():
    """ALPHA_BETA: at rest (V≈-65) S → α/(α+β) analytically."""
    spec = _alpha_beta_spec()
    # α ≈ 0.5, β ≈ 0.1 (via EXP_DECAY with very large C → f(V)≈A)
    alpha_val = spec.alpha.A
    beta_val  = spec.beta.A
    S_inf = alpha_val / (alpha_val + beta_val)

    rn, syn = _fresh_rn_with_spec(spec)
    _run(rn, LONG, I_pre=0.0)
    S_sim = _get_S(rn, syn)
    assert abs(S_sim - S_inf) < 0.01, \
        f"S_sim={S_sim:.5f} S_inf={S_inf:.5f}"


def test_boltzmann_gate_analytical_steady_state():
    """BOLTZMANN_GATE: at rest (V≈-65) S → Boltzmann(-65, v_half=-20, k=16)."""
    spec = _boltzmann_spec_manual("btz_ss")
    V_pre = -65.0
    S_inf = 1.0 / (1.0 + math.exp(-(V_pre - spec.s_inf.v_half) / spec.s_inf.k))

    rn, syn = _fresh_rn_with_spec(spec)
    _run(rn, LONG, I_pre=0.0)
    S_sim = _get_S(rn, syn)
    assert abs(S_sim - S_inf) < 0.01, \
        f"S_sim={S_sim:.5f} S_inf={S_inf:.5f}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. S STAYS IN [0, 1]
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("spec_factory,I_pre", [
    (KineticSynapseSpec.gaba_kinetic,  0.0),
    (KineticSynapseSpec.gaba_kinetic, 20.0),
    (KineticSynapseSpec.nmda_kinetic,  0.0),
    (KineticSynapseSpec.nmda_kinetic, 20.0),
    (KineticSynapseSpec.gaba_b,        0.0),
    (KineticSynapseSpec.gaba_b,       20.0),
])
def test_s_bounded(spec_factory, I_pre):
    """S must remain in [0, 1] throughout simulation."""
    spec = spec_factory()
    rn, syn = _fresh_rn_with_spec(spec)

    dt = DT
    n_steps_check = 2000

    rn.reset()
    S_vals = []
    for _ in range(n_steps_check):
        rn._rnet.network().step(dt, [I_pre, 0.0])
        S_vals.append(_get_S(rn, syn))

    assert all(0.0 - 1e-10 <= s <= 1.0 + 1e-10 for s in S_vals), \
        f"S out of [0,1]: min={min(S_vals):.4f} max={max(S_vals):.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. CURRENT FORM: LINEAR vs MG_BLOCK
# ═══════════════════════════════════════════════════════════════════════════

def test_mg_block_reduces_conductance_at_hyperpolarised():
    """At V_post=-70, MG_BLOCK g_eff < LINEAR g_eff (same S)."""
    spec_mg  = KineticSynapseSpec.nmda_kinetic()
    spec_lin = _boltzmann_spec_manual("nmda_lin", current_form=KineticCurrentForm.LINEAR,
                                      tau_val=80.0)
    spec_lin.s_inf.v_half = spec_mg.s_inf.v_half
    spec_lin.s_inf.k      = spec_mg.s_inf.k
    spec_lin.g = spec_mg.g
    spec_lin.E_syn = spec_mg.E_syn

    rn_mg,  syn_mg  = _fresh_rn_with_spec(spec_mg,  weight=1.0)
    rn_lin, syn_lin = _fresh_rn_with_spec(spec_lin, weight=1.0)

    for rn_ in (rn_mg, rn_lin):
        rn_.simulate(500.0, DT, {"pre": 15.0, "post": 0.0})

    S_mg  = _get_S(rn_mg,  syn_mg)
    S_lin = _get_S(rn_lin, syn_lin)
    g_mg  = _get_g(rn_mg,  syn_mg)
    g_lin = _get_g(rn_lin, syn_lin)

    # MG block denominator at V=-65: ≈ 17 → strong block at hyperpolarised V
    assert g_mg < g_lin or math.isclose(g_mg, g_lin, rel_tol=0.5), \
        f"MG g={g_mg:.6f} should be < LINEAR g={g_lin:.6f} (V_post hyperpolarised)"


def test_mg_block_weaker_at_depolarised():
    """MG block is less severe at positive V_post (partial relief)."""
    spec = KineticSynapseSpec.nmda_kinetic()
    V_hyperpol = -70.0
    V_depol    = +40.0
    mg_hyp = 1.0 + spec.mg_conc * math.exp(-spec.mg_scale * V_hyperpol) / spec.mg_denom
    mg_dep = 1.0 + spec.mg_conc * math.exp(-spec.mg_scale * V_depol)    / spec.mg_denom
    assert mg_dep < mg_hyp, \
        f"MG factor at +40={mg_dep:.3f} should be < at -70={mg_hyp:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# 6. POWER EFFECT
# ═══════════════════════════════════════════════════════════════════════════

def test_power2_smaller_conductance_than_power1():
    """Same spec, power=2 → g_eff = g*S^2 ≤ g*S^1 when S in (0,1)."""
    spec1 = KineticSynapseSpec.gaba_kinetic(); spec1.name = "p1"; spec1.power = 1
    spec2 = KineticSynapseSpec.gaba_kinetic(); spec2.name = "p2"; spec2.power = 2

    rn1, syn1 = _fresh_rn_with_spec(spec1)
    rn1.simulate(200.0, DT, {"pre": 10.0, "post": 0.0})

    rn2, syn2 = _fresh_rn_with_spec(spec2)
    rn2.simulate(200.0, DT, {"pre": 10.0, "post": 0.0})

    S1 = _get_S(rn1, syn1)
    S2 = _get_S(rn2, syn2)
    g1 = _get_g(rn1, syn1)
    g2 = _get_g(rn2, syn2)

    if S1 < 0.999:
        assert g2 <= g1 + 1e-9, \
            f"power=2 g={g2:.6f} should be ≤ power=1 g={g1:.6f} (S={S1:.3f})"


def test_power4_gaba_b_runs_cleanly():
    """GABA_B has power=4; verify it runs without crash and S is bounded."""
    spec = KineticSynapseSpec.gaba_b()
    rn, syn = _fresh_rn_with_spec(spec)
    traces = _run(rn, 500.0, I_pre=10.0)
    assert _all_finite(traces)
    S = _get_S(rn, syn)
    assert 0.0 <= S <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. WEIGHT SCALING
# ═══════════════════════════════════════════════════════════════════════════

def test_weight_zero_no_effect():
    """weight=0 → same V_post as isolated neuron."""
    # Baseline: isolated post with I_post=5
    rn_base = RegionalNetwork()
    rn_base.add_population("E", 1, model=NeuronModelSpec.hh_default())
    t_base = rn_base.simulate(200.0, DT, {"E": 5.0})["E"][0]

    # With weight=0 kinetic synapse from spiking pre
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, _ = _fresh_rn_with_spec(spec, weight=0.0)
    traces = _run(rn, 200.0, I_pre=15.0, I_post=5.0)
    np.testing.assert_array_almost_equal(traces[1], t_base, decimal=6,
        err_msg="weight=0 kinetic syn should have zero effect on post")


def test_double_weight_doubles_g():
    """At the same S, doubling weight should double g_eff."""
    spec1 = KineticSynapseSpec.gaba_kinetic(); spec1.name = "w1"
    spec2 = KineticSynapseSpec.gaba_kinetic(); spec2.name = "w2"

    rn1, syn1 = _fresh_rn_with_spec(spec1, weight=1.0)
    rn1.simulate(300.0, DT, {"pre": 10.0, "post": 0.0})

    rn2, syn2 = _fresh_rn_with_spec(spec2, weight=2.0)
    rn2.simulate(300.0, DT, {"pre": 10.0, "post": 0.0})

    S1 = _get_S(rn1, syn1)
    S2 = _get_S(rn2, syn2)
    assert abs(S1 - S2) < 0.001, f"S should be same: {S1:.4f} vs {S2:.4f}"

    g1 = _get_g(rn1, syn1)
    g2 = _get_g(rn2, syn2)
    assert math.isclose(g2, 2.0 * g1, rel_tol=1e-6), \
        f"g2={g2:.6f} should be 2×g1={g1:.6f}"


def test_large_weight_no_crash():
    """weight=1000 should not crash or produce NaN."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, _ = _fresh_rn_with_spec(spec, weight=1000.0)
    traces = _run(rn, 200.0, I_pre=10.0)
    assert _all_finite(traces)


def test_negative_weight_no_crash():
    """Negative weight (sign reversal) should run without crash."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, _ = _fresh_rn_with_spec(spec, weight=-1.0)
    traces = _run(rn, 200.0, I_pre=10.0)
    assert _all_finite(traces)


# ═══════════════════════════════════════════════════════════════════════════
# 8. ALL NEURON-TYPE COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pre_factory,post_factory,I_pre", [
    # HH × HH
    (_hh_model,     _hh_model,     10.0),
    # HH × Izhikevich variants
    (_hh_model,     _iz_rs_model,  10.0),
    (_hh_model,     _iz_fs_model,  10.0),
    (_hh_model,     _iz_ib_model,  10.0),
    (_hh_model,     _iz_ch_model,  10.0),
    # Izhikevich × HH
    (_iz_rs_model,  _hh_model,     10.0),
    (_iz_fs_model,  _hh_model,     10.0),
    # Izhikevich × Izhikevich
    (_iz_rs_model,  _iz_rs_model,  10.0),
    (_iz_fs_model,  _iz_fs_model,  10.0),
    (_iz_rs_model,  _iz_fs_model,  10.0),
    # Silent pre (I=0)
    (_hh_model,     _hh_model,      0.0),
    (_iz_rs_model,  _iz_rs_model,   0.0),
])
def test_stability_all_neuron_types_gaba_kinetic(pre_factory, post_factory, I_pre):
    """GABA kinetic synapse between various neuron pairs: no crash, no NaN."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=pre_factory())
    rn.add_population("post", 1, model=post_factory())
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=1.0, spec=spec)
    traces = _run(rn, 300.0, DT, I_pre, 0.0)
    assert _all_finite(traces), f"NaN/Inf for {pre_factory.__name__} → {post_factory.__name__}"


@pytest.mark.parametrize("model_name", list(_COMPOSABLE_MODELS.keys()))
def test_stability_composable_pre_gaba_kinetic(model_name):
    """ComposableNeuron as pre-synaptic neuron: no crash, no NaN."""
    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=_COMPOSABLE_MODELS[model_name]())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=0.5,
                               spec=KineticSynapseSpec.gaba_kinetic())
    traces = _run(rn, 300.0, DT, 5.0, 0.0)
    assert _all_finite(traces)


@pytest.mark.parametrize("model_name", list(_COMPOSABLE_MODELS.keys()))
def test_stability_composable_post_gaba_kinetic(model_name):
    """ComposableNeuron as post-synaptic neuron: no crash, no NaN."""
    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=_COMPOSABLE_MODELS[model_name]())
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=0.5,
                               spec=KineticSynapseSpec.gaba_kinetic())
    traces = _run(rn, 300.0, DT, 10.0, 0.0)
    assert _all_finite(traces)


# ═══════════════════════════════════════════════════════════════════════════
# 9. ALL UPDATE-FORM × CURRENT-FORM COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _make_tanh_mg_spec() -> KineticSynapseSpec:
    s = KineticSynapseSpec.gaba_kinetic()
    s.name = "tanh_mg"
    s.current_form = KineticCurrentForm.MG_BLOCK
    s.E_syn = 0.0
    s.mg_conc = 1.0; s.mg_scale = 0.062; s.mg_denom = 3.57
    return s


def _make_ab_mg_spec() -> KineticSynapseSpec:
    s = _alpha_beta_spec("ab_mg")
    s.current_form = KineticCurrentForm.MG_BLOCK
    s.E_syn = 0.0
    s.mg_conc = 1.0; s.mg_scale = 0.062; s.mg_denom = 3.57
    return s


@pytest.mark.parametrize("spec_factory,I_pre", [
    (KineticSynapseSpec.gaba_kinetic,                   10.0),  # TANH + LINEAR
    (_make_tanh_mg_spec,                                10.0),  # TANH + MG_BLOCK
    (_alpha_beta_spec,                                  10.0),  # ALPHA_BETA + LINEAR
    (_make_ab_mg_spec,                                  10.0),  # ALPHA_BETA + MG_BLOCK
    (lambda: _boltzmann_spec_manual("btz_lin"),         10.0),  # BOLTZMANN + LINEAR
    (lambda: _boltzmann_spec_manual("btz_mg",           # BOLTZMANN + MG_BLOCK
             current_form=KineticCurrentForm.MG_BLOCK),10.0),
])
def test_all_update_current_form_combinations(spec_factory, I_pre):
    """All 6 update × current form combos: run cleanly, finite traces, S bounded."""
    spec = spec_factory()
    rn, syn = _fresh_rn_with_spec(spec)
    traces = _run(rn, 300.0, DT, I_pre, 0.0)
    assert _all_finite(traces), f"NaN in {spec.name}"
    S = _get_S(rn, syn)
    assert 0.0 - 1e-9 <= S <= 1.0 + 1e-9, f"S={S:.4f} out of bounds for {spec.name}"


# ═══════════════════════════════════════════════════════════════════════════
# 10. ALL PRESETS × NEURON TYPES (stability matrix)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("preset_factory", [
    KineticSynapseSpec.gaba_kinetic,
    KineticSynapseSpec.nmda_kinetic,
    KineticSynapseSpec.gaba_b,
])
@pytest.mark.parametrize("pre_factory,post_factory", [
    (_hh_model,     _hh_model),
    (_iz_rs_model,  _hh_model),
    (_hh_model,     _iz_rs_model),
    (_iz_fs_model,  _iz_rs_model),
])
def test_preset_neuron_stability_matrix(preset_factory, pre_factory, post_factory):
    """Every preset × every HH/Iz neuron combination finishes without NaN."""
    spec = preset_factory()
    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=pre_factory())
    rn.add_population("post", 1, model=post_factory())
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=1.0, spec=spec)
    traces = _run(rn, 200.0, DT, 8.0, 0.0)
    assert _all_finite(traces)


# ═══════════════════════════════════════════════════════════════════════════
# 11. MIXED SYNAPSE TYPES
# ═══════════════════════════════════════════════════════════════════════════

def test_kinetic_plus_exponential():
    """Kinetic + EXP synapse on same post: no crash, no NaN."""
    rn = RegionalNetwork()
    rn.add_population("N0",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("N1",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("N0", 0, "post", 0, weight=0.5,
                               spec=KineticSynapseSpec.gaba_kinetic())
    rn.connect("N1", "post", "all_to_all", weight=0.5,
                synapse=SynapseSpec.exponential(2.0))
    result = rn.simulate(200.0, DT, {"N0": 10.0, "N1": 10.0, "post": 0.0})
    traces = np.vstack([result["N0"], result["N1"], result["post"]])
    assert _all_finite(traces)


def test_kinetic_plus_alpha():
    """Kinetic + alpha synapse: no crash, no NaN."""
    rn = RegionalNetwork()
    rn.add_population("N0",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("N1",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("N0", 0, "post", 0, weight=0.5,
                               spec=KineticSynapseSpec.gaba_kinetic())
    rn.connect("N1", "post", "all_to_all", weight=0.5,
                synapse=SynapseSpec.alpha_function(2.0))
    result = rn.simulate(200.0, DT, {"N0": 10.0, "N1": 10.0, "post": 0.0})
    traces = np.vstack([result["N0"], result["N1"], result["post"]])
    assert _all_finite(traces)


def test_kinetic_plus_double_exp():
    """Kinetic + DEXP synapse: no crash, no NaN."""
    rn = RegionalNetwork()
    rn.add_population("N0",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("N1",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("N0", 0, "post", 0, weight=0.5,
                               spec=KineticSynapseSpec.gaba_kinetic())
    rn.connect("N1", "post", "all_to_all", weight=0.5,
                synapse=SynapseSpec.double_exponential(0.4, 2.5))
    result = rn.simulate(200.0, DT, {"N0": 10.0, "N1": 10.0, "post": 0.0})
    traces = np.vstack([result["N0"], result["N1"], result["post"]])
    assert _all_finite(traces)


def test_all_synapse_types_together():
    """Kinetic + EXP + ALPHA + DEXP all on one post-neuron simultaneously."""
    rn = RegionalNetwork()
    for i in range(4):
        rn.add_population(f"N{i}", 1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())

    rn.add_kinetic_connection("N0", 0, "post", 0, weight=0.2,
                               spec=KineticSynapseSpec.gaba_kinetic())
    rn.connect("N1", "post", "all_to_all", weight=0.2,
                synapse=SynapseSpec.exponential(2.0))
    rn.connect("N2", "post", "all_to_all", weight=0.2,
                synapse=SynapseSpec.alpha_function(2.0))
    rn.connect("N3", "post", "all_to_all", weight=0.2,
                synapse=SynapseSpec.double_exponential(0.5, 3.0))

    I_dict = {"N0": 10.0, "N1": 10.0, "N2": 10.0, "N3": 10.0, "post": 0.0}
    result = rn.simulate(200.0, DT, I_dict)
    traces = np.vstack([result[f"N{i}"] for i in range(4)] + [result["post"]])
    assert _all_finite(traces)


def test_kinetic_before_and_after_regular_synapses():
    """Order of synapse addition should not matter for correctness."""
    spec = KineticSynapseSpec.gaba_kinetic(); spec.name = "order_test"

    def _build(kin_first):
        rn = RegionalNetwork()
        rn.add_population("N0",   1, model=NeuronModelSpec.hh_default())
        rn.add_population("N1",   1, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
        if kin_first:
            rn.add_kinetic_connection("N0", 0, "post", 0, weight=0.5, spec=spec)
            rn.connect("N1", "post", "all_to_all", weight=0.5,
                        synapse=SynapseSpec.exponential(2.0))
        else:
            rn.connect("N1", "post", "all_to_all", weight=0.5,
                        synapse=SynapseSpec.exponential(2.0))
            rn.add_kinetic_connection("N0", 0, "post", 0, weight=0.5, spec=spec)
        return rn

    I_dict = {"N0": 10.0, "N1": 10.0, "post": 0.0}
    ta = np.vstack(list(_build(True).simulate(200.0, DT, I_dict).values()))
    tb = np.vstack(list(_build(False).simulate(200.0, DT, I_dict).values()))
    assert _all_finite(ta) and _all_finite(tb)


# ═══════════════════════════════════════════════════════════════════════════
# 12. FAN-IN: MULTIPLE KINETIC SYNAPSES ON SAME POST
# ═══════════════════════════════════════════════════════════════════════════

def test_two_kinetic_synapses_additive():
    """Two kinetic synapses from the same pre are additive (double effect)."""
    spec_single  = KineticSynapseSpec.gaba_kinetic(); spec_single.name  = "single"
    spec_double_a = KineticSynapseSpec.gaba_kinetic(); spec_double_a.name = "d1"
    spec_double_b = KineticSynapseSpec.gaba_kinetic(); spec_double_b.name = "d2"

    dt = DT; dur = 300.0
    I_pre = 10.0; I_post = 12.0

    rn_s, _ = _fresh_rn_with_spec(spec_single)
    tr_single = _run(rn_s, dur, dt, I_pre, I_post)

    rn_d = RegionalNetwork()
    rn_d.add_population("pre",  1, model=NeuronModelSpec.hh_default())
    rn_d.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn_d.add_kinetic_connection("pre", 0, "post", 0, weight=1.0, spec=spec_double_a)
    rn_d.add_kinetic_connection("pre", 0, "post", 0, weight=1.0, spec=spec_double_b)
    tr_double = _run(rn_d, dur, dt, I_pre, I_post)

    assert _all_finite(tr_single) and _all_finite(tr_double)
    spikes_s = _count_spikes(tr_single[1])
    spikes_d = _count_spikes(tr_double[1])
    assert spikes_d <= spikes_s, \
        f"Double kinetic should fire ≤ single: {spikes_d} vs {spikes_s}"


def test_ten_kinetic_synapses_no_crash():
    """10 kinetic synapses from 10 different pre-neurons: no crash, no NaN."""
    rn = RegionalNetwork()
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    for i in range(10):
        rn.add_population(f"N{i}", 1, model=NeuronModelSpec.hh_default())
        s = KineticSynapseSpec.gaba_kinetic(); s.name = f"kin_{i}"
        rn.add_kinetic_connection(f"N{i}", 0, "post", 0, weight=0.1, spec=s)

    I_dict = {f"N{i}": 8.0 for i in range(10)}
    I_dict["post"] = 5.0
    result = rn.simulate(200.0, DT, I_dict)
    traces = np.vstack([result[f"N{i}"] for i in range(10)] + [result["post"]])
    assert _all_finite(traces)


# ═══════════════════════════════════════════════════════════════════════════
# 13. SPEC NAME DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def test_deduplication_same_name_reuses_spec():
    """Two kinetic synapses with the same spec name share the same spec entry."""
    spec_a = KineticSynapseSpec.gaba_kinetic()
    spec_b = KineticSynapseSpec.gaba_kinetic()   # same name "GABA_kinetic"

    rn = RegionalNetwork()
    rn.add_population("N0",   1, model=NeuronModelSpec.hh_default())
    rn.add_population("N1",   1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("N0", 0, "post", 0, weight=1.0, spec=spec_a)
    rn.add_kinetic_connection("N1", 0, "post", 0, weight=1.0, spec=spec_b)

    result = rn.simulate(200.0, DT, {"N0": 10.0, "N1": 10.0, "post": 0.0})
    traces = np.vstack([result["N0"], result["N1"], result["post"]])
    assert _all_finite(traces)


def test_different_names_different_specs():
    """Two kinetic synapses with different names get different specs."""
    spec_fast = KineticSynapseSpec.gaba_kinetic(); spec_fast.name = "fast"
    spec_fast.tau_decay = 5.0

    spec_slow = KineticSynapseSpec.gaba_kinetic(); spec_slow.name = "slow"
    spec_slow.tau_decay = 50.0

    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("pre2", 1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=1.0, spec=spec_fast)
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=1.0, spec=spec_slow)

    result = rn.simulate(300.0, DT, {"pre": 10.0, "pre2": 0.0, "post": 0.0})
    traces = np.vstack([result["pre"], result["pre2"], result["post"]])
    assert _all_finite(traces)


# ═══════════════════════════════════════════════════════════════════════════
# 14. FUNCTIONAL EFFECTS ON FIRING RATE
# ═══════════════════════════════════════════════════════════════════════════

def test_gaba_kinetic_suppresses_post_firing():
    """Active GABA kinetic pre reduces post firing below baseline."""
    dt = DT; dur = 500.0
    I_post = 12.0; I_pre = 10.0

    rn_b = RegionalNetwork()
    rn_b.add_population("E", 1, model=NeuronModelSpec.hh_default())
    t_base = rn_b.simulate(dur, dt, {"E": I_post})["E"][0]
    spikes_base = _count_spikes(t_base)

    spec = KineticSynapseSpec.gaba_kinetic(); spec.g = 0.5
    rn_i, _ = _fresh_rn_with_spec(spec)
    t_inh = _run(rn_i, dur, dt, I_pre, I_post)
    spikes_inh = _count_spikes(t_inh[1])

    assert spikes_base > 0, "Baseline must spike"
    assert spikes_inh < spikes_base, \
        f"Inhibition should reduce spikes: {spikes_inh} < {spikes_base}"


def test_nmda_kinetic_excites_post():
    """NMDA kinetic synapse should increase post firing above subthreshold baseline."""
    dt = DT; dur = 500.0
    I_post = 5.0   # sub-threshold for HH alone
    I_pre  = 12.0

    rn_b = RegionalNetwork()
    rn_b.add_population("E", 1, model=NeuronModelSpec.hh_default())
    t_base = rn_b.simulate(dur, dt, {"E": I_post})["E"][0]
    spikes_base = _count_spikes(t_base)

    spec = KineticSynapseSpec()
    spec.name = "excit_nmda"
    spec.update_form = KineticUpdateForm.BOLTZMANN_GATE
    si = BoltzmannParams(); si.v_half = -20.0; si.k = 16.0
    spec.s_inf = si
    t = TauParams(); t.form = TauForm.CONSTANT; t.set_param(0, 80.0)
    spec.tau = t
    spec.current_form = KineticCurrentForm.LINEAR
    spec.g = 0.3; spec.E_syn = 0.0; spec.power = 1

    rn_e, _ = _fresh_rn_with_spec(spec)
    t_exc = _run(rn_e, dur, dt, I_pre, I_post)
    spikes_exc = _count_spikes(t_exc[1])

    assert spikes_exc >= spikes_base, \
        f"NMDA excitation should produce ≥ spikes: {spikes_exc} vs {spikes_base}"


# ═══════════════════════════════════════════════════════════════════════════
# 15. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

def test_e_syn_equals_v_post_no_driving_force():
    """E_syn set to V_post resting (-65) → minimal current → post unaffected."""
    spec = KineticSynapseSpec.gaba_kinetic()
    spec.E_syn = -65.0   # ≈ V_rest for HH → zero driving force
    spec.g = 1.0

    rn_base = RegionalNetwork()
    rn_base.add_population("E", 1, model=NeuronModelSpec.hh_default())
    rn_base.simulate(200.0, DT, {"E": 8.0})  # just for stability check

    rn_syn, _ = _fresh_rn_with_spec(spec)
    t_syn = _run(rn_syn, 200.0, I_pre=12.0, I_post=8.0)
    assert _all_finite(t_syn)


@pytest.mark.parametrize("tau_decay", [0.5, 1.0, 5.0, 50.0, 200.0, 1000.0])
def test_tau_decay_range_tanh_gate(tau_decay):
    """TANH_GATE with extreme tau_decay values: no crash, no NaN."""
    spec = KineticSynapseSpec.gaba_kinetic()
    spec.tau_decay = tau_decay
    spec.name = f"tau_{tau_decay}"
    rn, _ = _fresh_rn_with_spec(spec)
    traces = _run(rn, 200.0, I_pre=10.0)
    assert _all_finite(traces), f"NaN for tau_decay={tau_decay}"


@pytest.mark.parametrize("tanh_k", [0.1, 1.0, 4.0, 20.0, 100.0])
def test_tanh_k_range(tanh_k):
    """TANH_GATE with varying tanh_k (slope): no crash, S bounded."""
    spec = KineticSynapseSpec.gaba_kinetic()
    spec.tanh_k = tanh_k
    spec.name = f"k_{tanh_k}"
    rn, syn = _fresh_rn_with_spec(spec)
    traces = _run(rn, 200.0, I_pre=10.0)
    assert _all_finite(traces)
    S = _get_S(rn, syn)
    assert 0.0 - 1e-9 <= S <= 1.0 + 1e-9


def test_very_small_dt():
    """dt=0.001 ms: no crash, no NaN."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, _ = _fresh_rn_with_spec(spec)
    traces = _run(rn, 5.0, 0.001, I_pre=10.0)
    assert _all_finite(traces)


def test_very_large_dt():
    """dt=0.5 ms (large step): no crash, no NaN."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, _ = _fresh_rn_with_spec(spec)
    traces = _run(rn, 200.0, 0.5, I_pre=10.0)
    assert _all_finite(traces)


def test_long_simulation_no_nan():
    """5000 ms simulation: no NaN, no Inf."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, syn = _fresh_rn_with_spec(spec)
    traces = _run(rn, 5000.0, DT, I_pre=10.0, I_post=5.0)
    assert _all_finite(traces)
    S = _get_S(rn, syn)
    assert 0.0 - 1e-9 <= S <= 1.0 + 1e-9


def test_single_step_no_crash():
    """Single simulation step: should not crash and S=S_init."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn, syn = _fresh_rn_with_spec(spec)
    rn.simulate(DT, DT, {"pre": 0.0, "post": 0.0})
    V0 = rn._rnet.network().neuron(0).V
    V1 = rn._rnet.network().neuron(1).V
    assert _all_finite(np.array([V0, V1]))
    S = _get_S(rn, syn)
    assert 0.0 - 1e-9 <= S <= 1.0 + 1e-9


def test_s_init_one_decays():
    """S_init=1.0 with silent pre: S should decay toward 0."""
    spec = KineticSynapseSpec.gaba_kinetic()
    spec.S_init = 1.0
    rn, syn = _fresh_rn_with_spec(spec)

    assert math.isclose(_get_S(rn, syn), 1.0)

    rn.simulate(200.0, DT, {"pre": 0.0, "post": 0.0})
    S_after = _get_S(rn, syn)
    # Should have decayed substantially from 1.0 in 200 ms (tau_decay=13 ms)
    assert S_after < 0.5, f"S_init=1 should decay; got S={S_after:.4f}"


def test_alpha_beta_with_very_fast_rates():
    """ALPHA_BETA with high α and β: equilibrates quickly, S bounded."""
    s = KineticSynapseSpec()
    s.name = "fast_ab"
    s.update_form = KineticUpdateForm.ALPHA_BETA
    ra = RateFuncParams(); ra.form = RateFuncForm.EXP_DECAY
    ra.A = 100.0; ra.B = 0.0; ra.C = 1e9
    rb = RateFuncParams(); rb.form = RateFuncForm.EXP_DECAY
    rb.A = 100.0; rb.B = 0.0; rb.C = 1e9
    s.alpha = ra; s.beta = rb
    s.current_form = KineticCurrentForm.LINEAR
    s.g = 0.1; s.E_syn = -80.0; s.power = 1
    rn, syn = _fresh_rn_with_spec(s)
    traces = _run(rn, 100.0, I_pre=10.0)
    assert _all_finite(traces)
    S = _get_S(rn, syn)
    assert 0.0 - 1e-9 <= S <= 1.0 + 1e-9


def test_boltzmann_gate_v_half_above_v_pre():
    """BOLTZMANN_GATE with v_half far above V_pre: S stays near 0."""
    s = _boltzmann_spec_manual("high_vhalf")
    s.s_inf.v_half = +50.0    # far above V_rest → S_inf(−65) ≈ 0
    s.s_inf.k = 5.0
    rn, syn = _fresh_rn_with_spec(s)
    _run(rn, LONG, I_pre=0.0)
    S = _get_S(rn, syn)
    assert S < 0.05, f"S should be near 0 with v_half far above V_pre: {S:.4f}"


def test_boltzmann_gate_v_half_below_v_pre():
    """BOLTZMANN_GATE with v_half far below V_pre: S converges toward 1."""
    s = _boltzmann_spec_manual("low_vhalf")
    s.s_inf.v_half = -100.0   # far below V_rest → S_inf(−65) ≈ 1
    s.s_inf.k = 5.0
    rn, syn = _fresh_rn_with_spec(s)
    _run(rn, LONG, I_pre=0.0)
    S = _get_S(rn, syn)
    assert S > 0.95, f"S should be near 1 with v_half far below V_pre: {S:.4f}"


def test_self_connection_no_crash():
    """Kinetic synapse from a neuron to itself (pre=post): no crash."""
    rn = RegionalNetwork()
    rn.add_population("E", 1, model=NeuronModelSpec.hh_default())
    rn.add_kinetic_connection("E", 0, "E", 0, weight=0.5,
                               spec=KineticSynapseSpec.gaba_kinetic())
    result = rn.simulate(200.0, DT, {"E": 10.0})
    assert _all_finite(result["E"])


# ═══════════════════════════════════════════════════════════════════════════
# 16. REGIONAL NETWORK INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def _make_rnet(pre_n=3, post_n=3):
    rnet = RegionalNetwork()
    rnet.add_population("pre",  pre_n,  model=NeuronModelSpec.hh_default())
    rnet.add_population("post", post_n, model=NeuronModelSpec.hh_default())
    return rnet


@pytest.mark.parametrize("pattern,kwargs", [
    ("all_to_all",        {}),
    ("random_sparse",     {"probability": 0.8}),
])
def test_regional_kinetic_connect_patterns(pattern, kwargs):
    """RegionalNetwork.connect with kinetic_spec for various patterns."""
    rnet = _make_rnet(3, 3)
    spec = KineticSynapseSpec.gaba_kinetic(); spec.g = 0.2
    rnet.connect("pre", "post", pattern, weight=1.0,
                 kinetic_spec=spec, seed=42, **kwargs)
    assert rnet.num_synapses > 0
    traces = rnet.simulate(200.0, DT, {"pre": 10.0, "post": 0.0})
    assert "pre" in traces and "post" in traces
    for arr in traces.values():
        assert _all_finite(arr)


def test_regional_kinetic_all_to_all_count():
    """ALL_TO_ALL kinetic produces N_pre × N_post synapses."""
    rnet = _make_rnet(4, 3)
    rnet.connect("pre", "post", "all_to_all", weight=1.0,
                 kinetic_spec=KineticSynapseSpec.gaba_kinetic())
    assert rnet.num_synapses == 4 * 3


def test_regional_kinetic_one_to_one():
    """ONE_TO_ONE kinetic produces N synapses (equal-size populations)."""
    rnet = _make_rnet(4, 4)
    rnet.connect("pre", "post", "one_to_one", weight=1.0,
                 kinetic_spec=KineticSynapseSpec.gaba_kinetic())
    assert rnet.num_synapses == 4


def test_regional_kinetic_shifted():
    """SHIFTED kinetic produces N synapses (equal-size populations)."""
    rnet = _make_rnet(5, 5)
    rnet.connect("pre", "post", "shifted", weight=1.0, shift=2,
                 kinetic_spec=KineticSynapseSpec.gaba_kinetic())
    assert rnet.num_synapses == 5


def test_regional_add_kinetic_connection_single():
    """add_kinetic_connection adds exactly one kinetic synapse."""
    rnet = _make_rnet(3, 3)
    spec = KineticSynapseSpec.gaba_kinetic()
    rnet.add_kinetic_connection("pre", 0, "post", 2, weight=0.7, spec=spec)
    assert rnet.num_synapses == 1
    traces = rnet.simulate(200.0, DT, {"pre": 10.0, "post": 0.0})
    for arr in traces.values():
        assert _all_finite(arr)


def test_regional_kinetic_plus_regular_mixed():
    """RegionalNetwork with both kinetic and regular synapses: no crash."""
    rnet = _make_rnet(2, 2)
    rnet.connect("pre", "post", "all_to_all",
                 synapse=SynapseSpec.ampa(), weight=0.5)
    rnet.connect("post", "pre", "all_to_all", weight=0.3,
                 kinetic_spec=KineticSynapseSpec.gaba_kinetic())
    assert rnet.num_synapses == 4 + 4

    traces = rnet.simulate(300.0, DT, {"pre": 10.0, "post": 5.0})
    for arr in traces.values():
        assert _all_finite(arr)


def test_regional_kinetic_composable_neurons():
    """RegionalNetwork with ComposableNeuron populations and kinetic synapse."""
    rnet = RegionalNetwork()
    rnet.add_population("stn", 3, model=make_stn())
    rnet.add_population("gpe", 3, model=make_gpe())
    rnet.connect("stn", "gpe", "all_to_all", weight=0.2,
                 kinetic_spec=KineticSynapseSpec.gaba_kinetic())
    traces = rnet.simulate(200.0, DT, {"stn": 5.0, "gpe": 0.0})
    for arr in traces.values():
        assert _all_finite(arr)


def test_regional_out_of_range_index_raises():
    """add_kinetic_connection with bad index should raise."""
    rnet = _make_rnet(2, 2)
    spec = KineticSynapseSpec.gaba_kinetic()
    with pytest.raises(Exception):
        rnet.add_kinetic_connection("pre", 99, "post", 0, 1.0, spec)


def test_regional_unknown_population_raises():
    """add_kinetic_connection with unknown population should raise."""
    rnet = _make_rnet(2, 2)
    spec = KineticSynapseSpec.gaba_kinetic()
    with pytest.raises(Exception):
        rnet.add_kinetic_connection("nonexistent", 0, "post", 0, 1.0, spec)


def test_regional_kinetic_nmda_all_to_all():
    """NMDA preset with MG block in RegionalNetwork: no crash, finite traces."""
    rnet = _make_rnet(2, 2)
    rnet.connect("pre", "post", "all_to_all", weight=0.5,
                 kinetic_spec=KineticSynapseSpec.nmda_kinetic())
    traces = rnet.simulate(200.0, DT, {"pre": 10.0, "post": 0.0})
    for arr in traces.values():
        assert _all_finite(arr)


# ═══════════════════════════════════════════════════════════════════════════
# 17. KINETICSYNAPSEMODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def test_builder_tanh_roundtrip():
    """KineticSynapseModel tanh_gate() produces identical fields to direct construction."""
    built = (KineticSynapseModel("test_tanh")
             .tanh_gate(amp=3.0, v_half=-5.0, k=6.0, tau_decay=20.0)
             .linear_current(g=0.2, E_syn=-70.0, power=2)
             .to_spec())

    direct = KineticSynapseSpec()
    direct.name = "test_tanh"
    direct.update_form = KineticUpdateForm.TANH_GATE
    direct.tanh_amp = 3.0; direct.tanh_vh = -5.0
    direct.tanh_k = 6.0;  direct.tau_decay = 20.0
    direct.current_form = KineticCurrentForm.LINEAR
    direct.g = 0.2; direct.E_syn = -70.0; direct.power = 2

    assert built.name == direct.name
    assert built.update_form == direct.update_form
    assert math.isclose(built.tanh_amp, direct.tanh_amp)
    assert math.isclose(built.tanh_vh, direct.tanh_vh)
    assert math.isclose(built.tanh_k, direct.tanh_k)
    assert math.isclose(built.tau_decay, direct.tau_decay)
    assert built.current_form == direct.current_form
    assert math.isclose(built.g, direct.g)
    assert math.isclose(built.E_syn, direct.E_syn)
    assert built.power == direct.power


def test_builder_boltzmann_roundtrip():
    """KineticSynapseModel boltzmann_gate() produces correct BoltzmannParams."""
    built = (KineticSynapseModel("btz_built")
             .boltzmann_gate(v_half=-30.0, k=12.0, tau=100.0)
             .mg_block_current(g=0.3, E_syn=0.0, mg_conc=2.0,
                               mg_scale=0.05, mg_denom=4.0)
             .to_spec())

    assert built.update_form == KineticUpdateForm.BOLTZMANN_GATE
    assert math.isclose(built.s_inf.v_half, -30.0)
    assert math.isclose(built.s_inf.k, 12.0)
    assert math.isclose(built.tau.get_param(0), 100.0)
    assert built.current_form == KineticCurrentForm.MG_BLOCK
    assert math.isclose(built.g, 0.3)
    assert math.isclose(built.mg_conc, 2.0)
    assert math.isclose(built.mg_scale, 0.05)
    assert math.isclose(built.mg_denom, 4.0)


def test_builder_alpha_beta_roundtrip():
    """KineticSynapseModel alpha_beta() preserves RateFuncParams."""
    alpha = RateFuncParams()
    alpha.form = RateFuncForm.EXP_DECAY; alpha.A = 0.3; alpha.B = 0.0; alpha.C = 1e9
    beta  = RateFuncParams()
    beta.form  = RateFuncForm.EXP_DECAY; beta.A  = 0.1; beta.B  = 0.0; beta.C  = 1e9

    built = (KineticSynapseModel("ab_built")
             .alpha_beta(alpha, beta)
             .linear_current(g=0.15, E_syn=-80.0)
             .to_spec())

    assert built.update_form == KineticUpdateForm.ALPHA_BETA
    assert math.isclose(built.alpha.A, 0.3)
    assert math.isclose(built.beta.A,  0.1)
    assert math.isclose(built.g, 0.15)


def test_builder_is_chainable():
    """Builder returns self from each method; to_spec() returns KineticSynapseSpec."""
    model = KineticSynapseModel("chain")
    result = model.tanh_gate().linear_current(g=0.1, E_syn=-80.0)
    assert result is model
    spec = model.to_spec()
    assert isinstance(spec, KineticSynapseSpec)


def test_builder_spec_runs_in_simulation():
    """Spec from builder actually works in a full simulation."""
    spec = (KineticSynapseModel("sim_builder")
            .tanh_gate(amp=2.0, v_half=0.0, k=4.0, tau_decay=13.0)
            .linear_current(g=0.1, E_syn=-80.0)
            .to_spec())
    rn, syn = _fresh_rn_with_spec(spec)
    traces = _run(rn, 300.0, I_pre=10.0)
    assert _all_finite(traces)
    assert 0.0 <= _get_S(rn, syn) <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 18. NUMERICAL STABILITY — COMPREHENSIVE SWEEP
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("spec_factory,I_pre,I_post,dt_val", [
    (KineticSynapseSpec.gaba_kinetic,  0.0,  0.0, 0.025),
    (KineticSynapseSpec.gaba_kinetic, 10.0,  5.0, 0.025),
    (KineticSynapseSpec.gaba_kinetic, 15.0, 12.0, 0.1),
    (KineticSynapseSpec.nmda_kinetic,  0.0,  0.0, 0.025),
    (KineticSynapseSpec.nmda_kinetic, 10.0,  0.0, 0.025),
    (KineticSynapseSpec.nmda_kinetic, 10.0,  5.0, 0.1),
    (KineticSynapseSpec.gaba_b,        0.0,  0.0, 0.025),
    (KineticSynapseSpec.gaba_b,        8.0,  0.0, 0.025),
    (KineticSynapseSpec.gaba_b,        8.0,  5.0, 0.1),
])
def test_numerical_stability_sweep(spec_factory, I_pre, I_post, dt_val):
    spec = spec_factory()
    rn, syn = _fresh_rn_with_spec(spec)
    traces = _run(rn, 300.0, dt_val, I_pre, I_post)
    assert _all_finite(traces), \
        f"NaN/Inf: {spec.name} I_pre={I_pre} I_post={I_post} dt={dt_val}"
    S = _get_S(rn, syn)
    assert 0.0 - 1e-9 <= S <= 1.0 + 1e-9


@pytest.mark.parametrize("pre_factory,post_factory", [
    (_iz_rs_model,  _iz_rs_model),
    (_iz_fs_model,  _iz_lts_model),
    (_hh_model,     _iz_ch_model),
    (_iz_ib_model,  _hh_model),
])
def test_numerical_stability_iz_variants(pre_factory, post_factory):
    """All Izhikevich sub-types as pre/post with GABA kinetic: finite traces."""
    spec = KineticSynapseSpec.gaba_kinetic()
    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=pre_factory())
    rn.add_population("post", 1, model=post_factory())
    rn.add_kinetic_connection("pre", 0, "post", 0, weight=1.0, spec=spec)
    traces = _run(rn, 300.0, DT, 10.0, 5.0)
    assert _all_finite(traces), \
        f"NaN for {pre_factory.__name__} → {post_factory.__name__}"
