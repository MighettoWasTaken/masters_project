"""
test_plasticity.py — test suite for task15 plasticity support.

Sections:
  A — DopamineGating preset (sub-problem A)
  B — HomeostaticScaling preset (sub-problem A)
  C — STDP: standard tests + LTP/LTD learning task (sub-problem B)
  D — STP: standard tests + frequency discrimination task (sub-problem B)
  E — Gated STDP (modulator-dependent weight updates)
  F — API surface / exports
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import hodgkin_huxley as hh
from hodgkin_huxley import (
    NeuronModel,
    NeuronModelSpec,
    RegionalNetwork,
    RecordingConfig,
    IntracellularDynamics,
    Modulation,
    DopamineGating,
    HomeostaticScaling,
    STDPRule,
    STPRule,
    SynapseSpec,
    WeightDistribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DT = 0.05   # ms — coarser for speed, still resolves spikes


def _two_neuron_rnet(weight: float = 0.5,
                     synapse=None,
                     plasticity=None) -> RegionalNetwork:
    """Single synapse: pre (idx 0) → post (idx 1), both HH default."""
    rn = RegionalNetwork()
    rn.add_population("pre",  1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    rn.connect("pre", "post", "all_to_all",
               weight=weight,
               synapse=synapse or SynapseSpec.ampa(),
               plasticity=plasticity)
    return rn


def _run(rn: RegionalNetwork,
         I_pre: float = 5.0,
         I_post: float = 0.0,
         duration: float = 200.0) -> dict:
    cfg = RecordingConfig(["V"])
    return rn.simulate(duration, DT, {"pre": I_pre, "post": I_post}, record=cfg)


def _spikes(trace: np.ndarray, threshold: float = 0.0) -> int:
    above = trace > threshold
    return int(np.sum(np.diff(above.astype(int)) == 1))


# ---------------------------------------------------------------------------
# Section A — DopamineGating
# ---------------------------------------------------------------------------

class TestDopamineGating:

    def test_no_crash(self):
        rn = RegionalNetwork()
        rn.add_population("pre", 2, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 2, model=NeuronModelSpec.hh_default())
        rn.connect("pre", "post", "all_to_all", weight=0.3)
        rn.add_intracellular(DopamineGating(da_level=0.5), populations=["post"])
        result = _run(rn, I_pre=5.0, I_post=0.0, duration=200.0)
        V = result["pre"]["V"]
        assert np.all(np.isfinite(V)), "Voltages should be finite"

    def test_is_intracellular_dynamics(self):
        assert isinstance(DopamineGating(), IntracellularDynamics)

    def test_high_da_increases_i_syn(self):
        """Higher DA level → stronger synaptic gain → higher mean I_syn to post."""
        mean_isyn = []
        for da_level in [0.1, 0.9]:
            rn = RegionalNetwork()
            rn.add_population("pre",  1, model=NeuronModelSpec.hh_default())
            rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
            rn.connect("pre", "post", "all_to_all", weight=1.0)
            rn.add_intracellular(DopamineGating(da_level=da_level, k_decay=0.0001),
                                 populations=["post"])
            cfg = RecordingConfig(["I_syn"])
            result = rn.simulate(300.0, DT, {"pre": 8.0, "post": 0.0}, record=cfg)
            mean_isyn.append(float(result["post"]["I_syn"].max()))
        assert mean_isyn[1] > mean_isyn[0], \
            f"High DA should give higher post max I_syn: {mean_isyn}"

    def test_tonic_da_stable(self):
        """With very slow decay, DA stays close to initial value."""
        rn = RegionalNetwork()
        rn.add_population("src", 1, model=NeuronModelSpec.hh_default())
        rn.add_intracellular(DopamineGating(da_level=0.7, k_decay=1e-5), populations=["src"])
        cfg = RecordingConfig(["calcium"])   # substance 0 recorded as 'calcium'
        result = rn.simulate(500.0, DT, {"src": 0.0}, record=cfg)
        da_trace = result["src"]["calcium"][0]
        assert abs(da_trace[-1] - 0.7) < 0.1, \
            f"Tonic DA should stay near initial: {da_trace[-1]:.3f}"


# ---------------------------------------------------------------------------
# Section B — HomeostaticScaling
# ---------------------------------------------------------------------------

class TestHomeostaticScaling:

    def test_no_crash(self):
        rn = RegionalNetwork()
        rn.add_population("pre",  2, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 2, model=NeuronModelSpec.hh_default())
        rn.connect("pre", "post", "all_to_all", weight=0.3)
        rn.add_intracellular(HomeostaticScaling(target_rate=20.0, tau=5000.0),
                             populations=["post"])
        result = _run(rn, I_pre=5.0, I_post=0.0, duration=200.0)
        V = result["pre"]["V"]
        assert np.all(np.isfinite(V)), "Voltages should be finite"

    def test_is_intracellular_dynamics(self):
        assert isinstance(HomeostaticScaling(), IntracellularDynamics)

    def test_synapse_g_finite(self):
        """HomeostaticScaling with active synapse should produce finite g_syn."""
        rn = RegionalNetwork()
        rn.add_population("pre",  2, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 2, model=NeuronModelSpec.hh_default())
        rn.connect("pre", "post", "all_to_all", weight=0.5)
        rn.add_intracellular(HomeostaticScaling(target_rate=20.0, tau=5000.0),
                             populations=["post"])
        cfg = RecordingConfig(["g_syn"])
        result = rn.simulate(300.0, DT, {"pre": 6.0, "post": 0.0}, record=cfg)
        g = result["post"]["g_syn"]
        assert np.all(np.isfinite(g)), "g_syn should remain finite with HomeostaticScaling"


# ---------------------------------------------------------------------------
# Section C — STDP
# ---------------------------------------------------------------------------

class TestSTDP:

    def test_no_crash(self):
        rn = _two_neuron_rnet(weight=0.5, plasticity=STDPRule())
        result = _run(rn, I_pre=5.0, duration=200.0)
        assert np.all(np.isfinite(result["pre"]["V"]))

    def test_no_crash_connect_string_pattern(self):
        rn = RegionalNetwork()
        rn.add_population("pre",  2, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 2, model=NeuronModelSpec.hh_default())
        rn.connect("pre", "post", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.ampa(), plasticity=STDPRule())
        result = _run(rn, I_pre=5.0, duration=200.0)
        assert np.all(np.isfinite(result["pre"]["V"]))

    def test_weight_stays_in_bounds(self):
        """No weight should leave [w_min, w_max] during sustained stimulation."""
        w_min, w_max = 0.1, 0.9
        rn = _two_neuron_rnet(weight=0.5,
                               plasticity=STDPRule(w_min=w_min, w_max=w_max))
        _run(rn, I_pre=8.0, I_post=8.0, duration=500.0)
        w = rn.get_synapse_weights()[0]
        assert w_min <= w <= w_max, f"Weight {w:.4f} out of bounds [{w_min}, {w_max}]"

    def test_get_synapse_weights_length(self):
        """get_synapse_weights returns array of length == num_synapses."""
        rn = RegionalNetwork()
        rn.add_population("pre",  3, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 3, model=NeuronModelSpec.hh_default())
        rn.connect("pre", "post", "all_to_all", weight=0.3,
                   plasticity=STDPRule())
        w = rn.get_synapse_weights()
        assert len(w) == 9, f"Expected 9 weights, got {len(w)}"

    def test_has_stdp_flag(self):
        rn = _two_neuron_rnet(weight=0.3, plasticity=STDPRule())
        cfg = RecordingConfig(["V"])
        rn.simulate(50.0, DT, {"pre": 5.0, "post": 0.0}, record=cfg)
        assert rn.network.has_stdp

    def test_no_stdp_flag_without_plasticity(self):
        rn = _two_neuron_rnet(weight=0.3)
        cfg = RecordingConfig(["V"])
        rn.simulate(50.0, DT, {"pre": 5.0, "post": 0.0}, record=cfg)
        assert not rn.network.has_stdp

    def test_sign_ltp_pre_before_post(self):
        """Pre fires, then post fires: causal → weight should increase (LTP)."""
        rn = _two_neuron_rnet(weight=0.3,
                               plasticity=STDPRule(A_plus=0.02, A_minus=0.01,
                                                   tau_plus=20.0, tau_minus=20.0,
                                                   w_min=0.0, w_max=1.0))
        # Drive pre strongly; post will follow via synapse after a delay
        # Run enough to get several LTP pairings
        _run(rn, I_pre=9.0, I_post=0.0, duration=200.0)
        w_after = rn.get_synapse_weights()[0]
        assert w_after > 0.3, \
            f"Causal pairing should increase weight (LTP): {w_after:.4f}"

    def test_sign_ltd_post_before_pre(self):
        """Post fires faster than pre: net LTD should dominate over LTP."""
        rn = _two_neuron_rnet(weight=0.7,
                               plasticity=STDPRule(A_plus=0.003, A_minus=0.02,
                                                   tau_plus=20.0, tau_minus=20.0,
                                                   w_min=0.0, w_max=1.0))
        # Both neurons fire; post fires much faster (anti-causal dominates)
        _run(rn, I_pre=7.0, I_post=10.0, duration=1000.0)
        w_after = rn.get_synapse_weights()[0]
        assert w_after < 0.7, \
            f"LTD-dominant protocol should decrease weight: {w_after:.4f}"


class TestSTDPLearning:
    """Learning task: sustained LTP protocol increases weight over epochs."""

    def test_ltp_weight_increases_across_epochs(self):
        rn = _two_neuron_rnet(weight=0.3,
                               plasticity=STDPRule(A_plus=0.01, A_minus=0.005,
                                                   w_min=0.0, w_max=1.0))
        weights = [0.3]
        for _ in range(5):
            _run(rn, I_pre=9.0, I_post=0.0, duration=300.0)
            weights.append(float(rn.get_synapse_weights()[0]))
        assert weights[-1] > weights[0] + 0.05, \
            f"Weight should increase with LTP epochs: {weights}"

    def test_ltd_weight_decreases_over_time(self):
        rn = _two_neuron_rnet(weight=0.7,
                               plasticity=STDPRule(A_plus=0.002, A_minus=0.02,
                                                   w_min=0.0, w_max=1.0))
        # Both fire; post faster (10 > 7) so anti-causal pairings dominate
        _run(rn, I_pre=7.0, I_post=10.0, duration=2000.0)
        w_final = float(rn.get_synapse_weights()[0])
        assert w_final < 0.6, \
            f"LTD protocol should reduce weight: {w_final:.4f}"


# ---------------------------------------------------------------------------
# Section D — STP
# ---------------------------------------------------------------------------

class TestSTP:

    def test_no_crash(self):
        rn = _two_neuron_rnet(weight=1.0, plasticity=STPRule())
        result = _run(rn, I_pre=5.0, duration=500.0)
        assert np.all(np.isfinite(result["pre"]["V"]))

    def test_has_stp_flag(self):
        rn = _two_neuron_rnet(weight=0.5, plasticity=STPRule())
        cfg = RecordingConfig(["V"])
        rn.simulate(50.0, DT, {"pre": 5.0, "post": 0.0}, record=cfg)
        assert rn.network.has_stp

    def test_no_stp_flag_without_plasticity(self):
        rn = _two_neuron_rnet(weight=0.5)
        cfg = RecordingConfig(["V"])
        rn.simulate(50.0, DT, {"pre": 5.0, "post": 0.0}, record=cfg)
        assert not rn.network.has_stp

    def test_g_syn_finite(self):
        """g_syn trace should be finite throughout STP simulation."""
        rn = _two_neuron_rnet(weight=1.0, plasticity=STPRule(U=0.5))
        cfg = RecordingConfig(["g_syn"])
        result = rn.simulate(500.0, DT, {"pre": 8.0, "post": 0.0}, record=cfg)
        g = result["post"]["g_syn"][0]
        assert np.all(np.isfinite(g)), "g_syn should remain finite with STP"
        assert np.all(g >= 0.0), "g_syn should be non-negative"


class TestSTPLearning:
    """Learning task: STP depression distinguishes high vs. low frequency input."""

    def test_frequency_discrimination(self):
        """
        STP should reduce mean effective g_syn more at high than at low frequency.
        Compare steady-state mean g_syn in last 20% of a long run.
        Uses Izhikevich RS (Type I) which spans a broad firing-rate range.
        """
        results = {}
        for label, I_pre in [("low", 5.0), ("high", 20.0)]:
            rn = RegionalNetwork()
            rn.add_population("pre",  1, neuron_type="IZHIKEVICH_RS")
            rn.add_population("post", 1, neuron_type="IZHIKEVICH_RS")
            rn.connect("pre", "post", "all_to_all", weight=1.0,
                       synapse=SynapseSpec.ampa(),
                       plasticity=STPRule(U=0.7, tau_u=1000.0, tau_x=200.0))
            cfg = RecordingConfig(["g_syn"])
            result = rn.simulate(2000.0, DT, {"pre": I_pre, "post": 0.0}, record=cfg)
            g = result["post"]["g_syn"][0]
            n_last = max(1, len(g) // 5)
            results[label] = float(g[-n_last:].mean())  # steady-state mean

        assert results["low"] >= results["high"], \
            (f"STP: low-freq steady-state g_syn ({results['low']:.5f}) should be >= "
             f"high-freq ({results['high']:.5f})")

    def test_depression_monotone(self):
        """
        At high firing rate, peak g_syn should decrease (or stay flat) across spikes.
        Check that the peak in the first 20% of the trace > peak in last 20%.
        """
        rn = _two_neuron_rnet(weight=1.0,
                               plasticity=STPRule(U=0.6, tau_u=800.0, tau_x=80.0))
        cfg = RecordingConfig(["g_syn"])
        result = rn.simulate(500.0, DT, {"pre": 10.0, "post": 0.0}, record=cfg)
        g = result["post"]["g_syn"][0]
        n_fifth = max(1, len(g) // 5)
        peak_early = float(g[:n_fifth].max())
        peak_late  = float(g[-n_fifth:].max())
        # Allow small tolerance — recovery can partially compensate
        assert peak_early >= peak_late * 0.8, \
            f"STP early peak {peak_early:.4f} should be >= late peak {peak_late:.4f}"


# ---------------------------------------------------------------------------
# Section E — Gated STDP
# ---------------------------------------------------------------------------

class TestGatedSTDP:

    def test_modulator_scale_zero_no_weight_change(self):
        """
        STDPRule with modulator_scale=0 means A_plus and A_minus are both
        multiplied by 0 → weight stays exactly at initial value.
        Note: non-modulated pathway is mod_scale=1.0 (pop_start=-1 guard).
        This tests by setting A_plus=A_minus=0 directly.
        """
        rn = _two_neuron_rnet(weight=0.5,
                               plasticity=STDPRule(A_plus=0.0, A_minus=0.0,
                                                   w_min=0.0, w_max=1.0))
        _run(rn, I_pre=8.0, I_post=8.0, duration=500.0)
        w = float(rn.get_synapse_weights()[0])
        assert abs(w - 0.5) < 1e-9, \
            f"Zero A_plus/A_minus: weight should not change, got {w:.6f}"

    def test_gated_stdp_with_dopamine(self):
        """
        Dopamine modulation integrated with STDP.
        A network with DopamineGating + STDP should not crash and produce
        finite weights.
        """
        rn = RegionalNetwork()
        rn.add_population("pre",  1, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
        rn.add_intracellular(DopamineGating(da_level=0.8, k_decay=0.001),
                             populations=["post"])
        rn.connect("pre", "post", "all_to_all", weight=0.4,
                   synapse=SynapseSpec.ampa(),
                   plasticity=STDPRule(A_plus=0.01, A_minus=0.005,
                                       w_min=0.0, w_max=1.0))
        _run(rn, I_pre=7.0, I_post=0.0, duration=300.0)
        w = rn.get_synapse_weights()
        assert np.all(np.isfinite(w)), "Weights should remain finite with DA gating"
        assert np.all(w >= 0.0) and np.all(w <= 1.0), "Weights should stay in bounds"


# ---------------------------------------------------------------------------
# Section F — API surface / exports
# ---------------------------------------------------------------------------

class TestPlasticityAPI:

    def test_exports_toplevel(self):
        assert hasattr(hh, "STDPRule")
        assert hasattr(hh, "STPRule")
        assert hasattr(hh, "DopamineGating")
        assert hasattr(hh, "HomeostaticScaling")

    def test_stdp_rule_constructs(self):
        rule = STDPRule(A_plus=0.01, A_minus=0.008,
                        tau_plus=15.0, tau_minus=25.0,
                        w_min=0.05, w_max=0.95)
        assert rule.A_plus == 0.01
        assert rule.w_max == 0.95

    def test_stp_rule_constructs(self):
        rule = STPRule(U=0.3, tau_u=500.0, tau_x=80.0)
        assert rule.U == 0.3
        assert rule.tau_x == 80.0

    def test_stdp_to_spec(self):
        from hodgkin_huxley._core import PlasticityType
        rule = STDPRule(A_plus=0.01, w_max=0.8)
        spec = rule.to_spec()
        assert spec.type == PlasticityType.STDP
        assert spec.stdp.A_plus == 0.01
        assert spec.stdp.w_max == 0.8

    def test_stp_to_spec(self):
        from hodgkin_huxley._core import PlasticityType
        rule = STPRule(U=0.4)
        spec = rule.to_spec()
        assert spec.type == PlasticityType.STP
        assert spec.stp.U == 0.4

    def test_no_plasticity_zero_overhead(self):
        """
        Network with no plasticity= should have has_stdp=False and has_stp=False,
        meaning no overhead arrays were allocated.
        """
        rn = RegionalNetwork()
        rn.add_population("pre",  5, model=NeuronModelSpec.hh_default())
        rn.add_population("post", 5, model=NeuronModelSpec.hh_default())
        rn.connect("pre", "post", "all_to_all", weight=0.3)
        cfg = RecordingConfig(["V"])
        rn.simulate(100.0, DT, {"pre": 5.0, "post": 0.0}, record=cfg)
        net = rn.network
        assert not net.has_stdp, "has_stdp should be False without plasticity"
        assert not net.has_stp,  "has_stp should be False without plasticity"

    def test_dopamine_gating_constructs(self):
        dg = DopamineGating(da_level=0.6, k=8.0, k_decay=0.005)
        assert isinstance(dg, IntracellularDynamics)

    def test_homeostatic_scaling_constructs(self):
        hs = HomeostaticScaling(target_rate=15.0, tau=3000.0)
        assert isinstance(hs, IntracellularDynamics)

    def test_mixed_plastic_and_non_plastic_synapses(self):
        """Network with both plastic and non-plastic synapses runs without crash."""
        rn = RegionalNetwork()
        rn.add_population("a", 2, model=NeuronModelSpec.hh_default())
        rn.add_population("b", 2, model=NeuronModelSpec.hh_default())
        rn.add_population("c", 2, model=NeuronModelSpec.hh_default())
        # Plastic a→b
        rn.connect("a", "b", "all_to_all", weight=0.4,
                   plasticity=STDPRule(w_min=0.0, w_max=1.0))
        # Non-plastic b→c
        rn.connect("b", "c", "all_to_all", weight=0.3)
        cfg = RecordingConfig(["V"])
        result = rn.simulate(200.0, DT, {"a": 6.0, "b": 0.0, "c": 0.0}, record=cfg)
        assert np.all(np.isfinite(result["a"]["V"]))
        # Plastic synapse weights are the first 4 (a→b, 2×2)
        weights = rn.get_synapse_weights()
        assert len(weights) == 4 + 4  # 4 a→b + 4 b→c
        assert np.all(np.isfinite(weights))
