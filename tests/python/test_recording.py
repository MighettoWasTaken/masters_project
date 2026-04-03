"""
Comprehensive tests for the Metrics & Recording System.
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    RegionalNetwork, NeuronModelSpec, SynapseSpec,
    RecordingConfig, MetricsResult, PopulationMetricsResult,
)
from neuron_specs import make_stn


# =============================================================================
# Constants
# =============================================================================

DURATION = 100.0   # ms
DT = 0.025         # ms
N_STEPS = int(DURATION / DT)


# =============================================================================
# Helpers
# =============================================================================

def _hh_rn(n=2, I_val=10.0):
    """Small HH RegionalNetwork with one synapse (neurons 0→1)."""
    rn = RegionalNetwork()
    rn.add_population("E", n, model=NeuronModelSpec.hh_default())
    if n >= 2:
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=0.1,
                   synapse=SynapseSpec.ampa())
    return rn, {"E": I_val}


def _stn_spec():
    return make_stn()


def _composable_rn(n=2, I_val=5.0):
    """Small composable (STN) RegionalNetwork."""
    rn = RegionalNetwork()
    rn.add_population("E", n, model=make_stn())
    return rn, {"E": I_val}


def _mixed_rn():
    """1 HH-default + 1 STN composable neuron in separate populations."""
    rn = RegionalNetwork()
    rn.add_population("HH", 1, model=NeuronModelSpec.hh_default())
    rn.add_population("STN", 1, model=make_stn())
    rn.connect("HH", "STN", "all_to_all", weight=0.05, synapse=SynapseSpec.ampa())
    return rn, {"HH": 10.0, "STN": 0.0}


def _regional_net():
    rn = RegionalNetwork()
    rn.add_population("STN", 4, neuron_type="HH")
    rn.add_population("GPe", 4, neuron_type="HH")
    rn.connect("STN", "GPe", "all_to_all", weight=0.1,
               synapse=SynapseSpec.ampa())
    return rn


# =============================================================================
# Backward-compatibility tests
# =============================================================================

def test_no_record_returns_dict():
    rn, I_ext = _hh_rn(2)
    result = rn.simulate(DURATION, DT, I_ext)
    assert isinstance(result, dict)
    assert result["E"].shape == (2, N_STEPS)


def test_no_record_regional_returns_dict():
    rn = _regional_net()
    n_steps = int(DURATION / DT)
    result = rn.simulate(DURATION, DT, {"STN": 10.0})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"STN", "GPe"}
    assert result["STN"].shape == (4, n_steps)
    assert result["GPe"].shape == (4, n_steps)


def test_no_record_matches_new_path():
    """Backward-compat path must give same traces as explicit RecordingConfig(["V"])."""
    rn, I_ext = _hh_rn(2)
    old = rn.simulate(DURATION, DT, I_ext)
    rn.reset()
    cfg = RecordingConfig.voltage_only(interval=1)
    new = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert isinstance(new, PopulationMetricsResult)
    np.testing.assert_array_equal(old["E"], new["E"]["V"])


# =============================================================================
# Shape tests
# =============================================================================

def test_voltage_only_shape():
    rn, I_ext = _hh_rn(3)
    cfg = RecordingConfig.voltage_only()
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert "V" in result["E"]
    assert result["E"]["V"].shape == (3, N_STEPS)


def test_interval_decimation_shape():
    rn, I_ext = _hh_rn(2)
    interval = 5
    cfg = RecordingConfig(["V"], interval=interval)
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    expected_n_rec = (N_STEPS + interval - 1) // interval
    assert result["E"]["V"].shape == (2, expected_n_rec)


def test_time_axis():
    rn, I_ext = _hh_rn(2)
    interval = 4
    cfg = RecordingConfig(["V"], interval=interval)
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    n_rec = result["E"]["V"].shape[1]
    expected = np.arange(n_rec) * interval * DT
    np.testing.assert_allclose(result["E"].time, expected)


# =============================================================================
# Spike / firing rate tests
# =============================================================================

def test_spike_count_matches_len_spikes():
    rn, I_ext = _hh_rn(2, I_val=10.0)
    cfg = RecordingConfig(["spikes", "spike_count", "firing_rate"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    sc = result["E"]["spike_count"]
    spikes = result["E"]["spikes"]
    for i in range(len(spikes)):
        assert sc[i] == len(spikes[i])


def test_firing_rate_hh_under_current():
    """HH neuron at I=10 should fire at roughly 30-200 Hz."""
    rn = RegionalNetwork()
    rn.add_population("E", 1, model=NeuronModelSpec.hh_default())
    cfg = RecordingConfig(["firing_rate"])
    result = rn.simulate(DURATION, DT, {"E": 10.0}, record=cfg)
    rate = result["E"]["firing_rate"][0]
    assert 30.0 < rate < 200.0, f"Unexpected firing rate: {rate:.1f} Hz"


def test_mean_V_reasonable():
    """At rest (I=0) a HH neuron stays near -65 mV."""
    rn = RegionalNetwork()
    rn.add_population("E", 1, model=NeuronModelSpec.hh_default())
    cfg = RecordingConfig(["mean_V"])
    result = rn.simulate(DURATION, DT, {"E": 0.0}, record=cfg)
    assert -80.0 < result["E"]["mean_V"][0] < -50.0


def test_isi_cv_regular():
    """Constant injected current → regular firing → ISI_cv near 0."""
    rn = RegionalNetwork()
    rn.add_population("E", 1, model=NeuronModelSpec.hh_default())
    cfg = RecordingConfig(["ISI_cv"], interval=1)
    result = rn.simulate(2000.0, DT, {"E": 10.0}, record=cfg)
    cv = result["E"]["ISI_cv"][0]
    assert cv < 0.05, f"ISI CV too high for regular firing: {cv:.4f}"


def test_spike_count_per_synapse():
    """spike_count_per_synapse[k] == spike_count of pre-synaptic neuron."""
    rn = RegionalNetwork()
    rn.add_population("E", 3, model=NeuronModelSpec.hh_default())
    # 0→1, 0→2, 1→2
    rn.connect("E", "E", lambda ns, nd: [(0, 1), (0, 2), (1, 2)],
               weight=0.1, synapse=SynapseSpec.ampa())
    I_ext = np.array([[10.0] * N_STEPS, [10.0] * N_STEPS, [0.0] * N_STEPS])
    cfg = RecordingConfig(["spike_count", "spike_count_per_synapse"])
    result = rn.simulate(DURATION, DT, {"E": I_ext}, record=cfg)
    sc = result["E"]["spike_count"]
    sc_syn = result["E"]["spike_count_per_synapse"]
    pre_indices = rn._rnet.network().get_synapse_pre_indices()
    for k, pre in enumerate(pre_indices):
        assert sc_syn[k] == sc[pre], f"Synapse {k}: pre={pre}"


# =============================================================================
# Gate / calcium tests (composable neurons)
# =============================================================================

def test_gate_shape_composable():
    rn, I_ext = _composable_rn(3)
    cfg = RecordingConfig(["gates"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    g = result["E"]["gates"]
    n_gates = rn._rnet.network().max_gate_count()
    assert g.shape == (3, n_gates, N_STEPS)
    # Gate values should be in [0, 1]
    assert g.min() >= 0.0
    assert g.max() <= 1.0


def test_gate_shape_hh_default():
    """NeuronModelSpec.hh_default() has 3 gates (m, h, n) via ComposableNeuron."""
    rn = RegionalNetwork()
    rn.add_population("E", 2, model=NeuronModelSpec.hh_default())
    n_gates = rn._rnet.network().max_gate_count()
    assert n_gates == 3
    cfg = RecordingConfig(["gates"])
    result = rn.simulate(DURATION, DT, {"E": np.array([[10.0] * N_STEPS, [0.0] * N_STEPS])}, record=cfg)
    g = result["E"]["gates"]
    assert g.shape == (2, 3, N_STEPS)
    assert g.min() >= 0.0
    assert g.max() <= 1.0


def test_calcium_shape_composable():
    rn, I_ext = _composable_rn(2, I_val=5.0)
    cfg = RecordingConfig(["calcium"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    ca = result["E"]["calcium"]
    assert ca.shape == (2, N_STEPS)
    assert (ca >= 0).all(), "Calcium must be non-negative"


def test_calcium_zeros_for_hh():
    """HH-default neurons have no active calcium dynamics: Ca stays constant at Ca_init."""
    rn = RegionalNetwork()
    rn.add_population("E", 2, model=NeuronModelSpec.hh_default())
    spec = NeuronModelSpec.hh_default()
    cfg = RecordingConfig(["calcium"])
    result = rn.simulate(DURATION, DT, {"E": np.array([[10.0] * N_STEPS, [0.0] * N_STEPS])}, record=cfg)
    ca = result["E"]["calcium"]
    np.testing.assert_allclose(ca, spec.calcium.Ca_init, atol=1e-10)


def test_calcium_increases_composable():
    """Under sustained drive, calcium should rise above zero."""
    rn, I_ext = _composable_rn(1, I_val=10.0)
    cfg = RecordingConfig(["calcium"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    spec = _stn_spec()
    if not spec.calcium.enabled:
        pytest.skip("STN model has no calcium dynamics")
    assert result["E"]["calcium"][0, -1] > 0.0, "Expected nonzero calcium under drive"


# =============================================================================
# g_syn / I_syn tests
# =============================================================================

def test_g_syn_shape():
    rn, I_ext = _hh_rn(2)
    n_synapses = rn.num_synapses
    cfg = RecordingConfig(["g_syn"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert result["E"]["g_syn"].shape == (n_synapses, N_STEPS)
    assert (result["E"]["g_syn"] >= 0).all()


def test_I_syn_zero_no_synapses():
    """Isolated neuron → I_syn ≈ 0 at all times."""
    rn = RegionalNetwork()
    rn.add_population("E", 1, model=NeuronModelSpec.hh_default())
    cfg = RecordingConfig(["I_syn"])
    result = rn.simulate(DURATION, DT, {"E": 5.0}, record=cfg)
    np.testing.assert_allclose(result["E"]["I_syn"], 0.0, atol=1e-12)


def test_I_syn_nonzero_with_synapses():
    """Connected network → post-synaptic neuron should see nonzero I_syn."""
    rn, I_ext = _hh_rn(2, I_val=10.0)
    cfg = RecordingConfig(["I_syn"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    # Neuron 1 (post-synaptic) should have some synaptic input
    assert np.any(result["E"]["I_syn"][1] != 0.0)


# =============================================================================
# Neuron selection tests
# =============================================================================

def test_neuron_selection():
    rn, I_ext = _hh_rn(4, I_val=10.0)
    cfg = RecordingConfig(["V"], neurons={"E": [0, 2]})
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert result["E"]["V"].shape == (2, N_STEPS)
    assert result["E"].neuron_indices == [0, 2]


def test_neuron_selection_traces_correct():
    """Selected neurons should match all-neuron traces at the same indices."""
    rn, I_ext = _hh_rn(4, I_val=10.0)
    cfg_all = RecordingConfig(["V"])
    result_all = rn.simulate(DURATION, DT, I_ext, record=cfg_all)
    rn.reset()
    cfg_sel = RecordingConfig(["V"], neurons={"E": [1, 3]})
    result_sel = rn.simulate(DURATION, DT, I_ext, record=cfg_sel)
    np.testing.assert_array_equal(result_all["E"]["V"][[1, 3]], result_sel["E"]["V"])


# =============================================================================
# Summary metrics (no V buffer)
# =============================================================================

def test_summary_metrics_no_V():
    rn, I_ext = _hh_rn(2, I_val=10.0)
    cfg = RecordingConfig.summary_metrics()
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert "V" not in result["E"]
    assert "spike_count" in result["E"]
    assert "firing_rate" in result["E"]
    assert "mean_V" in result["E"]


# =============================================================================
# MetricsResult API tests
# =============================================================================

def test_metrics_result_keys():
    rn, I_ext = _hh_rn(2, I_val=10.0)
    requested = ["V", "spikes", "spike_count"]
    cfg = RecordingConfig(requested)
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert set(result["E"].keys()) == set(requested)


def test_metrics_result_contains():
    rn, I_ext = _hh_rn(2)
    cfg = RecordingConfig(["V", "spike_count"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert "V" in result["E"]
    assert "spike_count" in result["E"]
    assert "gates" not in result["E"]


# =============================================================================
# Preset tests
# =============================================================================

def test_preset_voltage_only():
    rn, I_ext = _hh_rn(2)
    cfg = RecordingConfig.voltage_only()
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert set(result["E"].keys()) == {"V"}


def test_preset_spikes_only():
    rn, I_ext = _hh_rn(2, I_val=10.0)
    cfg = RecordingConfig.spikes_only()
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    assert set(result["E"].keys()) == {"spikes", "spike_count", "firing_rate"}
    assert cfg.interval == 1  # must be 1 for accurate spike detection


# =============================================================================
# NaN / Inf sanity
# =============================================================================

def test_all_neuron_metrics_no_nan():
    rn, I_ext = _composable_rn(2, I_val=5.0)
    cfg = RecordingConfig.all_neuron_metrics(interval=1)
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    for key in result["E"].keys():
        val = result["E"][key]
        if isinstance(val, np.ndarray):
            assert not np.any(np.isnan(val)), f"NaN in {key}"
            assert not np.any(np.isinf(val)), f"Inf in {key}"


def test_all_synapse_metrics_no_nan():
    rn, I_ext = _hh_rn(2, I_val=10.0)
    cfg = RecordingConfig.all_synapse_metrics(interval=1)
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    for key in result["E"].keys():
        val = result["E"][key]
        if isinstance(val, np.ndarray):
            assert not np.any(np.isnan(val)), f"NaN in {key}"
            assert not np.any(np.isinf(val)), f"Inf in {key}"


# =============================================================================
# RegionalNetwork tests
# =============================================================================

def test_regional_population_keys():
    rn = _regional_net()
    cfg = RecordingConfig.voltage_only()
    result = rn.simulate(DURATION, DT, {"STN": 10.0}, record=cfg)
    assert isinstance(result, PopulationMetricsResult)
    assert set(result.keys()) == {"STN", "GPe"}


def test_regional_neuron_selection():
    """neurons={"STN":"all","GPe":[0,1]} → GPe sub-result has shape (2, T)."""
    rn = _regional_net()
    cfg = RecordingConfig(["V"], neurons={"STN": "all", "GPe": [0, 1]})
    result = rn.simulate(DURATION, DT, {"STN": 10.0}, record=cfg)
    assert isinstance(result, PopulationMetricsResult)
    gpe_result = result["GPe"]
    assert gpe_result["V"].shape == (2, N_STEPS)
    stn_result = result["STN"]
    assert stn_result["V"].shape == (4, N_STEPS)


def test_for_population_preset():
    """for_population("STN") → only STN in PopulationMetricsResult."""
    rn = _regional_net()
    cfg = RecordingConfig.for_population("STN", metrics=["V", "spikes"])
    result = rn.simulate(DURATION, DT, {"STN": 10.0}, record=cfg)
    assert isinstance(result, PopulationMetricsResult)
    assert "STN" in result.populations
    # GPe was not selected → should not appear
    assert "GPe" not in result.populations


def test_regional_no_record_backward_compat():
    """RegionalNetwork.simulate() without record= must return dict of ndarrays."""
    rn = _regional_net()
    result = rn.simulate(DURATION, DT, {"STN": 10.0})
    assert isinstance(result, dict)
    for name in ("STN", "GPe"):
        assert isinstance(result[name], np.ndarray)
        assert result[name].ndim == 2


# =============================================================================
# Mixed neuron type tests
# =============================================================================

def test_mixed_net_gate_zeros_for_hh_rows():
    """In a mixed net, STN (composable) population has nonzero gates."""
    rn, I_ext = _mixed_rn()  # HH population + STN population
    cfg = RecordingConfig(["gates"])
    result = rn.simulate(DURATION, DT, I_ext, record=cfg)
    n_gates = rn._rnet.network().max_gate_count()
    assert n_gates > 3  # STN has more gates than HH-default (3)
    # STN (composable) population should have nonzero gate values
    g_stn = result["STN"]["gates"]
    assert np.any(g_stn != 0.0)
    # All gate values in valid range
    assert np.all(result["HH"]["gates"] >= 0.0)
    assert np.all(g_stn >= 0.0)
