"""
Comprehensive tests for the Metrics & Recording System.
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    Network, RegionalNetwork, NeuronModelSpec, SynapseSpec,
    RecordingConfig, MetricsResult, PopulationMetricsResult,
    NetworkNeuronType,
)


# =============================================================================
# Helpers
# =============================================================================

DURATION = 100.0   # ms
DT = 0.025         # ms
N_STEPS = int(DURATION / DT)


def _hh_net(n=2, I_val=10.0):
    """Small HH network with one synapse."""
    net = Network(n)
    if n >= 2:
        net.add_synapse(0, 1, weight=0.1)
    n_steps = int(DURATION / DT)
    I_ext = [[I_val] * n_steps for _ in range(n)]
    return net, I_ext


def _stn_spec():
    return NeuronModelSpec.stn()


def _composable_net(n=2, I_val=5.0):
    """Small composable (STN) network."""
    net = Network()
    spec = _stn_spec()
    for _ in range(n):
        net.add_neuron(spec)
    n_steps = int(DURATION / DT)
    I_ext = [[I_val] * n_steps for _ in range(n)]
    return net, I_ext


def _mixed_net():
    """1 HH + 1 composable neuron."""
    net = Network()
    net.add_hh_neuron()
    net.add_neuron(_stn_spec())
    net.add_synapse(0, 1, weight=0.05)
    n_steps = int(DURATION / DT)
    I_ext = [[10.0] * n_steps, [0.0] * n_steps]
    return net, I_ext


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

def test_no_record_returns_ndarray():
    net, I_ext = _hh_net(2)
    result = net.simulate(DURATION, DT, I_ext)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, N_STEPS)


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
    net, I_ext = _hh_net(2)
    old = net.simulate(DURATION, DT, I_ext)
    net.reset()
    cfg = RecordingConfig.voltage_only(interval=1)
    new = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert isinstance(new, MetricsResult)
    np.testing.assert_array_equal(old, new["V"])


# =============================================================================
# Shape tests
# =============================================================================

def test_voltage_only_shape():
    net, I_ext = _hh_net(3)
    cfg = RecordingConfig.voltage_only()
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert "V" in result
    assert result["V"].shape == (3, N_STEPS)


def test_interval_decimation_shape():
    net, I_ext = _hh_net(2)
    interval = 5
    cfg = RecordingConfig(["V"], interval=interval)
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    expected_n_rec = (N_STEPS + interval - 1) // interval
    assert result["V"].shape == (2, expected_n_rec)


def test_time_axis():
    net, I_ext = _hh_net(2)
    interval = 4
    cfg = RecordingConfig(["V"], interval=interval)
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    n_rec = result["V"].shape[1]
    expected = np.arange(n_rec) * interval * DT
    np.testing.assert_allclose(result.time, expected)


# =============================================================================
# Spike / firing rate tests
# =============================================================================

def test_spike_count_matches_len_spikes():
    net, I_ext = _hh_net(2, I_val=10.0)
    cfg = RecordingConfig(["spikes", "spike_count", "firing_rate"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    sc = result["spike_count"]
    spikes = result["spikes"]
    for i in range(len(spikes)):
        assert sc[i] == len(spikes[i])


def test_firing_rate_hh_under_current():
    """HH neuron at I=10 should fire at roughly 50-150 Hz."""
    net = Network(1)
    n_steps = int(DURATION / DT)
    I_ext = [[10.0] * n_steps]
    cfg = RecordingConfig(["firing_rate"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    rate = result["firing_rate"][0]
    assert 30.0 < rate < 200.0, f"Unexpected firing rate: {rate:.1f} Hz"


def test_mean_V_reasonable():
    """At rest (I=0) a HH neuron stays near -65 mV."""
    net = Network(1)
    n_steps = int(DURATION / DT)
    I_ext = [[0.0] * n_steps]
    cfg = RecordingConfig(["mean_V"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert -80.0 < result["mean_V"][0] < -50.0


def test_isi_cv_regular():
    """Constant injected current → regular firing → ISI_cv near 0."""
    net = Network(1)
    n_steps = int(2000.0 / DT)
    I_ext = [[10.0] * n_steps]
    cfg = RecordingConfig(["ISI_cv"], interval=1)
    result = net.simulate(2000.0, DT, I_ext, record=cfg)
    cv = result["ISI_cv"][0]
    assert cv < 0.05, f"ISI CV too high for regular firing: {cv:.4f}"


def test_spike_count_per_synapse():
    """spike_count_per_synapse[k] == spike_count of pre-synaptic neuron."""
    net = Network(3)
    net.add_synapse(0, 1, weight=0.1)
    net.add_synapse(0, 2, weight=0.1)
    net.add_synapse(1, 2, weight=0.1)
    n_steps = int(DURATION / DT)
    I_ext = [[10.0] * n_steps, [10.0] * n_steps, [0.0] * n_steps]
    cfg = RecordingConfig(["spike_count", "spike_count_per_synapse"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    sc = result["spike_count"]
    sc_syn = result["spike_count_per_synapse"]
    pre_indices = net._network.get_synapse_pre_indices()
    for k, pre in enumerate(pre_indices):
        # pre is global index; in selected (all), sc[pre] is correct
        assert sc_syn[k] == sc[pre], f"Synapse {k}: pre={pre}"


# =============================================================================
# Gate / calcium tests (composable neurons)
# =============================================================================

def test_gate_shape_composable():
    net, I_ext = _composable_net(3)
    cfg = RecordingConfig(["gates"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    g = result["gates"]
    n_gates = net._network.max_gate_count()
    assert g.shape == (3, n_gates, N_STEPS)
    # Gate values should be in [0, 1]
    assert g.min() >= 0.0
    assert g.max() <= 1.0


def test_gate_zeros_for_hh():
    """HH neurons are not composable → max_gate_count=0 → empty gate axis."""
    net = Network(2)  # pure HH
    assert net._network.max_gate_count() == 0
    n_steps = int(DURATION / DT)
    I_ext = [[10.0] * n_steps, [0.0] * n_steps]
    cfg = RecordingConfig(["gates"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    # Shape is (n_neurons, 0, n_rec) — no gate dimension for HH neurons
    assert result["gates"].shape == (2, 0, N_STEPS)


def test_calcium_shape_composable():
    net, I_ext = _composable_net(2, I_val=5.0)
    cfg = RecordingConfig(["calcium"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    ca = result["calcium"]
    assert ca.shape == (2, N_STEPS)
    assert (ca >= 0).all(), "Calcium must be non-negative"


def test_calcium_zeros_for_hh():
    """HH neurons leave calcium_buf rows as zero."""
    net = Network(2)
    n_steps = int(DURATION / DT)
    I_ext = [[10.0] * n_steps, [0.0] * n_steps]
    cfg = RecordingConfig(["calcium"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    np.testing.assert_array_equal(result["calcium"], 0.0)


def test_calcium_increases_composable():
    """Under sustained drive, calcium should rise above zero."""
    net, I_ext = _composable_net(1, I_val=10.0)
    cfg = RecordingConfig(["calcium"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    spec = _stn_spec()
    if not spec.calcium.enabled:
        pytest.skip("STN model has no calcium dynamics")
    assert result["calcium"][0, -1] > 0.0, "Expected nonzero calcium under drive"


# =============================================================================
# g_syn / I_syn tests
# =============================================================================

def test_g_syn_shape():
    net, I_ext = _hh_net(2)
    n_synapses = net.num_synapses
    cfg = RecordingConfig(["g_syn"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert result["g_syn"].shape == (n_synapses, N_STEPS)
    assert (result["g_syn"] >= 0).all()


def test_I_syn_zero_no_synapses():
    """Isolated neuron → I_syn ≈ 0 at all times."""
    net = Network(1)
    n_steps = int(DURATION / DT)
    I_ext = [[5.0] * n_steps]
    cfg = RecordingConfig(["I_syn"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    np.testing.assert_allclose(result["I_syn"], 0.0, atol=1e-12)


def test_I_syn_nonzero_with_synapses():
    """Connected network → post-synaptic neuron should see nonzero I_syn."""
    net, I_ext = _hh_net(2, I_val=10.0)
    cfg = RecordingConfig(["I_syn"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    # Neuron 1 (post-synaptic) should have some synaptic input
    assert np.any(result["I_syn"][1] != 0.0)


# =============================================================================
# Neuron selection tests
# =============================================================================

def test_neuron_selection():
    net, I_ext = _hh_net(4, I_val=10.0)
    cfg = RecordingConfig(["V"], neurons=[0, 2])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert result["V"].shape == (2, N_STEPS)
    assert result.neuron_indices == [0, 2]


def test_neuron_selection_traces_correct():
    """Selected neurons should match all-neuron traces at the same indices."""
    net, I_ext = _hh_net(4, I_val=10.0)
    cfg_all = RecordingConfig(["V"])
    result_all = net.simulate(DURATION, DT, I_ext, record=cfg_all)
    net.reset()
    cfg_sel = RecordingConfig(["V"], neurons=[1, 3])
    result_sel = net.simulate(DURATION, DT, I_ext, record=cfg_sel)
    np.testing.assert_array_equal(result_all["V"][[1, 3]], result_sel["V"])


# =============================================================================
# Summary metrics (no V buffer)
# =============================================================================

def test_summary_metrics_no_V():
    net, I_ext = _hh_net(2, I_val=10.0)
    cfg = RecordingConfig.summary_metrics()
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert "V" not in result
    assert "spike_count" in result
    assert "firing_rate" in result
    assert "mean_V" in result


# =============================================================================
# MetricsResult API tests
# =============================================================================

def test_metrics_result_keys():
    net, I_ext = _hh_net(2, I_val=10.0)
    requested = ["V", "spikes", "spike_count"]
    cfg = RecordingConfig(requested)
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert set(result.keys()) == set(requested)


def test_metrics_result_contains():
    net, I_ext = _hh_net(2)
    cfg = RecordingConfig(["V", "spike_count"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert "V" in result
    assert "spike_count" in result
    assert "gates" not in result


# =============================================================================
# Preset tests
# =============================================================================

def test_preset_voltage_only():
    net, I_ext = _hh_net(2)
    cfg = RecordingConfig.voltage_only()
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert set(result.keys()) == {"V"}


def test_preset_spikes_only():
    net, I_ext = _hh_net(2, I_val=10.0)
    cfg = RecordingConfig.spikes_only()
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    assert set(result.keys()) == {"spikes", "spike_count", "firing_rate"}
    assert cfg.interval == 1  # must be 1 for accurate spike detection


# =============================================================================
# NaN / Inf sanity
# =============================================================================

def test_all_neuron_metrics_no_nan():
    net, I_ext = _composable_net(2, I_val=5.0)
    cfg = RecordingConfig.all_neuron_metrics(interval=1)
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    for key in result.keys():
        val = result[key]
        if isinstance(val, np.ndarray):
            assert not np.any(np.isnan(val)), f"NaN in {key}"
            assert not np.any(np.isinf(val)), f"Inf in {key}"


def test_all_synapse_metrics_no_nan():
    net, I_ext = _hh_net(2, I_val=10.0)
    cfg = RecordingConfig.all_synapse_metrics(interval=1)
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    for key in result.keys():
        val = result[key]
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
    """In a mixed net, HH rows in gate_buf stay 0; composable rows are nonzero."""
    net, I_ext = _mixed_net()  # neuron 0 = HH, neuron 1 = composable
    cfg = RecordingConfig(["gates"])
    result = net.simulate(DURATION, DT, I_ext, record=cfg)
    g = result["gates"]
    n_gates = net._network.max_gate_count()
    assert n_gates > 0
    # Row 0 (HH) should be all zeros
    np.testing.assert_array_equal(g[0], 0.0)
    # Row 1 (composable) should have nonzero values
    assert np.any(g[1] != 0.0)
