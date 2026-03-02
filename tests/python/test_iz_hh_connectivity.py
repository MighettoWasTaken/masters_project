"""
Diagnostic tests for Izhikevich ↔ HH/Composable inter-region connectivity.

These tests progressively isolate where synaptic transmission breaks down
between HH/Composable presynaptic neurons and Izhikevich postsynaptic neurons
across separate brain regions (as seen in the CTX-BG-TH model).

Test hierarchy:
  Level 1 — Baseline: neurons fire correctly in isolation
  Level 2 — Single-Network HH→IZ  (plain HH pre, IZ post)
  Level 3 — RegionalNetwork HH→IZ  (two named populations)
  Level 4 — RegionalNetwork Composable→IZ  (mimics TH→CTX_e)
  Level 5 — IZ→IZ connectivity  (intra-type sanity check)
  Level 6 — Current magnitude: mean voltage must rise with synapse
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    Network,
    RegionalNetwork,
    IzhikevichType,
    IzhikevichParameters,
    SynapseSpec,
    NeuronModelSpec,
    RecordingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_spikes(trace: np.ndarray, threshold: float = 0.0) -> int:
    """Count upward threshold crossings in a 1-D voltage trace."""
    above = trace > threshold
    return int(np.sum(np.diff(above.astype(int)) == 1))


def firing_rate_hz(trace: np.ndarray, duration_ms: float,
                   threshold: float = 0.0) -> float:
    return count_spikes(trace, threshold) / (duration_ms / 1000.0)


def _iz_rs_params() -> IzhikevichParameters:
    """Standard Regular-Spiking (CTX_e) Izhikevich parameters."""
    p = IzhikevichParameters()
    p.a, p.b, p.c, p.d = 0.02, 0.2, -65.0, 8.0
    return p


def _iz_fs_params() -> IzhikevichParameters:
    """Fast-Spiking (CTX_i) Izhikevich parameters."""
    p = IzhikevichParameters()
    p.a, p.b, p.c, p.d = 0.10, 0.2, -65.0, 2.0
    return p


# ---------------------------------------------------------------------------
# Level 1 — Baseline: neurons fire correctly in isolation
# ---------------------------------------------------------------------------

class TestBaseline:
    """Verify each neuron type fires (or stays quiet) as expected alone."""

    def test_iz_rs_fires_with_dc_current(self):
        """RS Izhikevich fires >5 Hz with 10 mV/ms DC injection."""
        net = Network()
        net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[10.0] * n])
        assert firing_rate_hz(np.array(traces[0]), duration) > 5.0

    def test_iz_rs_silent_without_input(self):
        """RS Izhikevich stays silent with zero input."""
        net = Network()
        net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[0.0] * n])
        assert count_spikes(np.array(traces[0])) == 0

    def test_hh_fires_with_dc_current(self):
        """Default HH neuron fires >10 Hz with I=10."""
        net = Network(1)
        duration, dt = 200.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[10.0] * n])
        assert firing_rate_hz(np.array(traces[0]), duration) > 10.0

    def test_gpe_composable_fires_autonomously(self):
        """GPe composable neuron fires >10 Hz with I=3 (its natural drive)."""
        net = Network()
        net.add_neuron(model=NeuronModelSpec.gpe())
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[3.0] * n])
        assert firing_rate_hz(np.array(traces[0]), duration) > 10.0


# ---------------------------------------------------------------------------
# Level 2 — Single-Network HH→IZ (plain HH pre, IZ RS post)
# ---------------------------------------------------------------------------
#
# With weight=2.0 and E_syn=0, V_IZ≈-65:
#   I_peak = 2.0 × (0 − (−65)) = 130 mV/ms >> saddle threshold (4 mV/ms).
# If IZ does not respond, current delivery is broken.

class TestHHDrivesIZInNetwork:
    """Plain HH → Izhikevich RS in a single shared Network."""

    WEIGHT = 2.0

    def _setup(self, add_syn_fn, duration=500.0, dt=0.01):
        net = Network()
        pre  = net.add_hh_neuron()           # global index 0
        post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)  # index 1
        add_syn_fn(net, pre, post)
        n = int(duration / dt)
        I_ext = [[10.0] * n, [0.0] * n]     # only drive HH
        traces = net.simulate(duration, dt, I_ext)
        return np.array(traces[0]), np.array(traces[1])

    def test_exponential_synapse_drives_iz(self):
        """HH→IZ via exponential synapse: IZ must fire."""
        pre_t, post_t = self._setup(
            lambda net, pre, post: net.add_synapse(
                pre, post, self.WEIGHT, E_syn=0.0, tau=5.0))
        assert firing_rate_hz(pre_t, 500.0) > 10.0, "HH pre must fire"
        assert count_spikes(post_t) > 0, (
            f"IZ post silent with exp synapse weight={self.WEIGHT} "
            f"(pre fired {count_spikes(pre_t)}×)")

    def test_alpha_synapse_drives_iz(self):
        """HH→IZ via alpha synapse: IZ must fire."""
        pre_t, post_t = self._setup(
            lambda net, pre, post: net.add_alpha_synapse(
                pre, post, self.WEIGHT, E_syn=0.0, tau=5.0))
        assert firing_rate_hz(pre_t, 500.0) > 10.0, "HH pre must fire"
        assert count_spikes(post_t) > 0, (
            f"IZ post silent with alpha synapse weight={self.WEIGHT} "
            f"(pre fired {count_spikes(pre_t)}×)")

    def test_double_exp_synapse_drives_iz(self):
        """HH→IZ via double-exponential synapse: IZ must fire."""
        pre_t, post_t = self._setup(
            lambda net, pre, post: net.add_double_exp_synapse(
                pre, post, self.WEIGHT, E_syn=0.0, tau_rise=0.5, tau_decay=5.0))
        assert firing_rate_hz(pre_t, 500.0) > 10.0, "HH pre must fire"
        assert count_spikes(post_t) > 0, (
            f"IZ post silent with double_exp synapse weight={self.WEIGHT} "
            f"(pre fired {count_spikes(pre_t)}×)")

    def test_synapse_raises_iz_mean_voltage(self):
        """
        Mean IZ voltage must be higher WITH synapse than WITHOUT.
        Proves current is reaching IZ even if it doesn't spike.
        """
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        I_ext = [[10.0] * n, [0.0] * n]

        net_syn = Network()
        pre  = net_syn.add_hh_neuron()
        post = net_syn.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net_syn.add_alpha_synapse(pre, post, self.WEIGHT, E_syn=0.0, tau=5.0)
        traces_syn = net_syn.simulate(duration, dt, I_ext)

        net_ctrl = Network()
        net_ctrl.add_hh_neuron()
        net_ctrl.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        traces_ctrl = net_ctrl.simulate(duration, dt, I_ext)

        mean_syn  = float(np.mean(np.array(traces_syn[1])))
        mean_ctrl = float(np.mean(np.array(traces_ctrl[1])))
        assert mean_syn > mean_ctrl, (
            f"IZ mean voltage WITH synapse ({mean_syn:.2f} mV) must exceed "
            f"WITHOUT ({mean_ctrl:.2f} mV). "
            f"FAIL = synaptic current NOT reaching IZ neuron.")


# ---------------------------------------------------------------------------
# Level 3 — RegionalNetwork HH→IZ (two named populations)
# ---------------------------------------------------------------------------

class TestRegionalNetworkHHDrivesIZ:
    """
    RegionalNetwork with 'Drive' (HH, driven by DC) → 'Target' (IZ RS, no DC).
    Proves inter-population HH→IZ connectivity works.
    """

    N = 3
    WEIGHT = 2.0

    def _build_and_run(self, synapse: SynapseSpec, duration=500.0, dt=0.01):
        rnet = RegionalNetwork()
        rnet.add_population("Drive",  self.N, neuron_type="HH")
        rnet.add_population("Target", self.N, parameters=_iz_rs_params())
        rnet.connect("Drive", "Target", "all_to_all",
                     weight=self.WEIGHT, synapse=synapse)
        traces = rnet.simulate(duration, dt, {"Drive": 10.0})
        drive_traces  = np.array(traces["Drive"])   # (N, n_steps)
        target_traces = np.array(traces["Target"])  # (N, n_steps)
        return drive_traces, target_traces, duration

    def test_alpha_synapse(self):
        drive_t, target_t, dur = self._build_and_run(SynapseSpec.alpha(0.0, 5.0))
        drive_rate  = np.mean([firing_rate_hz(drive_t[i],  dur) for i in range(self.N)])
        target_rate = np.mean([firing_rate_hz(target_t[i], dur) for i in range(self.N)])
        assert drive_rate > 10.0, f"Drive must fire, got {drive_rate:.1f} Hz"
        assert target_rate > 0.0, (
            f"Target (IZ RS) must fire via HH→IZ alpha synapse "
            f"(drive={drive_rate:.1f} Hz, weight={self.WEIGHT})")

    def test_double_exp_synapse(self):
        drive_t, target_t, dur = self._build_and_run(
            SynapseSpec.double_exponential(0.0, 0.5, 5.0))
        drive_rate  = np.mean([firing_rate_hz(drive_t[i],  dur) for i in range(self.N)])
        target_rate = np.mean([firing_rate_hz(target_t[i], dur) for i in range(self.N)])
        assert drive_rate > 10.0, f"Drive must fire, got {drive_rate:.1f} Hz"
        assert target_rate > 0.0, (
            f"Target (IZ RS) must fire via HH→IZ double_exp synapse "
            f"(drive={drive_rate:.1f} Hz, weight={self.WEIGHT})")

    def test_mean_voltage_rises_with_synapse(self):
        """
        RegionalNetwork: mean IZ voltage must be higher WITH HH synapse than without.
        """
        duration, dt = 500.0, 0.01

        rnet_syn = RegionalNetwork()
        rnet_syn.add_population("Drive",  self.N, neuron_type="HH")
        rnet_syn.add_population("Target", self.N, parameters=_iz_rs_params())
        rnet_syn.connect("Drive", "Target", "all_to_all",
                         weight=self.WEIGHT, synapse=SynapseSpec.alpha(0.0, 5.0))
        traces_syn = rnet_syn.simulate(duration, dt, {"Drive": 10.0})
        mean_syn = float(np.mean(np.array(traces_syn["Target"])))

        rnet_ctrl = RegionalNetwork()
        rnet_ctrl.add_population("Drive",  self.N, neuron_type="HH")
        rnet_ctrl.add_population("Target", self.N, parameters=_iz_rs_params())
        traces_ctrl = rnet_ctrl.simulate(duration, dt, {"Drive": 10.0})
        mean_ctrl = float(np.mean(np.array(traces_ctrl["Target"])))

        assert mean_syn > mean_ctrl, (
            f"IZ mean voltage WITH synapse ({mean_syn:.2f} mV) must exceed "
            f"WITHOUT ({mean_ctrl:.2f} mV) in RegionalNetwork. "
            f"FAIL = current not crossing HH→IZ population boundary.")


# ---------------------------------------------------------------------------
# Level 4 — RegionalNetwork Composable→IZ (mimics TH→CTX_e exactly)
# ---------------------------------------------------------------------------

class TestRegionalNetworkComposableDrivesIZ:
    """
    GPe composable neuron fires autonomously; one-to-one alpha synapse to IZ RS.
    This is structurally identical to the TH→CTX_e pathway.
    """

    N = 3
    WEIGHT = 2.0   # far above the IZ RS saddle threshold of 4 mV/ms / 65 ≈ 0.06

    def _build_and_run(self, synapse: SynapseSpec, duration=500.0, dt=0.01):
        rnet = RegionalNetwork()
        rnet.add_population("Comp", self.N, model=NeuronModelSpec.gpe())
        rnet.add_population("IZ",   self.N, parameters=_iz_rs_params())
        rnet.connect("Comp", "IZ", "one_to_one",
                     weight=self.WEIGHT, synapse=synapse)
        traces = rnet.simulate(duration, dt, {"Comp": 3.0})
        comp_traces = np.array(traces["Comp"])
        iz_traces   = np.array(traces["IZ"])
        return comp_traces, iz_traces, duration

    def test_alpha_synapse_composable_to_iz(self):
        """Core replication: Composable→IZ alpha synapse must drive IZ."""
        comp_t, iz_t, dur = self._build_and_run(SynapseSpec.alpha(0.0, 5.0))
        comp_rate = np.mean([firing_rate_hz(comp_t[i], dur) for i in range(self.N)])
        iz_rate   = np.mean([firing_rate_hz(iz_t[i],  dur) for i in range(self.N)])
        assert comp_rate > 10.0, f"Composable pre must fire >10 Hz, got {comp_rate:.1f}"
        assert iz_rate > 0.0, (
            f"IZ post must fire via Composable→IZ alpha synapse. "
            f"Composable={comp_rate:.1f} Hz, IZ={iz_rate:.1f} Hz, weight={self.WEIGHT}. "
            f"FAIL = synaptic current not crossing neuron-type boundary.")

    def test_double_exp_synapse_composable_to_iz(self):
        comp_t, iz_t, dur = self._build_and_run(
            SynapseSpec.double_exponential(0.0, 0.5, 5.0))
        comp_rate = np.mean([firing_rate_hz(comp_t[i], dur) for i in range(self.N)])
        iz_rate   = np.mean([firing_rate_hz(iz_t[i],  dur) for i in range(self.N)])
        assert comp_rate > 10.0, f"Composable pre must fire >10 Hz, got {comp_rate:.1f}"
        assert iz_rate > 0.0, (
            f"IZ post must fire via Composable→IZ double_exp synapse. "
            f"Composable={comp_rate:.1f} Hz, IZ={iz_rate:.1f} Hz")

    def test_mean_voltage_rises_with_synapse(self):
        """
        IZ mean voltage must be higher WITH composable synapse than without.
        Proves current crosses the Composable→IZ type boundary.
        """
        duration, dt = 500.0, 0.01

        rnet_syn = RegionalNetwork()
        rnet_syn.add_population("Comp", self.N, model=NeuronModelSpec.gpe())
        rnet_syn.add_population("IZ",   self.N, parameters=_iz_rs_params())
        rnet_syn.connect("Comp", "IZ", "one_to_one",
                         weight=self.WEIGHT, synapse=SynapseSpec.alpha(0.0, 5.0))
        traces_syn = rnet_syn.simulate(duration, dt, {"Comp": 3.0})
        mean_syn = float(np.mean(np.array(traces_syn["IZ"])))

        rnet_ctrl = RegionalNetwork()
        rnet_ctrl.add_population("Comp", self.N, model=NeuronModelSpec.gpe())
        rnet_ctrl.add_population("IZ",   self.N, parameters=_iz_rs_params())
        traces_ctrl = rnet_ctrl.simulate(duration, dt, {"Comp": 3.0})
        mean_ctrl = float(np.mean(np.array(traces_ctrl["IZ"])))

        assert mean_syn > mean_ctrl, (
            f"IZ mean voltage WITH synapse ({mean_syn:.2f} mV) must exceed "
            f"WITHOUT ({mean_ctrl:.2f} mV). "
            f"FAIL = Composable→IZ current NOT delivered.")


# ---------------------------------------------------------------------------
# Level 5 — IZ→IZ connectivity (intra-type sanity check)
# ---------------------------------------------------------------------------

class TestIZDrivesIZ:
    """
    IZ pre (driven by DC) → IZ post (synapse only).
    Verifies IZ outgoing spike detection works.
    """

    WEIGHT = 2.0

    def test_iz_rs_drives_iz_rs_alpha(self):
        net = Network()
        pre  = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_alpha_synapse(pre, post, self.WEIGHT, E_syn=0.0, tau=5.0)
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[10.0] * n, [0.0] * n])
        pre_t, post_t = np.array(traces[0]), np.array(traces[1])
        assert firing_rate_hz(pre_t, duration) > 5.0, "IZ pre must fire with I=10"
        assert count_spikes(post_t) > 0, (
            f"IZ post must fire via IZ→IZ alpha synapse "
            f"(pre={firing_rate_hz(pre_t, duration):.1f} Hz, weight={self.WEIGHT})")

    def test_iz_rs_drives_iz_rs_double_exp(self):
        net = Network()
        pre  = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_double_exp_synapse(pre, post, self.WEIGHT, E_syn=0.0,
                                   tau_rise=0.5, tau_decay=5.0)
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[10.0] * n, [0.0] * n])
        pre_t, post_t = np.array(traces[0]), np.array(traces[1])
        assert firing_rate_hz(pre_t, duration) > 5.0, "IZ pre must fire with I=10"
        assert count_spikes(post_t) > 0, (
            f"IZ post must fire via IZ→IZ double_exp synapse "
            f"(pre={firing_rate_hz(pre_t, duration):.1f} Hz)")


# ---------------------------------------------------------------------------
# Level 6 — RegionalNetwork: composable→IZ current magnitude
# ---------------------------------------------------------------------------

class TestCurrentMagnitudeAcrossTypes:
    """
    Quantitative check: synapse weight correctly sets the peak conductance
    that IZ receives.  Compares IZ trajectories at different weights to
    confirm linear scaling (no silent gpeak/gpeak1 hidden rescaling).
    """

    def test_higher_weight_gives_higher_iz_voltage(self):
        """
        IZ mean voltage should increase monotonically with weight.
        If the weight is silently rescaled (e.g. ×0.43), the curve is the same
        shape but the absolute voltages would differ from expectations.
        """
        duration, dt = 200.0, 0.01
        weights = [0.1, 0.5, 1.0, 2.0]
        means = []

        for w in weights:
            rnet = RegionalNetwork()
            rnet.add_population("Pre",  2, model=NeuronModelSpec.gpe())
            rnet.add_population("Post", 2, parameters=_iz_rs_params())
            rnet.connect("Pre", "Post", "one_to_one",
                         weight=w, synapse=SynapseSpec.alpha(0.0, 5.0))
            traces = rnet.simulate(duration, dt, {"Pre": 3.0})
            means.append(float(np.mean(np.array(traces["Post"]))))

        for i in range(len(means) - 1):
            assert means[i] < means[i + 1], (
                f"IZ mean voltage should increase with weight. "
                f"Got means={[f'{m:.2f}' for m in means]} for weights={weights}. "
                f"Non-monotonic at index {i}.")

    def test_weight_above_saddle_threshold_fires_iz(self):
        """
        RS IZ neuron saddle-node at I_sustained = 4.0 mV/ms.
        With E_syn=0, V_IZ≈-65: g_saddle = 4.0/65 ≈ 0.062.
        weight=0.2 gives I_peak = 0.2×65 = 13 >> 4.0; IZ MUST fire.
        """
        rnet = RegionalNetwork()
        rnet.add_population("Pre",  3, model=NeuronModelSpec.gpe())
        rnet.add_population("Post", 3, parameters=_iz_rs_params())
        rnet.connect("Pre", "Post", "one_to_one",
                     weight=0.2, synapse=SynapseSpec.alpha(0.0, 5.0))
        duration, dt = 1000.0, 0.01
        traces = rnet.simulate(duration, dt, {"Pre": 3.0})
        post_traces = np.array(traces["Post"])
        pre_traces  = np.array(traces["Pre"])

        pre_rates  = [firing_rate_hz(pre_traces[i],  duration) for i in range(3)]
        post_rates = [firing_rate_hz(post_traces[i], duration) for i in range(3)]

        assert all(r > 10.0 for r in pre_rates), \
            f"GPe pre must fire >10 Hz, got {[f'{r:.1f}' for r in pre_rates]}"
        assert any(r > 0.0 for r in post_rates), (
            f"At least one IZ post neuron must fire with weight=0.2 "
            f"(well above saddle threshold). "
            f"IZ rates={[f'{r:.1f}' for r in post_rates]} Hz. "
            f"Pre rates={[f'{r:.1f}' for r in pre_rates]} Hz. "
            f"FAIL = synaptic current not reaching IZ.")


# ---------------------------------------------------------------------------
# Level 7 — Delayed synaptic transmission to IZ neurons
# ---------------------------------------------------------------------------
#
# The TH→CTX_e synapse in the CTX-BG-TH model uses delay=5 ms.
# These tests verify that the spike-delay buffer works correctly when
# the postsynaptic neuron is an Izhikevich neuron.

class TestDelayedSynapseToIZ:
    """
    Verify synaptic delays are correctly applied when an IZ neuron is the
    postsynaptic target — covering both single-Network and RegionalNetwork paths.
    """

    WEIGHT = 2.0   # far above IZ RS saddle threshold

    # ------------------------------------------------------------------
    # 7a — delay has no effect on whether IZ receives current (it still does)
    # ------------------------------------------------------------------

    def test_hh_drives_iz_with_delay_alpha(self):
        """Delayed alpha synapse: HH pre → IZ post still fires (single Network)."""
        net = Network()
        pre  = net.add_hh_neuron()
        post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_alpha_synapse(pre, post, self.WEIGHT, E_syn=0.0, tau=5.0, delay=5.0)
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[10.0] * n, [0.0] * n])
        pre_t, post_t = np.array(traces[0]), np.array(traces[1])
        assert firing_rate_hz(pre_t, duration) > 10.0, "HH pre must fire"
        assert count_spikes(post_t) > 0, (
            f"IZ post must fire via delayed (5 ms) alpha synapse "
            f"(pre={count_spikes(pre_t)} spikes, weight={self.WEIGHT})")

    def test_hh_drives_iz_with_delay_double_exp(self):
        """Delayed double-exp synapse: HH pre → IZ post still fires."""
        net = Network()
        pre  = net.add_hh_neuron()
        post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_double_exp_synapse(pre, post, self.WEIGHT, E_syn=0.0,
                                   tau_rise=0.5, tau_decay=5.0, delay=5.0)
        duration, dt = 500.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[10.0] * n, [0.0] * n])
        pre_t, post_t = np.array(traces[0]), np.array(traces[1])
        assert firing_rate_hz(pre_t, duration) > 10.0, "HH pre must fire"
        assert count_spikes(post_t) > 0, (
            f"IZ post must fire via delayed (5 ms) double_exp synapse "
            f"(pre={count_spikes(pre_t)} spikes)")

    def test_composable_drives_iz_with_delay_alpha(self):
        """Delayed alpha synapse: Composable pre → IZ post (mimics TH→CTX_e)."""
        rnet = RegionalNetwork()
        rnet.add_population("TH",    3, model=NeuronModelSpec.gpe())
        rnet.add_population("CTX_e", 3, parameters=_iz_rs_params())
        rnet.connect("TH", "CTX_e", "one_to_one",
                     weight=self.WEIGHT,
                     synapse=SynapseSpec.alpha(0.0, 5.0),
                     delay=5.0)
        duration, dt = 1000.0, 0.01
        traces = rnet.simulate(duration, dt, {"TH": 3.0})
        th_t   = np.array(traces["TH"])
        ctx_t  = np.array(traces["CTX_e"])
        th_rates  = [firing_rate_hz(th_t[i],  duration) for i in range(3)]
        ctx_rates = [firing_rate_hz(ctx_t[i], duration) for i in range(3)]
        assert all(r > 10.0 for r in th_rates), \
            f"TH must fire >10 Hz, got {[f'{r:.1f}' for r in th_rates]}"
        assert any(r > 0.0 for r in ctx_rates), (
            f"CTX_e must fire via delayed (5 ms) TH→CTX_e alpha synapse. "
            f"TH rates={[f'{r:.1f}' for r in th_rates]} Hz, "
            f"CTX_e rates={[f'{r:.1f}' for r in ctx_rates]} Hz, "
            f"weight={self.WEIGHT}. FAIL = delay buffer not working for IZ.")

    # ------------------------------------------------------------------
    # 7b — delay actually delays onset: IZ silent during delay window
    # ------------------------------------------------------------------

    def test_iz_silent_during_delay_window(self):
        """
        With delay=D ms, IZ must not fire in [0, D] ms even with
        a very strong pre-synaptic neuron spiking at t≈0.

        Strategy: drive HH with a huge current for just 2 ms (forces ≥1 spike),
        then turn it off; check IZ has zero spikes in [0, delay] ms.
        """
        delay = 10.0   # ms — generous window to test
        dt = 0.01
        duration = 100.0
        n = int(duration / dt)
        delay_steps = int(delay / dt)

        # Large I for first 2 ms forces HH spike; then silence
        I_pre = [0.0] * n
        for k in range(int(2.0 / dt)):
            I_pre[k] = 200.0

        net = Network()
        pre  = net.add_hh_neuron()
        post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        # Very strong weight so the post WOULD fire quickly if no delay
        net.add_alpha_synapse(pre, post, 5.0, E_syn=0.0, tau=5.0, delay=delay)

        traces = net.simulate(duration, dt, [I_pre, [0.0] * n])
        pre_t  = np.array(traces[0])
        post_t = np.array(traces[1])

        assert count_spikes(pre_t) >= 1, "HH pre must produce at least one spike"

        # IZ must be silent in the delay window
        spikes_in_window = count_spikes(post_t[:delay_steps])
        assert spikes_in_window == 0, (
            f"IZ post must be silent during [0, {delay}] ms delay window, "
            f"but spiked {spikes_in_window}× before delay elapsed.")

    def test_iz_fires_after_delay_not_before(self):
        """
        Compare IZ responses for delay=0 vs delay=D.
        With delay=0: first IZ spike arrives at t_zero.
        With delay=D: first IZ spike arrives at t_delay >= t_zero + D.
        """
        duration, dt = 300.0, 0.01
        n = int(duration / dt)
        I_ext_hh = [10.0] * n  # continuous HH drive

        def first_spike_time(delay_ms):
            net = Network()
            pre  = net.add_hh_neuron()
            post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
            net.add_alpha_synapse(pre, post, self.WEIGHT, E_syn=0.0,
                                  tau=5.0, delay=delay_ms)
            traces = net.simulate(duration, dt, [I_ext_hh, [0.0] * n])
            post_t = np.array(traces[0]), np.array(traces[1])
            # Find first upward crossing of 0 mV in post trace
            above = post_t[1] > 0.0
            crossings = np.where(np.diff(above.astype(int)) == 1)[0]
            return crossings[0] * dt if len(crossings) > 0 else None

        t_nodelay = first_spike_time(0.0)
        t_delayed = first_spike_time(5.0)

        assert t_nodelay is not None, "IZ must fire with delay=0 (weight is very strong)"
        assert t_delayed is not None, "IZ must also fire eventually with delay=5 ms"
        assert t_delayed >= t_nodelay + 4.0, (
            f"First IZ spike with delay=5 ms ({t_delayed:.1f} ms) must be "
            f"≥4 ms later than with delay=0 ({t_nodelay:.1f} ms). "
            f"FAIL = delay buffer not correctly offsetting spike delivery.")

    # ------------------------------------------------------------------
    # 7c — multiple delay values across the board
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("delay_ms", [1.0, 2.0, 5.0, 10.0, 20.0])
    def test_various_delays_hh_to_iz(self, delay_ms):
        """HH→IZ fires for every tested delay value (alpha synapse)."""
        net = Network()
        pre  = net.add_hh_neuron()
        post = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_alpha_synapse(pre, post, self.WEIGHT, E_syn=0.0,
                              tau=5.0, delay=delay_ms)
        duration, dt = 600.0, 0.01
        n = int(duration / dt)
        traces = net.simulate(duration, dt, [[10.0] * n, [0.0] * n])
        pre_t, post_t = np.array(traces[0]), np.array(traces[1])
        assert firing_rate_hz(pre_t, duration) > 10.0, "HH pre must fire"
        assert count_spikes(post_t) > 0, (
            f"IZ post must fire with delay={delay_ms} ms alpha synapse "
            f"(pre fired {count_spikes(pre_t)}×, weight={self.WEIGHT})")

    @pytest.mark.parametrize("delay_ms", [1.0, 2.0, 5.0, 10.0, 20.0])
    def test_various_delays_composable_to_iz(self, delay_ms):
        """Composable→IZ fires for every tested delay value (alpha synapse)."""
        rnet = RegionalNetwork()
        rnet.add_population("Pre",  2, model=NeuronModelSpec.gpe())
        rnet.add_population("Post", 2, parameters=_iz_rs_params())
        rnet.connect("Pre", "Post", "one_to_one",
                     weight=self.WEIGHT,
                     synapse=SynapseSpec.alpha(0.0, 5.0),
                     delay=delay_ms)
        duration, dt = 600.0, 0.01
        traces = rnet.simulate(duration, dt, {"Pre": 3.0})
        pre_t  = np.array(traces["Pre"])
        post_t = np.array(traces["Post"])
        pre_rates  = [firing_rate_hz(pre_t[i],  duration) for i in range(2)]
        post_rates = [firing_rate_hz(post_t[i], duration) for i in range(2)]
        assert all(r > 10.0 for r in pre_rates), \
            f"Composable pre must fire >10 Hz with delay={delay_ms} ms"
        assert any(r > 0.0 for r in post_rates), (
            f"IZ post must fire with delay={delay_ms} ms composable→IZ synapse. "
            f"Pre={[f'{r:.1f}' for r in pre_rates]} Hz, "
            f"Post={[f'{r:.1f}' for r in post_rates]} Hz")


# ---------------------------------------------------------------------------
# Level 8 — RecordingConfig.spike_stats() with IZ neurons
# ---------------------------------------------------------------------------
# These tests use the exact same code path as simulate_ctxbgth:
#   result = rnet.simulate(duration, dt, I_ext, record=RecordingConfig.spike_stats())
#   spike_times = result["CTX_e"]["spikes"]
# ---------------------------------------------------------------------------

class TestSpikeStatsRecordingWithIZ:
    """Verify RecordingConfig.spike_stats() correctly detects IZ spikes."""

    WEIGHT   = 2.0
    DURATION = 600.0
    DT       = 0.01

    # ------------------------------------------------------------------
    # 8a — Baseline: DC-driven IZ via spike_stats
    # ------------------------------------------------------------------

    def test_silent_iz_gives_zero_spikes(self):
        """IZ at rest (no input) → spike_stats reports 0 spikes."""
        rnet = RegionalNetwork()
        rnet.add_population("CTX_e", 4, parameters=_iz_rs_params())
        result = rnet.simulate(self.DURATION, self.DT, {"CTX_e": 0.0},
                               record=RecordingConfig.spike_stats())
        counts = result["CTX_e"]["spike_count"]
        assert np.all(counts == 0), \
            f"Silent IZ must produce 0 spikes, got {counts}"

    def test_dc_driven_iz_detected_by_spike_stats(self):
        """IZ with DC=6 fires; spike_stats must detect spikes."""
        rnet = RegionalNetwork()
        rnet.add_population("CTX_e", 4, parameters=_iz_rs_params())
        result = rnet.simulate(self.DURATION, self.DT, {"CTX_e": 6.0},
                               record=RecordingConfig.spike_stats())
        spikes = result["CTX_e"]["spikes"]
        counts = result["CTX_e"]["spike_count"]
        rates  = result["CTX_e"]["firing_rate"]
        assert len(spikes) == 4, "Must have one spike-time array per neuron"
        assert np.all(counts > 0), \
            f"All CTX_e neurons must spike under DC=6, got counts={counts}"
        assert np.all(rates > 0.0), "firing_rate must be positive"

    def test_spike_stats_consistency(self):
        """spike_count[i] == len(spikes[i]) and firing_rate == count/duration_s."""
        rnet = RegionalNetwork()
        rnet.add_population("CTX_e", 4, parameters=_iz_rs_params())
        result = rnet.simulate(self.DURATION, self.DT, {"CTX_e": 6.0},
                               record=RecordingConfig.spike_stats())
        spikes = result["CTX_e"]["spikes"]
        counts = result["CTX_e"]["spike_count"]
        rates  = result["CTX_e"]["firing_rate"]
        for i, sp in enumerate(spikes):
            assert counts[i] == len(sp), \
                f"Neuron {i}: spike_count={counts[i]} != len(spikes)={len(sp)}"
        expected_rates = counts / (self.DURATION / 1000.0)
        np.testing.assert_allclose(rates, expected_rates, rtol=1e-6,
                                   err_msg="firing_rate must equal spike_count/duration_s")

    # ------------------------------------------------------------------
    # 8b — HH→IZ with spike_stats
    # ------------------------------------------------------------------

    def test_hh_drives_iz_detected_by_spike_stats(self):
        """HH population drives IZ population; spike_stats detects IZ spikes."""
        rnet = RegionalNetwork()
        rnet.add_population("TH",    4, model=NeuronModelSpec.gpe())
        rnet.add_population("CTX_e", 4, parameters=_iz_rs_params())
        rnet.connect("TH", "CTX_e", "one_to_one",
                     weight=self.WEIGHT,
                     synapse=SynapseSpec.alpha(0.0, 5.0))
        result = rnet.simulate(self.DURATION, self.DT, {"TH": 3.0},
                               record=RecordingConfig.spike_stats())
        th_counts  = result["TH"]["spike_count"]
        ctx_counts = result["CTX_e"]["spike_count"]
        assert np.all(th_counts > 0), \
            f"TH neurons must spike, got {th_counts}"
        assert np.any(ctx_counts > 0), (
            f"spike_stats must detect CTX_e spikes driven by HH→IZ synapse. "
            f"TH counts={th_counts}, CTX_e counts={ctx_counts}")

    # ------------------------------------------------------------------
    # 8c — Composable→IZ with delay=5ms (exact TH→CTX_e path)
    # ------------------------------------------------------------------

    def test_composable_to_iz_with_delay_detected_by_spike_stats(self):
        """Composable→IZ with alpha synapse + delay=5ms; spike_stats detects IZ spikes.

        This replicates the exact code path from simulate_ctxbgth:
          net.add_connection("TH", i, "CTX_e", i, 0.15, SynapseSpec.alpha(0.0, tau), delay=5.0)
          result = net.simulate(..., record=RecordingConfig.spike_stats())
          spike_times = result["CTX_e"]["spikes"]
        """
        rnet = RegionalNetwork()
        rnet.add_population("TH",    4, model=NeuronModelSpec.gpe())
        rnet.add_population("CTX_e", 4, parameters=_iz_rs_params())
        rnet.connect("TH", "CTX_e", "one_to_one",
                     weight=self.WEIGHT,
                     synapse=SynapseSpec.alpha(0.0, 5.0),
                     delay=5.0)
        result = rnet.simulate(self.DURATION, self.DT, {"TH": 3.0},
                               record=RecordingConfig.spike_stats())
        th_spikes  = result["TH"]["spikes"]
        ctx_spikes = result["CTX_e"]["spikes"]
        th_counts  = result["TH"]["spike_count"]
        ctx_counts = result["CTX_e"]["spike_count"]
        assert np.all(th_counts > 0), \
            f"TH (composable) must spike, got {th_counts}"
        assert np.any(ctx_counts > 0), (
            f"spike_stats must detect CTX_e spikes from composable→IZ (delay=5ms). "
            f"TH counts={th_counts}, CTX_e counts={ctx_counts}. "
            f"TH spike times sample: {[s[:3].tolist() for s in th_spikes[:2]]}. "
            f"CTX_e spike times sample: {[s[:3].tolist() for s in ctx_spikes[:2]]}")

    def test_spike_stats_spikes_are_time_arrays(self):
        """spike_stats['spikes'] must be a list of numpy arrays (spike times in ms)."""
        rnet = RegionalNetwork()
        rnet.add_population("CTX_e", 3, parameters=_iz_rs_params())
        result = rnet.simulate(self.DURATION, self.DT, {"CTX_e": 6.0},
                               record=RecordingConfig.spike_stats())
        spikes = result["CTX_e"]["spikes"]
        assert isinstance(spikes, list), "spikes must be a list"
        assert len(spikes) == 3, "one entry per neuron"
        for sp in spikes:
            assert isinstance(sp, np.ndarray), "each entry must be ndarray"
            if len(sp) > 0:
                assert sp[0] >= 0.0, "spike times must be non-negative"
                assert sp[-1] <= self.DURATION, "spike times must be within duration"

    # ------------------------------------------------------------------
    # 8d — Multi-population result: exact simulate_ctxbgth pattern
    # ------------------------------------------------------------------

    def test_multi_pop_spike_stats_all_keys_present(self):
        """All expected populations appear in spike_stats result dict."""
        rnet = RegionalNetwork()
        rnet.add_population("TH",    2, model=NeuronModelSpec.gpe())
        rnet.add_population("CTX_e", 2, parameters=_iz_rs_params())
        rnet.add_population("CTX_i", 2, parameters=_iz_fs_params())
        rnet.connect("TH", "CTX_e", "one_to_one",
                     weight=self.WEIGHT, synapse=SynapseSpec.alpha(0.0, 5.0))
        rnet.connect("CTX_e", "CTX_i", "one_to_one",
                     weight=0.5, synapse=SynapseSpec.alpha(0.0, 5.0))
        result = rnet.simulate(self.DURATION, self.DT,
                               {"TH": 3.0, "CTX_e": 0.0, "CTX_i": 0.0},
                               record=RecordingConfig.spike_stats())
        # Exact pattern from simulate_ctxbgth:
        spike_times = {pop: result[pop]["spikes"] for pop in ("TH", "CTX_e", "CTX_i")}
        for pop, st in spike_times.items():
            assert isinstance(st, list), f"{pop}: spikes must be a list"
        th_counts = result["TH"]["spike_count"]
        assert np.all(th_counts > 0), f"TH must fire, got {th_counts}"
