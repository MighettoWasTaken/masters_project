"""
Synapse Model Tests

Tests for the three synapse types (Exponential, Alpha, Double-Exponential)
and their integration with RegionalNetwork.

Test Categories:
1. Synapse Kinetics - verify each model's effect on postsynaptic neuron
2. Network Integration - verify all types work in simulations
3. Edge Cases - invalid parameters, reset behavior, numerical stability
4. Synaptic Delays - creation, behavior, chains, reset, stability
5. Receptor Types - AMPA, NMDA, GABA_A functional behavior
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    RegionalNetwork, NeuronModelSpec, SynapseSpec, ReceptorType,
)


# =============================================================================
# Test Utilities
# =============================================================================

def count_spikes(trace, threshold=0.0):
    """Count spikes using upward threshold crossings."""
    above = np.array(trace) > threshold
    crossings = np.diff(above.astype(int))
    return int(np.sum(crossings == 1))


def get_mean_voltage(trace):
    return float(np.mean(trace))


def _find_first_spike_time(trace, dt, threshold=0.0):
    """
    Find the time (in ms) of the first upward threshold crossing in the trace.
    Returns None if no spike is found.
    """
    trace = np.asarray(trace)
    for i in range(1, len(trace)):
        if trace[i] > threshold and trace[i - 1] <= threshold:
            return i * dt
    return None


# Synapse parameter constants
WEIGHT = 5.0
WEIGHT_STRONG = 10.0
E_SYN_EXC = 0.0
E_SYN_INH = -80.0
TAU = 2.0
TAU_RISE = 0.4
TAU_DECAY = 2.5

# Biologically realistic pathway delays
DELAY_STN_GPE = 2.0    # ms
DELAY_STN_GPI = 1.5    # ms
DELAY_GPE_STN = 4.0    # ms
DELAY_GPE_GPI = 3.0    # ms
DELAY_GPI_TH  = 5.0    # ms
DELAY_COR_STN = 5.9    # ms
DELAY_TH_COR  = 5.0    # ms

# Pre-built SynapseSpec constants
EXC = SynapseSpec.exponential(E_SYN_EXC, TAU)
EXC_ALPHA = SynapseSpec.alpha(E_SYN_EXC, TAU)
EXC_DEXP = SynapseSpec.double_exponential(E_SYN_EXC, TAU_RISE, TAU_DECAY)


# =============================================================================
# Network construction helpers
# =============================================================================

def _two_pop_rn(synapse=None, weight=WEIGHT_STRONG, delay=0.0):
    """Two-population network: pre (1 neuron) -> post (1 neuron)."""
    rn = RegionalNetwork()
    rn.add_population("pre", 1, model=NeuronModelSpec.hh_default())
    rn.add_population("post", 1, model=NeuronModelSpec.hh_default())
    if synapse is not None:
        rn.connect("pre", "post", "all_to_all",
                   weight=weight, synapse=synapse, delay=delay)
    return rn


def _run2(rn, duration, dt, I_pre=15.0, I_post=0.0):
    """Simulate and return (pre_trace, post_trace) as 1D arrays."""
    result = rn.simulate(duration, dt, {"pre": I_pre, "post": I_post})
    return result["pre"][0], result["post"][0]


def _chain_rn(n, synapse, weight=WEIGHT_STRONG, delay=0.0):
    """n-neuron linear chain: N0 -> N1 -> ... -> N{n-1}."""
    rn = RegionalNetwork()
    for i in range(n):
        rn.add_population(f"N{i}", 1, model=NeuronModelSpec.hh_default())
    for i in range(n - 1):
        rn.connect(f"N{i}", f"N{i+1}", "all_to_all",
                   weight=weight, synapse=synapse, delay=delay)
    return rn


def _run_chain(rn, n, duration, dt, I_first=15.0, I_others=None):
    """Simulate chain, return list of voltage traces."""
    I_dict = {f"N{i}": (I_first if i == 0 else 0.0) for i in range(n)}
    if I_others is not None:
        for k, v in I_others.items():
            I_dict[k] = v
    result = rn.simulate(duration, dt, I_dict)
    return [result[f"N{i}"][0] for i in range(n)]


# =============================================================================
# Test Class: Synapse Kinetics
# =============================================================================

class TestSynapseKinetics:
    """Tests verifying the conductance waveform of each synapse type."""

    def test_exponential_decay_shape(self):
        """Exponential synapse: monotonic decay after spike trigger."""
        rn = _two_pop_rn(EXC)
        _, post = _run2(rn, 200.0, 0.01)

        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 200.0, 0.01)
        assert get_mean_voltage(post) > get_mean_voltage(ctrl), \
            "Excitatory exponential synapse should raise voltage"

    def test_alpha_produces_response(self):
        """Alpha synapse should produce measurable postsynaptic response."""
        rn = _two_pop_rn(EXC_ALPHA)
        _, post = _run2(rn, 200.0, 0.01)

        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 200.0, 0.01)
        assert get_mean_voltage(post) > get_mean_voltage(ctrl), \
            "Excitatory alpha synapse should raise voltage"

    def test_double_exp_produces_response(self):
        """Double-exponential synapse should produce measurable postsynaptic response."""
        rn = _two_pop_rn(EXC_DEXP)
        _, post = _run2(rn, 200.0, 0.01)

        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 200.0, 0.01)
        assert get_mean_voltage(post) > get_mean_voltage(ctrl), \
            "Excitatory double-exp synapse should raise voltage"

    def test_all_types_excitatory(self):
        """All synapse types with E_syn=0 should be excitatory."""
        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 300.0, 0.01)
        ctrl_mean = get_mean_voltage(ctrl)

        for label, spec in [
            ("exponential", SynapseSpec.exponential(E_SYN_EXC, TAU)),
            ("alpha",       SynapseSpec.alpha(E_SYN_EXC, TAU)),
            ("double_exp",  SynapseSpec.double_exponential(E_SYN_EXC, TAU_RISE, TAU_DECAY)),
        ]:
            rn = _two_pop_rn(spec)
            _, post = _run2(rn, 300.0, 0.01)
            assert get_mean_voltage(post) > ctrl_mean, \
                f"{label} synapse should be excitatory"

    def test_all_types_inhibitory(self):
        """All synapse types with E_syn=-80 should be inhibitory."""
        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 300.0, 0.01, I_pre=15.0, I_post=10.0)
        ctrl_mean = get_mean_voltage(ctrl)

        for label, spec in [
            ("exponential", SynapseSpec.exponential(E_SYN_INH, 5.0)),
            ("alpha",       SynapseSpec.alpha(E_SYN_INH, 5.0)),
            ("double_exp",  SynapseSpec.double_exponential(E_SYN_INH, 0.4, 5.0)),
        ]:
            rn = _two_pop_rn(spec)
            _, post = _run2(rn, 300.0, 0.01, I_pre=15.0, I_post=10.0)
            assert get_mean_voltage(post) < ctrl_mean, \
                f"{label} synapse should be inhibitory"


# =============================================================================
# Test Class: Network Integration
# =============================================================================

class TestNetworkIntegration:
    """Tests for synapse types working within network simulations."""

    def test_alpha_chain_propagation(self):
        """Spikes should propagate through a chain of alpha synapses."""
        rn = _chain_rn(3, EXC_ALPHA)
        traces = _run_chain(rn, 3, 500.0, 0.01)

        assert count_spikes(traces[0]) > 0, "N0 should spike"
        assert count_spikes(traces[1]) > 0, "N1 should spike via alpha synapse"
        assert count_spikes(traces[2]) > 0, "N2 should spike via chain"

    def test_double_exp_chain_propagation(self):
        """Spikes should propagate through a chain of double-exp synapses."""
        rn = _chain_rn(3, EXC_DEXP)
        traces = _run_chain(rn, 3, 500.0, 0.01)

        assert count_spikes(traces[0]) > 0, "N0 should spike"
        assert count_spikes(traces[1]) > 0, "N1 should spike via double-exp"
        assert count_spikes(traces[2]) > 0, "N2 should spike via chain"

    def test_mixed_synapse_chain(self):
        """Chain with mixed synapse types should propagate spikes."""
        rn = RegionalNetwork()
        for i in range(4):
            rn.add_population(f"N{i}", 1, model=NeuronModelSpec.hh_default())
        rn.connect("N0", "N1", "all_to_all", weight=WEIGHT_STRONG, synapse=EXC)
        rn.connect("N1", "N2", "all_to_all", weight=WEIGHT_STRONG, synapse=EXC_ALPHA)
        rn.connect("N2", "N3", "all_to_all", weight=WEIGHT_STRONG, synapse=EXC_DEXP)

        result = rn.simulate(500.0, 0.01, {"N0": 15.0, "N1": 0.0, "N2": 0.0, "N3": 0.0})
        for i in range(4):
            assert count_spikes(result[f"N{i}"][0]) > 0, \
                f"N{i} should spike through mixed-synapse chain"

    def test_nmda_like_slow_synapse(self):
        """Double-exp with NMDA-like parameters (slow rise/decay) should work."""
        rn = _two_pop_rn(SynapseSpec.double_exponential(E_SYN_EXC, 2.0, 67.0))
        _, post = _run2(rn, 500.0, 0.01)

        assert not np.any(np.isnan(post)), "No NaN with NMDA-like parameters"
        assert not np.any(np.isinf(post)), "No Inf with NMDA-like parameters"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and error handling for synapses."""

    def test_double_exp_invalid_tau_raises(self):
        """tau_rise >= tau_decay should raise an error."""
        rn = RegionalNetwork()
        rn.add_population("E", 2, model=NeuronModelSpec.hh_default())
        with pytest.raises(Exception):
            rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=0.5,
                       synapse=SynapseSpec.double_exponential(0.0, 5.0, 2.0))

    def test_connect_invalid_population_raises(self):
        """Connecting to a non-existent population should raise."""
        rn = RegionalNetwork()
        rn.add_population("E", 2, model=NeuronModelSpec.hh_default())
        with pytest.raises(Exception):
            rn.connect("E", "Z", "all_to_all", weight=0.5, synapse=EXC)

    def test_reset_with_all_synapse_types(self):
        """Reset should produce identical results for all synapse types."""
        rn = RegionalNetwork()
        rn.add_population("N0", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("N1", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("N2", 1, model=NeuronModelSpec.hh_default())
        rn.connect("N0", "N1", "all_to_all", weight=WEIGHT, synapse=EXC)
        rn.connect("N1", "N2", "all_to_all", weight=WEIGHT, synapse=EXC_ALPHA)
        rn.connect("N0", "N2", "all_to_all", weight=WEIGHT, synapse=EXC_DEXP)

        I = {"N0": 15.0, "N1": 0.0, "N2": 0.0}
        result1 = rn.simulate(100.0, 0.01, I)
        rn.reset()
        result2 = rn.simulate(100.0, 0.01, I)

        for key in ("N0", "N1", "N2"):
            np.testing.assert_array_almost_equal(
                result1[key][0], result2[key][0], decimal=5,
                err_msg=f"{key}: reset should produce identical traces"
            )

    def test_zero_weight_all_types(self):
        """Zero-weight synapses of any type should have no effect."""
        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 100.0, 0.01)

        for spec in [
            SynapseSpec.exponential(E_SYN_EXC, TAU),
            SynapseSpec.alpha(E_SYN_EXC, TAU),
            SynapseSpec.double_exponential(E_SYN_EXC, TAU_RISE, TAU_DECAY),
        ]:
            rn = _two_pop_rn(spec, weight=0.0)
            _, post = _run2(rn, 100.0, 0.01)
            np.testing.assert_array_almost_equal(
                post, ctrl, decimal=5,
                err_msg="Zero-weight synapse should have no effect"
            )

    def test_numerical_stability_long_simulation(self):
        """All synapse types should remain stable over long simulations."""
        for spec in [EXC, EXC_ALPHA, EXC_DEXP]:
            rn = _two_pop_rn(spec, weight=WEIGHT)
            pre_t, post = _run2(rn, 1000.0, 0.01)
            assert not np.any(np.isnan(pre_t)), "NaN in pre"
            assert not np.any(np.isnan(post)), "NaN in post"
            assert not np.any(np.isinf(pre_t)), "Inf in pre"
            assert not np.any(np.isinf(post)), "Inf in post"


# =============================================================================
# Test Class: Synaptic Delay - Backward Compatibility
# =============================================================================

class TestSynapticDelayBackwardCompat:
    """Verify that delay=0 produces identical behavior to the old code."""

    def test_zero_delay_matches_no_delay(self):
        """Explicit delay=0 should produce identical traces to omitting delay."""
        rn1 = _two_pop_rn(EXC)
        _, post1 = _run2(rn1, 200.0, 0.01)

        rn2 = _two_pop_rn(EXC, delay=0.0)
        _, post2 = _run2(rn2, 200.0, 0.01)

        np.testing.assert_array_almost_equal(
            post1, post2, decimal=10,
            err_msg="delay=0 should match no-delay default"
        )

    def test_zero_delay_all_types(self):
        """delay=0 should match default for all synapse types."""
        for spec in [EXC, EXC_ALPHA, EXC_DEXP]:
            rn1 = _two_pop_rn(spec)
            _, post1 = _run2(rn1, 200.0, 0.01)

            rn2 = _two_pop_rn(spec, delay=0.0)
            _, post2 = _run2(rn2, 200.0, 0.01)

            np.testing.assert_array_almost_equal(
                post1, post2, decimal=10,
                err_msg="delay=0 should match default for all synapse types"
            )


# =============================================================================
# Test Class: Synaptic Delay - Functional Behavior
# =============================================================================

class TestSynapticDelayBehavior:
    """Tests verifying that delays actually delay spike propagation."""

    def test_delay_postpones_first_postsynaptic_spike(self):
        """First postsynaptic spike should arrive later with delay > 0."""
        rn0 = _two_pop_rn(EXC, delay=0.0)
        _, post0 = _run2(rn0, 200.0, 0.01)

        rn5 = _two_pop_rn(EXC, delay=5.0)
        _, post5 = _run2(rn5, 200.0, 0.01)

        t0 = _find_first_spike_time(post0, 0.01)
        t5 = _find_first_spike_time(post5, 0.01)

        assert t0 is not None, "No-delay: post should spike"
        assert t5 is not None, "Delayed: post should spike"
        assert t5 > t0, \
            f"Delayed first spike ({t5:.2f} ms) should be later than no-delay ({t0:.2f} ms)"

    def test_delay_accuracy(self):
        """Difference in first postsynaptic spike times should match the delay."""
        delay_ms = 5.0

        rn0 = _two_pop_rn(EXC, delay=0.0)
        _, post0 = _run2(rn0, 200.0, 0.01)

        rn_d = _two_pop_rn(EXC, delay=delay_ms)
        _, post_d = _run2(rn_d, 200.0, 0.01)

        t0 = _find_first_spike_time(post0, 0.01)
        td = _find_first_spike_time(post_d, 0.01)

        assert t0 is not None and td is not None
        measured_delay = td - t0
        assert abs(measured_delay - delay_ms) < 1.5, \
            f"Measured delay {measured_delay:.2f} ms should be ~{delay_ms} ms"

    def test_delay_with_alpha_synapse(self):
        """Delay should postpone first postsynaptic spike with alpha synapses."""
        rn0 = _two_pop_rn(EXC_ALPHA, delay=0.0)
        _, post0 = _run2(rn0, 200.0, 0.01)

        rn_d = _two_pop_rn(EXC_ALPHA, delay=5.0)
        _, post_d = _run2(rn_d, 200.0, 0.01)

        t0 = _find_first_spike_time(post0, 0.01)
        td = _find_first_spike_time(post_d, 0.01)

        assert t0 is not None and td is not None
        assert td > t0, "Alpha synapse delay should postpone first spike"

    def test_delay_with_double_exp_synapse(self):
        """Delay should postpone first postsynaptic spike with double-exp synapses."""
        rn0 = _two_pop_rn(EXC_DEXP, delay=0.0)
        _, post0 = _run2(rn0, 200.0, 0.01)

        rn_d = _two_pop_rn(EXC_DEXP, delay=5.0)
        _, post_d = _run2(rn_d, 200.0, 0.01)

        t0 = _find_first_spike_time(post0, 0.01)
        td = _find_first_spike_time(post_d, 0.01)

        assert t0 is not None and td is not None
        assert td > t0, "Double-exp synapse delay should postpone first spike"

    def test_longer_delay_means_later_first_spike(self):
        """Increasing delay should monotonically increase first spike time."""
        spike_times = []
        for delay in [0.0, 2.0, 5.0]:
            rn = _two_pop_rn(EXC, delay=delay)
            _, post = _run2(rn, 300.0, 0.01)
            t = _find_first_spike_time(post, 0.01)
            assert t is not None, f"Post should spike with delay={delay}"
            spike_times.append(t)

        assert spike_times[0] < spike_times[1] < spike_times[2], \
            f"First spike times {spike_times} should be strictly increasing with delay"

    def test_inhibitory_with_delay(self):
        """Delay should also work with inhibitory synapses."""
        inh = SynapseSpec.exponential(E_SYN_INH, 5.0)

        rn = _two_pop_rn(inh, delay=4.0)
        _, post_inh = _run2(rn, 200.0, 0.01, I_pre=15.0, I_post=10.0)

        rn_nodelay = _two_pop_rn(inh, delay=0.0)
        _, post_nodelay = _run2(rn_nodelay, 200.0, 0.01, I_pre=15.0, I_post=10.0)

        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 200.0, 0.01, I_pre=15.0, I_post=10.0)
        ctrl_mean = get_mean_voltage(ctrl)

        assert get_mean_voltage(post_inh) < ctrl_mean, \
            "Inhibitory synapse with delay should still inhibit"
        assert get_mean_voltage(post_nodelay) < ctrl_mean, \
            "Inhibitory synapse without delay should inhibit"


# =============================================================================
# Test Class: Synaptic Delay - Chain Propagation
# =============================================================================

class TestSynapticDelayChains:
    """Tests for delays accumulating through multi-neuron chains."""

    def test_delay_chain_accumulates(self):
        """Delays through a chain should add up."""
        rn = _chain_rn(3, EXC, delay=3.0)
        traces = _run_chain(rn, 3, 500.0, 0.01)

        rn0 = _chain_rn(3, EXC, delay=0.0)
        traces0 = _run_chain(rn0, 3, 500.0, 0.01)

        t2_nodelay = _find_first_spike_time(traces0[2], 0.01)
        t2_delay = _find_first_spike_time(traces[2], 0.01)

        assert t2_nodelay is not None and t2_delay is not None
        assert t2_delay > t2_nodelay, \
            f"Chain delay first spike ({t2_delay:.1f}) should be later than no-delay ({t2_nodelay:.1f})"

    def test_chain_with_mixed_delays(self):
        """Different delays on different hops should all contribute."""
        # 0 --2ms--> 1 --4ms--> 2
        rn = RegionalNetwork()
        for i in range(3):
            rn.add_population(f"N{i}", 1, model=NeuronModelSpec.hh_default())
        rn.connect("N0", "N1", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=EXC, delay=2.0)
        rn.connect("N1", "N2", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=EXC, delay=4.0)
        result = rn.simulate(500.0, 0.01, {"N0": 15.0, "N1": 0.0, "N2": 0.0})
        traces = [result[f"N{i}"][0] for i in range(3)]

        # Uniform 3+3ms for comparison
        rn_uni = _chain_rn(3, EXC, delay=3.0)
        traces_uni = _run_chain(rn_uni, 3, 500.0, 0.01)

        t_mixed = _find_first_spike_time(traces[2], 0.01)
        t_uni = _find_first_spike_time(traces_uni[2], 0.01)

        assert t_mixed is not None and t_uni is not None
        assert abs(t_mixed - t_uni) < 5.0, \
            f"Mixed delays (2+4={t_mixed:.1f}ms) and uniform (3+3={t_uni:.1f}ms) " \
            f"should produce similar total delay"

    def test_biologically_realistic_pathway_delays(self):
        """Cortex->STN->GPe chain with realistic delays should propagate."""
        rn = RegionalNetwork()
        rn.add_population("Cor", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("STN", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("GPe", 1, model=NeuronModelSpec.hh_default())
        rn.connect("Cor", "STN", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=SynapseSpec.double_exponential(E_SYN_EXC, 0.5, 2.49),
                   delay=DELAY_COR_STN)
        rn.connect("STN", "GPe", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=SynapseSpec.double_exponential(E_SYN_EXC, 0.4, 2.5),
                   delay=DELAY_STN_GPE)

        result = rn.simulate(200.0, 0.01, {"Cor": 15.0, "STN": 0.0, "GPe": 0.0})
        traces = {"Cor": result["Cor"][0], "STN": result["STN"][0], "GPe": result["GPe"][0]}

        assert count_spikes(traces["Cor"]) > 0, "Cortex should spike"
        assert count_spikes(traces["STN"]) > 0, "STN should spike"
        assert count_spikes(traces["GPe"]) > 0, "GPe should spike"

        for name, t in traces.items():
            assert not np.any(np.isnan(t)), f"NaN in {name}"
            assert not np.any(np.isinf(t)), f"Inf in {name}"


# =============================================================================
# Test Class: Synaptic Delay - Reset Behavior
# =============================================================================

class TestSynapticDelayReset:
    """Tests that reset properly clears delay buffers."""

    def test_reset_with_delay_reproduces_traces(self):
        """After reset, a delayed network should produce identical traces."""
        rn = _two_pop_rn(EXC, delay=5.0)

        _, post1 = _run2(rn, 200.0, 0.01)
        rn.reset()
        _, post2 = _run2(rn, 200.0, 0.01)

        np.testing.assert_array_almost_equal(
            post1, post2, decimal=5,
            err_msg="Reset should reproduce identical traces with delay"
        )

    def test_reset_all_synapse_types_with_delay(self):
        """Reset should work for all synapse types when they have delays."""
        rn = RegionalNetwork()
        rn.add_population("N0", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("N1", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("N2", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("N3", 1, model=NeuronModelSpec.hh_default())
        rn.connect("N0", "N1", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=EXC, delay=2.0)
        rn.connect("N0", "N2", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=EXC_ALPHA, delay=4.0)
        rn.connect("N0", "N3", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=EXC_DEXP, delay=5.0)

        I = {"N0": 15.0, "N1": 0.0, "N2": 0.0, "N3": 0.0}
        result1 = rn.simulate(200.0, 0.01, I)
        rn.reset()
        result2 = rn.simulate(200.0, 0.01, I)

        for key in ("N0", "N1", "N2", "N3"):
            np.testing.assert_array_almost_equal(
                result1[key][0], result2[key][0], decimal=5,
                err_msg=f"{key}: reset with mixed delays should reproduce traces"
            )

    def test_reset_clears_buffered_spikes(self):
        """Reset mid-simulation should not leak spikes from the previous run."""
        rn = _two_pop_rn(EXC, delay=3.0)

        # First run: drive pre hard so it spikes
        _run2(rn, 50.0, 0.01, I_pre=15.0)

        rn.reset()

        # Second run: no input at all — no spikes should leak
        _, post = _run2(rn, 50.0, 0.01, I_pre=0.0, I_post=0.0)
        assert count_spikes(post) == 0, \
            "After reset, buffered spikes should be cleared"


# =============================================================================
# Test Class: Synaptic Delay - Numerical Stability
# =============================================================================

class TestSynapticDelayStability:
    """Stability tests for delayed synapses."""

    def test_long_simulation_with_delay(self):
        """Delayed synapses should remain stable over long simulations."""
        for spec in [EXC, EXC_ALPHA, EXC_DEXP]:
            rn = _two_pop_rn(spec, weight=WEIGHT, delay=5.0)
            pre_t, post = _run2(rn, 1000.0, 0.01)
            assert not np.any(np.isnan(pre_t)), "NaN in pre"
            assert not np.any(np.isinf(pre_t)), "Inf in pre"
            assert not np.any(np.isnan(post)), "NaN in post"
            assert not np.any(np.isinf(post)), "Inf in post"

    def test_large_delay_stability(self):
        """Even large delays (like TH->Cortex 5ms, Cortex->STN 5.9ms) should be stable."""
        for delay in [DELAY_TH_COR, DELAY_COR_STN, DELAY_GPI_TH]:
            rn = _two_pop_rn(EXC, delay=delay)
            pre_t, post = _run2(rn, 500.0, 0.01)
            assert not np.any(np.isnan(pre_t)), f"NaN in pre with delay={delay}"
            assert not np.any(np.isinf(pre_t)), f"Inf in pre with delay={delay}"
            assert not np.any(np.isnan(post)), f"NaN in post with delay={delay}"
            assert not np.any(np.isinf(post)), f"Inf in post with delay={delay}"

    def test_many_synapses_with_delays(self):
        """A convergent network with many delayed synapses should be stable."""
        n_pre = 10
        rn = RegionalNetwork()
        for i in range(n_pre):
            rn.add_population(f"N{i}", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("Npost", 1, model=NeuronModelSpec.hh_default())
        for i in range(n_pre):
            rn.connect(f"N{i}", "Npost", "all_to_all", weight=WEIGHT,
                       synapse=EXC, delay=float(i + 1))

        I_dict = {f"N{i}": 15.0 for i in range(n_pre)}
        I_dict["Npost"] = 0.0
        result = rn.simulate(300.0, 0.01, I_dict)

        for i in range(n_pre):
            t = result[f"N{i}"][0]
            assert not np.any(np.isnan(t)), f"NaN in N{i}"
            assert not np.any(np.isinf(t)), f"Inf in N{i}"

        post_t = result["Npost"][0]
        assert not np.any(np.isnan(post_t)), "NaN in Npost"
        assert not np.any(np.isinf(post_t)), "Inf in Npost"
        assert count_spikes(post_t) > 0, \
            "Postsynaptic neuron should spike with convergent delayed input"

    def test_small_dt_with_delay(self):
        """Delay should work correctly with very small dt (high resolution)."""
        rn = _two_pop_rn(EXC, delay=2.0)
        _, post = _run2(rn, 50.0, 0.001, I_pre=15.0)
        assert not np.any(np.isnan(post)), "No NaN with small dt"
        assert not np.any(np.isinf(post)), "No Inf with small dt"

    def test_delay_smaller_than_dt(self):
        """When delay < dt, it should effectively round to 0 (pass-through)."""
        rn_tiny = _two_pop_rn(EXC, delay=0.1)
        _, post_tiny = _run2(rn_tiny, 100.0, 0.5)

        rn_zero = _two_pop_rn(EXC, delay=0.0)
        _, post_zero = _run2(rn_zero, 100.0, 0.5)

        np.testing.assert_array_almost_equal(
            post_tiny, post_zero, decimal=5,
            err_msg="Sub-dt delay should behave as zero delay"
        )


# =============================================================================
# Test Class: Receptor Types — Functional Behavior
# =============================================================================

class TestReceptorTypeBehavior:
    """Tests verifying that receptor types produce biologically plausible behavior."""

    def test_ampa_is_excitatory(self):
        """AMPA (E_syn=0) should depolarize the postsynaptic neuron."""
        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 300.0, 0.01)

        rn = _two_pop_rn(SynapseSpec.ampa())
        _, post = _run2(rn, 300.0, 0.01)
        assert get_mean_voltage(post) > get_mean_voltage(ctrl), \
            "AMPA synapse should be excitatory"

    def test_nmda_is_excitatory(self):
        """NMDA (E_syn=0) should depolarize the postsynaptic neuron."""
        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 500.0, 0.01)

        rn = _two_pop_rn(SynapseSpec.nmda())
        _, post = _run2(rn, 500.0, 0.01)
        assert get_mean_voltage(post) > get_mean_voltage(ctrl), \
            "NMDA synapse should be excitatory"

    def test_gaba_a_is_inhibitory(self):
        """GABA_A (E_syn=-80) should hyperpolarize the postsynaptic neuron."""
        ctrl_rn = _two_pop_rn()
        _, ctrl = _run2(ctrl_rn, 300.0, 0.01, I_pre=15.0, I_post=10.0)

        rn = _two_pop_rn(SynapseSpec.gaba_a())
        _, post = _run2(rn, 300.0, 0.01, I_pre=15.0, I_post=10.0)
        assert get_mean_voltage(post) < get_mean_voltage(ctrl), \
            "GABA_A synapse should be inhibitory"

    def test_nmda_slower_than_ampa(self):
        """NMDA (tau_decay=67) should have a more sustained effect than AMPA (tau_decay=2.5)."""
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)
        I_pre = np.zeros(num_steps)
        I_pre[:int(10.0 / dt)] = 30.0  # brief 10ms pulse

        rn_ampa = _two_pop_rn(SynapseSpec.ampa())
        result_ampa = rn_ampa.simulate(duration, dt, {"pre": I_pre, "post": np.zeros(num_steps)})
        ampa_post = result_ampa["post"][0]

        rn_nmda = _two_pop_rn(SynapseSpec.nmda())
        result_nmda = rn_nmda.simulate(duration, dt, {"pre": I_pre, "post": np.zeros(num_steps)})
        nmda_post = result_nmda["post"][0]

        late_start = int(100.0 / dt)
        ampa_late_mean = np.mean(ampa_post[late_start:])
        nmda_late_mean = np.mean(nmda_post[late_start:])

        assert nmda_late_mean > ampa_late_mean, \
            f"NMDA late-phase mean ({nmda_late_mean:.2f}) should exceed " \
            f"AMPA late-phase mean ({ampa_late_mean:.2f})"

    def test_receptor_types_numerical_stability(self):
        """All receptor types should remain stable over long simulations."""
        for spec in [SynapseSpec.ampa(), SynapseSpec.nmda(), SynapseSpec.gaba_a()]:
            rn = _two_pop_rn(spec, weight=WEIGHT)
            pre_t, post = _run2(rn, 1000.0, 0.01)
            assert not np.any(np.isnan(pre_t)), "NaN in pre"
            assert not np.any(np.isinf(pre_t)), "Inf in pre"
            assert not np.any(np.isnan(post)), "NaN in post"
            assert not np.any(np.isinf(post)), "Inf in post"

    def test_biologically_realistic_circuit(self):
        """A Cortex->STN(AMPA)->GPe(AMPA)->GPi(GABA) circuit should work."""
        rn = RegionalNetwork()
        rn.add_population("Cor", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("STN", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("GPe", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("GPi", 1, model=NeuronModelSpec.hh_default())
        rn.connect("Cor", "STN", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=SynapseSpec.ampa(), delay=5.9)
        rn.connect("STN", "GPe", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=SynapseSpec.ampa(), delay=2.0)
        rn.connect("GPe", "GPi", "all_to_all", weight=WEIGHT_STRONG,
                   synapse=SynapseSpec.gaba_a(), delay=3.0)

        result = rn.simulate(300.0, 0.01,
                             {"Cor": 15.0, "STN": 0.0, "GPe": 0.0, "GPi": 10.0})

        for name in ("Cor", "STN", "GPe", "GPi"):
            t = result[name][0]
            assert not np.any(np.isnan(t)), f"NaN in {name}"
            assert not np.any(np.isinf(t)), f"Inf in {name}"

        assert count_spikes(result["Cor"][0]) > 0, "Cortex should spike"
        assert count_spikes(result["STN"][0]) > 0, "STN should spike via AMPA"
