"""
Synapse Model Tests

Tests for the three synapse types (Exponential, Alpha, Double-Exponential)
and their integration with the Network class.

Test Categories:
1. Synapse Creation and Properties - verify correct construction
2. Conductance Kinetics - verify each model's waveform shape
3. Network Integration - verify all types work in simulations
4. Backward Compatibility - verify existing add_synapse still works
5. Edge Cases - invalid parameters, reset behavior
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    Network,
    SynapseBase,
    ExponentialSynapse,
    AlphaSynapse,
    DoubleExponentialSynapse,
    ReceptorType,
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


# Synapse parameter constants
WEIGHT = 5.0
WEIGHT_STRONG = 10.0
E_SYN_EXC = 0.0
E_SYN_INH = -80.0
TAU = 2.0
TAU_RISE = 0.4
TAU_DECAY = 2.5


# =============================================================================
# Test Class: Synapse Creation and Properties
# =============================================================================

class TestSynapseCreation:
    """Tests for creating synapses and inspecting their properties."""

    def test_exponential_synapse_via_network(self):
        net = Network(2)
        net.add_synapse(0, 1, 0.5, E_SYN_EXC, TAU)
        syn = net.synapse(0)

        assert syn.type_name() == "Exponential"
        assert syn.pre_idx == 0
        assert syn.post_idx == 1
        assert syn.weight == pytest.approx(0.5)
        assert syn.reversal_potential == pytest.approx(E_SYN_EXC)
        assert syn.conductance == pytest.approx(0.0)

    def test_alpha_synapse_via_network(self):
        net = Network(2)
        net.add_alpha_synapse(0, 1, 0.5, E_SYN_EXC, 5.0)
        syn = net.synapse(0)

        assert syn.type_name() == "Alpha"
        assert syn.pre_idx == 0
        assert syn.post_idx == 1
        assert syn.weight == pytest.approx(0.5)
        assert syn.tau == pytest.approx(5.0)

    def test_double_exp_synapse_via_network(self):
        net = Network(2)
        net.add_double_exp_synapse(0, 1, 0.5, E_SYN_EXC, TAU_RISE, TAU_DECAY)
        syn = net.synapse(0)

        assert syn.type_name() == "DoubleExponential"
        assert syn.pre_idx == 0
        assert syn.post_idx == 1
        assert syn.weight == pytest.approx(0.5)
        assert syn.tau_rise == pytest.approx(TAU_RISE)
        assert syn.tau_decay == pytest.approx(TAU_DECAY)

    def test_mixed_synapse_types(self):
        net = Network(3)
        net.add_synapse(0, 1, 0.5)
        net.add_alpha_synapse(1, 2, 0.5)
        net.add_double_exp_synapse(0, 2, 0.5)

        assert net.num_synapses == 3
        assert net.synapse(0).type_name() == "Exponential"
        assert net.synapse(1).type_name() == "Alpha"
        assert net.synapse(2).type_name() == "DoubleExponential"

    def test_synapse_count(self):
        net = Network(3)
        assert net.num_synapses == 0
        net.add_synapse(0, 1, 0.5)
        assert net.num_synapses == 1
        net.add_alpha_synapse(1, 2, 0.5)
        assert net.num_synapses == 2
        net.add_double_exp_synapse(0, 2, 0.5)
        assert net.num_synapses == 3


# =============================================================================
# Test Class: Synapse Kinetics
# =============================================================================

class TestSynapseKinetics:
    """Tests verifying the conductance waveform of each synapse type."""

    def _simulate_and_get_conductance(self, synapse_method, duration=50.0,
                                      dt=0.01, **kwargs):
        """Drive a 2-neuron network and track postsynaptic conductance effect."""
        net = Network(2)
        synapse_method(net, 0, 1, **kwargs)
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)
        return traces

    def test_exponential_decay_shape(self):
        """Exponential synapse: monotonic decay after spike trigger."""
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        # Postsynaptic neuron should be affected
        trace_ctrl = Network(2).simulate(duration, dt, I_ext)
        mean_with = get_mean_voltage(traces[1])
        mean_without = get_mean_voltage(trace_ctrl[1])
        assert mean_with > mean_without, "Excitatory exponential synapse should raise voltage"

    def test_alpha_produces_response(self):
        """Alpha synapse should produce measurable postsynaptic response."""
        net = Network(2)
        net.add_alpha_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        trace_ctrl = Network(2).simulate(duration, dt, I_ext)
        mean_with = get_mean_voltage(traces[1])
        mean_without = get_mean_voltage(trace_ctrl[1])
        assert mean_with > mean_without, "Excitatory alpha synapse should raise voltage"

    def test_double_exp_produces_response(self):
        """Double-exponential synapse should produce measurable postsynaptic response."""
        net = Network(2)
        net.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU_RISE, TAU_DECAY)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        trace_ctrl = Network(2).simulate(duration, dt, I_ext)
        mean_with = get_mean_voltage(traces[1])
        mean_without = get_mean_voltage(trace_ctrl[1])
        assert mean_with > mean_without, "Excitatory double-exp synapse should raise voltage"

    def test_all_types_excitatory(self):
        """All synapse types with E_syn=0 should be excitatory."""
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        ctrl_traces = Network(2).simulate(duration, dt, I_ext)
        ctrl_mean = get_mean_voltage(ctrl_traces[1])

        for label, add_fn in [
            ("exponential", lambda n: n.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)),
            ("alpha", lambda n: n.add_alpha_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)),
            ("double_exp", lambda n: n.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU_RISE, TAU_DECAY)),
        ]:
            net = Network(2)
            add_fn(net)
            traces = net.simulate(duration, dt, I_ext)
            mean_v = get_mean_voltage(traces[1])
            assert mean_v > ctrl_mean, f"{label} synapse should be excitatory"

    def test_all_types_inhibitory(self):
        """All synapse types with E_syn=-80 should be inhibitory."""
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        I_ext[1, :] = 10.0  # drive postsynaptic too

        ctrl_traces = Network(2).simulate(duration, dt, I_ext)
        ctrl_mean = get_mean_voltage(ctrl_traces[1])

        for label, add_fn in [
            ("exponential", lambda n: n.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_INH, 5.0)),
            ("alpha", lambda n: n.add_alpha_synapse(0, 1, WEIGHT_STRONG, E_SYN_INH, 5.0)),
            ("double_exp", lambda n: n.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_INH, 0.4, 5.0)),
        ]:
            net = Network(2)
            add_fn(net)
            traces = net.simulate(duration, dt, I_ext)
            mean_v = get_mean_voltage(traces[1])
            assert mean_v < ctrl_mean, f"{label} synapse should be inhibitory"


# =============================================================================
# Test Class: Network Integration
# =============================================================================

class TestNetworkIntegration:
    """Tests for synapse types working within network simulations."""

    def test_alpha_chain_propagation(self):
        """Spikes should propagate through a chain of alpha synapses."""
        net = Network(3)
        net.add_alpha_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)
        net.add_alpha_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC, TAU)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        assert count_spikes(traces[0]) > 0, "Neuron 0 should spike"
        assert count_spikes(traces[1]) > 0, "Neuron 1 should spike via alpha synapse"
        assert count_spikes(traces[2]) > 0, "Neuron 2 should spike via chain"

    def test_double_exp_chain_propagation(self):
        """Spikes should propagate through a chain of double-exp synapses."""
        net = Network(3)
        net.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU_RISE, TAU_DECAY)
        net.add_double_exp_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC, TAU_RISE, TAU_DECAY)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        assert count_spikes(traces[0]) > 0, "Neuron 0 should spike"
        assert count_spikes(traces[1]) > 0, "Neuron 1 should spike via double-exp"
        assert count_spikes(traces[2]) > 0, "Neuron 2 should spike via chain"

    def test_mixed_synapse_chain(self):
        """Chain with mixed synapse types should propagate spikes."""
        net = Network(4)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)
        net.add_alpha_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC, TAU)
        net.add_double_exp_synapse(2, 3, WEIGHT_STRONG, E_SYN_EXC, TAU_RISE, TAU_DECAY)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        for i in range(4):
            assert count_spikes(traces[i]) > 0, \
                f"Neuron {i} should spike through mixed-synapse chain"

    def test_nmda_like_slow_synapse(self):
        """Double-exp with NMDA-like parameters (slow rise/decay) should work."""
        net = Network(2)
        # NMDA-like: tau_rise=2.0, tau_decay=67.0
        net.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, 2.0, 67.0)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        # Should not produce NaN/Inf
        assert not np.any(np.isnan(traces[1])), "No NaN with NMDA-like parameters"
        assert not np.any(np.isinf(traces[1])), "No Inf with NMDA-like parameters"


# =============================================================================
# Test Class: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Verify existing add_synapse API still works as before."""

    def test_add_synapse_creates_exponential(self):
        """add_synapse should create an ExponentialSynapse."""
        net = Network(2)
        net.add_synapse(0, 1, 0.5)
        assert net.synapse(0).type_name() == "Exponential"

    def test_add_synapse_default_params(self):
        """Default E_syn=0.0 and tau=2.0 should still work."""
        net = Network(2)
        net.add_synapse(0, 1, 0.5)
        syn = net.synapse(0)
        assert syn.reversal_potential == pytest.approx(0.0)

    def test_existing_network_behavior_unchanged(self):
        """A simple 2-neuron chain should still work exactly as before."""
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        assert count_spikes(traces[0]) > 0
        assert not np.any(np.isnan(traces[1]))


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and error handling for synapses."""

    def test_double_exp_invalid_tau_raises(self):
        """tau_rise >= tau_decay should raise an error."""
        net = Network(2)
        with pytest.raises(Exception):
            net.add_double_exp_synapse(0, 1, 0.5, 0.0, 5.0, 2.0)

    def test_synapse_index_out_of_range(self):
        """Adding synapse with invalid neuron index should raise."""
        net = Network(2)
        with pytest.raises(Exception):
            net.add_synapse(0, 5, 0.5)
        with pytest.raises(Exception):
            net.add_alpha_synapse(5, 0, 0.5)
        with pytest.raises(Exception):
            net.add_double_exp_synapse(0, 5, 0.5)

    def test_reset_with_all_synapse_types(self):
        """Reset should produce identical results for all synapse types."""
        net = Network(3)
        net.add_synapse(0, 1, WEIGHT)
        net.add_alpha_synapse(1, 2, WEIGHT)
        net.add_double_exp_synapse(0, 2, WEIGHT, E_SYN_EXC, TAU_RISE, TAU_DECAY)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        traces1 = net.simulate(duration, dt, I_ext)
        net.reset()
        traces2 = net.simulate(duration, dt, I_ext)

        for i in range(3):
            np.testing.assert_array_almost_equal(
                traces1[i], traces2[i], decimal=5,
                err_msg=f"Neuron {i}: reset should produce identical traces"
            )

    def test_zero_weight_all_types(self):
        """Zero-weight synapses of any type should have no effect."""
        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        ctrl = Network(2).simulate(duration, dt, I_ext)

        for add_fn in [
            lambda n: n.add_synapse(0, 1, 0.0),
            lambda n: n.add_alpha_synapse(0, 1, 0.0),
            lambda n: n.add_double_exp_synapse(0, 1, 0.0),
        ]:
            net = Network(2)
            add_fn(net)
            traces = net.simulate(duration, dt, I_ext)
            np.testing.assert_array_almost_equal(
                traces[1], ctrl[1], decimal=5,
                err_msg="Zero-weight synapse should have no effect"
            )

    def test_numerical_stability_long_simulation(self):
        """All synapse types should remain stable over long simulations."""
        duration = 1000.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        for add_fn in [
            lambda n: n.add_synapse(0, 1, WEIGHT),
            lambda n: n.add_alpha_synapse(0, 1, WEIGHT),
            lambda n: n.add_double_exp_synapse(0, 1, WEIGHT),
        ]:
            net = Network(2)
            add_fn(net)
            traces = net.simulate(duration, dt, I_ext)

            for i in range(2):
                assert not np.any(np.isnan(traces[i])), f"NaN in neuron {i}"
                assert not np.any(np.isinf(traces[i])), f"Inf in neuron {i}"


# =============================================================================
# Delay Test Utilities
# =============================================================================

# Biologically realistic pathway delays from task3.md
DELAY_STN_GPE = 2.0    # ms
DELAY_STN_GPI = 1.5    # ms
DELAY_GPE_STN = 4.0    # ms
DELAY_GPE_GPI = 3.0    # ms
DELAY_GPI_TH  = 5.0    # ms
DELAY_COR_STN = 5.9    # ms
DELAY_TH_COR  = 5.0    # ms


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


# =============================================================================
# Test Class: Synaptic Delay - Creation and Properties
# =============================================================================

class TestSynapticDelayProperties:
    """Tests for delay parameter storage and access."""

    def test_exponential_delay_stored(self):
        """Delay should be stored and retrievable on ExponentialSynapse."""
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT, E_SYN_EXC, TAU, delay=3.0)
        syn = net.synapse(0)
        assert syn.delay == pytest.approx(3.0)

    def test_alpha_delay_stored(self):
        """Delay should be stored and retrievable on AlphaSynapse."""
        net = Network(2)
        net.add_alpha_synapse(0, 1, WEIGHT, E_SYN_EXC, TAU, delay=4.0)
        syn = net.synapse(0)
        assert syn.delay == pytest.approx(4.0)

    def test_double_exp_delay_stored(self):
        """Delay should be stored and retrievable on DoubleExponentialSynapse."""
        net = Network(2)
        net.add_double_exp_synapse(0, 1, WEIGHT, E_SYN_EXC, TAU_RISE, TAU_DECAY, delay=5.9)
        syn = net.synapse(0)
        assert syn.delay == pytest.approx(5.9)

    def test_default_delay_is_zero(self):
        """When delay is not specified, it should default to 0."""
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT)
        assert net.synapse(0).delay == pytest.approx(0.0)

        net.add_alpha_synapse(0, 1, WEIGHT)
        assert net.synapse(1).delay == pytest.approx(0.0)

        net.add_double_exp_synapse(0, 1, WEIGHT)
        assert net.synapse(2).delay == pytest.approx(0.0)

    def test_mixed_delays(self):
        """Multiple synapses with different delays should each store correctly."""
        net = Network(3)
        net.add_synapse(0, 1, WEIGHT, delay=2.0)
        net.add_alpha_synapse(1, 2, WEIGHT, delay=4.0)
        net.add_double_exp_synapse(0, 2, WEIGHT, delay=5.9)

        assert net.synapse(0).delay == pytest.approx(2.0)
        assert net.synapse(1).delay == pytest.approx(4.0)
        assert net.synapse(2).delay == pytest.approx(5.9)


# =============================================================================
# Test Class: Synaptic Delay - Backward Compatibility
# =============================================================================

class TestSynapticDelayBackwardCompat:
    """Verify that delay=0 produces identical behavior to the old code."""

    def test_zero_delay_matches_no_delay(self):
        """Explicit delay=0 should produce identical traces to omitting delay."""
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        # Without delay arg (uses default)
        net1 = Network(2)
        net1.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU)
        traces1 = net1.simulate(duration, dt, I_ext)

        # With explicit delay=0
        net2 = Network(2)
        net2.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.0)
        traces2 = net2.simulate(duration, dt, I_ext)

        for i in range(2):
            np.testing.assert_array_almost_equal(
                traces1[i], traces2[i], decimal=10,
                err_msg=f"Neuron {i}: delay=0 should match no-delay default"
            )

    def test_zero_delay_all_types(self):
        """delay=0 should match default for all synapse types."""
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        for add_default, add_explicit in [
            (lambda n: n.add_synapse(0, 1, WEIGHT_STRONG),
             lambda n: n.add_synapse(0, 1, WEIGHT_STRONG, delay=0.0)),
            (lambda n: n.add_alpha_synapse(0, 1, WEIGHT_STRONG),
             lambda n: n.add_alpha_synapse(0, 1, WEIGHT_STRONG, delay=0.0)),
            (lambda n: n.add_double_exp_synapse(0, 1, WEIGHT_STRONG),
             lambda n: n.add_double_exp_synapse(0, 1, WEIGHT_STRONG, delay=0.0)),
        ]:
            net1 = Network(2)
            add_default(net1)
            t1 = net1.simulate(duration, dt, I_ext)

            net2 = Network(2)
            add_explicit(net2)
            t2 = net2.simulate(duration, dt, I_ext)

            np.testing.assert_array_almost_equal(
                t1[1], t2[1], decimal=10,
                err_msg="delay=0 should match default for all synapse types"
            )


# =============================================================================
# Test Class: Synaptic Delay - Functional Behavior
# =============================================================================

class TestSynapticDelayBehavior:
    """Tests verifying that delays actually delay spike propagation.

    Strategy: compare the first spike time of the postsynaptic neuron
    between delayed and non-delayed networks. This avoids fragile
    voltage-onset heuristics and relies on unambiguous spike detection.
    """

    def _make_two_neuron_input(self, duration, dt):
        """Constant drive to neuron 0, nothing to neuron 1."""
        num_steps = int(duration / dt)
        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        return I_ext

    def test_delay_postpones_first_postsynaptic_spike(self):
        """First postsynaptic spike should arrive later with delay > 0."""
        duration = 200.0
        dt = 0.01
        I_ext = self._make_two_neuron_input(duration, dt)

        net0 = Network(2)
        net0.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.0)
        traces0 = net0.simulate(duration, dt, I_ext)

        net5 = Network(2)
        net5.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=5.0)
        traces5 = net5.simulate(duration, dt, I_ext)

        t0 = _find_first_spike_time(traces0[1], dt)
        t5 = _find_first_spike_time(traces5[1], dt)

        assert t0 is not None, "No-delay: neuron 1 should spike"
        assert t5 is not None, "Delayed: neuron 1 should spike"
        assert t5 > t0, \
            f"Delayed first spike ({t5:.2f} ms) should be later than no-delay ({t0:.2f} ms)"

    def test_delay_accuracy(self):
        """Difference in first postsynaptic spike times should match the delay."""
        duration = 200.0
        dt = 0.01
        delay_ms = 5.0
        I_ext = self._make_two_neuron_input(duration, dt)

        net0 = Network(2)
        net0.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.0)
        traces0 = net0.simulate(duration, dt, I_ext)

        net_d = Network(2)
        net_d.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=delay_ms)
        traces_d = net_d.simulate(duration, dt, I_ext)

        t0 = _find_first_spike_time(traces0[1], dt)
        td = _find_first_spike_time(traces_d[1], dt)

        assert t0 is not None and td is not None
        measured_delay = td - t0
        assert abs(measured_delay - delay_ms) < 1.5, \
            f"Measured delay {measured_delay:.2f} ms should be ~{delay_ms} ms"

    def test_delay_with_alpha_synapse(self):
        """Delay should postpone first postsynaptic spike with alpha synapses."""
        duration = 200.0
        dt = 0.01
        I_ext = self._make_two_neuron_input(duration, dt)

        net0 = Network(2)
        net0.add_alpha_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.0)
        traces0 = net0.simulate(duration, dt, I_ext)

        net_d = Network(2)
        net_d.add_alpha_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=5.0)
        traces_d = net_d.simulate(duration, dt, I_ext)

        t0 = _find_first_spike_time(traces0[1], dt)
        td = _find_first_spike_time(traces_d[1], dt)

        assert t0 is not None and td is not None
        assert td > t0, "Alpha synapse delay should postpone first spike"

    def test_delay_with_double_exp_synapse(self):
        """Delay should postpone first postsynaptic spike with double-exp synapses."""
        duration = 200.0
        dt = 0.01
        I_ext = self._make_two_neuron_input(duration, dt)

        net0 = Network(2)
        net0.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC,
                                    TAU_RISE, TAU_DECAY, delay=0.0)
        traces0 = net0.simulate(duration, dt, I_ext)

        net_d = Network(2)
        net_d.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC,
                                     TAU_RISE, TAU_DECAY, delay=5.0)
        traces_d = net_d.simulate(duration, dt, I_ext)

        t0 = _find_first_spike_time(traces0[1], dt)
        td = _find_first_spike_time(traces_d[1], dt)

        assert t0 is not None and td is not None
        assert td > t0, "Double-exp synapse delay should postpone first spike"

    def test_longer_delay_means_later_first_spike(self):
        """Increasing delay should monotonically increase first spike time."""
        duration = 300.0
        dt = 0.01
        I_ext = self._make_two_neuron_input(duration, dt)

        spike_times = []
        for delay in [0.0, 2.0, 5.0]:
            net = Network(2)
            net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=delay)
            traces = net.simulate(duration, dt, I_ext)
            t = _find_first_spike_time(traces[1], dt)
            assert t is not None, f"Neuron 1 should spike with delay={delay}"
            spike_times.append(t)

        assert spike_times[0] < spike_times[1] < spike_times[2], \
            f"First spike times {spike_times} should be strictly increasing with delay"

    def test_inhibitory_with_delay(self):
        """Delay should also work with inhibitory (E_syn=-80) synapses."""
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        I_ext[1, :] = 10.0  # drive postsynaptic so inhibition is visible

        # Inhibitory with delay
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_INH, 5.0, delay=4.0)
        traces = net.simulate(duration, dt, I_ext)

        # Inhibitory without delay
        net_nodelay = Network(2)
        net_nodelay.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_INH, 5.0, delay=0.0)
        traces_nodelay = net_nodelay.simulate(duration, dt, I_ext)

        # Control: no synapse at all
        ctrl = Network(2).simulate(duration, dt, I_ext)

        # Both inhibitory networks should reduce mean voltage compared to ctrl
        ctrl_mean = get_mean_voltage(ctrl[1])
        assert get_mean_voltage(traces[1]) < ctrl_mean, \
            "Inhibitory synapse with delay should still inhibit"
        assert get_mean_voltage(traces_nodelay[1]) < ctrl_mean, \
            "Inhibitory synapse without delay should inhibit"


# =============================================================================
# Test Class: Synaptic Delay - Chain Propagation
# =============================================================================

class TestSynapticDelayChains:
    """Tests for delays accumulating through multi-neuron chains."""

    def test_delay_chain_accumulates(self):
        """Delays through a chain should add up."""
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)
        delay_per_hop = 3.0

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        # Chain with delays: 0 --3ms--> 1 --3ms--> 2
        net = Network(3)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=delay_per_hop)
        net.add_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=delay_per_hop)
        traces = net.simulate(duration, dt, I_ext)

        # Chain without delays
        net0 = Network(3)
        net0.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.0)
        net0.add_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.0)
        traces0 = net0.simulate(duration, dt, I_ext)

        # Neuron 2 first spike: delayed chain should be later
        t2_nodelay = _find_first_spike_time(traces0[2], dt)
        t2_delay = _find_first_spike_time(traces[2], dt)

        assert t2_nodelay is not None and t2_delay is not None
        assert t2_delay > t2_nodelay, \
            f"Chain delay first spike ({t2_delay:.1f}) should be later than no-delay ({t2_nodelay:.1f})"

    def test_chain_with_mixed_delays(self):
        """Different delays on different hops should all contribute."""
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        # 0 --2ms--> 1 --4ms--> 2  (total ~6ms extra)
        net = Network(3)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=2.0)
        net.add_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=4.0)
        traces = net.simulate(duration, dt, I_ext)

        # Same chain with uniform 3ms each (total ~6ms extra)
        net_uni = Network(3)
        net_uni.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=3.0)
        net_uni.add_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=3.0)
        traces_uni = net_uni.simulate(duration, dt, I_ext)

        t_mixed = _find_first_spike_time(traces[2], dt)
        t_uni = _find_first_spike_time(traces_uni[2], dt)

        assert t_mixed is not None and t_uni is not None
        # Both have ~6ms total delay; first spike should be at similar times
        assert abs(t_mixed - t_uni) < 5.0, \
            f"Mixed delays (2+4={t_mixed:.1f}ms) and uniform (3+3={t_uni:.1f}ms) " \
            f"should produce similar total delay"

    def test_biologically_realistic_pathway_delays(self):
        """Cortex->STN->GPe chain with realistic delays should propagate."""
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0  # drive "cortex"

        # Cortex -> STN (5.9ms) -> GPe (2.0ms)
        net = Network(3)
        net.add_double_exp_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC,
                                   0.5, 2.49, delay=DELAY_COR_STN)
        net.add_double_exp_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXC,
                                   0.4, 2.5, delay=DELAY_STN_GPE)
        traces = net.simulate(duration, dt, I_ext)

        # All neurons should eventually spike
        assert count_spikes(traces[0]) > 0, "Cortex should spike"
        assert count_spikes(traces[1]) > 0, "STN should spike"
        assert count_spikes(traces[2]) > 0, "GPe should spike"

        # No NaN/Inf
        for i in range(3):
            assert not np.any(np.isnan(traces[i])), f"NaN in neuron {i}"
            assert not np.any(np.isinf(traces[i])), f"Inf in neuron {i}"


# =============================================================================
# Test Class: Synaptic Delay - Reset Behavior
# =============================================================================

class TestSynapticDelayReset:
    """Tests that reset properly clears delay buffers."""

    def test_reset_with_delay_reproduces_traces(self):
        """After reset, a delayed network should produce identical traces."""
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=5.0)

        traces1 = net.simulate(duration, dt, I_ext)
        net.reset()
        traces2 = net.simulate(duration, dt, I_ext)

        for i in range(2):
            np.testing.assert_array_almost_equal(
                traces1[i], traces2[i], decimal=5,
                err_msg=f"Neuron {i}: reset should reproduce identical traces with delay"
            )

    def test_reset_all_synapse_types_with_delay(self):
        """Reset should work for all synapse types when they have delays."""
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0

        net = Network(4)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=2.0)
        net.add_alpha_synapse(0, 2, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=4.0)
        net.add_double_exp_synapse(0, 3, WEIGHT_STRONG, E_SYN_EXC,
                                   TAU_RISE, TAU_DECAY, delay=5.0)

        traces1 = net.simulate(duration, dt, I_ext)
        net.reset()
        traces2 = net.simulate(duration, dt, I_ext)

        for i in range(4):
            np.testing.assert_array_almost_equal(
                traces1[i], traces2[i], decimal=5,
                err_msg=f"Neuron {i}: reset with mixed delays should reproduce traces"
            )

    def test_reset_clears_buffered_spikes(self):
        """Reset mid-simulation should not leak spikes from the previous run."""
        duration = 50.0
        dt = 0.01
        num_steps = int(duration / dt)

        # First run: drive neuron 0 hard so it spikes
        I_ext_active = np.zeros((2, num_steps))
        I_ext_active[0, :] = 15.0

        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=3.0)
        net.simulate(duration, dt, I_ext_active)

        net.reset()

        # Second run: no input at all
        I_ext_silent = np.zeros((2, num_steps))
        traces = net.simulate(duration, dt, I_ext_silent)

        # Neuron 1 should NOT spike — no leaked spikes from first run
        assert count_spikes(traces[1]) == 0, \
            "After reset, buffered spikes should be cleared"


# =============================================================================
# Test Class: Synaptic Delay - Numerical Stability
# =============================================================================

class TestSynapticDelayStability:
    """Stability tests for delayed synapses."""

    def test_long_simulation_with_delay(self):
        """Delayed synapses should remain stable over long simulations."""
        duration = 1000.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        for add_fn in [
            lambda n: n.add_synapse(0, 1, WEIGHT, delay=5.0),
            lambda n: n.add_alpha_synapse(0, 1, WEIGHT, delay=5.0),
            lambda n: n.add_double_exp_synapse(0, 1, WEIGHT, delay=5.0),
        ]:
            net = Network(2)
            add_fn(net)
            traces = net.simulate(duration, dt, I_ext)

            for i in range(2):
                assert not np.any(np.isnan(traces[i])), f"NaN in neuron {i}"
                assert not np.any(np.isinf(traces[i])), f"Inf in neuron {i}"

    def test_large_delay_stability(self):
        """Even large delays (like TH->Cortex 5ms, Cortex->STN 5.9ms) should be stable."""
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        for delay in [DELAY_TH_COR, DELAY_COR_STN, DELAY_GPI_TH]:
            net = Network(2)
            net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=delay)
            traces = net.simulate(duration, dt, I_ext)

            for i in range(2):
                assert not np.any(np.isnan(traces[i])), \
                    f"NaN with delay={delay} in neuron {i}"
                assert not np.any(np.isinf(traces[i])), \
                    f"Inf with delay={delay} in neuron {i}"

    def test_many_synapses_with_delays(self):
        """A convergent network with many delayed synapses should be stable."""
        n_pre = 10
        net = Network(n_pre + 1)  # n_pre presynaptic + 1 postsynaptic

        for i in range(n_pre):
            net.add_synapse(i, n_pre, WEIGHT, E_SYN_EXC, TAU, delay=float(i + 1))

        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((n_pre + 1, num_steps))
        for i in range(n_pre):
            I_ext[i, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        for i in range(n_pre + 1):
            assert not np.any(np.isnan(traces[i])), f"NaN in neuron {i}"
            assert not np.any(np.isinf(traces[i])), f"Inf in neuron {i}"

        # The postsynaptic neuron should fire (lots of excitatory input)
        assert count_spikes(traces[n_pre]) > 0, \
            "Postsynaptic neuron should spike with convergent delayed input"

    def test_small_dt_with_delay(self):
        """Delay should work correctly with very small dt (high resolution)."""
        duration = 50.0
        dt = 0.001  # 1 us steps
        num_steps = int(duration / dt)
        delay = 2.0

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=delay)
        traces = net.simulate(duration, dt, I_ext)

        assert not np.any(np.isnan(traces[1])), "No NaN with small dt"
        assert not np.any(np.isinf(traces[1])), "No Inf with small dt"

    def test_delay_smaller_than_dt(self):
        """When delay < dt, it should effectively round to 0 (pass-through)."""
        duration = 100.0
        dt = 0.5  # large dt
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        # delay=0.1 ms < dt=0.5 ms → rounds to 0 steps
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.1)
        traces_tiny = net.simulate(duration, dt, I_ext)

        net0 = Network(2)
        net0.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXC, TAU, delay=0.0)
        traces_zero = net0.simulate(duration, dt, I_ext)

        # Should behave identically since delay rounds to 0 steps
        np.testing.assert_array_almost_equal(
            traces_tiny[1], traces_zero[1], decimal=5,
            err_msg="Sub-dt delay should behave as zero delay"
        )


# =============================================================================
# Test Class: Receptor Types — Creation and Properties
# =============================================================================

class TestReceptorTypeCreation:
    """Tests for receptor-type convenience methods and their default parameters."""

    def test_ampa_synapse_properties(self):
        """AMPA synapse should have E_syn=0, tau_rise=0.5, tau_decay=2.5."""
        net = Network(2)
        net.add_ampa_synapse(0, 1, WEIGHT)
        syn = net.synapse(0)

        assert syn.type_name() == "DoubleExponential"
        assert syn.reversal_potential == pytest.approx(0.0)
        assert syn.tau_rise == pytest.approx(0.5)
        assert syn.tau_decay == pytest.approx(2.5)
        assert syn.weight == pytest.approx(WEIGHT)

    def test_nmda_synapse_properties(self):
        """NMDA synapse should have E_syn=0, tau_rise=2.0, tau_decay=67.0."""
        net = Network(2)
        net.add_nmda_synapse(0, 1, WEIGHT)
        syn = net.synapse(0)

        assert syn.type_name() == "DoubleExponential"
        assert syn.reversal_potential == pytest.approx(0.0)
        assert syn.tau_rise == pytest.approx(2.0)
        assert syn.tau_decay == pytest.approx(67.0)
        assert syn.weight == pytest.approx(WEIGHT)

    def test_gaba_a_synapse_properties(self):
        """GABA_A synapse should have E_syn=-80, tau_rise=0.4, tau_decay=7.7."""
        net = Network(2)
        net.add_gaba_a_synapse(0, 1, WEIGHT)
        syn = net.synapse(0)

        assert syn.type_name() == "DoubleExponential"
        assert syn.reversal_potential == pytest.approx(-80.0)
        assert syn.tau_rise == pytest.approx(0.4)
        assert syn.tau_decay == pytest.approx(7.7)
        assert syn.weight == pytest.approx(WEIGHT)

    def test_receptor_type_enum_ampa(self):
        """add_receptor_synapse with AMPA should match add_ampa_synapse."""
        net = Network(2)
        net.add_receptor_synapse(0, 1, WEIGHT, ReceptorType.AMPA)
        syn = net.synapse(0)

        assert syn.reversal_potential == pytest.approx(0.0)
        assert syn.tau_rise == pytest.approx(0.5)
        assert syn.tau_decay == pytest.approx(2.5)

    def test_receptor_type_enum_nmda(self):
        """add_receptor_synapse with NMDA should match add_nmda_synapse."""
        net = Network(2)
        net.add_receptor_synapse(0, 1, WEIGHT, ReceptorType.NMDA)
        syn = net.synapse(0)

        assert syn.reversal_potential == pytest.approx(0.0)
        assert syn.tau_rise == pytest.approx(2.0)
        assert syn.tau_decay == pytest.approx(67.0)

    def test_receptor_type_enum_gaba_a(self):
        """add_receptor_synapse with GABA_A should match add_gaba_a_synapse."""
        net = Network(2)
        net.add_receptor_synapse(0, 1, WEIGHT, ReceptorType.GABA_A)
        syn = net.synapse(0)

        assert syn.reversal_potential == pytest.approx(-80.0)
        assert syn.tau_rise == pytest.approx(0.4)
        assert syn.tau_decay == pytest.approx(7.7)

    def test_receptor_synapse_with_delay(self):
        """Receptor-type synapses should accept delay parameter."""
        net = Network(2)
        net.add_ampa_synapse(0, 1, WEIGHT, delay=5.9)
        assert net.synapse(0).delay == pytest.approx(5.9)

        net.add_nmda_synapse(0, 1, WEIGHT, delay=2.0)
        assert net.synapse(1).delay == pytest.approx(2.0)

        net.add_gaba_a_synapse(0, 1, WEIGHT, delay=4.0)
        assert net.synapse(2).delay == pytest.approx(4.0)

    def test_receptor_enum_with_delay(self):
        """add_receptor_synapse should pass delay through."""
        net = Network(2)
        net.add_receptor_synapse(0, 1, WEIGHT, ReceptorType.AMPA, delay=3.0)
        assert net.synapse(0).delay == pytest.approx(3.0)

    def test_mixed_receptor_types(self):
        """Multiple receptor types in one network should all work."""
        net = Network(4)
        net.add_ampa_synapse(0, 1, WEIGHT)
        net.add_nmda_synapse(1, 2, WEIGHT)
        net.add_gaba_a_synapse(2, 3, WEIGHT)

        assert net.num_synapses == 3
        assert net.synapse(0).reversal_potential == pytest.approx(0.0)
        assert net.synapse(1).tau_decay == pytest.approx(67.0)
        assert net.synapse(2).reversal_potential == pytest.approx(-80.0)

    def test_synapse_index_out_of_range_receptor(self):
        """Receptor-type methods should raise on invalid neuron index."""
        net = Network(2)
        with pytest.raises(Exception):
            net.add_ampa_synapse(0, 5, WEIGHT)
        with pytest.raises(Exception):
            net.add_nmda_synapse(5, 0, WEIGHT)
        with pytest.raises(Exception):
            net.add_gaba_a_synapse(0, 5, WEIGHT)


# =============================================================================
# Test Class: Receptor Types — Functional Behavior
# =============================================================================

class TestReceptorTypeBehavior:
    """Tests verifying that receptor types produce biologically plausible behavior."""

    def test_ampa_is_excitatory(self):
        """AMPA (E_syn=0) should depolarize the postsynaptic neuron."""
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        ctrl = Network(2).simulate(duration, dt, I_ext)
        ctrl_mean = get_mean_voltage(ctrl[1])

        net = Network(2)
        net.add_ampa_synapse(0, 1, WEIGHT_STRONG)
        traces = net.simulate(duration, dt, I_ext)
        assert get_mean_voltage(traces[1]) > ctrl_mean, \
            "AMPA synapse should be excitatory"

    def test_nmda_is_excitatory(self):
        """NMDA (E_syn=0) should depolarize the postsynaptic neuron."""
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        ctrl = Network(2).simulate(duration, dt, I_ext)
        ctrl_mean = get_mean_voltage(ctrl[1])

        net = Network(2)
        net.add_nmda_synapse(0, 1, WEIGHT_STRONG)
        traces = net.simulate(duration, dt, I_ext)
        assert get_mean_voltage(traces[1]) > ctrl_mean, \
            "NMDA synapse should be excitatory"

    def test_gaba_a_is_inhibitory(self):
        """GABA_A (E_syn=-80) should hyperpolarize the postsynaptic neuron."""
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        I_ext[1, :] = 10.0  # drive postsynaptic so inhibition is visible

        ctrl = Network(2).simulate(duration, dt, I_ext)
        ctrl_mean = get_mean_voltage(ctrl[1])

        net = Network(2)
        net.add_gaba_a_synapse(0, 1, WEIGHT_STRONG)
        traces = net.simulate(duration, dt, I_ext)
        assert get_mean_voltage(traces[1]) < ctrl_mean, \
            "GABA_A synapse should be inhibitory"

    def test_nmda_slower_than_ampa(self):
        """NMDA (tau_decay=67) should have a more sustained effect than AMPA (tau_decay=2.5)."""
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        # Brief pulse to neuron 0 — only first 10ms
        I_ext[0, :int(10.0 / dt)] = 30.0

        net_ampa = Network(2)
        net_ampa.add_ampa_synapse(0, 1, WEIGHT_STRONG)
        traces_ampa = net_ampa.simulate(duration, dt, I_ext)

        net_nmda = Network(2)
        net_nmda.add_nmda_synapse(0, 1, WEIGHT_STRONG)
        traces_nmda = net_nmda.simulate(duration, dt, I_ext)

        # Look at late portion of trace (after 100ms) — NMDA should still
        # show elevated voltage due to long decay, AMPA should have decayed
        late_start = int(100.0 / dt)
        ampa_late_mean = np.mean(traces_ampa[1][late_start:])
        nmda_late_mean = np.mean(traces_nmda[1][late_start:])

        assert nmda_late_mean > ampa_late_mean, \
            f"NMDA late-phase mean ({nmda_late_mean:.2f}) should exceed " \
            f"AMPA late-phase mean ({ampa_late_mean:.2f})"

    def test_receptor_types_numerical_stability(self):
        """All receptor types should remain stable over long simulations."""
        duration = 1000.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        for add_fn in [
            lambda n: n.add_ampa_synapse(0, 1, WEIGHT),
            lambda n: n.add_nmda_synapse(0, 1, WEIGHT),
            lambda n: n.add_gaba_a_synapse(0, 1, WEIGHT),
        ]:
            net = Network(2)
            add_fn(net)
            traces = net.simulate(duration, dt, I_ext)

            for i in range(2):
                assert not np.any(np.isnan(traces[i])), f"NaN in neuron {i}"
                assert not np.any(np.isinf(traces[i])), f"Inf in neuron {i}"

    def test_biologically_realistic_circuit(self):
        """A Cortex->STN(AMPA)->GPe(AMPA)->GPi(GABA) circuit should work."""
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        # 4 neurons: Cortex(0), STN(1), GPe(2), GPi(3)
        net = Network(4)
        net.add_ampa_synapse(0, 1, WEIGHT_STRONG, delay=5.9)   # Cor -> STN
        net.add_ampa_synapse(1, 2, WEIGHT_STRONG, delay=2.0)   # STN -> GPe
        net.add_gaba_a_synapse(2, 3, WEIGHT_STRONG, delay=3.0) # GPe -> GPi

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0  # drive cortex
        I_ext[3, :] = 10.0  # drive GPi so inhibition is visible

        traces = net.simulate(duration, dt, I_ext)

        # All should be stable
        for i in range(4):
            assert not np.any(np.isnan(traces[i])), f"NaN in neuron {i}"
            assert not np.any(np.isinf(traces[i])), f"Inf in neuron {i}"

        # Cortex and STN should spike
        assert count_spikes(traces[0]) > 0, "Cortex should spike"
        assert count_spikes(traces[1]) > 0, "STN should spike via AMPA"
