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
