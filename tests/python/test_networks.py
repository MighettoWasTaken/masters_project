"""
Comprehensive Network Verification Tests

This module provides rigorous testing of neural network behavior, verifying that
networks of neurons connected via synapses exhibit correct emergent behaviors.

Test Categories:
1. Basic Synaptic Transmission - verify spikes propagate through synapses
2. Excitatory vs Inhibitory - verify E_syn polarity effects
3. Weight Effects - verify synaptic strength scaling
4. Network Topologies - verify behavior in various network structures
5. Timing and Causality - verify spike timing relationships
6. Network Dynamics - verify emergent network behaviors
7. Mixed Neuron Types - verify HH and Izhikevich work together
8. Edge Cases and Stability - verify robustness

These tests serve as the primary verification that the simulation produces
biologically plausible and mathematically correct network dynamics.
"""

import numpy as np
import pytest
from typing import List, Tuple, Optional
from dataclasses import dataclass

from hodgkin_huxley import (
    HHNeuron,
    Network,
    HHParameters,
    IntegrationMethod,
    IzhikevichNeuron,
    IzhikevichType,
    IzhikevichParameters,
    NetworkNeuronType,
)


# =============================================================================
# Test Utilities
# =============================================================================

def count_spikes(trace: np.ndarray, threshold: float = 0.0) -> int:
    """Count spikes using upward threshold crossings."""
    above = trace > threshold
    crossings = np.diff(above.astype(int))
    return int(np.sum(crossings == 1))


def get_spike_times(trace: np.ndarray, dt: float, threshold: float = 0.0) -> np.ndarray:
    """Get spike times in ms using upward threshold crossings."""
    above = trace > threshold
    crossings = np.diff(above.astype(int))
    spike_indices = np.where(crossings == 1)[0]
    return spike_indices * dt


def get_firing_rate(trace: np.ndarray, duration: float, threshold: float = 0.0) -> float:
    """Calculate firing rate in Hz."""
    spikes = count_spikes(trace, threshold)
    return spikes / (duration / 1000.0)  # Convert ms to s


def get_mean_voltage(trace: np.ndarray) -> float:
    """Get mean membrane voltage."""
    return float(np.mean(trace))


def spike_follows(
    pre_trace: np.ndarray,
    post_trace: np.ndarray,
    dt: float,
    max_delay: float = 20.0,
    threshold: float = 0.0
) -> bool:
    """
    Check if postsynaptic spikes follow presynaptic spikes within max_delay.
    Returns True if at least one postsynaptic spike follows a presynaptic spike.
    """
    pre_times = get_spike_times(pre_trace, dt, threshold)
    post_times = get_spike_times(post_trace, dt, threshold)

    if len(pre_times) == 0 or len(post_times) == 0:
        return False

    for pre_t in pre_times:
        for post_t in post_times:
            delay = post_t - pre_t
            if 0 < delay <= max_delay:
                return True
    return False


# =============================================================================
# Synapse Parameter Constants
# =============================================================================

# Standard excitatory synapse (AMPA-like)
E_SYN_EXCITATORY = 0.0      # mV - reversal potential for excitation
TAU_EXCITATORY = 2.0        # ms - fast excitatory decay

# Standard inhibitory synapse (GABA-A-like)
E_SYN_INHIBITORY = -80.0    # mV - reversal potential for inhibition
TAU_INHIBITORY = 5.0        # ms - slower inhibitory decay

# Synaptic weights - adjusted for stronger effects
WEIGHT_WEAK = 0.5
WEIGHT_MODERATE = 2.0
WEIGHT_STRONG = 5.0
WEIGHT_VERY_STRONG = 10.0


# =============================================================================
# Test Class: Basic Synaptic Transmission
# =============================================================================

class TestBasicSynapticTransmission:
    """Tests for basic spike propagation through synapses."""

    def test_single_excitatory_synapse_causes_response(self):
        """
        A single excitatory synapse should cause measurable postsynaptic response.

        Network: [0] --exc--> [1]
        Drive neuron 0, verify neuron 1 responds.
        """
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        # Drive neuron 0
        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        # Neuron 0 should spike
        spikes_0 = count_spikes(traces[0])
        assert spikes_0 > 0, "Presynaptic neuron should spike"

        # Compare to control without synapse
        net_ctrl = Network(2)
        traces_ctrl = net_ctrl.simulate(duration, dt, I_ext)

        # Postsynaptic neuron should have higher mean voltage with synapse
        mean_with_synapse = get_mean_voltage(traces[1])
        mean_without_synapse = get_mean_voltage(traces_ctrl[1])

        assert mean_with_synapse > mean_without_synapse, \
            f"Excitatory synapse should increase postsynaptic voltage ({mean_with_synapse:.2f} vs {mean_without_synapse:.2f})"

    def test_excitatory_synapse_can_trigger_spike(self):
        """
        A strong excitatory synapse should be able to trigger postsynaptic spikes.
        """
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        spikes_0 = count_spikes(traces[0])
        spikes_1 = count_spikes(traces[1])

        assert spikes_0 > 0, "Presynaptic neuron should spike"
        assert spikes_1 > 0, "Strong excitatory synapse should trigger postsynaptic spikes"

    def test_inhibitory_synapse_reduces_activity(self):
        """
        An inhibitory synapse should reduce postsynaptic firing.
        """
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        # With inhibition
        net_inh = Network(2)
        net_inh.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        I_ext[1, :] = 10.0  # Drive both, inhibition should reduce neuron 1

        traces_inh = net_inh.simulate(duration, dt, I_ext)

        # Without inhibition
        net_ctrl = Network(2)
        traces_ctrl = net_ctrl.simulate(duration, dt, I_ext)

        rate_with_inh = get_firing_rate(traces_inh[1], duration)
        rate_without_inh = get_firing_rate(traces_ctrl[1], duration)

        assert rate_with_inh < rate_without_inh, \
            f"Inhibition should reduce firing rate ({rate_with_inh:.1f} Hz vs {rate_without_inh:.1f} Hz)"

    def test_no_synapse_no_interaction(self):
        """
        Without synaptic connections, neurons should be independent.
        """
        net = Network(2)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        spikes_0 = count_spikes(traces[0])
        spikes_1 = count_spikes(traces[1])

        assert spikes_0 > 0, "Driven neuron should spike"
        assert spikes_1 == 0, "Unconnected, undriven neuron should not spike"


# =============================================================================
# Test Class: Synaptic Weight Effects
# =============================================================================

class TestSynapticWeightEffects:
    """Tests verifying that synaptic weights scale effects appropriately."""

    def test_stronger_weight_stronger_response(self):
        """
        Stronger synaptic weights should produce stronger postsynaptic responses.
        """
        weights = [WEIGHT_WEAK, WEIGHT_MODERATE, WEIGHT_STRONG]
        mean_voltages = []

        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        for weight in weights:
            net = Network(2)
            net.add_synapse(0, 1, weight, E_SYN_EXCITATORY, TAU_EXCITATORY)

            I_ext = np.zeros((2, num_steps))
            I_ext[0, :] = 15.0

            traces = net.simulate(duration, dt, I_ext)
            mean_voltages.append(get_mean_voltage(traces[1]))

        # Each stronger weight should produce higher mean voltage
        for i in range(len(weights) - 1):
            assert mean_voltages[i+1] > mean_voltages[i], \
                f"Weight {weights[i+1]} should produce higher voltage than {weights[i]}"

    def test_zero_weight_no_effect(self):
        """
        A synapse with zero weight should have no effect.
        """
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        net_zero = Network(2)
        net_zero.add_synapse(0, 1, 0.0, E_SYN_EXCITATORY, TAU_EXCITATORY)

        net_none = Network(2)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces_zero = net_zero.simulate(duration, dt, I_ext)
        traces_none = net_none.simulate(duration, dt, I_ext)

        # Should be essentially identical
        np.testing.assert_array_almost_equal(
            traces_zero[1], traces_none[1], decimal=5,
            err_msg="Zero weight synapse should have no effect"
        )


# =============================================================================
# Test Class: Chain Propagation
# =============================================================================

class TestChainPropagation:
    """Tests for spike propagation through chains of neurons."""

    def test_two_neuron_chain(self):
        """
        Spikes should propagate through a 2-neuron chain: [0] -> [1]
        """
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        assert count_spikes(traces[0]) > 0, "Neuron 0 should spike"
        assert count_spikes(traces[1]) > 0, "Neuron 1 should spike from synaptic input"
        assert spike_follows(traces[0], traces[1], dt), "Neuron 1 spikes should follow neuron 0"

    def test_three_neuron_chain(self):
        """
        Spikes should propagate through a 3-neuron chain: [0] -> [1] -> [2]
        """
        net = Network(3)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        spikes = [count_spikes(traces[i]) for i in range(3)]

        assert spikes[0] > 0, "Neuron 0 should spike"
        assert spikes[1] > 0, "Neuron 1 should spike"
        assert spikes[2] > 0, "Neuron 2 should spike (chain propagation)"

    def test_chain_maintains_causality(self):
        """
        In a chain, spike times should increase along the chain.
        """
        net = Network(3)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        # Get first spike time for each neuron
        spike_times = []
        for i in range(3):
            times = get_spike_times(traces[i], dt)
            if len(times) > 0:
                spike_times.append(times[0])
            else:
                spike_times.append(float('inf'))

        assert spike_times[0] < spike_times[1] < spike_times[2], \
            f"Spike times should increase along chain: {spike_times}"


# =============================================================================
# Test Class: Network Topologies
# =============================================================================

class TestNetworkTopologies:
    """Tests for various network structures."""

    def test_divergent_connectivity(self):
        """
        One neuron driving multiple targets.

        [0] --> [1]
            --> [2]
            --> [3]
        """
        net = Network(4)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(0, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(0, 3, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        spikes_0 = count_spikes(traces[0])
        assert spikes_0 > 0, "Source neuron should spike"

        # All targets should receive input and respond
        for i in [1, 2, 3]:
            spikes_i = count_spikes(traces[i])
            assert spikes_i > 0, f"Target neuron {i} should spike"

    def test_convergent_connectivity(self):
        """
        Multiple neurons driving one target.

        [0] -->
        [1] --> [3]
        [2] -->
        """
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        # Convergent: 3 inputs to neuron 3
        net_conv = Network(4)
        net_conv.add_synapse(0, 3, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net_conv.add_synapse(1, 3, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net_conv.add_synapse(2, 3, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 12.0
        I_ext[1, :] = 12.0
        I_ext[2, :] = 12.0

        traces_conv = net_conv.simulate(duration, dt, I_ext)

        # Single input
        net_single = Network(2)
        net_single.add_synapse(0, 1, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)

        I_ext_single = np.zeros((2, num_steps))
        I_ext_single[0, :] = 12.0

        traces_single = net_single.simulate(duration, dt, I_ext_single)

        # Convergent should produce more activity
        mean_conv = get_mean_voltage(traces_conv[3])
        mean_single = get_mean_voltage(traces_single[1])

        assert mean_conv > mean_single, \
            f"Convergent input should produce larger effect ({mean_conv:.1f} vs {mean_single:.1f} mV)"

    def test_bidirectional_connection(self):
        """
        Two neurons with mutual excitation should both be active.

        [0] <--> [1]
        """
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(1, 0, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        # Drive both slightly
        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 10.0
        I_ext[1, :] = 10.0

        traces = net.simulate(duration, dt, I_ext)

        # Both should spike
        assert count_spikes(traces[0]) > 0, "Neuron 0 should spike"
        assert count_spikes(traces[1]) > 0, "Neuron 1 should spike"


# =============================================================================
# Test Class: Excitatory-Inhibitory Balance
# =============================================================================

class TestExcitatoryInhibitoryBalance:
    """Tests for E-I balance and interactions."""

    def test_inhibition_vs_no_inhibition(self):
        """
        Adding inhibition to a driven neuron should reduce its activity.
        """
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        # Driven neuron with inhibition from another spiking neuron
        net_inh = Network(2)
        net_inh.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Inhibitory source
        I_ext[1, :] = 10.0  # Target neuron

        traces_inh = net_inh.simulate(duration, dt, I_ext)

        # Same but without inhibition
        net_ctrl = Network(2)

        traces_ctrl = net_ctrl.simulate(duration, dt, I_ext)

        rate_inh = get_firing_rate(traces_inh[1], duration)
        rate_ctrl = get_firing_rate(traces_ctrl[1], duration)

        assert rate_inh < rate_ctrl, \
            f"Inhibition should reduce firing ({rate_inh:.1f} vs {rate_ctrl:.1f} Hz)"

    def test_mutual_inhibition_asymmetric_drive(self):
        """
        With mutual inhibition and asymmetric drive, stronger-driven should win.
        """
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY)
        net.add_synapse(1, 0, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        # Asymmetric drive - neuron 0 gets more
        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Strong drive
        I_ext[1, :] = 8.0   # Weak drive

        traces = net.simulate(duration, dt, I_ext)

        rate_0 = get_firing_rate(traces[0], duration)
        rate_1 = get_firing_rate(traces[1], duration)

        # Neuron 0 (stronger drive) should have higher rate
        assert rate_0 > rate_1, \
            f"Stronger driven neuron should dominate ({rate_0:.1f} vs {rate_1:.1f} Hz)"


# =============================================================================
# Test Class: Timing and Delays
# =============================================================================

class TestTimingAndDelays:
    """Tests for temporal relationships in network activity."""

    def test_postsynaptic_spike_timing(self):
        """
        Postsynaptic spikes should occur after presynaptic spikes.
        """
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        pre_times = get_spike_times(traces[0], dt)
        post_times = get_spike_times(traces[1], dt)

        assert len(pre_times) > 0, "Should have presynaptic spikes"
        assert len(post_times) > 0, "Should have postsynaptic spikes"

        # First postsynaptic spike should come after first presynaptic spike
        assert post_times[0] > pre_times[0], \
            f"Post spike ({post_times[0]:.2f} ms) should follow pre spike ({pre_times[0]:.2f} ms)"

    def test_spike_latency_in_chain(self):
        """
        Spike latency should accumulate through a chain.
        """
        net = Network(4)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(2, 3, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 150.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        first_spike_times = []
        for i in range(4):
            times = get_spike_times(traces[i], dt)
            if len(times) > 0:
                first_spike_times.append(times[0])
            else:
                first_spike_times.append(float('inf'))

        # Each subsequent neuron should spike later
        for i in range(3):
            if first_spike_times[i] < float('inf') and first_spike_times[i+1] < float('inf'):
                assert first_spike_times[i+1] > first_spike_times[i], \
                    f"Neuron {i+1} should spike after neuron {i}"


# =============================================================================
# Test Class: Network Stability
# =============================================================================

class TestNetworkStability:
    """Tests for numerical stability and edge cases."""

    def test_large_network_stability(self):
        """
        Large networks should remain numerically stable.
        """
        n = 20
        net = Network(n)
        for i in range(n):
            net.add_synapse(i, (i+1) % n, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((n, num_steps))
        I_ext[0, :] = 12.0

        traces = net.simulate(duration, dt, I_ext)

        # Check no NaN or Inf
        for i in range(n):
            assert not np.any(np.isnan(traces[i])), f"Neuron {i} has NaN"
            assert not np.any(np.isinf(traces[i])), f"Neuron {i} has Inf"

    def test_no_input_quiescence(self):
        """
        Without input, connected network should remain quiescent.
        """
        net = Network(3)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(1, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))

        traces = net.simulate(duration, dt, I_ext)

        # No spikes should occur
        for i in range(3):
            assert count_spikes(traces[i]) == 0, f"Neuron {i} should not spike without input"

    def test_reset_clears_state(self):
        """
        Network reset should clear all state and synaptic variables.
        """
        net = Network(2)
        net.add_synapse(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces1 = net.simulate(duration, dt, I_ext)
        net.reset()

        # Run again from fresh state
        traces2 = net.simulate(duration, dt, I_ext)

        # Should be identical
        np.testing.assert_array_almost_equal(
            traces1, traces2, decimal=5,
            err_msg="Reset should produce identical simulation"
        )


# =============================================================================
# Test Class: Mixed Neuron Types
# =============================================================================

class TestMixedNeuronTypes:
    """Tests for networks with mixed HH and Izhikevich neurons."""

    def test_create_mixed_network(self):
        """
        Can create a network with both HH and Izhikevich neurons.
        """
        net = Network()
        idx_hh = net.add_hh_neuron()
        idx_iz = net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)

        assert net.num_neurons == 2
        assert net.neuron_type(idx_hh) == "HH"
        assert net.neuron_type(idx_iz) == "Izhikevich"

    def test_hh_to_izhikevich_synapse(self):
        """
        HH neuron can drive Izhikevich neuron through synapse.
        """
        net = Network()
        net.add_hh_neuron()
        net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Drive HH neuron

        traces = net.simulate(duration, dt, I_ext)

        # Both should spike
        assert count_spikes(traces[0]) > 0, "HH neuron should spike"
        assert count_spikes(traces[1], threshold=0) > 0, "Izhikevich neuron should spike"

    def test_izhikevich_to_hh_synapse(self):
        """
        Izhikevich neuron can drive HH neuron through synapse.
        """
        net = Network()
        net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_hh_neuron()
        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        # Note: Izhikevich needs different dt, but we use HH dt for network
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 10.0  # Drive Izhikevich neuron

        traces = net.simulate(duration, dt, I_ext)

        # Both should spike
        assert count_spikes(traces[0], threshold=0) > 0, "Izhikevich neuron should spike"
        assert count_spikes(traces[1]) > 0, "HH neuron should spike"

    def test_mixed_network_chain(self):
        """
        Chain of alternating HH and Izhikevich neurons.
        """
        net = Network()
        net.add_hh_neuron()        # 0
        net.add_izhikevich_neuron()  # 1
        net.add_hh_neuron()        # 2
        net.add_izhikevich_neuron()  # 3

        net.add_synapse(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net.add_synapse(2, 3, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0

        traces = net.simulate(duration, dt, I_ext)

        # All should eventually spike
        assert count_spikes(traces[0]) > 0, "Neuron 0 (HH) should spike"
        assert count_spikes(traces[1], threshold=0) > 0, "Neuron 1 (Iz) should spike"
        assert count_spikes(traces[2]) > 0, "Neuron 2 (HH) should spike"
        assert count_spikes(traces[3], threshold=0) > 0, "Neuron 3 (Iz) should spike"

    def test_izhikevich_network_types(self):
        """
        Network with different Izhikevich neuron types.
        """
        net = Network()
        net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
        net.add_izhikevich_neuron(IzhikevichType.FAST_SPIKING)
        net.add_izhikevich_neuron(IzhikevichType.INTRINSICALLY_BURSTING)

        duration = 500.0
        dt = 0.1  # Larger dt ok for Izhikevich
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 10.0
        I_ext[1, :] = 10.0
        I_ext[2, :] = 10.0

        traces = net.simulate(duration, dt, I_ext)

        # All should spike
        for i in range(3):
            assert count_spikes(traces[i], threshold=0) > 0, f"Neuron {i} should spike"

        # FS should spike faster than RS with same input
        rate_rs = get_firing_rate(traces[0], duration, threshold=0)
        rate_fs = get_firing_rate(traces[1], duration, threshold=0)

        assert rate_fs > rate_rs, \
            f"Fast spiking should have higher rate ({rate_fs:.1f} vs {rate_rs:.1f} Hz)"

    def test_network_neuron_type_enum(self):
        """
        Test using NetworkNeuronType enum to create networks.
        """
        net = Network(3, NetworkNeuronType.IZHIKEVICH_FS)

        assert net.num_neurons == 3
        for i in range(3):
            assert net.neuron_type(i) == "Izhikevich"

    def test_add_neuron_with_type(self):
        """
        Test adding neurons using NetworkNeuronType.
        """
        net = Network()
        net.add_neuron(neuron_type=NetworkNeuronType.HH)
        net.add_neuron(neuron_type=NetworkNeuronType.IZHIKEVICH_RS)
        net.add_neuron(neuron_type=NetworkNeuronType.IZHIKEVICH_FS)

        assert net.num_neurons == 3
        assert net.neuron_type(0) == "HH"
        assert net.neuron_type(1) == "Izhikevich"
        assert net.neuron_type(2) == "Izhikevich"


# =============================================================================
# Test Class: Quantitative Verification
# =============================================================================

class TestQuantitativeVerification:
    """Tests with specific quantitative expectations."""

    def test_reversal_potential_determines_polarity(self):
        """
        E_syn > V_rest should be excitatory, E_syn < V_rest should be inhibitory.
        """
        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        # Excitatory (E_syn = 0 > V_rest ~ -65)
        net_exc = Network(2)
        net_exc.add_synapse(0, 1, WEIGHT_STRONG, 0.0, TAU_EXCITATORY)

        # Inhibitory (E_syn = -80 < V_rest ~ -65)
        net_inh = Network(2)
        net_inh.add_synapse(0, 1, WEIGHT_STRONG, -80.0, TAU_INHIBITORY)

        # Control (no synapse)
        net_ctrl = Network(2)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        traces_exc = net_exc.simulate(duration, dt, I_ext)
        traces_inh = net_inh.simulate(duration, dt, I_ext)
        traces_ctrl = net_ctrl.simulate(duration, dt, I_ext)

        mean_exc = get_mean_voltage(traces_exc[1])
        mean_inh = get_mean_voltage(traces_inh[1])
        mean_ctrl = get_mean_voltage(traces_ctrl[1])

        assert mean_exc > mean_ctrl, \
            f"Excitatory should depolarize ({mean_exc:.1f} vs {mean_ctrl:.1f} mV)"
        assert mean_inh < mean_ctrl, \
            f"Inhibitory should hyperpolarize ({mean_inh:.1f} vs {mean_ctrl:.1f} mV)"

    def test_synaptic_summation(self):
        """
        Multiple synaptic inputs should sum (approximately).
        """
        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        # Single input
        net_1 = Network(2)
        net_1.add_synapse(0, 1, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)

        I_ext_1 = np.zeros((2, num_steps))
        I_ext_1[0, :] = 15.0

        traces_1 = net_1.simulate(duration, dt, I_ext_1)

        # Double input (two presynaptic neurons)
        net_2 = Network(3)
        net_2.add_synapse(0, 2, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)
        net_2.add_synapse(1, 2, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)

        I_ext_2 = np.zeros((3, num_steps))
        I_ext_2[0, :] = 15.0
        I_ext_2[1, :] = 15.0

        traces_2 = net_2.simulate(duration, dt, I_ext_2)

        mean_1 = get_mean_voltage(traces_1[1])
        mean_2 = get_mean_voltage(traces_2[2])

        # Double input should produce larger effect
        assert mean_2 > mean_1, \
            f"Two inputs should produce larger effect ({mean_2:.1f} vs {mean_1:.1f} mV)"
