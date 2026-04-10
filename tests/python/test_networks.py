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
    RegionalNetwork,
    NeuronModelSpec,
    SynapseSpec,
    IzhikevichType,
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


def _hh_rn(n: int) -> RegionalNetwork:
    """Create a RegionalNetwork with n HH neurons in population 'E'."""
    rn = RegionalNetwork()
    rn.add_population("E", n, model=NeuronModelSpec.hh_default())
    return rn


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

EXC = SynapseSpec.exponential(TAU_EXCITATORY, E_syn=E_SYN_EXCITATORY)
INH = SynapseSpec.exponential(TAU_INHIBITORY, E_syn=E_SYN_INHIBITORY)


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
        rn = _hh_rn(2)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_STRONG, synapse=EXC)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert count_spikes(result["E"][0]) > 0, "Presynaptic neuron should spike"

        rn_ctrl = _hh_rn(2)
        result_ctrl = rn_ctrl.simulate(duration, dt, {"E": I_ext})

        mean_with_synapse = get_mean_voltage(result["E"][1])
        mean_without_synapse = get_mean_voltage(result_ctrl["E"][1])

        assert mean_with_synapse > mean_without_synapse, \
            f"Excitatory synapse should increase postsynaptic voltage ({mean_with_synapse:.2f} vs {mean_without_synapse:.2f})"

    def test_excitatory_synapse_can_trigger_spike(self):
        """
        A strong excitatory synapse should be able to trigger postsynaptic spikes.
        """
        rn = _hh_rn(2)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert count_spikes(result["E"][0]) > 0, "Presynaptic neuron should spike"
        assert count_spikes(result["E"][1]) > 0, "Strong excitatory synapse should trigger postsynaptic spikes"

    def test_inhibitory_synapse_reduces_activity(self):
        """
        An inhibitory synapse should reduce postsynaptic firing.
        """
        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        I_ext[1, :] = 10.0  # Drive both, inhibition should reduce neuron 1

        rn_inh = _hh_rn(2)
        rn_inh.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_STRONG, synapse=INH)
        result_inh = rn_inh.simulate(duration, dt, {"E": I_ext})

        rn_ctrl = _hh_rn(2)
        result_ctrl = rn_ctrl.simulate(duration, dt, {"E": I_ext})

        rate_with_inh = get_firing_rate(result_inh["E"][1], duration)
        rate_without_inh = get_firing_rate(result_ctrl["E"][1], duration)

        assert rate_with_inh < rate_without_inh, \
            f"Inhibition should reduce firing rate ({rate_with_inh:.1f} Hz vs {rate_without_inh:.1f} Hz)"

    def test_no_synapse_no_interaction(self):
        """
        Without synaptic connections, neurons should be independent.
        """
        rn = _hh_rn(2)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert count_spikes(result["E"][0]) > 0, "Driven neuron should spike"
        assert count_spikes(result["E"][1]) == 0, "Unconnected, undriven neuron should not spike"


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

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        for weight in weights:
            rn = _hh_rn(2)
            rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=weight, synapse=EXC)
            result = rn.simulate(duration, dt, {"E": I_ext})
            mean_voltages.append(get_mean_voltage(result["E"][1]))

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

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        rn_zero = _hh_rn(2)
        rn_zero.connect("E", "E", lambda ns, nd: [(0, 1)], weight=0.0, synapse=EXC)
        result_zero = rn_zero.simulate(duration, dt, {"E": I_ext})

        rn_none = _hh_rn(2)
        result_none = rn_none.simulate(duration, dt, {"E": I_ext})

        np.testing.assert_array_almost_equal(
            result_zero["E"][1], result_none["E"][1], decimal=5,
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
        rn = _hh_rn(2)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert count_spikes(result["E"][0]) > 0, "Neuron 0 should spike"
        assert count_spikes(result["E"][1]) > 0, "Neuron 1 should spike from synaptic input"
        assert spike_follows(result["E"][0], result["E"][1], dt), "Neuron 1 spikes should follow neuron 0"

    def test_three_neuron_chain(self):
        """
        Spikes should propagate through a 3-neuron chain: [0] -> [1] -> [2]
        """
        rn = _hh_rn(3)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_VERY_STRONG, synapse=EXC)
        rn.connect("E", "E", lambda ns, nd: [(1, 2)], weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert count_spikes(result["E"][0]) > 0, "Neuron 0 should spike"
        assert count_spikes(result["E"][1]) > 0, "Neuron 1 should spike"
        assert count_spikes(result["E"][2]) > 0, "Neuron 2 should spike (chain propagation)"

    def test_chain_maintains_causality(self):
        """
        In a chain, spike times should increase along the chain.
        """
        rn = _hh_rn(3)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_VERY_STRONG, synapse=EXC)
        rn.connect("E", "E", lambda ns, nd: [(1, 2)], weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        spike_times = []
        for i in range(3):
            times = get_spike_times(result["E"][i], dt)
            spike_times.append(times[0] if len(times) > 0 else float('inf'))

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
        rn = _hh_rn(4)
        rn.connect("E", "E", lambda ns, nd: [(0, 1), (0, 2), (0, 3)],
                   weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 300.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert count_spikes(result["E"][0]) > 0, "Source neuron should spike"
        for i in [1, 2, 3]:
            assert count_spikes(result["E"][i]) > 0, f"Target neuron {i} should spike"

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

        rn_conv = _hh_rn(4)
        rn_conv.connect("E", "E", lambda ns, nd: [(0, 3), (1, 3), (2, 3)],
                        weight=WEIGHT_MODERATE, synapse=EXC)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 12.0
        I_ext[1, :] = 12.0
        I_ext[2, :] = 12.0

        result_conv = rn_conv.simulate(duration, dt, {"E": I_ext})

        rn_single = _hh_rn(2)
        rn_single.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_MODERATE, synapse=EXC)

        I_ext_single = np.zeros((2, num_steps))
        I_ext_single[0, :] = 12.0

        result_single = rn_single.simulate(duration, dt, {"E": I_ext_single})

        mean_conv = get_mean_voltage(result_conv["E"][3])
        mean_single = get_mean_voltage(result_single["E"][1])

        assert mean_conv > mean_single, \
            f"Convergent input should produce larger effect ({mean_conv:.1f} vs {mean_single:.1f} mV)"

    def test_bidirectional_connection(self):
        """
        Two neurons with mutual excitation should both be active.

        [0] <--> [1]
        """
        rn = _hh_rn(2)
        rn.connect("E", "E", lambda ns, nd: [(0, 1), (1, 0)], weight=WEIGHT_STRONG, synapse=EXC)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 10.0
        I_ext[1, :] = 10.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert count_spikes(result["E"][0]) > 0, "Neuron 0 should spike"
        assert count_spikes(result["E"][1]) > 0, "Neuron 1 should spike"


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

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Inhibitory source
        I_ext[1, :] = 10.0  # Target neuron

        rn_inh = _hh_rn(2)
        rn_inh.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_STRONG, synapse=INH)
        result_inh = rn_inh.simulate(duration, dt, {"E": I_ext})

        rn_ctrl = _hh_rn(2)
        result_ctrl = rn_ctrl.simulate(duration, dt, {"E": I_ext})

        rate_inh = get_firing_rate(result_inh["E"][1], duration)
        rate_ctrl = get_firing_rate(result_ctrl["E"][1], duration)

        assert rate_inh < rate_ctrl, \
            f"Inhibition should reduce firing ({rate_inh:.1f} vs {rate_ctrl:.1f} Hz)"

    def test_mutual_inhibition_asymmetric_drive(self):
        """
        With mutual inhibition and asymmetric drive, stronger-driven should win.
        """
        rn = _hh_rn(2)
        rn.connect("E", "E", lambda ns, nd: [(0, 1), (1, 0)], weight=WEIGHT_STRONG, synapse=INH)

        duration = 500.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Strong drive
        I_ext[1, :] = 8.0   # Weak drive

        result = rn.simulate(duration, dt, {"E": I_ext})

        rate_0 = get_firing_rate(result["E"][0], duration)
        rate_1 = get_firing_rate(result["E"][1], duration)

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
        rn = _hh_rn(2)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        pre_times = get_spike_times(result["E"][0], dt)
        post_times = get_spike_times(result["E"][1], dt)

        assert len(pre_times) > 0, "Should have presynaptic spikes"
        assert len(post_times) > 0, "Should have postsynaptic spikes"

        assert post_times[0] > pre_times[0], \
            f"Post spike ({post_times[0]:.2f} ms) should follow pre spike ({pre_times[0]:.2f} ms)"

    def test_spike_latency_in_chain(self):
        """
        Spike latency should accumulate through a chain.
        """
        rn = _hh_rn(4)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_VERY_STRONG, synapse=EXC)
        rn.connect("E", "E", lambda ns, nd: [(1, 2)], weight=WEIGHT_VERY_STRONG, synapse=EXC)
        rn.connect("E", "E", lambda ns, nd: [(2, 3)], weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 150.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((4, num_steps))
        I_ext[0, :] = 15.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        first_spike_times = []
        for i in range(4):
            times = get_spike_times(result["E"][i], dt)
            first_spike_times.append(times[0] if len(times) > 0 else float('inf'))

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
        rn = _hh_rn(n)
        rn.connect("E", "E", lambda ns, nd: [(i, (i+1) % n) for i in range(n)],
                   weight=WEIGHT_MODERATE, synapse=EXC)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((n, num_steps))
        I_ext[0, :] = 12.0

        result = rn.simulate(duration, dt, {"E": I_ext})

        assert np.all(np.isfinite(result["E"])), "All neurons should remain finite"

    def test_no_input_quiescence(self):
        """
        Without input, connected network should remain quiescent.
        """
        rn = _hh_rn(3)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_STRONG, synapse=EXC)
        rn.connect("E", "E", lambda ns, nd: [(1, 2)], weight=WEIGHT_STRONG, synapse=EXC)

        duration = 200.0
        dt = 0.01

        result = rn.simulate(duration, dt, {"E": 0.0})

        for i in range(3):
            assert count_spikes(result["E"][i]) == 0, f"Neuron {i} should not spike without input"

    def test_reset_clears_state(self):
        """
        Network reset should clear all state and synaptic variables.
        """
        rn = _hh_rn(2)
        rn.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_STRONG, synapse=EXC)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        result1 = rn.simulate(duration, dt, {"E": I_ext})
        rn.reset()
        result2 = rn.simulate(duration, dt, {"E": I_ext})

        np.testing.assert_array_almost_equal(
            result1["E"], result2["E"], decimal=5,
            err_msg="Reset should produce identical simulation"
        )


# =============================================================================
# Test Class: Mixed Neuron Types
# =============================================================================

class TestMixedNeuronTypes:
    """Tests for networks with mixed HH and Izhikevich neurons."""

    def test_hh_to_izhikevich_synapse(self):
        """
        HH neuron can drive Izhikevich neuron through synapse.
        """
        rn = RegionalNetwork()
        rn.add_population("HH", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("IZ", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING))
        rn.connect("HH", "IZ", "all_to_all", weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 300.0
        dt = 0.01

        result = rn.simulate(duration, dt, {"HH": 15.0, "IZ": 0.0})

        assert count_spikes(result["HH"][0]) > 0, "HH neuron should spike"
        assert count_spikes(result["IZ"][0]) > 0, "Izhikevich neuron should spike"

    def test_izhikevich_to_hh_synapse(self):
        """
        Izhikevich neuron can drive HH neuron through synapse.
        """
        rn = RegionalNetwork()
        rn.add_population("IZ", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING))
        rn.add_population("HH", 1, model=NeuronModelSpec.hh_default())
        rn.connect("IZ", "HH", "all_to_all", weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 300.0
        dt = 0.01

        result = rn.simulate(duration, dt, {"IZ": 10.0, "HH": 0.0})

        assert count_spikes(result["IZ"][0]) > 0, "Izhikevich neuron should spike"
        assert count_spikes(result["HH"][0]) > 0, "HH neuron should spike"

    def test_mixed_network_chain(self):
        """
        Chain of alternating HH and Izhikevich neurons.
        """
        rn = RegionalNetwork()
        rn.add_population("N0", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("N1", 1, model=NeuronModelSpec.izhikevich())
        rn.add_population("N2", 1, model=NeuronModelSpec.hh_default())
        rn.add_population("N3", 1, model=NeuronModelSpec.izhikevich())

        rn.connect("N0", "N1", "all_to_all", weight=WEIGHT_VERY_STRONG, synapse=EXC)
        rn.connect("N1", "N2", "all_to_all", weight=WEIGHT_VERY_STRONG, synapse=EXC)
        rn.connect("N2", "N3", "all_to_all", weight=WEIGHT_VERY_STRONG, synapse=EXC)

        duration = 500.0
        dt = 0.01

        result = rn.simulate(duration, dt, {"N0": 15.0, "N1": 0.0, "N2": 0.0, "N3": 0.0})

        assert count_spikes(result["N0"][0]) > 0, "Neuron 0 (HH) should spike"
        assert count_spikes(result["N1"][0]) > 0, "Neuron 1 (Iz) should spike"
        assert count_spikes(result["N2"][0]) > 0, "Neuron 2 (HH) should spike"
        assert count_spikes(result["N3"][0]) > 0, "Neuron 3 (Iz) should spike"

    def test_izhikevich_network_types(self):
        """
        Network with different Izhikevich neuron types.
        """
        rn = RegionalNetwork()
        rn.add_population("RS", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING))
        rn.add_population("FS", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING))
        rn.add_population("IB", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.INTRINSICALLY_BURSTING))

        duration = 500.0
        dt = 0.1

        result = rn.simulate(duration, dt, {"RS": 10.0, "FS": 10.0, "IB": 10.0})

        assert count_spikes(result["RS"][0]) > 0, "RS should spike"
        assert count_spikes(result["FS"][0]) > 0, "FS should spike"
        assert count_spikes(result["IB"][0]) > 0, "IB should spike"

        rate_rs = get_firing_rate(result["RS"][0], duration)
        rate_fs = get_firing_rate(result["FS"][0], duration)

        assert rate_fs > rate_rs, \
            f"Fast spiking should have higher rate ({rate_fs:.1f} vs {rate_rs:.1f} Hz)"


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

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0

        rn_exc = _hh_rn(2)
        rn_exc.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_STRONG,
                       synapse=SynapseSpec.exponential(TAU_EXCITATORY, E_syn=0.0))
        result_exc = rn_exc.simulate(duration, dt, {"E": I_ext})

        rn_inh = _hh_rn(2)
        rn_inh.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_STRONG,
                       synapse=SynapseSpec.exponential(TAU_INHIBITORY, E_syn=-80.0))
        result_inh = rn_inh.simulate(duration, dt, {"E": I_ext})

        rn_ctrl = _hh_rn(2)
        result_ctrl = rn_ctrl.simulate(duration, dt, {"E": I_ext})

        mean_exc = get_mean_voltage(result_exc["E"][1])
        mean_inh = get_mean_voltage(result_inh["E"][1])
        mean_ctrl = get_mean_voltage(result_ctrl["E"][1])

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
        rn_1 = _hh_rn(2)
        rn_1.connect("E", "E", lambda ns, nd: [(0, 1)], weight=WEIGHT_MODERATE, synapse=EXC)

        I_ext_1 = np.zeros((2, num_steps))
        I_ext_1[0, :] = 15.0

        result_1 = rn_1.simulate(duration, dt, {"E": I_ext_1})

        # Double input (two presynaptic neurons driving neuron 2)
        rn_2 = _hh_rn(3)
        rn_2.connect("E", "E", lambda ns, nd: [(0, 2), (1, 2)], weight=WEIGHT_MODERATE, synapse=EXC)

        I_ext_2 = np.zeros((3, num_steps))
        I_ext_2[0, :] = 15.0
        I_ext_2[1, :] = 15.0

        result_2 = rn_2.simulate(duration, dt, {"E": I_ext_2})

        mean_1 = get_mean_voltage(result_1["E"][1])
        mean_2 = get_mean_voltage(result_2["E"][2])

        assert mean_2 > mean_1, \
            f"Two inputs should produce larger effect ({mean_2:.1f} vs {mean_1:.1f} mV)"
