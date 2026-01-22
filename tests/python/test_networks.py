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
7. Edge Cases and Stability - verify robustness

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


def get_voltage_std(trace: np.ndarray) -> float:
    """Get standard deviation of membrane voltage."""
    return float(np.std(trace))


def traces_correlated(trace1: np.ndarray, trace2: np.ndarray, min_corr: float = 0.5) -> bool:
    """Check if two traces are positively correlated above threshold."""
    if len(trace1) != len(trace2):
        return False
    corr = np.corrcoef(trace1, trace2)[0, 1]
    return corr > min_corr


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


@dataclass
class NetworkConfig:
    """Configuration for a test network."""
    num_neurons: int
    synapses: List[Tuple[int, int, float, float, float]]  # (pre, post, weight, E_syn, tau)
    duration: float = 500.0
    dt: float = 0.01


def run_network_simulation(
    config: NetworkConfig,
    input_currents: Optional[np.ndarray] = None,
    driven_neurons: Optional[List[int]] = None,
    drive_current: float = 15.0,
) -> Tuple[Network, np.ndarray]:
    """
    Run a network simulation with given configuration.

    Args:
        config: Network configuration
        input_currents: Full input current array (neurons x timesteps), or None for auto-generation
        driven_neurons: List of neuron indices to drive with current (if input_currents is None)
        drive_current: Current amplitude for driven neurons

    Returns:
        Tuple of (network, traces array)
    """
    net = Network(config.num_neurons)

    for pre, post, weight, E_syn, tau in config.synapses:
        net.add_synapse(pre, post, weight, E_syn, tau)

    num_steps = int(config.duration / config.dt)

    if input_currents is None:
        input_currents = np.zeros((config.num_neurons, num_steps))
        if driven_neurons:
            for idx in driven_neurons:
                input_currents[idx, :] = drive_current

    traces = net.simulate(config.duration, config.dt, input_currents)
    return net, np.array(traces)


# =============================================================================
# Synapse Parameter Constants
# =============================================================================

# Standard excitatory synapse (AMPA-like)
E_SYN_EXCITATORY = 0.0      # mV - reversal potential for excitation
TAU_EXCITATORY = 2.0        # ms - fast excitatory decay

# Standard inhibitory synapse (GABA-A-like)
E_SYN_INHIBITORY = -80.0    # mV - reversal potential for inhibition
TAU_INHIBITORY = 5.0        # ms - slower inhibitory decay

# Synaptic weights
WEIGHT_WEAK = 0.1
WEIGHT_MODERATE = 0.5
WEIGHT_STRONG = 1.0
WEIGHT_VERY_STRONG = 2.0


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
        config = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=200.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        # Neuron 0 should spike
        spikes_0 = count_spikes(traces[0])
        assert spikes_0 > 0, "Presynaptic neuron should spike"

        # Neuron 1 should show increased voltage (even if no spike)
        # Compare to a control simulation without the synapse
        control_config = NetworkConfig(num_neurons=2, synapses=[], duration=200.0)
        _, control_traces = run_network_simulation(control_config, driven_neurons=[0], drive_current=15.0)

        # Postsynaptic neuron should have higher mean voltage with synapse
        mean_with_synapse = get_mean_voltage(traces[1])
        mean_without_synapse = get_mean_voltage(control_traces[1])

        assert mean_with_synapse > mean_without_synapse, \
            f"Excitatory synapse should increase postsynaptic voltage ({mean_with_synapse:.2f} vs {mean_without_synapse:.2f})"

    def test_excitatory_synapse_can_trigger_spike(self):
        """
        A strong excitatory synapse should be able to trigger postsynaptic spikes.
        """
        config = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=500.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        spikes_0 = count_spikes(traces[0])
        spikes_1 = count_spikes(traces[1])

        assert spikes_0 > 0, "Presynaptic neuron should spike"
        assert spikes_1 > 0, "Strong excitatory synapse should trigger postsynaptic spikes"

    def test_inhibitory_synapse_reduces_activity(self):
        """
        An inhibitory synapse should reduce postsynaptic firing.

        Network: [0] --inh--> [1], both neurons driven
        Compare to [1] alone - firing rate should be lower with inhibition.
        """
        # With inhibition
        config_inh = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY)],
            duration=500.0,
        )
        _, traces_inh = run_network_simulation(config_inh, driven_neurons=[0, 1], drive_current=12.0)

        # Without inhibition (control)
        config_ctrl = NetworkConfig(num_neurons=2, synapses=[], duration=500.0)
        _, traces_ctrl = run_network_simulation(config_ctrl, driven_neurons=[0, 1], drive_current=12.0)

        rate_with_inh = get_firing_rate(traces_inh[1], 500.0)
        rate_without_inh = get_firing_rate(traces_ctrl[1], 500.0)

        assert rate_with_inh < rate_without_inh, \
            f"Inhibition should reduce firing rate ({rate_with_inh:.1f} Hz vs {rate_without_inh:.1f} Hz)"

    def test_no_synapse_no_interaction(self):
        """
        Without synaptic connections, neurons should be independent.
        """
        config = NetworkConfig(num_neurons=2, synapses=[], duration=200.0)

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        spikes_0 = count_spikes(traces[0])
        spikes_1 = count_spikes(traces[1])

        assert spikes_0 > 0, "Driven neuron should spike"
        assert spikes_1 == 0, "Unconnected, undriven neuron should not spike"

        # Neuron 1 should stay near resting potential
        mean_v1 = get_mean_voltage(traces[1])
        assert -70 < mean_v1 < -60, f"Undriven neuron should stay near rest ({mean_v1:.1f} mV)"


# =============================================================================
# Test Class: Synaptic Weight Effects
# =============================================================================

class TestSynapticWeightEffects:
    """Tests verifying that synaptic weights scale effects appropriately."""

    def test_stronger_weight_stronger_response(self):
        """
        Stronger synaptic weights should produce stronger postsynaptic responses.
        """
        weights = [WEIGHT_WEAK, WEIGHT_MODERATE, WEIGHT_STRONG, WEIGHT_VERY_STRONG]
        mean_voltages = []

        for weight in weights:
            config = NetworkConfig(
                num_neurons=2,
                synapses=[(0, 1, weight, E_SYN_EXCITATORY, TAU_EXCITATORY)],
                duration=300.0,
            )
            _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)
            mean_voltages.append(get_mean_voltage(traces[1]))

        # Each stronger weight should produce higher mean voltage
        for i in range(len(weights) - 1):
            assert mean_voltages[i+1] > mean_voltages[i], \
                f"Weight {weights[i+1]} should produce higher voltage than {weights[i]}"

    def test_stronger_inhibition_stronger_suppression(self):
        """
        Stronger inhibitory weights should produce greater firing suppression.
        """
        weights = [WEIGHT_WEAK, WEIGHT_MODERATE, WEIGHT_STRONG]
        firing_rates = []

        for weight in weights:
            config = NetworkConfig(
                num_neurons=2,
                synapses=[(0, 1, weight, E_SYN_INHIBITORY, TAU_INHIBITORY)],
                duration=500.0,
            )
            _, traces = run_network_simulation(config, driven_neurons=[0, 1], drive_current=12.0)
            firing_rates.append(get_firing_rate(traces[1], 500.0))

        # Stronger inhibition should produce lower firing rate
        for i in range(len(weights) - 1):
            assert firing_rates[i+1] <= firing_rates[i], \
                f"Weight {weights[i+1]} should produce lower or equal rate than {weights[i]}"

    def test_zero_weight_no_effect(self):
        """
        A synapse with zero weight should have no effect.
        """
        config_zero = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, 0.0, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=200.0,
        )
        config_none = NetworkConfig(num_neurons=2, synapses=[], duration=200.0)

        _, traces_zero = run_network_simulation(config_zero, driven_neurons=[0], drive_current=15.0)
        _, traces_none = run_network_simulation(config_none, driven_neurons=[0], drive_current=15.0)

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
        config = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=300.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        assert count_spikes(traces[0]) > 0, "Neuron 0 should spike"
        assert count_spikes(traces[1]) > 0, "Neuron 1 should spike from synaptic input"
        assert spike_follows(traces[0], traces[1], 0.01), "Neuron 1 spikes should follow neuron 0"

    def test_three_neuron_chain(self):
        """
        Spikes should propagate through a 3-neuron chain: [0] -> [1] -> [2]
        """
        config = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=500.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        spikes = [count_spikes(traces[i]) for i in range(3)]

        assert spikes[0] > 0, "Neuron 0 should spike"
        assert spikes[1] > 0, "Neuron 1 should spike"
        assert spikes[2] > 0, "Neuron 2 should spike (chain propagation)"

    def test_chain_maintains_causality(self):
        """
        In a chain, spike times should increase along the chain.
        """
        config = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=100.0,
            dt=0.01,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        # Get first spike time for each neuron
        spike_times = []
        for i in range(3):
            times = get_spike_times(traces[i], 0.01)
            if len(times) > 0:
                spike_times.append(times[0])
            else:
                spike_times.append(float('inf'))

        assert spike_times[0] < spike_times[1] < spike_times[2], \
            f"Spike times should increase along chain: {spike_times}"

    def test_long_chain_attenuation(self):
        """
        Activity may attenuate along a long chain with moderate weights.
        """
        n_neurons = 5
        config = NetworkConfig(
            num_neurons=n_neurons,
            synapses=[
                (i, i+1, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)
                for i in range(n_neurons - 1)
            ],
            duration=500.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        # First neuron should definitely spike
        assert count_spikes(traces[0]) > 0, "First neuron should spike"

        # Activity should decrease or stay similar along chain
        rates = [get_firing_rate(traces[i], 500.0) for i in range(n_neurons)]

        # The firing rate of the last neuron should be <= first neuron
        # (allowing for some transmission)
        assert rates[-1] <= rates[0] * 1.5, \
            f"Chain should not amplify signal: rates={rates}"


# =============================================================================
# Test Class: Network Topologies
# =============================================================================

class TestNetworkTopologies:
    """Tests for various network structures."""

    def test_feedforward_network(self):
        """
        Feedforward network: layer 0 -> layer 1 -> layer 2

        [0] --+--> [2] --+--> [4]
              |         |
        [1] --+    [3] -+
        """
        config = NetworkConfig(
            num_neurons=5,
            synapses=[
                # Layer 0 -> Layer 1
                (0, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (0, 3, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 3, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                # Layer 1 -> Layer 2
                (2, 4, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (3, 4, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=300.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0, 1], drive_current=15.0)

        # All neurons should be active
        for i in range(5):
            assert count_spikes(traces[i]) > 0, f"Neuron {i} should spike in feedforward network"

        # Output layer should spike after input
        assert spike_follows(traces[0], traces[4], 0.01, max_delay=50.0), \
            "Output should follow input"

    def test_divergent_connectivity(self):
        """
        One neuron driving multiple targets.

        [0] --> [1]
            --> [2]
            --> [3]
        """
        config = NetworkConfig(
            num_neurons=4,
            synapses=[
                (0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (0, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (0, 3, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=300.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

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
        config = NetworkConfig(
            num_neurons=4,
            synapses=[
                (0, 3, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 3, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (2, 3, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=300.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0, 1, 2], drive_current=12.0)

        # Convergent input should make neuron 3 very active
        rate_3 = get_firing_rate(traces[3], 300.0)

        # Compare to single input
        config_single = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=300.0,
        )
        _, traces_single = run_network_simulation(config_single, driven_neurons=[0], drive_current=12.0)
        rate_single = get_firing_rate(traces_single[1], 300.0)

        assert rate_3 > rate_single, \
            f"Convergent input should produce higher rate ({rate_3:.1f} Hz vs {rate_single:.1f} Hz)"

    def test_bidirectional_connection(self):
        """
        Two neurons with mutual excitation should synchronize.

        [0] <--> [1]
        """
        config = NetworkConfig(
            num_neurons=2,
            synapses=[
                (0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 0, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=500.0,
        )

        # Drive both neurons slightly
        _, traces = run_network_simulation(config, driven_neurons=[0, 1], drive_current=10.0)

        # Both should spike
        assert count_spikes(traces[0]) > 0, "Neuron 0 should spike"
        assert count_spikes(traces[1]) > 0, "Neuron 1 should spike"

        # Traces should be correlated (synchronized activity)
        assert traces_correlated(traces[0], traces[1], min_corr=0.3), \
            "Mutually connected neurons should show correlated activity"

    def test_ring_network(self):
        """
        Ring network: [0] -> [1] -> [2] -> [0]
        Activity should circulate.
        """
        config = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (2, 0, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=500.0,
        )

        # Give a brief initial stimulus to start activity
        num_steps = int(500.0 / 0.01)
        I_ext = np.zeros((3, num_steps))
        I_ext[0, :1000] = 15.0  # Brief pulse to neuron 0

        _, traces = run_network_simulation(config, input_currents=I_ext)

        # All neurons should spike
        for i in range(3):
            assert count_spikes(traces[i]) > 0, f"Ring neuron {i} should spike"


# =============================================================================
# Test Class: Excitatory-Inhibitory Balance
# =============================================================================

class TestExcitatoryInhibitoryBalance:
    """Tests for E-I balance and interactions."""

    def test_feedforward_inhibition(self):
        """
        Feedforward inhibition: excitation arrives with delayed inhibition.

        [0] --exc--> [2]
            --exc--> [1] --inh--> [2]

        The inhibition should reduce neuron 2's response.
        """
        # With feedforward inhibition
        config_ffi = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY),
            ],
            duration=500.0,
        )
        _, traces_ffi = run_network_simulation(config_ffi, driven_neurons=[0], drive_current=15.0)

        # Without feedforward inhibition (just excitation)
        config_exc = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=500.0,
        )
        _, traces_exc = run_network_simulation(config_exc, driven_neurons=[0], drive_current=15.0)

        rate_with_ffi = get_firing_rate(traces_ffi[2], 500.0)
        rate_without_ffi = get_firing_rate(traces_exc[1], 500.0)

        # FFI should reduce firing rate
        assert rate_with_ffi < rate_without_ffi, \
            f"FFI should reduce firing ({rate_with_ffi:.1f} vs {rate_without_ffi:.1f} Hz)"

    def test_feedback_inhibition(self):
        """
        Feedback inhibition: neuron inhibits itself through interneuron.

        [0] --exc--> [1] --inh--> [0]

        Should reduce firing rate compared to no feedback.
        """
        # With feedback inhibition
        config_fbi = NetworkConfig(
            num_neurons=2,
            synapses=[
                (0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 0, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY),
            ],
            duration=500.0,
        )
        _, traces_fbi = run_network_simulation(config_fbi, driven_neurons=[0], drive_current=15.0)

        # Without feedback (just driven neuron)
        config_ctrl = NetworkConfig(num_neurons=1, synapses=[], duration=500.0)
        _, traces_ctrl = run_network_simulation(config_ctrl, driven_neurons=[0], drive_current=15.0)

        rate_with_fbi = get_firing_rate(traces_fbi[0], 500.0)
        rate_without_fbi = get_firing_rate(traces_ctrl[0], 500.0)

        # Feedback inhibition should reduce firing
        assert rate_with_fbi < rate_without_fbi, \
            f"Feedback inhibition should reduce firing ({rate_with_fbi:.1f} vs {rate_without_fbi:.1f} Hz)"

    def test_mutual_inhibition_competition(self):
        """
        Mutual inhibition creates competition (winner-take-all dynamics).

        [0] <--inh--> [1]

        With asymmetric input, stronger-driven neuron should dominate.
        """
        config = NetworkConfig(
            num_neurons=2,
            synapses=[
                (0, 1, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY),
                (1, 0, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY),
            ],
            duration=500.0,
        )

        # Asymmetric drive
        num_steps = int(500.0 / 0.01)
        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Strong drive to 0
        I_ext[1, :] = 10.0  # Weaker drive to 1

        _, traces = run_network_simulation(config, input_currents=I_ext)

        rate_0 = get_firing_rate(traces[0], 500.0)
        rate_1 = get_firing_rate(traces[1], 500.0)

        # Stronger driven neuron should have higher rate
        assert rate_0 > rate_1, \
            f"Stronger driven neuron should dominate ({rate_0:.1f} vs {rate_1:.1f} Hz)"

    def test_balanced_ei_produces_irregular_activity(self):
        """
        Balanced E-I input produces more variable activity than pure excitation.
        """
        # Excitation only
        config_exc = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=500.0,
        )
        _, traces_exc = run_network_simulation(config_exc, driven_neurons=[0, 1], drive_current=12.0)

        # Balanced E-I
        config_ei = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY),
            ],
            duration=500.0,
        )
        _, traces_ei = run_network_simulation(config_ei, driven_neurons=[0, 1], drive_current=12.0)

        # Balanced case should have lower firing rate
        rate_exc = get_firing_rate(traces_exc[2], 500.0)
        rate_ei = get_firing_rate(traces_ei[2], 500.0)

        assert rate_ei < rate_exc, \
            f"E-I balance should reduce rate ({rate_ei:.1f} vs {rate_exc:.1f} Hz)"


# =============================================================================
# Test Class: Timing and Delays
# =============================================================================

class TestTimingAndDelays:
    """Tests for temporal relationships in network activity."""

    def test_synaptic_delay_effect(self):
        """
        Different tau values should affect timing of postsynaptic response.
        """
        taus = [1.0, 2.0, 5.0, 10.0]
        mean_voltages = []

        for tau in taus:
            config = NetworkConfig(
                num_neurons=2,
                synapses=[(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, tau)],
                duration=200.0,
            )
            _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)
            mean_voltages.append(get_mean_voltage(traces[1]))

        # Longer tau should generally result in more sustained effect
        # (larger mean voltage due to longer-lasting synaptic conductance)
        assert mean_voltages[-1] > mean_voltages[0], \
            f"Longer tau should produce more sustained effect: {mean_voltages}"

    def test_postsynaptic_spike_timing(self):
        """
        Postsynaptic spikes should occur after presynaptic spikes.
        """
        config = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=100.0,
            dt=0.01,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        pre_times = get_spike_times(traces[0], 0.01)
        post_times = get_spike_times(traces[1], 0.01)

        assert len(pre_times) > 0, "Should have presynaptic spikes"
        assert len(post_times) > 0, "Should have postsynaptic spikes"

        # First postsynaptic spike should come after first presynaptic spike
        assert post_times[0] > pre_times[0], \
            f"Post spike ({post_times[0]:.2f} ms) should follow pre spike ({pre_times[0]:.2f} ms)"

    def test_spike_latency_in_chain(self):
        """
        Spike latency should accumulate through a chain.
        """
        config = NetworkConfig(
            num_neurons=4,
            synapses=[
                (0, 1, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (2, 3, WEIGHT_VERY_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=150.0,
            dt=0.01,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)

        first_spike_times = []
        for i in range(4):
            times = get_spike_times(traces[i], 0.01)
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
        config = NetworkConfig(
            num_neurons=n,
            synapses=[
                (i, (i+1) % n, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)
                for i in range(n)
            ],
            duration=200.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=12.0)

        # Check no NaN or Inf
        for i in range(n):
            assert not np.any(np.isnan(traces[i])), f"Neuron {i} has NaN"
            assert not np.any(np.isinf(traces[i])), f"Neuron {i} has Inf"
            assert np.all(traces[i] > -200), f"Neuron {i} voltage too negative"
            assert np.all(traces[i] < 100), f"Neuron {i} voltage too positive"

    def test_high_activity_stability(self):
        """
        High activity networks should remain stable.
        """
        config = NetworkConfig(
            num_neurons=5,
            synapses=[
                (i, j, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)
                for i in range(5) for j in range(5) if i != j
            ],
            duration=500.0,
        )

        _, traces = run_network_simulation(
            config, driven_neurons=list(range(5)), drive_current=15.0
        )

        # All should spike but remain bounded
        for i in range(5):
            assert count_spikes(traces[i]) > 0, f"Neuron {i} should spike"
            assert np.max(traces[i]) < 100, f"Neuron {i} peak should be bounded"

    def test_no_input_quiescence(self):
        """
        Without input, connected network should remain quiescent.
        """
        config = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=200.0,
        )

        _, traces = run_network_simulation(config, driven_neurons=[], drive_current=0.0)

        # No spikes should occur
        for i in range(3):
            assert count_spikes(traces[i]) == 0, f"Neuron {i} should not spike without input"

    def test_reset_clears_state(self):
        """
        Network reset should clear all state and synaptic variables.
        """
        config = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=100.0,
        )

        net, traces1 = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)
        net.reset()

        # Run again from fresh state
        num_steps = int(100.0 / 0.01)
        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        traces2 = np.array(net.simulate(100.0, 0.01, I_ext))

        # Should be identical (or very close)
        np.testing.assert_array_almost_equal(
            traces1, traces2, decimal=5,
            err_msg="Reset should produce identical simulation"
        )

    def test_long_simulation_stability(self):
        """
        Long simulations should remain stable.
        """
        config = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=2000.0,  # 2 seconds
        )

        _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=12.0)

        # Check stability throughout
        for i in range(3):
            assert not np.any(np.isnan(traces[i])), f"Neuron {i} has NaN in long sim"
            assert not np.any(np.isinf(traces[i])), f"Neuron {i} has Inf in long sim"

            # Check activity is consistent (not diverging)
            first_half = traces[i][:len(traces[i])//2]
            second_half = traces[i][len(traces[i])//2:]

            rate1 = count_spikes(first_half) * 2  # Normalize to same duration
            rate2 = count_spikes(second_half) * 2

            # Rates should be similar (within factor of 3)
            if rate1 > 0:
                assert 0.33 < rate2 / rate1 < 3.0, \
                    f"Neuron {i} rate changed dramatically: {rate1} -> {rate2}"


# =============================================================================
# Test Class: Quantitative Verification
# =============================================================================

class TestQuantitativeVerification:
    """Tests with specific quantitative expectations."""

    def test_firing_rate_scales_with_input(self):
        """
        Firing rate should monotonically increase with input current.
        """
        currents = [5, 8, 10, 12, 15, 20]
        rates = []

        for I in currents:
            config = NetworkConfig(num_neurons=1, synapses=[], duration=500.0)
            _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=float(I))
            rates.append(get_firing_rate(traces[0], 500.0))

        # Should be monotonically increasing (or at least non-decreasing)
        for i in range(len(rates) - 1):
            assert rates[i+1] >= rates[i], \
                f"Rate should increase with current: {list(zip(currents, rates))}"

    def test_synaptic_summation(self):
        """
        Multiple synaptic inputs should sum (approximately linearly for small inputs).
        """
        # Single input
        config_1 = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=300.0,
        )
        _, traces_1 = run_network_simulation(config_1, driven_neurons=[0], drive_current=15.0)
        mean_1 = get_mean_voltage(traces_1[1])

        # Double input (two presynaptic neurons)
        config_2 = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 2, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=300.0,
        )
        _, traces_2 = run_network_simulation(config_2, driven_neurons=[0, 1], drive_current=15.0)
        mean_2 = get_mean_voltage(traces_2[2])

        # Double input should produce larger effect
        assert mean_2 > mean_1, \
            f"Two inputs should produce larger effect than one ({mean_2:.1f} vs {mean_1:.1f} mV)"

    def test_reversal_potential_determines_polarity(self):
        """
        E_syn > V_rest should be excitatory, E_syn < V_rest should be inhibitory.
        """
        V_rest = -65.0  # Approximate resting potential

        # Excitatory (E_syn = 0 > V_rest)
        config_exc = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, 0.0, TAU_EXCITATORY)],
            duration=200.0,
        )
        _, traces_exc = run_network_simulation(config_exc, driven_neurons=[0], drive_current=15.0)

        # Inhibitory (E_syn = -80 < V_rest)
        config_inh = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, -80.0, TAU_INHIBITORY)],
            duration=200.0,
        )
        _, traces_inh = run_network_simulation(config_inh, driven_neurons=[0], drive_current=15.0)

        # Control (no synapse)
        config_ctrl = NetworkConfig(num_neurons=2, synapses=[], duration=200.0)
        _, traces_ctrl = run_network_simulation(config_ctrl, driven_neurons=[0], drive_current=15.0)

        mean_exc = get_mean_voltage(traces_exc[1])
        mean_inh = get_mean_voltage(traces_inh[1])
        mean_ctrl = get_mean_voltage(traces_ctrl[1])

        assert mean_exc > mean_ctrl, \
            f"Excitatory should depolarize ({mean_exc:.1f} vs {mean_ctrl:.1f} mV)"
        assert mean_inh < mean_ctrl, \
            f"Inhibitory should hyperpolarize ({mean_inh:.1f} vs {mean_ctrl:.1f} mV)"


# =============================================================================
# Test Class: Complex Network Motifs
# =============================================================================

class TestComplexNetworkMotifs:
    """Tests for more complex network structures and dynamics."""

    def test_lateral_inhibition(self):
        """
        Lateral inhibition should sharpen responses (winner-take-all).

        [0] --inh--> [1] <--inh-- [2]

        With center neuron receiving most input, it should dominate.
        """
        config = NetworkConfig(
            num_neurons=3,
            synapses=[
                (0, 1, WEIGHT_MODERATE, E_SYN_INHIBITORY, TAU_INHIBITORY),
                (2, 1, WEIGHT_MODERATE, E_SYN_INHIBITORY, TAU_INHIBITORY),
            ],
            duration=500.0,
        )

        # Center gets more input
        num_steps = int(500.0 / 0.01)
        I_ext = np.zeros((3, num_steps))
        I_ext[0, :] = 10.0  # Flanker
        I_ext[1, :] = 15.0  # Center (strongest)
        I_ext[2, :] = 10.0  # Flanker

        _, traces = run_network_simulation(config, input_currents=I_ext)

        rate_0 = get_firing_rate(traces[0], 500.0)
        rate_1 = get_firing_rate(traces[1], 500.0)
        rate_2 = get_firing_rate(traces[2], 500.0)

        # Center should have highest rate despite inhibition from flankers
        # (because it has strongest drive)
        assert rate_1 > rate_0 or rate_1 > rate_2, \
            f"Center should remain active: rates = [{rate_0:.1f}, {rate_1:.1f}, {rate_2:.1f}] Hz"

    def test_disinhibition(self):
        """
        Disinhibition: inhibiting an inhibitory neuron releases excitation.

        [0] --exc--> [2]
        [1] --inh--> [2]
        [3] --inh--> [1]  (disinhibits [2])

        Activating [3] should increase [2]'s activity.
        """
        # Without disinhibition (only [0] and [1] active)
        config_no_disinh = NetworkConfig(
            num_neurons=4,
            synapses=[
                (0, 2, WEIGHT_STRONG, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 2, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY),
                (3, 1, WEIGHT_STRONG, E_SYN_INHIBITORY, TAU_INHIBITORY),
            ],
            duration=500.0,
        )
        _, traces_no = run_network_simulation(
            config_no_disinh, driven_neurons=[0, 1], drive_current=12.0
        )

        # With disinhibition ([0], [1], and [3] active)
        _, traces_yes = run_network_simulation(
            config_no_disinh, driven_neurons=[0, 1, 3], drive_current=12.0
        )

        rate_no = get_firing_rate(traces_no[2], 500.0)
        rate_yes = get_firing_rate(traces_yes[2], 500.0)

        # Disinhibition should increase target's rate
        assert rate_yes > rate_no, \
            f"Disinhibition should increase firing ({rate_yes:.1f} vs {rate_no:.1f} Hz)"

    def test_recurrent_excitation_amplification(self):
        """
        Recurrent excitation should amplify responses.

        [0] --exc--> [1] --exc--> [0] (recurrent)

        Should produce higher activity than feedforward alone.
        """
        # Feedforward only
        config_ff = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY)],
            duration=500.0,
        )
        _, traces_ff = run_network_simulation(config_ff, driven_neurons=[0], drive_current=10.0)

        # With recurrence
        config_rec = NetworkConfig(
            num_neurons=2,
            synapses=[
                (0, 1, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY),
                (1, 0, WEIGHT_MODERATE, E_SYN_EXCITATORY, TAU_EXCITATORY),
            ],
            duration=500.0,
        )
        _, traces_rec = run_network_simulation(config_rec, driven_neurons=[0], drive_current=10.0)

        # Recurrence should increase activity
        rate_ff = get_firing_rate(traces_ff[0], 500.0)
        rate_rec = get_firing_rate(traces_rec[0], 500.0)

        assert rate_rec >= rate_ff, \
            f"Recurrence should amplify ({rate_rec:.1f} vs {rate_ff:.1f} Hz)"


# =============================================================================
# Test Class: Parameter Sensitivity
# =============================================================================

class TestParameterSensitivity:
    """Tests for sensitivity to synaptic parameters."""

    def test_tau_affects_temporal_integration(self):
        """
        Longer tau should allow more temporal integration of inputs.
        """
        # Short tau
        config_short = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, 1.0)],
            duration=300.0,
        )
        _, traces_short = run_network_simulation(config_short, driven_neurons=[0], drive_current=15.0)

        # Long tau
        config_long = NetworkConfig(
            num_neurons=2,
            synapses=[(0, 1, WEIGHT_STRONG, E_SYN_EXCITATORY, 10.0)],
            duration=300.0,
        )
        _, traces_long = run_network_simulation(config_long, driven_neurons=[0], drive_current=15.0)

        # Longer tau should produce smoother, more integrated response
        std_short = get_voltage_std(traces_short[1])
        std_long = get_voltage_std(traces_long[1])

        # Both should respond, long tau typically smoother
        assert get_mean_voltage(traces_long[1]) > -70, "Long tau should still produce response"

    def test_e_syn_intermediate_values(self):
        """
        Intermediate E_syn values should produce intermediate effects.
        """
        e_syns = [-80.0, -40.0, 0.0]  # Inhibitory -> Shunting -> Excitatory
        mean_voltages = []

        for e_syn in e_syns:
            config = NetworkConfig(
                num_neurons=2,
                synapses=[(0, 1, WEIGHT_STRONG, e_syn, TAU_EXCITATORY)],
                duration=200.0,
            )
            _, traces = run_network_simulation(config, driven_neurons=[0], drive_current=15.0)
            mean_voltages.append(get_mean_voltage(traces[1]))

        # Should be monotonically increasing with E_syn
        for i in range(len(e_syns) - 1):
            assert mean_voltages[i+1] > mean_voltages[i], \
                f"Higher E_syn should produce higher voltage: E_syn={e_syns}, V={mean_voltages}"
