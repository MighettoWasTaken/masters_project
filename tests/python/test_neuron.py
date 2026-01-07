"""Tests for the Hodgkin-Huxley neuron model."""

import numpy as np
import pytest

from hodgkin_huxley import HHNeuron, Network, Parameters, State


class TestHHNeuron:
    """Tests for the HHNeuron class."""

    def test_default_creation(self):
        """Test creating a neuron with default parameters."""
        neuron = HHNeuron()
        assert neuron.V == pytest.approx(-65.0, abs=1.0)

    def test_custom_parameters(self):
        """Test creating a neuron with custom parameters."""
        params = Parameters()
        params.g_Na = 100.0
        neuron = HHNeuron(params)
        assert neuron.parameters.g_Na == 100.0

    def test_reset(self):
        """Test reset returns neuron to resting state."""
        neuron = HHNeuron()
        neuron.V = 0.0
        neuron.reset()
        assert neuron.V == pytest.approx(-65.0, abs=1.0)

    def test_step(self):
        """Test single step integration."""
        neuron = HHNeuron()
        initial_V = neuron.V
        neuron.step(dt=0.01, I_ext=10.0)
        # With positive current, voltage should increase
        assert neuron.V > initial_V

    def test_simulate_constant_current(self):
        """Test simulation with constant current."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=10.0, dt=0.01, I_ext=10.0)
        assert len(trace) == 1000
        assert isinstance(trace, np.ndarray)

    def test_simulate_time_varying_current(self):
        """Test simulation with time-varying current."""
        neuron = HHNeuron()
        I_ext = np.ones(1000) * 10.0
        trace = neuron.simulate(duration=10.0, dt=0.01, I_ext=I_ext)
        assert len(trace) == 1000

    def test_action_potential(self):
        """Test that sufficient current produces an action potential."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=50.0, dt=0.01, I_ext=15.0)
        # Action potential should exceed 0 mV
        assert np.max(trace) > 0.0
        # Resting potential should be around -65 mV
        assert trace[0] < -60.0

    def test_no_spike_below_threshold(self):
        """Test that subthreshold current doesn't produce a spike."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=50.0, dt=0.01, I_ext=2.0)
        # Voltage should stay below threshold
        assert np.max(trace) < -40.0


class TestNetwork:
    """Tests for the Network class."""

    def test_empty_network(self):
        """Test creating an empty network."""
        net = Network()
        assert net.num_neurons == 0
        assert len(net) == 0

    def test_network_with_neurons(self):
        """Test creating a network with neurons."""
        net = Network(3)
        assert net.num_neurons == 3

    def test_add_neuron(self):
        """Test adding neurons to network."""
        net = Network()
        idx = net.add_neuron()
        assert idx == 0
        assert net.num_neurons == 1

    def test_add_synapse(self):
        """Test adding synaptic connections."""
        net = Network(2)
        net.add_synapse(0, 1, weight=0.5)
        assert net.num_synapses == 1

    def test_get_potentials(self):
        """Test getting all membrane potentials."""
        net = Network(3)
        potentials = net.get_potentials()
        assert len(potentials) == 3
        assert all(p == pytest.approx(-65.0, abs=1.0) for p in potentials)

    def test_network_simulate(self):
        """Test network simulation."""
        net = Network(2)
        duration = 10.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 10.0  # Only stimulate first neuron

        traces = net.simulate(duration, dt, I_ext)
        assert traces.shape == (2, num_steps)

    def test_synapse_propagation(self):
        """Test that synaptic connections work."""
        net = Network(2)
        net.add_synapse(0, 1, weight=1.0, E_syn=0.0, tau=2.0)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Strong current to first neuron

        traces = net.simulate(duration, dt, I_ext)

        # First neuron should spike
        assert np.max(traces[0]) > 0.0


class TestParameters:
    """Tests for the Parameters class."""

    def test_default_values(self):
        """Test default parameter values."""
        params = Parameters()
        assert params.C_m == pytest.approx(1.0)
        assert params.g_Na == pytest.approx(120.0)
        assert params.g_K == pytest.approx(36.0)
        assert params.g_L == pytest.approx(0.3)

    def test_modify_parameters(self):
        """Test modifying parameters."""
        params = Parameters()
        params.g_Na = 150.0
        assert params.g_Na == 150.0


class TestState:
    """Tests for the State class."""

    def test_default_values(self):
        """Test default state values."""
        state = State()
        assert state.V == pytest.approx(-65.0)

    def test_modify_state(self):
        """Test modifying state."""
        state = State()
        state.V = -50.0
        assert state.V == -50.0
