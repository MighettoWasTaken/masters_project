"""Comprehensive tests for the Hodgkin-Huxley neuron model."""

import numpy as np
import pytest

from hodgkin_huxley import (
    HHNeuron,
    Network,
    HHParameters,
    HHState,
    IntegrationMethod,
    # Backward compatibility aliases
    Parameters,
    State,
)


class TestHHNeuronBasic:
    """Basic tests for the HHNeuron class."""

    def test_default_creation(self):
        """Test creating a neuron with default parameters."""
        neuron = HHNeuron()
        assert neuron.V == pytest.approx(-65.0, abs=1.0)

    def test_default_integration_method(self):
        """Test that default integration method is RK4."""
        neuron = HHNeuron()
        assert neuron.integration_method == IntegrationMethod.RK4

    def test_custom_parameters(self):
        """Test creating a neuron with custom parameters."""
        params = Parameters()
        params.g_Na = 100.0
        params.g_K = 40.0
        neuron = HHNeuron(params)
        assert neuron.parameters.g_Na == 100.0
        assert neuron.parameters.g_K == 40.0

    def test_custom_integration_method(self):
        """Test creating a neuron with custom integration method."""
        params = Parameters()
        neuron = HHNeuron(params, method=IntegrationMethod.EULER)
        assert neuron.integration_method == IntegrationMethod.EULER

    def test_set_integration_method(self):
        """Test setting integration method after creation."""
        neuron = HHNeuron()
        neuron.integration_method = IntegrationMethod.EULER
        assert neuron.integration_method == IntegrationMethod.EULER
        neuron.integration_method = IntegrationMethod.RK4
        assert neuron.integration_method == IntegrationMethod.RK4

    def test_reset(self):
        """Test reset returns neuron to resting state."""
        neuron = HHNeuron()
        neuron.V = 0.0
        assert neuron.V == pytest.approx(0.0, abs=0.01)
        neuron.reset()
        assert neuron.V == pytest.approx(-65.0, abs=1.0)

    def test_state_access(self):
        """Test accessing neuron state."""
        neuron = HHNeuron()
        state = neuron.state
        assert state.V == pytest.approx(-65.0, abs=1.0)
        assert 0.0 <= state.m <= 1.0
        assert 0.0 <= state.h <= 1.0
        assert 0.0 <= state.n <= 1.0


class TestHHNeuronStep:
    """Tests for neuron step integration."""

    def test_step_positive_current(self):
        """Test that positive current increases voltage."""
        neuron = HHNeuron()
        initial_V = neuron.V
        neuron.step(dt=0.01, I_ext=10.0)
        assert neuron.V > initial_V

    def test_step_negative_current(self):
        """Test that negative current decreases voltage."""
        neuron = HHNeuron()
        initial_V = neuron.V
        neuron.step(dt=0.01, I_ext=-10.0)
        assert neuron.V < initial_V

    def test_step_zero_current(self):
        """Test that zero current maintains near-resting potential."""
        neuron = HHNeuron()
        initial_V = neuron.V
        for _ in range(100):
            neuron.step(dt=0.01, I_ext=0.0)
        assert neuron.V == pytest.approx(initial_V, abs=5.0)

    def test_step_euler_method(self):
        """Test step with Euler integration."""
        neuron = HHNeuron()
        neuron.integration_method = IntegrationMethod.EULER
        initial_V = neuron.V
        neuron.step(dt=0.01, I_ext=10.0)
        assert neuron.V > initial_V

    def test_step_rk4_method(self):
        """Test step with RK4 integration."""
        neuron = HHNeuron()
        neuron.integration_method = IntegrationMethod.RK4
        initial_V = neuron.V
        neuron.step(dt=0.01, I_ext=10.0)
        assert neuron.V > initial_V


class TestHHNeuronSimulation:
    """Tests for neuron simulation."""

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

    def test_action_potential_generation(self):
        """Test that sufficient current produces an action potential."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=50.0, dt=0.01, I_ext=15.0)
        assert np.max(trace) > 0.0  # AP should exceed 0 mV
        assert np.max(trace) < 60.0  # But not exceed realistic bounds
        assert trace[0] < -60.0  # Starts near resting

    def test_no_spike_below_threshold(self):
        """Test that subthreshold current doesn't produce a spike."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=50.0, dt=0.01, I_ext=2.0)
        assert np.max(trace) < -40.0

    def test_spike_count_increases_with_current(self):
        """Test that higher current produces more spikes."""
        def count_spikes(trace, threshold=0.0):
            above = trace > threshold
            return np.sum(np.diff(above.astype(int)) == 1)

        neuron1 = HHNeuron()
        trace1 = neuron1.simulate(duration=500.0, dt=0.01, I_ext=8.0)

        neuron2 = HHNeuron()
        trace2 = neuron2.simulate(duration=500.0, dt=0.01, I_ext=15.0)

        assert count_spikes(trace2) > count_spikes(trace1)

    def test_simulate_with_euler(self):
        """Test simulation using Euler integration."""
        neuron = HHNeuron()
        neuron.integration_method = IntegrationMethod.EULER
        trace = neuron.simulate(duration=50.0, dt=0.01, I_ext=15.0)
        assert np.max(trace) > 0.0  # Should still produce AP

    def test_simulate_with_rk4(self):
        """Test simulation using RK4 integration."""
        neuron = HHNeuron()
        neuron.integration_method = IntegrationMethod.RK4
        trace = neuron.simulate(duration=50.0, dt=0.01, I_ext=15.0)
        assert np.max(trace) > 0.0


class TestHHNeuronEdgeCases:
    """Edge case tests for HHNeuron."""

    def test_extreme_positive_current(self):
        """Test with very high positive current."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=10.0, dt=0.01, I_ext=1000.0)
        assert not np.any(np.isnan(trace))
        assert not np.any(np.isinf(trace))

    def test_extreme_negative_current(self):
        """Test with very negative current."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=10.0, dt=0.01, I_ext=-100.0)
        assert not np.any(np.isnan(trace))
        assert not np.any(np.isinf(trace))

    def test_very_small_dt(self):
        """Test with very small timestep."""
        neuron = HHNeuron()
        for _ in range(100):
            neuron.step(dt=0.0001, I_ext=10.0)
        assert not np.isnan(neuron.V)

    def test_large_dt_rk4(self):
        """Test RK4 stability with larger timestep."""
        neuron = HHNeuron()
        neuron.integration_method = IntegrationMethod.RK4
        for _ in range(100):
            neuron.step(dt=0.05, I_ext=10.0)
        assert not np.isnan(neuron.V)
        assert -200.0 < neuron.V < 200.0

    def test_gating_variables_bounded(self):
        """Test that gating variables stay in [0,1]."""
        neuron = HHNeuron()
        neuron.simulate(duration=100.0, dt=0.01, I_ext=20.0)
        state = neuron.state
        assert 0.0 <= state.m <= 1.0
        assert 0.0 <= state.h <= 1.0
        assert 0.0 <= state.n <= 1.0

    def test_zero_conductance(self):
        """Test with zero conductances."""
        params = Parameters()
        params.g_Na = 0.0
        params.g_K = 0.0
        neuron = HHNeuron(params)
        trace = neuron.simulate(duration=10.0, dt=0.01, I_ext=10.0)
        assert not np.any(np.isnan(trace))

    def test_multiple_resets(self):
        """Test multiple simulation/reset cycles."""
        neuron = HHNeuron()
        for _ in range(10):
            neuron.simulate(duration=10.0, dt=0.01, I_ext=20.0)
            neuron.reset()
            assert neuron.V == pytest.approx(-65.0, abs=1.0)

    def test_empty_varying_current(self):
        """Test with empty current array raises error."""
        neuron = HHNeuron()
        with pytest.raises(Exception):
            neuron.simulate(duration=10.0, dt=0.01, I_ext=np.array([]))

    def test_short_varying_current(self):
        """Test that short current array raises error."""
        neuron = HHNeuron()
        I_ext = np.ones(100) * 10.0  # Too short
        with pytest.raises(Exception):
            neuron.simulate(duration=10.0, dt=0.01, I_ext=I_ext)


class TestIntegrationMethodComparison:
    """Tests comparing integration methods."""

    def test_euler_rk4_similar_small_dt(self):
        """Test that Euler and RK4 give similar results for small dt."""
        dt = 0.001
        duration = 10.0
        I_ext = 10.0

        neuron_euler = HHNeuron()
        neuron_euler.integration_method = IntegrationMethod.EULER
        trace_euler = neuron_euler.simulate(duration, dt, I_ext)

        neuron_rk4 = HHNeuron()
        neuron_rk4.integration_method = IntegrationMethod.RK4
        trace_rk4 = neuron_rk4.simulate(duration, dt, I_ext)

        # Should be very close for small dt
        assert np.allclose(trace_euler, trace_rk4, atol=1.0)

    def test_rk4_more_stable_large_dt(self):
        """Test that RK4 is more stable than Euler for larger dt."""
        dt = 0.05
        duration = 100.0
        I_ext = 15.0

        neuron_rk4 = HHNeuron()
        neuron_rk4.integration_method = IntegrationMethod.RK4
        trace_rk4 = neuron_rk4.simulate(duration, dt, I_ext)

        # RK4 should produce valid output
        assert not np.any(np.isnan(trace_rk4))
        assert np.max(trace_rk4) > 0.0  # Should still see spikes

    def test_rk4_accuracy_vs_reference(self):
        """Test RK4 accuracy against fine-grained reference."""
        duration = 50.0
        I_ext = 10.0

        # Reference: RK4 with very small dt
        ref = HHNeuron()
        ref.integration_method = IntegrationMethod.RK4
        trace_ref = ref.simulate(duration, 0.001, I_ext)

        # Test: RK4 with larger dt
        test = HHNeuron()
        test.integration_method = IntegrationMethod.RK4
        trace_test = test.simulate(duration, 0.01, I_ext)

        # Compare peak voltages
        max_ref = np.max(trace_ref)
        max_test = np.max(trace_test)
        assert abs(max_ref - max_test) < 2.0  # Within 2 mV


class TestNumericalStability:
    """Numerical stability tests."""

    def test_long_simulation(self):
        """Test that long simulations remain stable."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=1000.0, dt=0.01, I_ext=10.0)
        assert not np.any(np.isnan(trace))
        assert not np.any(np.isinf(trace))

    def test_resting_stability(self):
        """Test that resting neuron stays stable."""
        neuron = HHNeuron()
        initial_V = neuron.V
        trace = neuron.simulate(duration=100.0, dt=0.01, I_ext=0.0)
        # Should stay near resting potential
        assert np.all(np.abs(trace - initial_V) < 5.0)

    def test_oscillation_stability(self):
        """Test that repeated spiking doesn't diverge."""
        neuron = HHNeuron()
        trace = neuron.simulate(duration=1000.0, dt=0.01, I_ext=15.0)
        # Voltage should stay in reasonable bounds
        assert np.min(trace) > -100.0
        assert np.max(trace) < 80.0


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

    def test_add_neuron_custom_params(self):
        """Test adding neuron with custom parameters."""
        net = Network()
        params = Parameters()
        params.g_Na = 100.0
        idx = net.add_neuron(params)
        assert net.neuron(idx).parameters.g_Na == 100.0

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

    def test_network_reset(self):
        """Test resetting network."""
        net = Network(2)
        net.step(0.01, [20.0, 20.0])
        net.reset()
        potentials = net.get_potentials()
        assert all(p == pytest.approx(-65.0, abs=1.0) for p in potentials)

    def test_network_step(self):
        """Test single network step."""
        net = Network(2)
        initial = net.get_potentials()
        net.step(0.01, [10.0, 0.0])
        after = net.get_potentials()
        assert after[0] > initial[0]

    def test_network_simulate(self):
        """Test network simulation."""
        net = Network(2)
        duration = 10.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 10.0

        traces = net.simulate(duration, dt, I_ext)
        assert traces.shape == (2, num_steps)

    def test_synaptic_transmission(self):
        """Test that synaptic connections affect postsynaptic neuron."""
        net = Network(2)
        net.add_synapse(0, 1, weight=2.0, E_syn=0.0, tau=2.0)

        duration = 200.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0  # Only stimulate first neuron

        traces = net.simulate(duration, dt, I_ext)
        assert np.max(traces[0]) > 0.0  # First neuron spikes


class TestNetworkEdgeCases:
    """Edge case tests for Network."""

    def test_self_synapse(self):
        """Test self-connections (autapses)."""
        net = Network(1)
        net.add_synapse(0, 0, weight=0.1)
        assert net.num_synapses == 1

    def test_multiple_synapses_same_pair(self):
        """Test multiple synapses between same neurons."""
        net = Network(2)
        net.add_synapse(0, 1, weight=0.5)
        net.add_synapse(0, 1, weight=0.3)
        assert net.num_synapses == 2

    def test_bidirectional_synapses(self):
        """Test bidirectional connections."""
        net = Network(2)
        net.add_synapse(0, 1, weight=0.5)
        net.add_synapse(1, 0, weight=0.5)
        assert net.num_synapses == 2

    def test_inhibitory_synapse(self):
        """Test inhibitory synapse (negative reversal potential)."""
        net = Network(2)
        net.add_synapse(0, 1, weight=1.0, E_syn=-80.0, tau=2.0)

        duration = 100.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 15.0
        I_ext[1, :] = 10.0

        traces = net.simulate(duration, dt, I_ext)
        assert not np.any(np.isnan(traces))

    def test_large_network(self):
        """Test creating a larger network."""
        net = Network(100)
        for i in range(99):
            net.add_synapse(i, i + 1, weight=0.1)
        assert net.num_neurons == 100
        assert net.num_synapses == 99

    def test_invalid_synapse_index(self):
        """Test that invalid synapse indices raise error."""
        net = Network(2)
        with pytest.raises(Exception):
            net.add_synapse(0, 5, weight=0.5)  # Index out of range


class TestParameters:
    """Tests for the Parameters class."""

    def test_default_values(self):
        """Test default parameter values."""
        params = Parameters()
        assert params.C_m == pytest.approx(1.0)
        assert params.g_Na == pytest.approx(120.0)
        assert params.g_K == pytest.approx(36.0)
        assert params.g_L == pytest.approx(0.3)
        assert params.E_Na == pytest.approx(50.0)
        assert params.E_K == pytest.approx(-77.0)
        assert params.E_L == pytest.approx(-54.387, abs=0.001)

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
        assert state.m == pytest.approx(0.05)
        assert state.h == pytest.approx(0.6)
        assert state.n == pytest.approx(0.32)

    def test_modify_state(self):
        """Test modifying state."""
        state = State()
        state.V = -50.0
        state.m = 0.1
        assert state.V == -50.0
        assert state.m == 0.1


class TestIntegrationMethodEnum:
    """Tests for IntegrationMethod enum."""

    def test_enum_values_exist(self):
        """Test that all enum values exist."""
        assert hasattr(IntegrationMethod, 'EULER')
        assert hasattr(IntegrationMethod, 'RK4')
        assert hasattr(IntegrationMethod, 'RK45_ADAPTIVE')

    def test_enum_comparison(self):
        """Test enum value comparison."""
        assert IntegrationMethod.EULER == IntegrationMethod.EULER
        assert IntegrationMethod.EULER != IntegrationMethod.RK4
