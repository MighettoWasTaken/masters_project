"""Comprehensive tests for the Hodgkin-Huxley neuron model."""

import numpy as np
import pytest

from hodgkin_huxley import (
    IntegrationMethod,
    # Backward compatibility aliases
    Parameters,
    State,
    NeuronModelSpec,
    RegionalNetwork,
    SynapseSpec,
    RecordingConfig,
)


def _make_hh_rn(n=1):
    """Create a RegionalNetwork with n HH-default neurons."""
    rn = RegionalNetwork()
    rn.add_population("E", n, model=NeuronModelSpec.hh_default())
    return rn


class TestHHNeuronBasic:
    """Basic tests for HH-type neurons via NeuronModelSpec.hh_default()."""

    def test_default_creation(self):
        """Test that hh_default() initialises near resting potential."""
        spec = NeuronModelSpec.hh_default()
        assert spec.V_init == pytest.approx(-65.0, abs=1.0)

    def test_custom_parameters(self):
        """Test creating a neuron with custom parameters via spec."""
        spec = NeuronModelSpec.hh_default()
        spec.channels[0].g = 100.0  # Na channel
        spec.channels[1].g = 40.0   # K channel
        assert spec.channels[0].g == pytest.approx(100.0)
        assert spec.channels[1].g == pytest.approx(40.0)

    def test_reset(self):
        """Test reset returns network to initial state."""
        rn = _make_hh_rn()
        rn.simulate(50.0, 0.01, {"E": 20.0})
        rn.reset()
        V = rn._rnet.network().neuron(0).V
        assert V == pytest.approx(-65.0, abs=1.0)

    def test_state_access(self):
        """Test that initial gate values are in valid range."""
        rn = _make_hh_rn()
        # First recorded point is the initial state (before any step)
        result = rn.simulate(0.01, 0.01, {"E": 0.0},
                             record=RecordingConfig(["V", "gates"]))
        assert result["E"]["V"][0, 0] == pytest.approx(-65.0, abs=1.0)
        gates = result["E"]["gates"][0, :, 0]
        assert 0.0 <= gates[0] <= 1.0  # m
        assert 0.0 <= gates[1] <= 1.0  # h
        assert 0.0 <= gates[2] <= 1.0  # n


class TestHHNeuronStep:
    """Tests for neuron step integration."""

    def test_step_positive_current(self):
        """Test that positive current increases voltage."""
        rn = _make_hh_rn()
        spec = NeuronModelSpec.hh_default()
        result = rn.simulate(0.02, 0.01, {"E": 10.0})
        assert result["E"][0, -1] > spec.V_init

    def test_step_negative_current(self):
        """Test that negative current decreases voltage."""
        rn = _make_hh_rn()
        spec = NeuronModelSpec.hh_default()
        result = rn.simulate(0.02, 0.01, {"E": -10.0})
        assert result["E"][0, -1] < spec.V_init

    def test_step_zero_current(self):
        """Test that zero current maintains near-resting potential."""
        rn = _make_hh_rn()
        result = rn.simulate(1.0, 0.01, {"E": 0.0})
        assert result["E"][0, -1] == pytest.approx(-65.0, abs=5.0)


class TestHHNeuronSimulation:
    """Tests for neuron simulation."""

    def test_simulate_constant_current(self):
        """Test simulation with constant current returns correct shape."""
        rn = _make_hh_rn()
        result = rn.simulate(10.0, 0.01, {"E": 10.0})
        assert result["E"].shape == (1, 1000)
        assert isinstance(result["E"], np.ndarray)

    def test_simulate_time_varying_current(self):
        """Test simulation with time-varying (2D) current."""
        rn = _make_hh_rn()
        n_steps = 1000
        I_ext = np.ones((1, n_steps)) * 10.0
        result = rn.simulate(10.0, 0.01, {"E": I_ext})
        assert result["E"].shape == (1, n_steps)

    def test_action_potential_generation(self):
        """Test that sufficient current produces an action potential."""
        rn = _make_hh_rn()
        result = rn.simulate(50.0, 0.01, {"E": 15.0})
        trace = result["E"][0]
        assert np.max(trace) > 0.0   # AP should exceed 0 mV
        assert np.max(trace) < 60.0  # But not exceed realistic bounds
        assert trace[0] < -60.0      # Starts near resting

    def test_no_spike_below_threshold(self):
        """Test that subthreshold current doesn't produce a spike."""
        rn = _make_hh_rn()
        result = rn.simulate(50.0, 0.01, {"E": 2.0})
        assert np.max(result["E"][0]) < -40.0

    def test_spike_count_increases_with_current(self):
        """Test that higher current produces more spikes."""
        def count_spikes(trace, threshold=0.0):
            above = trace > threshold
            return np.sum(np.diff(above.astype(int)) == 1)

        rn1 = _make_hh_rn()
        trace1 = rn1.simulate(500.0, 0.01, {"E": 8.0})["E"][0]

        rn2 = _make_hh_rn()
        trace2 = rn2.simulate(500.0, 0.01, {"E": 15.0})["E"][0]

        assert count_spikes(trace2) > count_spikes(trace1)


class TestHHNeuronEdgeCases:
    """Edge case tests for HH-type neurons."""

    def test_extreme_positive_current(self):
        """Test with very high positive current."""
        rn = _make_hh_rn()
        result = rn.simulate(10.0, 0.01, {"E": 1000.0})
        assert not np.any(np.isnan(result["E"]))
        assert not np.any(np.isinf(result["E"]))

    def test_extreme_negative_current(self):
        """Test with very negative current."""
        rn = _make_hh_rn()
        result = rn.simulate(10.0, 0.01, {"E": -100.0})
        assert not np.any(np.isnan(result["E"]))
        assert not np.any(np.isinf(result["E"]))

    def test_very_small_dt(self):
        """Test with very small timestep."""
        rn = _make_hh_rn()
        result = rn.simulate(0.01, 0.0001, {"E": 10.0})  # 100 steps
        assert not np.any(np.isnan(result["E"]))

    def test_gating_variables_bounded(self):
        """Test that gating variables stay in [0,1]."""
        rn = _make_hh_rn()
        result = rn.simulate(100.0, 0.01, {"E": 20.0},
                             record=RecordingConfig(["gates"]))
        gates = result["E"]["gates"][0]  # (3, n_rec)
        assert np.all(gates >= 0.0)
        assert np.all(gates <= 1.0 + 1e-6)

    def test_zero_conductance(self):
        """Test with zero conductances — no crash, no NaN."""
        spec = NeuronModelSpec.hh_default()
        spec.channels[0].g = 0.0  # Na
        spec.channels[1].g = 0.0  # K
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        result = rn.simulate(10.0, 0.01, {"E": 10.0})
        assert not np.any(np.isnan(result["E"]))

    def test_multiple_resets(self):
        """Test multiple simulation/reset cycles."""
        rn = _make_hh_rn()
        for _ in range(10):
            rn.simulate(10.0, 0.01, {"E": 20.0})
            rn.reset()
            V = rn._rnet.network().neuron(0).V
            assert V == pytest.approx(-65.0, abs=1.0)


class TestNumericalStability:
    """Numerical stability tests."""

    def test_long_simulation(self):
        """Test that long simulations remain stable."""
        rn = _make_hh_rn()
        result = rn.simulate(1000.0, 0.01, {"E": 10.0})
        assert not np.any(np.isnan(result["E"]))
        assert not np.any(np.isinf(result["E"]))

    def test_resting_stability(self):
        """Test that resting neuron stays stable."""
        spec = NeuronModelSpec.hh_default()
        rn = _make_hh_rn()
        result = rn.simulate(100.0, 0.01, {"E": 0.0})
        # Should stay near resting potential
        assert np.all(np.abs(result["E"][0] - spec.V_init) < 5.0)

    def test_oscillation_stability(self):
        """Test that repeated spiking doesn't diverge."""
        rn = _make_hh_rn()
        result = rn.simulate(1000.0, 0.01, {"E": 15.0})
        trace = result["E"][0]
        # Voltage should stay in reasonable bounds
        assert np.min(trace) > -100.0
        assert np.max(trace) < 80.0


class TestNetwork:
    """Tests for RegionalNetwork (replaces legacy Network class)."""

    def test_empty_network(self):
        """Test creating an empty network."""
        rn = RegionalNetwork()
        assert rn.num_neurons == 0
        assert rn.num_synapses == 0

    def test_network_with_neurons(self):
        """Test creating a network with a population of neurons."""
        rn = RegionalNetwork()
        rn.add_population("E", 3)
        assert rn.num_neurons == 3

    def test_add_neuron(self):
        """Test adding a neuron population to network."""
        rn = RegionalNetwork()
        rn.add_population("E", 1)
        assert rn.num_neurons == 1

    def test_add_neuron_custom_params(self):
        """Test adding neuron with custom parameters via spec."""
        spec = NeuronModelSpec.hh_default()
        spec.channels[0].g = 100.0  # custom g_Na
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        assert spec.channels[0].g == pytest.approx(100.0)

    def test_add_synapse(self):
        """Test adding synaptic connections."""
        rn = RegionalNetwork()
        rn.add_population("pre", 1)
        rn.add_population("post", 1)
        rn.connect("pre", "post", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        assert rn.num_synapses == 1

    def test_get_potentials(self):
        """Test getting all membrane potentials."""
        rn = RegionalNetwork()
        rn.add_population("E", 3)
        potentials = rn._rnet.network().get_potentials()
        assert len(potentials) == 3
        assert all(p == pytest.approx(-65.0, abs=1.0) for p in potentials)

    def test_network_reset(self):
        """Test resetting network."""
        rn = RegionalNetwork()
        rn.add_population("E", 2)
        rn._rnet.network().step(0.01, [20.0, 20.0])
        rn.reset()
        potentials = rn._rnet.network().get_potentials()
        assert all(p == pytest.approx(-65.0, abs=1.0) for p in potentials)

    def test_network_step(self):
        """Test single network step via internal API."""
        rn = RegionalNetwork()
        rn.add_population("E", 2)
        net = rn._rnet.network()
        initial = list(net.get_potentials())
        net.step(0.01, [10.0, 0.0])
        after = list(net.get_potentials())
        assert after[0] > initial[0]

    def test_network_simulate(self):
        """Test network simulation."""
        rn = RegionalNetwork()
        rn.add_population("E", 2)
        duration = 10.0
        dt = 0.01
        num_steps = int(duration / dt)

        I_ext = np.zeros((2, num_steps))
        I_ext[0, :] = 10.0

        result = rn.simulate(duration, dt, {"E": I_ext})
        assert result["E"].shape == (2, num_steps)

    def test_synaptic_transmission(self):
        """Test that synaptic connections affect postsynaptic neuron."""
        rn = RegionalNetwork()
        rn.add_population("pre", 1)
        rn.add_population("post", 1)
        rn.connect("pre", "post", "all_to_all", weight=2.0,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))

        result = rn.simulate(200.0, 0.01, {"pre": 15.0, "post": 0.0})
        assert np.max(result["pre"][0]) > 0.0  # pre neuron spikes


class TestNetworkEdgeCases:
    """Edge case tests for RegionalNetwork."""

    def test_self_synapse(self):
        """Test self-connections (autapses)."""
        rn = RegionalNetwork()
        rn.add_population("E", 1)
        rn.connect("E", "E", lambda ns, nd: [(0, 0)], weight=0.1,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        assert rn.num_synapses == 1

    def test_multiple_synapses_same_pair(self):
        """Test multiple synapses between same neurons."""
        rn = RegionalNetwork()
        rn.add_population("pre", 1)
        rn.add_population("post", 1)
        rn.connect("pre", "post", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        rn.connect("pre", "post", "all_to_all", weight=0.3,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        assert rn.num_synapses == 2

    def test_bidirectional_synapses(self):
        """Test bidirectional connections."""
        rn = RegionalNetwork()
        rn.add_population("A", 1)
        rn.add_population("B", 1)
        rn.connect("A", "B", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        rn.connect("B", "A", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        assert rn.num_synapses == 2

    def test_inhibitory_synapse(self):
        """Test inhibitory synapse (negative reversal potential)."""
        rn = RegionalNetwork()
        rn.add_population("pre", 1)
        rn.add_population("post", 1)
        rn.connect("pre", "post", "all_to_all", weight=1.0,
                   synapse=SynapseSpec.exponential(E_syn=-80.0, tau=2.0))

        result = rn.simulate(100.0, 0.01, {"pre": 15.0, "post": 10.0})
        assert not np.any(np.isnan(result["pre"]))
        assert not np.any(np.isnan(result["post"]))

    def test_large_network(self):
        """Test creating a larger network with chain connections."""
        rn = RegionalNetwork()
        rn.add_population("chain", 100)
        rn.connect("chain", "chain",
                   lambda ns, nd: [(i, i + 1) for i in range(99)],
                   weight=0.1,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        assert rn.num_neurons == 100
        assert rn.num_synapses == 99

    def test_invalid_synapse_population(self):
        """Test that connecting to a non-existent population raises error."""
        rn = RegionalNetwork()
        rn.add_population("E", 2)
        with pytest.raises(Exception):
            rn.connect("E", "F", "all_to_all", weight=0.5,
                       synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))


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
