"""Comprehensive tests for the Izhikevich neuron model."""

import numpy as np
import pytest

from hodgkin_huxley import (
    IzhikevichType,
    NeuronModelSpec,
    RegionalNetwork,
    RecordingConfig,
)


def _make_iz_rn(neuron_type=IzhikevichType.REGULAR_SPIKING):
    """Create a single-neuron RegionalNetwork with given Izhikevich type."""
    rn = RegionalNetwork()
    rn.add_population("E", 1, model=NeuronModelSpec.izhikevich(neuron_type))
    return rn


class TestIzhikevichNeuronBasic:
    """Basic tests for IzhikevichNeuron via NeuronModelSpec.izhikevich()."""

    def test_default_creation(self):
        """Test that izhikevich() initialises at resting potential."""
        spec = NeuronModelSpec.izhikevich()
        # Default is RS: V_init == c == -65
        assert spec.V_init == pytest.approx(spec.iz_params.c, abs=1.0)

    def test_default_is_regular_spiking(self):
        """Test that default parameters are regular spiking."""
        spec = NeuronModelSpec.izhikevich()
        assert spec.iz_params.a == pytest.approx(0.02)
        assert spec.iz_params.b == pytest.approx(0.2)
        assert spec.iz_params.c == pytest.approx(-65.0)
        assert spec.iz_params.d == pytest.approx(8.0)

    def test_create_with_type(self):
        """Test creating neurons with preset types."""
        for neuron_type in [
            IzhikevichType.REGULAR_SPIKING,
            IzhikevichType.FAST_SPIKING,
            IzhikevichType.INTRINSICALLY_BURSTING,
            IzhikevichType.CHATTERING,
            IzhikevichType.LOW_THRESHOLD_SPIKING,
        ]:
            spec = NeuronModelSpec.izhikevich(neuron_type)
            assert spec.V_init == pytest.approx(spec.iz_params.c, abs=1.0)

    def test_create_with_custom_parameters(self):
        """Test creating a neuron spec with custom parameters."""
        spec = NeuronModelSpec.izhikevich()
        spec.iz_params.a = 0.05
        spec.iz_params.b = 0.25
        spec.iz_params.c = -60.0
        spec.iz_params.d = 4.0
        assert spec.iz_params.a == pytest.approx(0.05)
        assert spec.iz_params.b == pytest.approx(0.25)
        assert spec.iz_params.c == pytest.approx(-60.0)
        assert spec.iz_params.d == pytest.approx(4.0)

    def test_reset(self):
        """Test reset returns network to initial state."""
        rn = _make_iz_rn()
        rn.simulate(100.0, 0.1, {"E": 20.0})
        rn.reset()
        V = rn._rnet.network().neuron(0).V
        spec = NeuronModelSpec.izhikevich()
        assert V == pytest.approx(spec.iz_params.c, abs=1.0)

    def test_state_access(self):
        """Test that initial state values are correct."""
        spec = NeuronModelSpec.izhikevich()
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        # First recorded point is the initial state (before any step)
        result = rn.simulate(0.1, 0.1, {"E": 0.0},
                             record=RecordingConfig(["V", "u"]))
        assert result["E"]["V"][0, 0] == pytest.approx(spec.iz_params.c, abs=1.0)
        expected_u = spec.iz_params.b * spec.iz_params.c
        assert result["E"]["u"][0, 0] == pytest.approx(expected_u, abs=1.0)

    def test_recovery_variable(self):
        """Test that initial recovery variable u ≈ b * c."""
        spec = NeuronModelSpec.izhikevich()
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        # First recorded point is the initial state
        result = rn.simulate(0.1, 0.1, {"E": 0.0}, record=RecordingConfig(["u"]))
        expected_u = spec.iz_params.b * spec.iz_params.c  # 0.2 * -65 = -13
        assert result["E"]["u"][0, 0] == pytest.approx(expected_u, abs=1.0)


class TestIzhikevichPresets:
    """Tests for preset neuron types via NeuronModelSpec.izhikevich(type)."""

    def test_regular_spiking_preset(self):
        """Test regular spiking preset parameters."""
        spec = NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING)
        assert spec.iz_params.a == pytest.approx(0.02)
        assert spec.iz_params.b == pytest.approx(0.2)
        assert spec.iz_params.c == pytest.approx(-65.0)
        assert spec.iz_params.d == pytest.approx(8.0)

    def test_fast_spiking_preset(self):
        """Test fast spiking preset parameters."""
        spec = NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING)
        assert spec.iz_params.a == pytest.approx(0.1)
        assert spec.iz_params.b == pytest.approx(0.2)
        assert spec.iz_params.c == pytest.approx(-65.0)
        assert spec.iz_params.d == pytest.approx(2.0)

    def test_intrinsically_bursting_preset(self):
        """Test intrinsically bursting preset parameters."""
        spec = NeuronModelSpec.izhikevich(IzhikevichType.INTRINSICALLY_BURSTING)
        assert spec.iz_params.a == pytest.approx(0.02)
        assert spec.iz_params.b == pytest.approx(0.2)
        assert spec.iz_params.c == pytest.approx(-55.0)
        assert spec.iz_params.d == pytest.approx(4.0)

    def test_chattering_preset(self):
        """Test chattering preset parameters."""
        spec = NeuronModelSpec.izhikevich(IzhikevichType.CHATTERING)
        assert spec.iz_params.a == pytest.approx(0.02)
        assert spec.iz_params.b == pytest.approx(0.2)
        assert spec.iz_params.c == pytest.approx(-50.0)
        assert spec.iz_params.d == pytest.approx(2.0)

    def test_low_threshold_spiking_preset(self):
        """Test low threshold spiking preset parameters."""
        spec = NeuronModelSpec.izhikevich(IzhikevichType.LOW_THRESHOLD_SPIKING)
        assert spec.iz_params.a == pytest.approx(0.02)
        assert spec.iz_params.b == pytest.approx(0.25)
        assert spec.iz_params.c == pytest.approx(-65.0)
        assert spec.iz_params.d == pytest.approx(2.0)


class TestIzhikevichNeuronStep:
    """Tests for neuron step integration."""

    def test_step_positive_current(self):
        """Test that positive current increases voltage."""
        spec = NeuronModelSpec.izhikevich()
        rn = _make_iz_rn()
        result = rn.simulate(0.2, 0.1, {"E": 10.0})
        assert result["E"][0, -1] > spec.iz_params.c

    def test_step_negative_current(self):
        """Test that negative current decreases voltage."""
        spec = NeuronModelSpec.izhikevich()
        rn = _make_iz_rn()
        result = rn.simulate(0.2, 0.1, {"E": -10.0})
        assert result["E"][0, -1] < spec.iz_params.c

    def test_step_zero_current(self):
        """Test that zero current maintains near-resting potential."""
        spec = NeuronModelSpec.izhikevich()
        rn = _make_iz_rn()
        result = rn.simulate(10.0, 0.1, {"E": 0.0})
        assert result["E"][0, -1] == pytest.approx(spec.iz_params.c, abs=10.0)

    def test_spike_detection(self):
        """Test that spikes are detected correctly."""
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": 15.0},
                             record=RecordingConfig(["spikes"]))
        assert len(result["E"]["spikes"][0]) > 0

    def test_spike_reset(self):
        """Test that voltage resets after spike — voltage stays bounded."""
        spec = NeuronModelSpec.izhikevich()
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": 15.0})
        trace = result["E"][0]
        # After reset, voltage drops back near c (not a sustained high plateau)
        below_c = trace < spec.iz_params.c + 5.0
        assert np.any(below_c)


class TestIzhikevichNeuronSimulation:
    """Tests for neuron simulation."""

    def test_simulate_constant_current(self):
        """Test simulation with constant current returns correct shape."""
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": 10.0})
        assert result["E"].shape == (1, 1000)
        assert isinstance(result["E"], np.ndarray)

    def test_simulate_time_varying_current(self):
        """Test simulation with time-varying (2D) current."""
        rn = _make_iz_rn()
        n_steps = 1000
        I_ext = np.ones((1, n_steps)) * 10.0
        result = rn.simulate(100.0, 0.1, {"E": I_ext})
        assert result["E"].shape == (1, n_steps)

    def test_spike_generation(self):
        """Test that sufficient current produces spikes."""
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": 15.0})
        assert np.max(result["E"][0]) > 0.0

    def test_no_spike_below_threshold(self):
        """Test that subthreshold current doesn't produce spikes."""
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": 1.0})
        assert np.max(result["E"][0]) < 0.0

    def test_spike_count_increases_with_current(self):
        """Test that higher current produces more spikes."""
        def count_spikes(trace, threshold=0.0):
            above = trace > threshold
            return np.sum(np.diff(above.astype(int)) == 1)

        rn1 = _make_iz_rn()
        trace1 = rn1.simulate(500.0, 0.1, {"E": 8.0})["E"][0]

        rn2 = _make_iz_rn()
        trace2 = rn2.simulate(500.0, 0.1, {"E": 20.0})["E"][0]

        assert count_spikes(trace2) > count_spikes(trace1)


class TestIzhikevichSpikingPatterns:
    """Tests for different spiking patterns."""

    def test_regular_spiking_pattern(self):
        """Test that RS neurons show regular spiking."""
        rn = _make_iz_rn(IzhikevichType.REGULAR_SPIKING)
        result = rn.simulate(500.0, 0.1, {"E": 10.0})
        trace = result["E"][0]
        assert np.max(trace) > 0.0
        # Should not be bursting (voltage shouldn't stay high)
        high_voltage_count = np.sum(trace > 0.0)
        assert high_voltage_count < len(trace) * 0.3

    def test_fast_spiking_pattern(self):
        """Test that FS neurons spike faster than RS."""
        def count_spikes(trace, threshold=0.0):
            above = trace > threshold
            return np.sum(np.diff(above.astype(int)) == 1)

        rn_rs = _make_iz_rn(IzhikevichType.REGULAR_SPIKING)
        rn_fs = _make_iz_rn(IzhikevichType.FAST_SPIKING)

        trace_rs = rn_rs.simulate(500.0, 0.1, {"E": 15.0})["E"][0]
        trace_fs = rn_fs.simulate(500.0, 0.1, {"E": 15.0})["E"][0]

        assert count_spikes(trace_fs) > count_spikes(trace_rs)

    def test_bursting_pattern(self):
        """Test that IB neurons show bursting behavior."""
        rn = _make_iz_rn(IzhikevichType.INTRINSICALLY_BURSTING)
        result = rn.simulate(500.0, 0.1, {"E": 10.0})
        assert np.max(result["E"][0]) > 0.0


class TestIzhikevichEdgeCases:
    """Edge case tests for Izhikevich neurons."""

    def test_extreme_positive_current(self):
        """Test with very high positive current."""
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": 100.0})
        assert not np.any(np.isnan(result["E"]))
        assert not np.any(np.isinf(result["E"]))

    def test_extreme_negative_current(self):
        """Test with very negative current."""
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": -100.0})
        assert not np.any(np.isnan(result["E"]))
        assert not np.any(np.isinf(result["E"]))

    def test_very_small_dt(self):
        """Test with very small timestep via internal step loop."""
        rn = _make_iz_rn()
        net = rn._rnet.network()
        for _ in range(1000):
            net.step(0.01, [10.0])
        assert not np.isnan(net.neuron(0).V)

    def test_large_dt(self):
        """Test stability with larger timestep."""
        rn = _make_iz_rn()
        net = rn._rnet.network()
        for _ in range(100):
            net.step(1.0, [10.0])
        V = net.neuron(0).V
        assert not np.isnan(V)
        assert -200.0 < V < 200.0

    def test_voltage_bounded(self):
        """Test that voltage stays bounded during simulation."""
        rn = _make_iz_rn()
        result = rn.simulate(1000.0, 0.1, {"E": 20.0})
        trace = result["E"][0]
        # Safety clamp at 100 mV is in IzPool::step_euler
        assert np.max(trace) <= 100.0
        assert np.min(trace) > -200.0

    def test_multiple_resets(self):
        """Test multiple simulation/reset cycles."""
        spec = NeuronModelSpec.izhikevich()
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        reset_V = spec.iz_params.c
        for _ in range(10):
            rn.simulate(50.0, 0.1, {"E": 20.0})
            rn.reset()
            V = rn._rnet.network().neuron(0).V
            assert V == pytest.approx(reset_V, abs=1.0)


class TestIzhikevichNumericalStability:
    """Numerical stability tests."""

    def test_long_simulation(self):
        """Test that long simulations remain stable."""
        rn = _make_iz_rn()
        result = rn.simulate(10000.0, 0.1, {"E": 10.0})
        assert not np.any(np.isnan(result["E"]))
        assert not np.any(np.isinf(result["E"]))

    def test_resting_stability(self):
        """Test that resting neuron stays stable."""
        spec = NeuronModelSpec.izhikevich()
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        result = rn.simulate(1000.0, 0.1, {"E": 0.0})
        initial_V = spec.iz_params.c
        assert np.all(np.abs(result["E"][0] - initial_V) < 20.0)

    def test_oscillation_stability(self):
        """Test that repeated spiking doesn't diverge."""
        rn = _make_iz_rn()
        result = rn.simulate(5000.0, 0.1, {"E": 15.0})
        trace = result["E"][0]
        assert np.min(trace) > -150.0
        assert np.max(trace) <= 100.0

    def test_u_recording_finite(self):
        """Test that recovery variable u remains finite throughout."""
        rn = _make_iz_rn()
        result = rn.simulate(100.0, 0.1, {"E": 15.0},
                             record=RecordingConfig(["u"]))
        u_trace = result["E"]["u"][0]
        assert np.all(np.isfinite(u_trace))


class TestIzhikevichParameters:
    """Tests for Izhikevich parameter access via NeuronModelSpec."""

    def test_default_values(self):
        """Test default parameter values."""
        iz_params = NeuronModelSpec.izhikevich().iz_params
        assert iz_params.a == pytest.approx(0.02)
        assert iz_params.b == pytest.approx(0.2)
        assert iz_params.c == pytest.approx(-65.0)
        assert iz_params.d == pytest.approx(8.0)

    def test_modify_parameters(self):
        """Test modifying parameters via spec."""
        spec = NeuronModelSpec.izhikevich()
        spec.iz_params.a = 0.1
        spec.iz_params.b = 0.25
        spec.iz_params.c = -60.0
        spec.iz_params.d = 4.0
        assert spec.iz_params.a == pytest.approx(0.1)
        assert spec.iz_params.b == pytest.approx(0.25)
        assert spec.iz_params.c == pytest.approx(-60.0)
        assert spec.iz_params.d == pytest.approx(4.0)


class TestIzhikevichURecording:
    """Tests for Izhikevich recovery variable u via RecordingConfig."""

    def test_initial_u_value(self):
        """Test initial u value ≈ b * c = 0.2 * -65 = -13."""
        spec = NeuronModelSpec.izhikevich()
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        result = rn.simulate(0.1, 0.1, {"E": 0.0}, record=RecordingConfig(["u"]))
        expected_u = spec.iz_params.b * spec.iz_params.c
        assert result["E"]["u"][0, 0] == pytest.approx(expected_u, abs=1.0)

    def test_u_changes_during_spiking(self):
        """Test that u changes during spiking activity."""
        spec = NeuronModelSpec.izhikevich()
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=spec)
        result = rn.simulate(100.0, 0.1, {"E": 15.0}, record=RecordingConfig(["u"]))
        u_trace = result["E"]["u"][0]
        initial_u = spec.iz_params.b * spec.iz_params.c
        # u should change from its initial value during spiking
        assert not np.allclose(u_trace, initial_u, atol=1e-3)

    def test_u_shape(self):
        """Test that u recording has correct shape."""
        rn = _make_iz_rn()
        duration = 100.0
        dt = 0.1
        result = rn.simulate(duration, dt, {"E": 10.0},
                             record=RecordingConfig(["u"]))
        n_steps = int(duration / dt)
        assert result["E"]["u"].shape == (1, n_steps)

    def test_u_zero_for_hh_neurons(self):
        """Test that u recording is all zeros for non-Izhikevich neurons."""
        rn = RegionalNetwork()
        rn.add_population("E", 1, model=NeuronModelSpec.hh_default())
        result = rn.simulate(10.0, 0.1, {"E": 10.0}, record=RecordingConfig(["u"]))
        assert np.all(result["E"]["u"] == 0.0)


class TestIzhikevichTypeEnum:
    """Tests for IzhikevichType enum."""

    def test_enum_values_exist(self):
        """Test that all enum values exist."""
        assert hasattr(IzhikevichType, 'REGULAR_SPIKING')
        assert hasattr(IzhikevichType, 'FAST_SPIKING')
        assert hasattr(IzhikevichType, 'INTRINSICALLY_BURSTING')
        assert hasattr(IzhikevichType, 'CHATTERING')
        assert hasattr(IzhikevichType, 'LOW_THRESHOLD_SPIKING')
        assert hasattr(IzhikevichType, 'CUSTOM')

    def test_enum_comparison(self):
        """Test enum value comparison."""
        assert IzhikevichType.REGULAR_SPIKING == IzhikevichType.REGULAR_SPIKING
        assert IzhikevichType.REGULAR_SPIKING != IzhikevichType.FAST_SPIKING
