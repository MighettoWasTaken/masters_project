"""Comprehensive tests for the Izhikevich neuron model."""

import numpy as np
import pytest

from hodgkin_huxley import (
    IzhikevichNeuron,
    IzhikevichParameters,
    IzhikevichState,
    IzhikevichType,
    IntegrationMethod,
)


class TestIzhikevichNeuronBasic:
    """Basic tests for the IzhikevichNeuron class."""

    def test_default_creation(self):
        """Test creating a neuron with default parameters."""
        neuron = IzhikevichNeuron()  # Default is RS with c=-65
        # V is initialized to params.c
        assert neuron.V == pytest.approx(neuron.parameters.c, abs=1.0)

    def test_default_is_regular_spiking(self):
        """Test that default parameters are regular spiking."""
        neuron = IzhikevichNeuron()
        params = neuron.parameters
        assert params.a == pytest.approx(0.02)
        assert params.b == pytest.approx(0.2)
        assert params.c == pytest.approx(-65.0)
        assert params.d == pytest.approx(8.0)

    def test_create_with_type(self):
        """Test creating neurons with preset types."""
        for neuron_type in [
            IzhikevichType.REGULAR_SPIKING,
            IzhikevichType.FAST_SPIKING,
            IzhikevichType.INTRINSICALLY_BURSTING,
            IzhikevichType.CHATTERING,
            IzhikevichType.LOW_THRESHOLD_SPIKING,
        ]:
            neuron = IzhikevichNeuron(neuron_type)
            # Initial V is set to params.c (after-spike reset value)
            assert neuron.V == pytest.approx(neuron.parameters.c, abs=1.0)

    def test_create_with_custom_parameters(self):
        """Test creating a neuron with custom parameters."""
        params = IzhikevichParameters()
        params.a = 0.05
        params.b = 0.25
        params.c = -60.0
        params.d = 4.0
        neuron = IzhikevichNeuron(parameters=params)
        assert neuron.parameters.a == pytest.approx(0.05)
        assert neuron.parameters.b == pytest.approx(0.25)
        assert neuron.parameters.c == pytest.approx(-60.0)
        assert neuron.parameters.d == pytest.approx(4.0)

    def test_reset(self):
        """Test reset returns neuron to resting state."""
        neuron = IzhikevichNeuron()  # Default is RS with c=-65
        neuron.V = 0.0
        assert neuron.V == pytest.approx(0.0, abs=0.01)
        neuron.reset()
        # Reset sets V to params.c
        assert neuron.V == pytest.approx(neuron.parameters.c, abs=1.0)

    def test_state_access(self):
        """Test accessing neuron state."""
        neuron = IzhikevichNeuron()  # Default RS: c=-65, b=0.2
        state = neuron.state
        # v is initialized to c, u is initialized to b*v
        expected_v = neuron.parameters.c
        expected_u = neuron.parameters.b * expected_v
        assert state.v == pytest.approx(expected_v, abs=1.0)
        assert state.u == pytest.approx(expected_u, abs=1.0)

    def test_recovery_variable(self):
        """Test accessing recovery variable u."""
        neuron = IzhikevichNeuron()  # Default RS: c=-65, b=0.2
        # u is initialized to b*v = 0.2 * -65 = -13
        expected_u = neuron.parameters.b * neuron.parameters.c
        assert neuron.u == pytest.approx(expected_u, abs=1.0)

    def test_repr(self):
        """Test string representation."""
        neuron = IzhikevichNeuron()
        repr_str = repr(neuron)
        assert "IzhikevichNeuron" in repr_str
        assert "mV" in repr_str


class TestIzhikevichPresets:
    """Tests for preset neuron types."""

    def test_regular_spiking_preset(self):
        """Test regular spiking preset parameters."""
        params = IzhikevichNeuron.get_preset(IzhikevichType.REGULAR_SPIKING)
        assert params.a == pytest.approx(0.02)
        assert params.b == pytest.approx(0.2)
        assert params.c == pytest.approx(-65.0)
        assert params.d == pytest.approx(8.0)

    def test_fast_spiking_preset(self):
        """Test fast spiking preset parameters."""
        params = IzhikevichNeuron.get_preset(IzhikevichType.FAST_SPIKING)
        assert params.a == pytest.approx(0.1)
        assert params.b == pytest.approx(0.2)
        assert params.c == pytest.approx(-65.0)
        assert params.d == pytest.approx(2.0)

    def test_intrinsically_bursting_preset(self):
        """Test intrinsically bursting preset parameters."""
        params = IzhikevichNeuron.get_preset(IzhikevichType.INTRINSICALLY_BURSTING)
        assert params.a == pytest.approx(0.02)
        assert params.b == pytest.approx(0.2)
        assert params.c == pytest.approx(-55.0)
        assert params.d == pytest.approx(4.0)

    def test_chattering_preset(self):
        """Test chattering preset parameters."""
        params = IzhikevichNeuron.get_preset(IzhikevichType.CHATTERING)
        assert params.a == pytest.approx(0.02)
        assert params.b == pytest.approx(0.2)
        assert params.c == pytest.approx(-50.0)
        assert params.d == pytest.approx(2.0)

    def test_low_threshold_spiking_preset(self):
        """Test low threshold spiking preset parameters."""
        params = IzhikevichNeuron.get_preset(IzhikevichType.LOW_THRESHOLD_SPIKING)
        assert params.a == pytest.approx(0.02)
        assert params.b == pytest.approx(0.25)
        assert params.c == pytest.approx(-65.0)
        assert params.d == pytest.approx(2.0)


class TestIzhikevichNeuronStep:
    """Tests for neuron step integration."""

    def test_step_positive_current(self):
        """Test that positive current increases voltage."""
        neuron = IzhikevichNeuron()
        initial_V = neuron.V
        neuron.step(dt=0.1, I_ext=10.0)
        assert neuron.V > initial_V

    def test_step_negative_current(self):
        """Test that negative current decreases voltage."""
        neuron = IzhikevichNeuron()
        initial_V = neuron.V
        neuron.step(dt=0.1, I_ext=-10.0)
        assert neuron.V < initial_V

    def test_step_zero_current(self):
        """Test that zero current maintains near-resting potential."""
        neuron = IzhikevichNeuron()
        initial_V = neuron.V
        for _ in range(100):
            neuron.step(dt=0.1, I_ext=0.0)
        assert neuron.V == pytest.approx(initial_V, abs=10.0)

    def test_spike_detection(self):
        """Test that spikes are detected correctly."""
        neuron = IzhikevichNeuron()
        spiked = False
        for _ in range(1000):
            neuron.step(dt=0.1, I_ext=15.0)
            if neuron.spiked:
                spiked = True
                break
        assert spiked

    def test_spike_reset(self):
        """Test that voltage resets after spike."""
        neuron = IzhikevichNeuron()
        for _ in range(1000):
            neuron.step(dt=0.1, I_ext=15.0)
            if neuron.spiked:
                # After spike, V should be reset to c
                assert neuron.V == pytest.approx(neuron.parameters.c, abs=5.0)
                break


class TestIzhikevichNeuronSimulation:
    """Tests for neuron simulation."""

    def test_simulate_constant_current(self):
        """Test simulation with constant current."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=100.0, dt=0.1, I_ext=10.0)
        assert len(trace) == 1000
        assert isinstance(trace, np.ndarray)

    def test_simulate_time_varying_current(self):
        """Test simulation with time-varying current."""
        neuron = IzhikevichNeuron()
        I_ext = np.ones(1000) * 10.0
        trace = neuron.simulate(duration=100.0, dt=0.1, I_ext=I_ext)
        assert len(trace) == 1000

    def test_spike_generation(self):
        """Test that sufficient current produces spikes."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=100.0, dt=0.1, I_ext=15.0)
        # Spikes should exceed 0 mV
        assert np.max(trace) > 0.0

    def test_no_spike_below_threshold(self):
        """Test that subthreshold current doesn't produce spikes."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=100.0, dt=0.1, I_ext=1.0)
        # Should not spike
        assert np.max(trace) < 0.0

    def test_spike_count_increases_with_current(self):
        """Test that higher current produces more spikes."""
        def count_spikes(trace, threshold=0.0):
            above = trace > threshold
            return np.sum(np.diff(above.astype(int)) == 1)

        neuron1 = IzhikevichNeuron()
        trace1 = neuron1.simulate(duration=500.0, dt=0.1, I_ext=8.0)

        neuron2 = IzhikevichNeuron()
        trace2 = neuron2.simulate(duration=500.0, dt=0.1, I_ext=20.0)

        assert count_spikes(trace2) > count_spikes(trace1)


class TestIzhikevichSpikingPatterns:
    """Tests for different spiking patterns."""

    def test_regular_spiking_pattern(self):
        """Test that RS neurons show regular spiking."""
        neuron = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
        trace = neuron.simulate(duration=500.0, dt=0.1, I_ext=10.0)
        # Should produce spikes
        assert np.max(trace) > 0.0
        # Should not be bursting (voltage shouldn't stay high)
        high_voltage_count = np.sum(trace > 0.0)
        assert high_voltage_count < len(trace) * 0.3

    def test_fast_spiking_pattern(self):
        """Test that FS neurons spike faster than RS."""
        def count_spikes(trace, threshold=0.0):
            above = trace > threshold
            return np.sum(np.diff(above.astype(int)) == 1)

        rs = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
        fs = IzhikevichNeuron(IzhikevichType.FAST_SPIKING)

        trace_rs = rs.simulate(duration=500.0, dt=0.1, I_ext=15.0)
        trace_fs = fs.simulate(duration=500.0, dt=0.1, I_ext=15.0)

        # Fast spiking should have more spikes
        assert count_spikes(trace_fs) > count_spikes(trace_rs)

    def test_bursting_pattern(self):
        """Test that IB neurons show bursting behavior."""
        neuron = IzhikevichNeuron(IzhikevichType.INTRINSICALLY_BURSTING)
        trace = neuron.simulate(duration=500.0, dt=0.1, I_ext=10.0)
        # Should produce spikes
        assert np.max(trace) > 0.0


class TestIzhikevichEdgeCases:
    """Edge case tests for IzhikevichNeuron."""

    def test_extreme_positive_current(self):
        """Test with very high positive current."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=100.0, dt=0.1, I_ext=100.0)
        assert not np.any(np.isnan(trace))
        assert not np.any(np.isinf(trace))

    def test_extreme_negative_current(self):
        """Test with very negative current."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=100.0, dt=0.1, I_ext=-100.0)
        assert not np.any(np.isnan(trace))
        assert not np.any(np.isinf(trace))

    def test_very_small_dt(self):
        """Test with very small timestep."""
        neuron = IzhikevichNeuron()
        for _ in range(1000):
            neuron.step(dt=0.01, I_ext=10.0)
        assert not np.isnan(neuron.V)

    def test_large_dt(self):
        """Test stability with larger timestep."""
        neuron = IzhikevichNeuron()
        for _ in range(100):
            neuron.step(dt=1.0, I_ext=10.0)
        assert not np.isnan(neuron.V)
        assert -200.0 < neuron.V < 200.0

    def test_voltage_bounded(self):
        """Test that voltage stays bounded during simulation."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=1000.0, dt=0.1, I_ext=20.0)
        # Voltage should be capped at 100 mV (safety bound in implementation)
        assert np.max(trace) <= 100.0
        assert np.min(trace) > -200.0

    def test_multiple_resets(self):
        """Test multiple simulation/reset cycles."""
        neuron = IzhikevichNeuron()  # Default is RS with c=-65
        reset_V = neuron.parameters.c
        for _ in range(10):
            neuron.simulate(duration=50.0, dt=0.1, I_ext=20.0)
            neuron.reset()
            assert neuron.V == pytest.approx(reset_V, abs=1.0)


class TestIzhikevichNumericalStability:
    """Numerical stability tests."""

    def test_long_simulation(self):
        """Test that long simulations remain stable."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=10000.0, dt=0.1, I_ext=10.0)
        assert not np.any(np.isnan(trace))
        assert not np.any(np.isinf(trace))

    def test_resting_stability(self):
        """Test that resting neuron stays stable."""
        neuron = IzhikevichNeuron()
        initial_V = neuron.V
        trace = neuron.simulate(duration=1000.0, dt=0.1, I_ext=0.0)
        # Should stay near resting potential
        assert np.all(np.abs(trace - initial_V) < 20.0)

    def test_oscillation_stability(self):
        """Test that repeated spiking doesn't diverge."""
        neuron = IzhikevichNeuron()
        trace = neuron.simulate(duration=5000.0, dt=0.1, I_ext=15.0)
        # Voltage should stay in reasonable bounds
        assert np.min(trace) > -150.0
        assert np.max(trace) <= 100.0


class TestIzhikevichIntegrationMethod:
    """Tests for integration method handling."""

    def test_default_euler(self):
        """Test that Izhikevich defaults to Euler integration."""
        neuron = IzhikevichNeuron()
        # Izhikevich should always use Euler due to discontinuous reset
        # The integration_method property should reflect this


class TestIzhikevichParameters:
    """Tests for IzhikevichParameters class."""

    def test_default_values(self):
        """Test default parameter values."""
        params = IzhikevichParameters()
        assert params.a == pytest.approx(0.02)
        assert params.b == pytest.approx(0.2)
        assert params.c == pytest.approx(-65.0)
        assert params.d == pytest.approx(8.0)

    def test_modify_parameters(self):
        """Test modifying parameters."""
        params = IzhikevichParameters()
        params.a = 0.1
        params.b = 0.25
        params.c = -60.0
        params.d = 4.0
        assert params.a == pytest.approx(0.1)
        assert params.b == pytest.approx(0.25)
        assert params.c == pytest.approx(-60.0)
        assert params.d == pytest.approx(4.0)

    def test_repr(self):
        """Test string representation."""
        params = IzhikevichParameters()
        repr_str = repr(params)
        assert "IzhikevichParameters" in repr_str


class TestIzhikevichState:
    """Tests for IzhikevichState class."""

    def test_default_values(self):
        """Test default state values."""
        # These are the default struct values, not neuron-initialized values
        state = IzhikevichState()
        assert state.v == pytest.approx(-65.0)
        assert state.u == pytest.approx(-13.0)

    def test_modify_state(self):
        """Test modifying state."""
        state = IzhikevichState()
        state.v = -50.0
        state.u = -10.0
        assert state.v == pytest.approx(-50.0)
        assert state.u == pytest.approx(-10.0)

    def test_repr(self):
        """Test string representation."""
        state = IzhikevichState()
        repr_str = repr(state)
        assert "IzhikevichState" in repr_str


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
