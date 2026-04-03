"""
Tests for Task 12: API Structural Cleanup.

Verifies:
  1. NeuronModelSpec.hh_default() produces correct HH dynamics
  2. NeuronModelSpec.izhikevich() fires correctly in RegionalNetwork
  3. DeprecationWarning is raised for all legacy imports
  4. dict I_ext routing: scalar, stimulator, 1D array
  5. Public __all__ does NOT contain removed symbols
"""

import warnings

import numpy as np
import pytest

import hodgkin_huxley as hh
from hodgkin_huxley import (
    RegionalNetwork,
    NeuronModelSpec,
    IzhikevichType,
    SynapseSpec,
    RecordingConfig,
    DBSStimulator,
    DBSParameters,
)



# =============================================================================
# 1. NeuronModelSpec.hh_default()
# =============================================================================

class TestHHDefault:
    def test_hh_default_exists(self):
        spec = NeuronModelSpec.hh_default()
        assert spec is not None

    def test_hh_default_has_gates_and_channels(self):
        spec = NeuronModelSpec.hh_default()
        assert len(spec.gates) == 3        # m, h, n
        assert len(spec.channels) == 3     # Na, K, Leak

    def test_hh_default_fires_with_current(self):
        """Classic HH neuron should spike at I_ext=10 µA/cm²."""
        rn = RegionalNetwork()
        rn.add_population("E", 1, NeuronModelSpec.hh_default())
        result = rn.simulate(duration=100.0, dt=0.01, I_ext={"E": 10.0})
        v = result["E"][0]
        spikes = np.sum((v[:-1] < 0) & (v[1:] >= 0))  # upward zero-crossings
        assert spikes >= 3, f"Expected ≥3 spikes, got {spikes}"

    def test_hh_default_silent_without_current(self):
        """HH neuron should not spike spontaneously at rest."""
        rn = RegionalNetwork()
        rn.add_population("E", 1, NeuronModelSpec.hh_default())
        result = rn.simulate(duration=100.0, dt=0.01, I_ext={"E": 0.0})
        v = result["E"][0]
        assert v.max() < 0.0, "Neuron should not spike without input"


# =============================================================================
# 2. NeuronModelSpec.izhikevich()
# =============================================================================

class TestIzhikevichSpec:
    def test_izhikevich_spec_is_izhikevich_flag(self):
        spec = NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING)
        assert spec.is_izhikevich is True

    def test_izhikevich_spec_fast_spiking(self):
        rn = RegionalNetwork()
        rn.add_population("FS", 1, NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING))
        result = rn.simulate(duration=200.0, dt=0.1, I_ext={"FS": 10.0})
        v = result["FS"][0]
        spikes = np.sum((v[:-1] < 0) & (v[1:] >= 0))
        assert spikes >= 2, f"Expected ≥2 spikes, got {spikes}"

    def test_izhikevich_default_type_is_rs(self):
        spec_default = NeuronModelSpec.izhikevich()
        spec_rs = NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING)
        # Both should have is_izhikevich set
        assert spec_default.is_izhikevich
        assert spec_rs.is_izhikevich


# =============================================================================
# 3. DeprecationWarning for legacy imports
# =============================================================================

class TestDeprecationWarnings:
    @pytest.mark.parametrize("name", [
        "Network",
        "NetworkNeuronType",
        "HHNeuron",
        "IzhikevichNeuron",
        "HHParameters",
        "HHState",
        "IzhikevichParameters",
        "IzhikevichState",
        "SynapseBase",
        "ExponentialSynapse",
        "AlphaSynapse",
        "DoubleExponentialSynapse",
    ])
    def test_legacy_module_emits_deprecation_warning(self, name):
        from hodgkin_huxley import legacy
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            getattr(legacy, name)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
            f"Expected DeprecationWarning for hodgkin_huxley.legacy.{name}"

    def test_main_module_network_emits_deprecation(self):
        """from hodgkin_huxley import Network should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = hh.Network
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
            "Expected DeprecationWarning when accessing hh.Network"

    def test_main_module_hhneuron_emits_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = hh.HHNeuron
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# =============================================================================
# 4. __all__ does not contain removed symbols
# =============================================================================

class TestPublicAll:
    _REMOVED = {
        "HHNeuron", "IzhikevichNeuron",
        "HHParameters", "HHState",
        "IzhikevichParameters", "IzhikevichState",
        "Network", "NetworkNeuronType",
        "SynapseBase", "ExponentialSynapse",
        "AlphaSynapse", "DoubleExponentialSynapse",
    }

    def test_removed_names_not_in_all(self):
        for name in self._REMOVED:
            assert name not in hh.__all__, \
                f"'{name}' should not be in hodgkin_huxley.__all__"

    def test_regional_network_in_all(self):
        assert "RegionalNetwork" in hh.__all__

    def test_neuron_model_spec_in_all(self):
        assert "NeuronModelSpec" in hh.__all__


# =============================================================================
# 5. dict I_ext routing
# =============================================================================

class TestIExtRouting:
    def _make_network(self, n=2):
        rn = RegionalNetwork()
        rn.add_population("A", n, NeuronModelSpec.hh_default())
        return rn

    def test_scalar_iext(self):
        rn = self._make_network()
        result = rn.simulate(duration=50.0, dt=0.1, I_ext={"A": 10.0})
        # result is {pop_name: ndarray(n_neurons, n_steps)}
        assert result["A"].shape[0] == 2

    def test_1d_array_iext(self):
        rn = self._make_network(n=2)
        n_steps = int(50.0 / 0.1)
        I = np.ones(n_steps) * 10.0
        result = rn.simulate(duration=50.0, dt=0.1, I_ext={"A": I})
        assert result["A"].shape[0] == 2

    def test_stimulator_iext(self):
        rn = self._make_network()
        stim = DBSStimulator(frequency=100.0, pulse_width=0.06, amplitude=5.0)
        rn.attach_stimulator("A", stim)
        result = rn.simulate(duration=50.0, dt=0.1, I_ext={})
        assert result["A"].shape[0] == 2

    def test_missing_population_defaults_to_zero(self):
        rn = RegionalNetwork()
        rn.add_population("A", 1, NeuronModelSpec.hh_default())
        rn.add_population("B", 1, NeuronModelSpec.hh_default())
        # Only specify A; B should default to 0
        result = rn.simulate(duration=50.0, dt=0.1, I_ext={"A": 5.0})
        assert "A" in result and "B" in result
