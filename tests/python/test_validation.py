"""
Tests for NeuronModelSpec validation (Task 10.6).

Verifies that validate() raises ValueError (via std::invalid_argument) for:
- Gate index out of range in channel gates
- derived_source_gate out of range
- calcium.source_channels index out of range
- Negative channel conductance
- Non-positive C_m

And that valid specs do not raise.
"""

import pytest

from hodgkin_huxley import (
    NeuronModelSpec, GateSpec, GateUpdateForm, ChannelSpec,
    BoltzmannParams, TauParams, TauForm,
    Network,
)
import hodgkin_huxley._core as _core


# ---------------------------------------------------------------------------
# Helper: minimal valid spec
# ---------------------------------------------------------------------------

def make_valid_spec(name="test"):
    """One gate, one channel referencing gate 0 — structurally valid."""
    spec = NeuronModelSpec()
    spec.name = name
    spec.C_m = 1.0
    spec.V_init = -65.0

    g = GateSpec()
    g.name = "m"
    g.update_form = GateUpdateForm.INF_TAU
    g.initial_value = 0.5
    spec.gates.append(g)

    ch = ChannelSpec()
    ch.name = "Na"
    ch.g = 5.0
    ch.E_rev = 50.0
    ch.gates = [(0, 3)]   # gate 0, power 3
    spec.channels.append(ch)

    return spec


# ---------------------------------------------------------------------------
# Valid specs — should not raise
# ---------------------------------------------------------------------------

class TestValidSpecsPass:

    def test_minimal_valid(self):
        spec = make_valid_spec()
        spec.validate()  # must not raise

    def test_stn_preset_valid(self):
        NeuronModelSpec.stn().validate()

    def test_thalamic_preset_valid(self):
        NeuronModelSpec.thalamic().validate()

    def test_gpe_preset_valid(self):
        NeuronModelSpec.gpe().validate()

    def test_gpi_preset_valid(self):
        NeuronModelSpec.gpi().validate()

    def test_empty_spec_valid(self):
        """No gates, no channels, C_m=1 → valid."""
        spec = NeuronModelSpec()
        spec.name = "empty"
        spec.C_m = 1.0
        spec.validate()


# ---------------------------------------------------------------------------
# Negative conductance
# ---------------------------------------------------------------------------

class TestNegativeConductance:

    def test_negative_g_raises(self):
        spec = make_valid_spec()
        spec.channels[0].g = -1.0
        with pytest.raises(ValueError, match="negative conductance"):
            spec.validate()

    def test_zero_g_is_allowed(self):
        """g=0 means channel is off but is structurally valid."""
        spec = make_valid_spec()
        spec.channels[0].g = 0.0
        spec.validate()  # should not raise


# ---------------------------------------------------------------------------
# Non-positive C_m
# ---------------------------------------------------------------------------

class TestCmValidation:

    def test_zero_C_m_raises(self):
        spec = make_valid_spec()
        spec.C_m = 0.0
        with pytest.raises(ValueError, match="C_m"):
            spec.validate()

    def test_negative_C_m_raises(self):
        spec = make_valid_spec()
        spec.C_m = -1.0
        with pytest.raises(ValueError, match="C_m"):
            spec.validate()

    def test_positive_C_m_ok(self):
        spec = make_valid_spec()
        spec.C_m = 0.5
        spec.validate()


# ---------------------------------------------------------------------------
# Gate index out of range in ChannelSpec
# ---------------------------------------------------------------------------

class TestGateIndexOutOfRange:

    def test_channel_gate_index_too_large(self):
        spec = make_valid_spec()
        # spec has 1 gate (index 0); reference index 5
        spec.channels[0].gates = [(5, 1)]
        with pytest.raises(ValueError, match="gate index"):
            spec.validate()

    def test_channel_gate_index_negative(self):
        spec = make_valid_spec()
        spec.channels[0].gates = [(-1, 1)]
        with pytest.raises(ValueError, match="gate index"):
            spec.validate()

    def test_valid_gate_index(self):
        spec = make_valid_spec()
        spec.channels[0].gates = [(0, 2)]  # index 0 exists
        spec.validate()


# ---------------------------------------------------------------------------
# derived_source_gate out of range
# ---------------------------------------------------------------------------

class TestDerivedSourceGateOutOfRange:

    def test_derived_source_too_large(self):
        spec = NeuronModelSpec()
        spec.name = "derived_test"
        spec.C_m = 1.0

        g0 = GateSpec()
        g0.name = "m"
        g0.update_form = GateUpdateForm.INF_TAU
        spec.gates.append(g0)

        g1 = GateSpec()
        g1.name = "h_derived"
        g1.update_form = GateUpdateForm.DERIVED
        g1.derived_source_gate = 99   # out of range — only 2 gates
        spec.gates.append(g1)

        with pytest.raises(ValueError, match="derived_source_gate"):
            spec.validate()

    def test_derived_source_negative(self):
        spec = NeuronModelSpec()
        spec.name = "derived_neg"
        spec.C_m = 1.0

        g0 = GateSpec()
        g0.name = "m"
        g0.update_form = GateUpdateForm.INF_TAU
        spec.gates.append(g0)

        g1 = GateSpec()
        g1.name = "h"
        g1.update_form = GateUpdateForm.DERIVED
        g1.derived_source_gate = -5
        spec.gates.append(g1)

        with pytest.raises(ValueError, match="derived_source_gate"):
            spec.validate()

    def test_valid_derived_source(self):
        spec = NeuronModelSpec()
        spec.name = "derived_ok"
        spec.C_m = 1.0

        g0 = GateSpec()
        g0.name = "m"
        g0.update_form = GateUpdateForm.INF_TAU
        spec.gates.append(g0)

        g1 = GateSpec()
        g1.name = "h"
        g1.update_form = GateUpdateForm.DERIVED
        g1.derived_source_gate = 0   # valid: refers to g0
        spec.gates.append(g1)

        spec.validate()  # should not raise


# ---------------------------------------------------------------------------
# calcium.source_channels out of range
# ---------------------------------------------------------------------------

class TestCalciumSourceChannels:

    def test_source_channel_index_too_large(self):
        spec = make_valid_spec()   # has 1 channel (index 0)
        spec.calcium.enabled = True
        spec.calcium.source_channels = [99]
        with pytest.raises(ValueError, match="source_channels"):
            spec.validate()

    def test_source_channel_index_negative(self):
        spec = make_valid_spec()
        spec.calcium.enabled = True
        spec.calcium.source_channels = [-1]
        with pytest.raises(ValueError, match="source_channels"):
            spec.validate()

    def test_valid_source_channel(self):
        spec = make_valid_spec()   # channel 0 exists
        spec.calcium.enabled = True
        spec.calcium.source_channels = [0]
        spec.validate()  # should not raise

    def test_empty_source_channels_ok(self):
        spec = make_valid_spec()
        spec.calcium.enabled = True
        spec.calcium.source_channels = []
        spec.validate()  # should not raise


# ---------------------------------------------------------------------------
# Validation triggered automatically by Network.add_neuron
# ---------------------------------------------------------------------------

class TestAutoValidationViaNetwork:

    def test_invalid_spec_add_neuron_raises(self):
        spec = make_valid_spec()
        spec.channels[0].gates = [(99, 1)]  # gate 99 doesn't exist
        net = Network()
        with pytest.raises(ValueError, match="gate index"):
            net.add_neuron(spec)

    def test_negative_g_add_neuron_raises(self):
        spec = make_valid_spec()
        spec.channels[0].g = -2.0
        net = Network()
        with pytest.raises(ValueError, match="negative conductance"):
            net.add_neuron(spec)

    def test_valid_spec_add_neuron_ok(self):
        spec = make_valid_spec()
        net = Network()
        idx = net.add_neuron(spec)
        assert idx == 0


# ---------------------------------------------------------------------------
# Validation triggered automatically by ComposableNeuron constructor
# ---------------------------------------------------------------------------

class TestAutoValidationViaComposableNeuron:

    def test_invalid_spec_constructor_raises(self):
        spec = make_valid_spec()
        spec.C_m = -1.0
        with pytest.raises(ValueError, match="C_m"):
            _core.ComposableNeuron(spec)

    def test_valid_spec_constructor_ok(self):
        spec = make_valid_spec()
        neuron = _core.ComposableNeuron(spec)
        assert neuron is not None
