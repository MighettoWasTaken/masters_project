"""
Tests for PulseStimulator — pulse.py

Covers:
  1. Single pulse — timing, amplitude, zeros elsewhere
  2. Multiple / irregular onsets
  3. Train constructor — n_pulses and total_duration variants
  4. Burst constructor
  5. Biphasic waveform — cathodic + anodic phases, interphase gap
  6. apply_to() — scalar and array base currents
  7. Properties — n_pulses, charge_per_phase, is_charge_balanced
  8. Validation — ValueError on bad inputs
  9. Integration — waveforms passed to Network / RegionalNetwork simulate()
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    PulseStimulator,
    Network,
    RegionalNetwork,
    NeuronModelSpec,
    RecordingConfig,
)

DT   = 0.1   # ms  — fast enough for most tests, avoids huge arrays
TOTAL = 100.0  # ms


# =============================================================================
# Helpers
# =============================================================================

def steps(duration_ms, dt=DT):
    return int(duration_ms / dt)


def make_single(onset=20.0, duration=5.0, amplitude=10.0, **kw):
    return PulseStimulator.single(onset, duration, amplitude, **kw)


# =============================================================================
# 1. Single pulse — timing and values
# =============================================================================

class TestSinglePulse:

    def test_shape(self):
        I = make_single().generate(TOTAL, DT)
        assert I.shape == (steps(TOTAL),)
        assert I.dtype == np.float64

    def test_amplitude_during_pulse(self):
        onset, dur, amp = 20.0, 5.0, 10.0
        I = PulseStimulator.single(onset, dur, amp).generate(TOTAL, DT)
        on_start = int(round(onset / DT))
        on_end   = int(round((onset + dur) / DT))
        assert np.all(I[on_start:on_end] == pytest.approx(amp))

    def test_zero_before_onset(self):
        I = make_single(onset=30.0).generate(TOTAL, DT)
        assert np.all(I[:steps(30.0)] == 0.0)

    def test_zero_after_pulse(self):
        onset, dur = 20.0, 5.0
        I = make_single(onset=onset, duration=dur).generate(TOTAL, DT)
        after = int(round((onset + dur) / DT))
        assert np.all(I[after:] == 0.0)

    def test_onset_at_zero(self):
        """Pulse starting at t=0 should fill the first dur/dt steps."""
        I = PulseStimulator.single(0.0, 5.0, 7.0).generate(50.0, DT)
        assert np.all(I[:steps(5.0)] == pytest.approx(7.0))
        assert np.all(I[steps(5.0):] == 0.0)

    def test_onset_beyond_duration_is_silent(self):
        """A pulse that starts after total_duration produces all zeros."""
        I = PulseStimulator.single(200.0, 5.0, 10.0).generate(TOTAL, DT)
        assert np.all(I == 0.0)

    def test_negative_amplitude(self):
        """Negative amplitude (hyperpolarising pulse) is supported."""
        I = PulseStimulator.single(10.0, 5.0, -8.0).generate(50.0, DT)
        on_start = int(round(10.0 / DT))
        on_end   = int(round(15.0 / DT))
        assert np.all(I[on_start:on_end] == pytest.approx(-8.0))

    def test_zero_amplitude(self):
        I = PulseStimulator.single(10.0, 5.0, 0.0).generate(50.0, DT)
        assert np.all(I == 0.0)


# =============================================================================
# 2. Multiple / irregular onsets
# =============================================================================

class TestMultipleOnsets:

    def test_two_pulses_distinct_windows(self):
        p = PulseStimulator.from_onsets([10.0, 50.0], duration=4.0, amplitude=5.0)
        I = p.generate(TOTAL, DT)
        for onset in [10.0, 50.0]:
            s = int(round(onset / DT))
            e = int(round((onset + 4.0) / DT))
            assert np.all(I[s:e] == pytest.approx(5.0)), f"Pulse at {onset} ms failed"

    def test_region_between_pulses_is_zero(self):
        p = PulseStimulator.from_onsets([10.0, 50.0], duration=4.0, amplitude=5.0)
        I = p.generate(TOTAL, DT)
        between_start = int(round(14.0 / DT))
        between_end   = int(round(50.0 / DT))
        assert np.all(I[between_start:between_end] == 0.0)

    def test_onsets_sorted_automatically(self):
        """Constructor sorts onsets; result is same regardless of input order."""
        I_a = PulseStimulator.from_onsets([50.0, 10.0], 4.0, 5.0).generate(TOTAL, DT)
        I_b = PulseStimulator.from_onsets([10.0, 50.0], 4.0, 5.0).generate(TOTAL, DT)
        assert np.allclose(I_a, I_b)

    def test_n_pulses_property(self):
        p = PulseStimulator.from_onsets([5.0, 15.0, 25.0], 2.0, 1.0)
        assert p.n_pulses == 3

    def test_onsets_property(self):
        p = PulseStimulator.from_onsets([30.0, 10.0, 20.0], 2.0, 1.0)
        assert p.onsets == [10.0, 20.0, 30.0]


# =============================================================================
# 3. Train constructor
# =============================================================================

class TestTrain:

    def test_n_pulses_count(self):
        p = PulseStimulator.train(100.0, 1.0, 5.0, onset=0.0, n_pulses=10)
        assert p.n_pulses == 10

    def test_period_spacing(self):
        freq = 100.0   # Hz  → period = 10 ms
        p = PulseStimulator.train(freq, 1.0, 5.0, onset=0.0, n_pulses=5)
        expected = [0.0, 10.0, 20.0, 30.0, 40.0]
        assert np.allclose(p.onsets, expected)

    def test_total_duration_variant(self):
        """Pulses generated while onset_i < total_duration."""
        freq = 100.0  # period = 10 ms
        p = PulseStimulator.train(freq, 1.0, 5.0, onset=0.0, total_duration=50.0)
        # Onsets: 0, 10, 20, 30, 40  (50 is excluded)
        assert p.n_pulses == 5
        assert p.onsets[-1] == pytest.approx(40.0)

    def test_onset_offset(self):
        """Train starting at non-zero onset."""
        p = PulseStimulator.train(100.0, 1.0, 5.0, onset=25.0, n_pulses=3)
        assert np.allclose(p.onsets, [25.0, 35.0, 45.0])

    def test_pulses_in_correct_positions(self):
        """Generated array has non-zero values only at pulse windows."""
        p = PulseStimulator.train(100.0, 2.0, 7.0, onset=0.0, n_pulses=3)
        I = p.generate(TOTAL, DT)
        for onset in p.onsets:
            s = int(round(onset / DT))
            e = int(round((onset + 2.0) / DT))
            assert np.all(I[s:e] == pytest.approx(7.0))

    def test_raises_both_n_pulses_and_total_duration(self):
        with pytest.raises(ValueError):
            PulseStimulator.train(100.0, 1.0, 5.0, n_pulses=5, total_duration=100.0)

    def test_raises_neither_n_pulses_nor_total_duration(self):
        with pytest.raises(ValueError):
            PulseStimulator.train(100.0, 1.0, 5.0)

    def test_raises_zero_frequency(self):
        with pytest.raises(ValueError):
            PulseStimulator.train(0.0, 1.0, 5.0, n_pulses=5)


# =============================================================================
# 4. Burst constructor
# =============================================================================

class TestBurst:

    def test_n_pulses(self):
        b = PulseStimulator.burst(5, 300.0, 0.06, 200.0, onset=0.0)
        assert b.n_pulses == 5

    def test_period_spacing(self):
        intra = 300.0   # Hz  → period ≈ 3.333 ms
        b = PulseStimulator.burst(4, intra, 0.06, 200.0, onset=10.0)
        period = 1000.0 / intra
        expected = [10.0 + i * period for i in range(4)]
        assert np.allclose(b.onsets, expected)

    def test_burst_generates_correct_shape(self):
        b = PulseStimulator.burst(3, 200.0, 1.0, 50.0, onset=20.0)
        I = b.generate(TOTAL, DT)
        assert I.shape == (steps(TOTAL),)
        assert np.all(np.isfinite(I))


# =============================================================================
# 5. Biphasic waveform
# =============================================================================

class TestBiphasic:

    def _make_biphasic(self, onset=10.0, duration=2.0, amplitude=5.0,
                        gap=0.0, anodic_amp=None, anodic_dur=None):
        return PulseStimulator.single(
            onset, duration, amplitude,
            waveform="biphasic",
            interphase_gap=gap,
            anodic_amplitude=anodic_amp,
            anodic_duration=anodic_dur,
        )

    def test_cathodic_phase_amplitude(self):
        p = self._make_biphasic(onset=10.0, duration=2.0, amplitude=5.0)
        I = p.generate(50.0, DT)
        s = int(round(10.0 / DT))
        e = int(round(12.0 / DT))
        assert np.all(I[s:e] == pytest.approx(5.0))

    def test_anodic_phase_default_symmetric(self):
        """Default anodic phase: same duration, opposite sign → -amplitude."""
        p = self._make_biphasic(onset=10.0, duration=2.0, amplitude=5.0, gap=0.0)
        I = p.generate(50.0, DT)
        a_s = int(round(12.0 / DT))
        a_e = int(round(14.0 / DT))
        assert np.all(I[a_s:a_e] == pytest.approx(-5.0))

    def test_interphase_gap_is_zero(self):
        """Zero gap: anodic phase immediately follows cathodic."""
        p = self._make_biphasic(onset=10.0, duration=2.0, amplitude=5.0, gap=0.0)
        I = p.generate(50.0, DT)
        gap_idx = int(round(12.0 / DT))
        # The sample right at the boundary belongs to the anodic phase
        assert I[gap_idx] == pytest.approx(-5.0)

    def test_interphase_gap_separates_phases(self):
        """Non-zero gap: samples in gap should be zero."""
        p = self._make_biphasic(onset=10.0, duration=2.0, amplitude=5.0, gap=1.0)
        I = p.generate(50.0, DT)
        gap_start = int(round(12.0 / DT))
        gap_end   = int(round(13.0 / DT))
        assert np.all(I[gap_start:gap_end] == 0.0)
        # Anodic phase should be present after gap
        a_s = int(round(13.0 / DT))
        a_e = int(round(15.0 / DT))
        assert np.all(I[a_s:a_e] == pytest.approx(-5.0))

    def test_asymmetric_anodic_amplitude(self):
        """Custom anodic amplitude (longer, lower — common for charge balancing)."""
        p = self._make_biphasic(
            onset=10.0, duration=2.0, amplitude=10.0,
            anodic_amp=5.0, anodic_dur=4.0
        )
        I = p.generate(50.0, DT)
        a_s = int(round(12.0 / DT))
        a_e = int(round(16.0 / DT))
        assert np.all(I[a_s:a_e] == pytest.approx(-5.0))

    def test_charge_balanced_symmetric(self):
        """Symmetric biphasic pulse is charge-balanced."""
        p = self._make_biphasic(onset=10.0, duration=2.0, amplitude=5.0)
        assert p.is_charge_balanced

    def test_charge_balanced_asymmetric_equal_charge(self):
        """Asymmetric but equal charge (10×2 = 5×4) → balanced."""
        p = self._make_biphasic(
            onset=10.0, duration=2.0, amplitude=10.0,
            anodic_amp=5.0, anodic_dur=4.0
        )
        assert p.is_charge_balanced

    def test_charge_imbalanced(self):
        """Unequal anodic charge → not balanced."""
        p = self._make_biphasic(
            onset=10.0, duration=2.0, amplitude=10.0,
            anodic_amp=3.0, anodic_dur=2.0   # 10×2 ≠ 3×2
        )
        assert not p.is_charge_balanced

    def test_rectangular_always_charge_balanced(self):
        p = PulseStimulator.single(10.0, 2.0, 5.0, waveform="rectangular")
        assert p.is_charge_balanced

    def test_net_charge_is_zero_for_balanced(self):
        """Integral of a balanced biphasic trace is (approximately) 0."""
        p = self._make_biphasic(onset=5.0, duration=2.0, amplitude=6.0, gap=0.0)
        I = p.generate(30.0, 0.01)
        assert abs(I.sum()) < 1.0   # small residual due to rounding only


# =============================================================================
# 6. apply_to()
# =============================================================================

class TestApplyTo:

    def test_scalar_base_produces_correct_shape(self):
        I = make_single().apply_to(3.0, TOTAL, DT)
        assert I.shape == (steps(TOTAL),)

    def test_scalar_base_adds_correctly(self):
        onset, dur, amp = 20.0, 5.0, 10.0
        I = PulseStimulator.single(onset, dur, amp).apply_to(3.0, TOTAL, DT)
        # Background should be 3 everywhere
        before = int(round(onset / DT))
        assert np.all(I[:before] == pytest.approx(3.0))
        # During pulse: background + amplitude
        s = int(round(onset / DT))
        e = int(round((onset + dur) / DT))
        assert np.all(I[s:e] == pytest.approx(3.0 + amp))

    def test_array_base_adds_correctly(self):
        base = np.ones(steps(TOTAL)) * 2.0
        I = make_single(onset=20.0, duration=5.0, amplitude=8.0).apply_to(base, TOTAL, DT)
        s = int(round(20.0 / DT))
        e = int(round(25.0 / DT))
        assert np.all(I[s:e] == pytest.approx(2.0 + 8.0))

    def test_zero_base(self):
        """apply_to(0.0, ...) == generate(...)"""
        p = make_single()
        assert np.allclose(p.apply_to(0.0, TOTAL, DT), p.generate(TOTAL, DT))


# =============================================================================
# 7. Properties
# =============================================================================

class TestProperties:

    def test_charge_per_phase(self):
        p = PulseStimulator.single(0.0, 2.0, 5.0)
        assert p.charge_per_phase == pytest.approx(10.0)

    def test_waveform_property(self):
        r = PulseStimulator.single(0.0, 1.0, 1.0, waveform="rectangular")
        b = PulseStimulator.single(0.0, 1.0, 1.0, waveform="biphasic")
        assert r.waveform == "rectangular"
        assert b.waveform == "biphasic"

    def test_duration_property(self):
        p = PulseStimulator.single(0.0, 3.5, 1.0)
        assert p.duration == pytest.approx(3.5)

    def test_amplitude_property(self):
        p = PulseStimulator.single(0.0, 1.0, 42.0)
        assert p.amplitude == pytest.approx(42.0)


# =============================================================================
# 8. Validation
# =============================================================================

class TestValidation:

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError, match="duration"):
            PulseStimulator.single(0.0, 0.0, 1.0)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration"):
            PulseStimulator.single(0.0, -1.0, 1.0)

    def test_negative_onset_raises(self):
        with pytest.raises(ValueError):
            PulseStimulator([-10.0], 1.0, 1.0)

    def test_invalid_waveform_raises(self):
        with pytest.raises(ValueError, match="waveform"):
            PulseStimulator.single(0.0, 1.0, 1.0, waveform="sinusoidal")

    def test_negative_interphase_gap_raises(self):
        with pytest.raises(ValueError, match="interphase_gap"):
            PulseStimulator.single(0.0, 1.0, 1.0, waveform="biphasic",
                                   interphase_gap=-0.5)


# =============================================================================
# 9. Integration — Network and RegionalNetwork
# =============================================================================

class TestIntegration:

    def test_pulse_evokes_response_in_hh_neuron(self):
        """
        HH neuron at rest (I_ext=0) shouldn't spike; with a strong pulse
        it should.  Verifies the waveform is accepted and applied correctly.
        """
        dt       = 0.1
        duration = 200.0
        n_steps  = int(duration / dt)

        # No pulse → no spikes
        net_rest = Network()
        net_rest.add_hh_neuron()
        V_rest = net_rest.simulate(duration, dt, [[0.0] * n_steps])
        assert np.max(V_rest) < -20.0, "HH at rest should not spike"

        # Strong pulse → at least one spike
        pulse  = PulseStimulator.single(onset=50.0, duration=5.0, amplitude=50.0)
        I_ext  = pulse.generate(duration, dt)
        net_stim = Network()
        net_stim.add_hh_neuron()
        V_stim = net_stim.simulate(duration, dt, [I_ext.tolist()])
        assert np.max(V_stim) > 0.0, "HH should spike under strong pulse"

    def test_train_drives_multiple_spikes(self):
        """A regular 50 Hz rectangular train should evoke several spikes."""
        dt       = 0.1
        duration = 500.0
        n_steps  = int(duration / dt)

        train = PulseStimulator.train(
            frequency=50, duration=2.0, amplitude=40.0,
            onset=0.0, total_duration=duration
        )
        I_ext = train.generate(duration, dt)

        net = Network()
        net.add_hh_neuron()
        V = net.simulate(duration, dt, [I_ext.tolist()])

        above = (np.array(V) > 0.0).astype(int)
        n_spikes = int(np.sum(np.diff(above) == 1))
        assert n_spikes >= 3, f"Expected multiple spikes from 50 Hz train, got {n_spikes}"

    def test_biphasic_accepted_by_regional_network(self):
        """Biphasic pulse array is accepted as I_ext in RegionalNetwork."""
        dt       = 0.1
        duration = 200.0

        pulse = PulseStimulator.single(50.0, 5.0, 30.0, waveform="biphasic")
        I_arr = pulse.generate(duration, dt)

        rn = RegionalNetwork()
        rn.add_population("TH", 3, model=NeuronModelSpec.thalamic())
        traces = rn.simulate(duration, dt, {"TH": I_arr})
        assert np.all(np.isfinite(traces["TH"])), "RegionalNetwork produced NaN/Inf"

    def test_apply_to_with_regional_network(self):
        """apply_to() output is accepted as I_ext in RegionalNetwork."""
        dt       = 0.1
        duration = 200.0

        pulse = PulseStimulator.single(50.0, 5.0, 30.0)
        I_net = pulse.apply_to(base_current=3.0, total_duration=duration, dt=dt)

        rn = RegionalNetwork()
        rn.add_population("STN", 3, model=NeuronModelSpec.stn())
        traces = rn.simulate(duration, dt, {"STN": I_net})
        assert np.all(np.isfinite(traces["STN"]))

    def test_benchmark_cortical_stimulus_pattern(self):
        """
        Exactly reproduce the benchmark cortical stimulus:
        onset=1000 ms, duration=0.3 ms, amplitude=350 µA/cm².
        """
        dt       = 0.01
        duration = 2000.0
        n_steps  = int(duration / dt)

        pulse = PulseStimulator.single(onset=1000.0, duration=0.3, amplitude=350.0)
        I = pulse.generate(duration, dt)

        assert I.shape == (n_steps,)
        assert np.all(I[:int(1000.0 / dt)] == 0.0), "Should be zero before onset"

        on_start = int(round(1000.0 / dt))
        on_end   = int(round(1000.3 / dt))
        assert np.all(I[on_start:on_end] == pytest.approx(350.0)), "Amplitude during pulse"
        assert np.all(I[on_end:] == 0.0), "Should be zero after pulse"
