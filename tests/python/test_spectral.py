"""
Tests for src/hodgkin_huxley/spectral.py

Covers:
  1. beta_band_power — direct integration tests with synthetic spectra
  2. mtspectrumpt   — algorithm correctness with known pulse trains
  3. analyze_beta_power — end-to-end pipeline on real simulation output
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    beta_band_power,
    mtspectrumpt,
    analyze_beta_power,
    RecordingConfig,
    Network,
    RegionalNetwork,
    NeuronModelSpec,
)

# Use a reduced sampling rate for test speed (matches dt=0.1 ms).
# The algorithm is Fs-invariant; beta band (7–35 Hz) is well within Nyquist.
TEST_FS = 10_000   # Hz  (dt = 0.1 ms)
TEST_DT = 0.1      # ms


def make_regular_train(rate_hz: float, duration_s: float) -> np.ndarray:
    """Regular spike train: n = floor(duration * rate) spikes, first at 1/rate."""
    n = int(duration_s * rate_hz)
    return np.arange(1, n + 1) / rate_hz


# =============================================================================
# 1. beta_band_power
# =============================================================================

class TestBetaBandPower:

    def test_flat_spectrum_area(self):
        """Flat spectrum at height 1 → area = fmax - fmin = 28 Hz."""
        f = np.linspace(0, 100, 100_001)   # 0.001 Hz resolution
        S = np.ones_like(f)
        power = beta_band_power(S, f, fmin=7, fmax=35)
        assert abs(power - 28.0) < 0.05

    def test_zero_spectrum(self):
        """All-zero spectrum → zero power."""
        f = np.linspace(0, 100, 1000)
        assert beta_band_power(np.zeros_like(f), f) == pytest.approx(0.0, abs=1e-10)

    def test_in_band_peak_larger_than_out_of_band(self):
        """Peak inside beta band gives more power than identical peak outside."""
        f = np.linspace(1, 100, 10_000)
        S_in  = np.where((f >= 7)  & (f <= 35), 10.0, 0.0)
        S_out = np.where((f >= 50) & (f <= 70), 10.0, 0.0)
        assert beta_band_power(S_in, f) > beta_band_power(S_out, f)

    def test_custom_band_edges(self):
        """Custom band edges are respected."""
        f = np.linspace(0, 100, 10_001)
        S = np.where((f >= 10) & (f <= 20), 5.0, 0.0)
        # [10, 20] fully inside [5, 25] → power ≈ 5 * 10 = 50
        assert abs(beta_band_power(S, f, fmin=5, fmax=25) - 50.0) < 0.5
        # [10, 20] fully outside [30, 40] → power ≈ 0
        assert beta_band_power(S, f, fmin=30, fmax=40) == pytest.approx(0.0, abs=0.1)


# =============================================================================
# 2. mtspectrumpt
# =============================================================================

class TestMtspectrumpt:

    def test_output_shapes_match(self):
        """S and f must be 1-D arrays of equal length within fpass."""
        spikes = [make_regular_train(20, 1.0)]
        S, f = mtspectrumpt(spikes, duration=1.0, Fs=TEST_FS, fpass=(1, 100))
        assert S.ndim == 1 and f.ndim == 1
        assert S.shape == f.shape
        assert S.size > 0

    def test_frequency_range_respected(self):
        """All returned frequencies must lie within fpass."""
        S, f = mtspectrumpt(
            [make_regular_train(20, 1.0)], duration=1.0,
            Fs=TEST_FS, fpass=(5, 80)
        )
        assert f.min() >= 5.0
        assert f.max() <= 80.0

    def test_finite_outputs_normal(self):
        """No NaN or Inf for typical spike trains."""
        spikes = [make_regular_train(20, 1.0), make_regular_train(40, 1.0)]
        S, f = mtspectrumpt(spikes, duration=1.0, Fs=TEST_FS)
        assert np.all(np.isfinite(S))
        assert np.all(np.isfinite(f))

    def test_no_crash_empty_train(self):
        """A trial with zero spikes must not raise and must produce finite output."""
        S, f = mtspectrumpt(
            [np.array([]), make_regular_train(20, 1.0)],
            duration=1.0, Fs=TEST_FS
        )
        assert np.all(np.isfinite(S))

    def test_no_crash_single_spike(self):
        """A single spike must not raise."""
        S, f = mtspectrumpt([[0.5]], duration=1.0, Fs=TEST_FS)
        assert np.all(np.isfinite(S))

    def test_peak_near_input_frequency(self):
        """Regular 20 Hz train → spectral peak within 5 Hz of 20 Hz."""
        # Use 2 s for cleaner frequency resolution (df = 0.5 Hz at TEST_FS)
        spikes = [make_regular_train(20, 2.0)]
        S, f = mtspectrumpt(spikes, duration=2.0, Fs=TEST_FS, fpass=(1, 100))
        peak_freq = f[np.argmax(S)]
        assert abs(peak_freq - 20.0) < 5.0, (
            f"Peak at {peak_freq:.1f} Hz; expected near 20 Hz"
        )

    def test_beta_train_has_more_beta_power_than_high_freq_train(self):
        """20 Hz (beta) train must produce higher beta-band power than 60 Hz train."""
        dur = 2.0
        S_beta, f = mtspectrumpt([make_regular_train(20, dur)], duration=dur, Fs=TEST_FS)
        S_high, _ = mtspectrumpt([make_regular_train(60, dur)], duration=dur, Fs=TEST_FS)
        bp_beta = beta_band_power(S_beta, f)
        bp_high = beta_band_power(S_high, f)
        assert bp_beta > bp_high, (
            f"Beta train power {bp_beta:.4f} should exceed "
            f"high-freq train power {bp_high:.4f}"
        )

    def test_averaging_identical_trials_unchanged(self):
        """Averaging N identical trials must give the same spectrum as one trial."""
        train = make_regular_train(20, 1.0)
        S1, f  = mtspectrumpt([train],      duration=1.0, Fs=TEST_FS)
        S5, _  = mtspectrumpt([train] * 5,  duration=1.0, Fs=TEST_FS)
        assert np.allclose(S1, S5, atol=1e-10)

    def test_higher_rate_train_has_more_total_power(self):
        """Denser spike train carries more spectral energy than a sparser one."""
        dur = 2.0
        S_lo, f = mtspectrumpt([make_regular_train(5,  dur)], duration=dur, Fs=TEST_FS)
        S_hi, _ = mtspectrumpt([make_regular_train(50, dur)], duration=dur, Fs=TEST_FS)
        # Total integrated power across fpass should be larger for the denser train
        assert np.trapezoid(np.abs(S_hi), f) > np.trapezoid(np.abs(S_lo), f)


# =============================================================================
# 3. analyze_beta_power — end-to-end pipeline on simulation output
# =============================================================================

class TestAnalyzeBetaPower:

    def test_raises_without_spikes_key(self):
        """ValueError if result doesn't contain 'spikes'."""
        net = Network()
        net.add_hh_neuron()
        cfg = RecordingConfig(["V"])
        result = net.simulate(100.0, TEST_DT, [[5.0] * 1000], record=cfg)
        with pytest.raises(ValueError, match="spikes"):
            analyze_beta_power(result, duration_ms=100.0, Fs=TEST_FS)

    def test_return_keys(self):
        """Output dict must contain exactly 'power', 'spectrum', 'frequencies'."""
        net = Network()
        net.add_hh_neuron()
        cfg = RecordingConfig(["spikes"])
        result = net.simulate(500.0, TEST_DT, [[10.0] * 5000], record=cfg)
        out = analyze_beta_power(result, duration_ms=500.0, Fs=TEST_FS)
        assert set(out.keys()) == {"power", "spectrum", "frequencies"}

    def test_finite_output_hh_neuron(self):
        """HH neuron under current: all outputs finite."""
        net = Network()
        net.add_hh_neuron()
        cfg = RecordingConfig(["spikes"])
        result = net.simulate(1000.0, TEST_DT, [[10.0] * 10000], record=cfg)
        out = analyze_beta_power(result, duration_ms=1000.0, Fs=TEST_FS)
        assert np.isfinite(out["power"])
        assert np.all(np.isfinite(out["spectrum"]))
        assert np.all(np.isfinite(out["frequencies"]))

    def test_frequencies_within_fpass(self):
        """Returned frequencies lie within the default fpass=[1, 100] Hz."""
        net = Network()
        net.add_hh_neuron()
        cfg = RecordingConfig(["spikes"])
        result = net.simulate(500.0, TEST_DT, [[10.0] * 5000], record=cfg)
        out = analyze_beta_power(result, duration_ms=500.0, Fs=TEST_FS)
        assert out["frequencies"].min() >= 1.0
        assert out["frequencies"].max() <= 100.0

    def test_spectrum_and_frequencies_same_shape(self):
        """'spectrum' and 'frequencies' arrays must have the same length."""
        net = Network()
        net.add_hh_neuron()
        cfg = RecordingConfig(["spikes"])
        result = net.simulate(500.0, TEST_DT, [[10.0] * 5000], record=cfg)
        out = analyze_beta_power(result, duration_ms=500.0, Fs=TEST_FS)
        assert out["spectrum"].shape == out["frequencies"].shape

    def test_regional_network_gpi_pipeline(self):
        """Full pipeline on a RegionalNetwork GPi population."""
        rn = RegionalNetwork()
        rn.add_population("GPi", 5, model=NeuronModelSpec.gpi())
        cfg = RecordingConfig(["spikes"])
        result = rn.simulate(1000.0, TEST_DT, {"GPi": 5.0}, record=cfg)
        gpi = result["GPi"]
        out = analyze_beta_power(gpi, duration_ms=1000.0, Fs=TEST_FS)
        assert np.isfinite(out["power"])
        assert np.all(np.isfinite(out["spectrum"]))

    def test_no_spike_population_finite(self):
        """Population that produces no spikes must not crash or return NaN."""
        net = Network()
        net.add_hh_neuron()
        cfg = RecordingConfig(["spikes"])
        # Zero current → no spikes
        result = net.simulate(500.0, TEST_DT, [[0.0] * 5000], record=cfg)
        out = analyze_beta_power(result, duration_ms=500.0, Fs=TEST_FS)
        assert np.isfinite(out["power"])
        assert np.all(np.isfinite(out["spectrum"]))

    def test_power_scalar(self):
        """'power' must be a plain Python float (or 0-d array castable to float)."""
        net = Network()
        net.add_hh_neuron()
        cfg = RecordingConfig(["spikes"])
        result = net.simulate(500.0, TEST_DT, [[10.0] * 5000], record=cfg)
        out = analyze_beta_power(result, duration_ms=500.0, Fs=TEST_FS)
        assert float(out["power"]) == out["power"]
