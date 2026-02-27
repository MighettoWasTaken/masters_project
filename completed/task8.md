# Task 8: Spectral Analysis

## Priority: 2 (Needed)

## Overview
Implement multitaper spectral analysis for computing power spectra from spike train data. The benchmark uses this to compute the GPi pathological low-frequency oscillatory power (7-35 Hz beta band), which is the primary output metric for evaluating DBS efficacy.

---

## 8.1 Analysis Pipeline

The benchmark analysis pipeline:
1. **Detect spikes** from voltage traces (threshold crossing at -20 mV)
2. **Convert to spike times** (in seconds)
3. **Compute multitaper point-process spectrum** using DPSS tapers
4. **Integrate power** in the 7-35 Hz beta band

### Spike Detection
Spike detection is handled by the recording system (Task 10.3). Use:
```python
cfg = RecordingConfig(["spikes"], spike_threshold=-20.0)
result = rnet.simulate(duration, dt, I_ext, record=cfg)
# result["spikes"] → list of ndarrays, spike times in ms per neuron
```

### Multitaper Spectrum (Point Process)
`mtspectrumpt()` in `src/hodgkin_huxley/spectral.py` — a multitaper spectral estimator for point processes (spike times), based on the Chronux MATLAB toolbox.

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Fs | 100,000 Hz (= 1 / (0.01ms × 1e-3)) |
| fpass | [1, 100] Hz |
| tapers | [3, 5] (time-bandwidth product = 3, num tapers = 5) |

**Algorithm:**
1. Generate DPSS (Slepian) tapers: `scipy.signal.windows.dpss(N, NW=3, Kmax=5)`
2. For each neuron's spike times, interpolate each taper at the exact spike times
3. Compute the non-uniform DFT: `J_k(f) = Σ h_k(t_i) · exp(-2πi·f·t_i)`
4. Trial power: `S_n(f) = mean_k(|J_k(f)|²)` across tapers
5. Average across neurons
6. Subtract bias: `S(f) = S(f) - mean_rate` (point-process correction)

### Beta Band Power
```python
beta_band_power(S, f, fmin=7, fmax=35)
# → np.trapezoid(S[mask], f[mask])
```

---

## 8.2 Implementation

Implemented in Python (`src/hodgkin_huxley/spectral.py`), since:
- It uses scipy/numpy for DPSS tapers, FFT, and integration
- It runs once after simulation (not in the hot loop)
- It matches the benchmark's Chronux-style `mtspectrumpt` algorithm

### Exported API

```python
from hodgkin_huxley import mtspectrumpt, beta_band_power, analyze_beta_power
```

#### `mtspectrumpt(spike_times_list, duration, Fs, fpass, tapers)`
Core spectral estimator. Takes spike times **in seconds**, returns `(S, f)`.

#### `beta_band_power(S, f, fmin=7, fmax=35)`
Trapezoidal integration of PSD over the beta band.

#### `analyze_beta_power(result, duration_ms, Fs, fpass, tapers, band)`
High-level convenience wrapper. Reads `result["spikes"]` directly from a
`MetricsResult` (recording system output), converts ms → seconds internally.

```python
cfg = RecordingConfig(["spikes"])
result = rnet.simulate(1000.0, 0.01, I_ext, record=cfg)

analysis = analyze_beta_power(result["GPi"], duration_ms=1000.0)
print(f"GPi beta power: {analysis['power']:.4f}")
# analysis["spectrum"]    → PSD array
# analysis["frequencies"] → frequency array in Hz
```

---

## 8.3 Implementation Checklist
- [x] Spike detection — provided by `RecordingConfig(["spikes"])` (Task 10.3)
- [x] Implement `mtspectrumpt()` — multitaper point-process spectrum (Chronux-compatible)
- [x] Implement `beta_band_power()` — trapezoidal integration in frequency band
- [x] Implement `analyze_beta_power()` — convenience wrapper over recording system output
- [x] Add to `__init__.py` exports
- [x] Unit tests: flat spectrum area, out-of-band vs in-band power (`test_spectral.py`)
- [x] Unit tests: spectrum of regular pulse train (peak near known frequency)
- [x] Unit tests: beta power higher for 20 Hz train than 60 Hz train
- [x] Integration tests: end-to-end on HH neuron and RegionalNetwork GPi output
- [ ] Verification: compare spectral output against benchmark for same input data
