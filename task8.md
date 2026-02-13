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
```python
# From benchmark:
spike_times = time[np.diff((voltage[k, :] > -20).astype(int), prepend=0) == 1]
```
For each neuron, find upward threshold crossings at -20 mV, return times in seconds.

### Multitaper Spectrum (Point Process)
The benchmark implements `mtspectrumpt()` — a multitaper spectral estimator for point processes (spike times), not continuous signals. This is based on the Chronux MATLAB toolbox.

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Fs | 100,000 Hz (= 1 / (0.01ms * 1e-3)) |
| fpass | [1, 100] Hz |
| tapers | [3, 5] (time-bandwidth product = 3, num tapers = 5) |
| trialave | 1 (average across neurons/trials) |

**Algorithm:**
1. Generate DPSS (Slepian) tapers: `scipy.signal.windows.dpss(N, NW=3, Kmax=5)`
2. For each neuron's spike times, compute the Fourier transform of the taper-weighted point process
3. Compute the power spectrum: `S(f) = mean(|J(f)|^2)` across tapers
4. Average across neurons (trialave=1)
5. Subtract bias: `S(f) = S(f) - mean_rate` (point process correction)

### Beta Band Power
```python
beta = S[(f > 7) & (f < 35)]
betaf = f[(f > 7) & (f < 35)]
area = np.trapz(beta, betaf)
```

---

## 8.2 Implementation Approach

This is best implemented in **Python** (not C++), since:
- It uses scipy/numpy for DPSS tapers, FFT, and integration
- It runs once after simulation (not in the hot loop)
- It needs to match the benchmark's Chronux-style `mtspectrumpt` exactly

### Python Module

```python
# src/hodgkin_huxley/spectral.py

def detect_spikes(voltage_traces, time_array, threshold=-20.0):
    """
    Detect spike times from voltage traces.

    Args:
        voltage_traces: dict {pop_name: ndarray (n_neurons, n_steps)}
                       or ndarray (n_neurons, n_steps)
        time_array: 1D array of time points (ms)
        threshold: spike detection threshold (mV)

    Returns:
        list of lists: spike times (in seconds) for each neuron
    """

def mtspectrumpt(spike_times_list, params):
    """
    Multitaper spectrum for point process data.

    Args:
        spike_times_list: list of dicts [{'times': [t1, t2, ...]}, ...]
                         or list of arrays
        params: dict with keys:
            'Fs': sampling frequency (Hz)
            'fpass': [fmin, fmax] frequency range
            'tapers': [NW, K] time-bandwidth product and number of tapers
            'trialave': 1 to average across trials/neurons

    Returns:
        S: power spectrum array
        f: frequency array
    """

def beta_band_power(S, f, fmin=7, fmax=35):
    """
    Compute integrated power in the beta frequency band.

    Args:
        S: power spectrum
        f: frequencies
        fmin, fmax: band edges (Hz)

    Returns:
        area: integrated power (trapz)
    """

def analyze_oscillatory_power(traces, dt, threshold=-20.0,
                               fpass=(1, 100), tapers=(3, 5),
                               band=(7, 35)):
    """
    Convenience function: spike detection + spectrum + band power.

    Args:
        traces: voltage traces dict {pop_name: ndarray} or ndarray
        dt: time step (ms)
        threshold: spike detection threshold
        fpass: frequency range for spectrum
        tapers: (NW, K) for DPSS
        band: (fmin, fmax) for band power integration

    Returns:
        dict with keys: 'power', 'spectrum', 'frequencies'
    """
```

---

## 8.3 Implementation Checklist
- [ ] Implement `detect_spikes()` — threshold crossing detection from voltage traces
- [ ] Implement `mtspectrumpt()` — multitaper point-process spectrum (matching Chronux)
- [ ] Implement `beta_band_power()` — trapezoidal integration in frequency band
- [ ] Implement `analyze_oscillatory_power()` — convenience wrapper
- [ ] Add to `__init__.py` exports
- [ ] Unit tests: spike detection on synthetic traces with known spike times
- [ ] Unit tests: spectrum of regular pulse train (known frequency peaks)
- [ ] Unit tests: beta band power computation
- [ ] Verification: compare spectral output against benchmark for same input data
