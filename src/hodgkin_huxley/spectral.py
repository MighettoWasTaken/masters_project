"""
Spectral analysis for neural simulation output.

Provides a multitaper point-process spectral estimator (Chronux-compatible)
and helpers for computing beta-band oscillatory power — the primary metric
for evaluating DBS efficacy in the basal-ganglia benchmark.

Typical usage
-------------
    cfg = RecordingConfig(["spikes"])
    result = rnet.simulate(1000.0, 0.01, I_ext, record=cfg)

    analysis = analyze_beta_power(result["GPi"], duration_ms=1000.0)
    print(f"GPi beta power: {analysis['power']:.4f}")
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import dpss
from scipy.interpolate import interp1d as _interp1d


# =============================================================================
# Core algorithm: multitaper point-process spectrum
# =============================================================================

def mtspectrumpt(spike_times_list: list,
                 duration: float,
                 Fs: float = 100_000.0,
                 fpass: tuple = (1.0, 100.0),
                 tapers: tuple = (3, 5)) -> tuple[np.ndarray, np.ndarray]:
    """
    Multitaper spectral estimator for point-process (spike train) data.

    Implements the Chronux ``mtspectrumpt`` algorithm (Mitra & Bokil 2008),
    matching simulate_network_model.py / MATLAB Chronux exactly:

      J_k(f) = Σ_i h_k(t_i) exp(−2πif(t_i − t₀)) − H_k(f) · Msp

    where H_k(f) = FFT(h_k)[f] (tapers scaled by √Fs), Msp = Nsp/N.

    Parameters
    ----------
    spike_times_list : list of array_like
        Spike times **in seconds** for each neuron/trial.
    duration : float
        Total recording duration in seconds.
    Fs : float
        Sampling frequency in Hz (default 100 000 Hz = 1/(0.01 ms)).
    fpass : (fmin, fmax)
        Frequency band of interest in Hz.
    tapers : (NW, K)
        DPSS time-bandwidth product and number of tapers.

    Returns
    -------
    S : ndarray, shape (n_freqs,)
        Bias-corrected, trial-averaged power spectral density.
    f : ndarray, shape (n_freqs,)
        Corresponding frequencies in Hz.
    """
    NW, K = tapers
    dt = 1.0 / Fs

    # Time grid covering the full recording duration
    t = np.arange(0, duration, dt)
    N = len(t)

    # Zero-pad to next power of 2 (matches benchmark pad=0)
    nfft = int(2 ** np.ceil(np.log2(max(N, 1))))
    nfft = max(nfft, N)

    # Frequency grid: uniform from 0 to Fs, restricted to fpass
    df = Fs / nfft
    f_all = np.arange(0, Fs, df)[:nfft]
    findx = np.where((f_all >= fpass[0]) & (f_all <= fpass[-1]))[0]
    f_out = f_all[findx]
    w = 2.0 * np.pi * f_out

    # DPSS tapers scaled by sqrt(Fs) — Chronux convention
    tap, _ = dpss(N, NW, K, return_ratios=True)   # (K, N)
    tap = tap * np.sqrt(Fs)

    # FFT of tapers for mean-rate bias correction: shape (nfreq, K)
    H = np.fft.fft(tap.T, nfft, axis=0)[findx, :]

    nfreq = len(f_out)
    C = len(spike_times_list)
    J = np.zeros((nfreq, K, C), dtype=complex)
    Msp = np.zeros(C)

    for ch, spikes in enumerate(spike_times_list):
        spikes = np.asarray(spikes, dtype=np.float64).ravel()
        spikes = spikes[(spikes >= t[0]) & (spikes <= t[-1])]
        Nsp = len(spikes)
        Msp[ch] = Nsp / N   # mean spike count per time-grid point

        if Nsp > 0:
            # Taper values at spike times (linear interpolation)
            data_proj = _interp1d(t, tap.T, axis=0, kind='linear',
                                  fill_value='extrapolate')(spikes)  # type: ignore[call-arg]  # (Nsp, K)
            # Phase at spike times relative to t[0]
            exponential = np.exp(-1j * np.outer(w, spikes - t[0]))  # (nfreq, Nsp)
            # J = tapered NUFFT − mean-rate bias (Chronux eq.)
            J[:, :, ch] = exponential @ data_proj - H * Msp[ch]

    # Power: mean over tapers → (nfreq, C)
    S = np.mean(np.real(np.conj(J) * J), axis=1)

    # Average over trials (neurons) → (nfreq,)
    S = np.mean(S, axis=-1)

    return S, f_out


# =============================================================================
# Beta-band integration
# =============================================================================

def beta_band_power(S: np.ndarray, f: np.ndarray,
                    fmin: float = 7.0, fmax: float = 35.0) -> float:
    """
    Integrate power spectral density over the beta frequency band.

    Uses the trapezoidal rule, matching the benchmark's
    ``np.trapz(beta, betaf)`` computation.

    Parameters
    ----------
    S : ndarray
        Power spectrum (output of ``mtspectrumpt``).
    f : ndarray
        Corresponding frequency array in Hz.
    fmin, fmax : float
        Band edges in Hz (default 7–35 Hz).

    Returns
    -------
    float
        Integrated beta-band power.
    """
    mask = (f >= fmin) & (f <= fmax)
    return float(np.trapezoid(S[mask], f[mask]))


# =============================================================================
# High-level convenience wrapper
# =============================================================================

def analyze_beta_power(result,
                       duration_ms: float | None = None,
                       Fs: float = 100_000.0,
                       fpass: tuple = (1.0, 100.0),
                       tapers: tuple = (3, 5),
                       band: tuple = (7.0, 35.0)) -> dict:
    """
    Compute beta-band oscillatory power from a ``MetricsResult``.

    Reads ``result["spikes"]`` (spike times in ms from the recording system),
    converts to seconds, runs the multitaper spectrum, and integrates over
    the beta band.

    Parameters
    ----------
    result : MetricsResult
        Must contain the ``"spikes"`` metric
        (use ``RecordingConfig(["spikes"])`` or any preset that includes it).
    duration_ms : float, optional
        Simulation duration in ms — read from ``result.duration`` if omitted.
    Fs : float
        Sampling frequency in Hz (default 100 000 Hz for dt=0.01 ms).
    fpass : (fmin, fmax)
        Frequency range for the spectrum in Hz.
    tapers : (NW, K)
        DPSS parameters.
    band : (fmin, fmax)
        Beta-band edges in Hz.

    Returns
    -------
    dict with keys:
        ``"power"``       — scalar beta-band integrated power
        ``"spectrum"``    — full PSD array (n_freqs,)
        ``"frequencies"`` — frequency array in Hz (n_freqs,)
    """
    if "spikes" not in result:
        raise ValueError(
            'result must contain "spikes" — use RecordingConfig(["spikes", ...])'
        )

    duration_s = (duration_ms if duration_ms is not None else result.duration) / 1000.0

    # Convert spike times from ms → seconds
    spike_times_s = [sp / 1000.0 for sp in result["spikes"]]

    S, f = mtspectrumpt(spike_times_s, duration_s,
                        Fs=Fs, fpass=fpass, tapers=tapers)
    power = beta_band_power(S, f, fmin=band[0], fmax=band[1])

    return {"power": power, "spectrum": S, "frequencies": f}
