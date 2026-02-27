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

    Implements the Chronux ``mtspectrumpt`` algorithm:
    for each taper, evaluate the taper at spike times (via interpolation),
    compute the non-uniform DFT, average |J|^2 across tapers and trials,
    then subtract the mean firing rate as a bias correction.

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
    N = int(duration * Fs)
    dt = 1.0 / Fs

    # DPSS tapers — shape (K, N)
    H = dpss(N, NW, Kmax=K)

    # Continuous time axis for taper interpolation (seconds)
    t_taper = np.arange(N) * dt

    # Frequency grid matching an N-point DFT, restricted to fpass
    f_all = np.fft.rfftfreq(N, d=dt)          # Hz
    f_mask = (f_all >= fpass[0]) & (f_all <= fpass[1])
    f_out = f_all[f_mask]

    S_trials = []
    total_rate = 0.0

    for spikes in spike_times_list:
        spikes = np.asarray(spikes, dtype=np.float64)
        spikes = spikes[(spikes >= 0) & (spikes < duration)]
        total_rate += len(spikes) / duration

        if len(spikes) == 0:
            S_trials.append(np.zeros(f_out.size))
            continue

        # For each taper k, compute the tapered Fourier transform at the
        # exact spike times (non-uniform DFT):
        #   J_k(f) = Σ_i  h_k(t_i) · exp(−2πi · f · t_i)
        J = np.empty((K, f_out.size), dtype=complex)
        for k in range(K):
            h = np.interp(spikes, t_taper, H[k])       # (n_spikes,)
            phase = -2.0 * np.pi * np.outer(spikes, f_out)   # (n_spikes, n_freq)
            J[k] = h @ np.exp(1j * phase)               # (n_freq,)

        # Trial power: mean over tapers of |J|^2
        S_trials.append(np.mean(np.abs(J) ** 2, axis=0))

    # Average across neurons/trials
    S = np.mean(S_trials, axis=0) if S_trials else np.zeros(f_out.size)

    # Point-process bias correction: subtract mean firing rate
    mean_rate = total_rate / len(spike_times_list) if spike_times_list else 0.0
    S = S - mean_rate

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
