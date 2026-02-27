"""
Flexible current pulse generator for neural stimulation experiments.

``PulseStimulator`` produces monophasic or biphasic rectangular current
pulses at arbitrary onset times, compatible with the ``I_ext`` arrays
accepted by ``Network.simulate()`` and ``RegionalNetwork.simulate()``.

Typical usage
-------------
    # Benchmark cortical stimulus
    pulse = PulseStimulator.single(onset=1000.0, duration=0.3, amplitude=350.0)
    I = pulse.generate(2000.0, dt=0.01)                # 1-D ndarray

    # Charge-balanced DBS-style biphasic train
    train = PulseStimulator.train(
        frequency=130, duration=0.06, amplitude=300,
        onset=0.0, total_duration=1000.0, waveform="biphasic"
    )

    # Apply on top of a tonic background current
    I_net = pulse.apply_to(base_current=3.0, total_duration=2000.0, dt=0.01)

    # Use directly in RegionalNetwork
    rn.simulate(2000.0, 0.01, {
        "Cortex_E": pulse.generate(2000.0, 0.01),
        "STN":      dbs_train.generate(2000.0, 0.01),
        "GPe":      3.0,
    })
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PulseStimulator:
    """
    Flexible rectangular / biphasic current-pulse generator.

    Supports single pulses, arbitrary-onset multi-pulse sequences,
    frequency-based trains, and burst patterns.  Biphasic mode appends
    a charge-balancing anodic phase after each cathodic pulse.

    All time parameters are in **milliseconds**; amplitudes in **µA/cm²**.

    Parameters
    ----------
    onsets : float or list of float
        Pulse onset time(s) in ms.  Sorted automatically.
    duration : float
        Cathodic-phase duration in ms.  Must be > 0.
    amplitude : float
        Cathodic-phase amplitude in µA/cm².
    waveform : {"rectangular", "biphasic"}
        ``"rectangular"`` — monophasic pulse.
        ``"biphasic"`` — cathodic phase then anodic phase (charge-balanced
        by default).
    interphase_gap : float
        Dead-time between cathodic and anodic phases in ms (biphasic only).
        Default 0.
    anodic_amplitude : float, optional
        Anodic-phase amplitude (positive value, µA/cm²).
        Defaults to ``abs(amplitude)`` (symmetric, charge-balanced).
    anodic_duration : float, optional
        Anodic-phase duration in ms.
        Defaults to ``duration`` (symmetric, charge-balanced).

    Examples
    --------
    >>> # Benchmark cortical pulse (task 9)
    >>> pulse = PulseStimulator.single(onset=1000.0, duration=0.3, amplitude=350.0)
    >>> I = pulse.generate(2000.0, dt=0.01)

    >>> # 130 Hz biphasic DBS train for 1 second
    >>> train = PulseStimulator.train(
    ...     frequency=130, duration=0.06, amplitude=300,
    ...     onset=0.0, total_duration=1000.0, waveform="biphasic"
    ... )

    >>> # 5-pulse burst at 300 Hz starting at t=500 ms
    >>> burst = PulseStimulator.burst(
    ...     n_pulses=5, intra_freq=300, duration=0.06, amplitude=200, onset=500.0
    ... )

    >>> # Add pulse on top of tonic background
    >>> I_net = pulse.apply_to(base_current=3.0, total_duration=2000.0, dt=0.01)
    """

    def __init__(
        self,
        onsets: float | list[float],
        duration: float,
        amplitude: float,
        waveform: str = "rectangular",
        interphase_gap: float = 0.0,
        anodic_amplitude: float | None = None,
        anodic_duration: float | None = None,
    ):
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        if waveform not in ("rectangular", "biphasic"):
            raise ValueError(
                f"waveform must be 'rectangular' or 'biphasic', got {waveform!r}"
            )
        if interphase_gap < 0:
            raise ValueError(f"interphase_gap must be >= 0, got {interphase_gap}")

        if isinstance(onsets, (int, float)):
            onsets = [float(onsets)]
        self._onsets = sorted(float(t) for t in onsets)
        if any(t < 0 for t in self._onsets):
            raise ValueError("All onset times must be >= 0")

        self._duration          = float(duration)
        self._amplitude         = float(amplitude)
        self._waveform          = waveform
        self._interphase_gap    = float(interphase_gap)
        self._anodic_amplitude  = (
            float(anodic_amplitude) if anodic_amplitude is not None
            else abs(self._amplitude)
        )
        self._anodic_duration   = (
            float(anodic_duration) if anodic_duration is not None
            else self._duration
        )

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def single(
        cls,
        onset: float,
        duration: float,
        amplitude: float,
        **kwargs,
    ) -> "PulseStimulator":
        """Single pulse at ``onset`` ms."""
        return cls([onset], duration, amplitude, **kwargs)

    @classmethod
    def from_onsets(
        cls,
        onsets: list[float],
        duration: float,
        amplitude: float,
        **kwargs,
    ) -> "PulseStimulator":
        """
        Pulses at an explicit list of onset times.

        Use this when you want pulses at irregular intervals
        (e.g., matching a recorded spike train).
        """
        return cls(onsets, duration, amplitude, **kwargs)

    @classmethod
    def train(
        cls,
        frequency: float,
        duration: float,
        amplitude: float,
        onset: float = 0.0,
        n_pulses: int | None = None,
        total_duration: float | None = None,
        **kwargs,
    ) -> "PulseStimulator":
        """
        Regular pulse train at a fixed repetition frequency.

        Parameters
        ----------
        frequency : float
            Pulse repetition rate in Hz.
        duration : float
            Individual pulse duration in ms.
        amplitude : float
            Pulse amplitude in µA/cm².
        onset : float
            Time of the first pulse in ms.  Default 0.
        n_pulses : int, optional
            Total number of pulses.  Mutually exclusive with ``total_duration``.
        total_duration : float, optional
            Generate pulses while ``onset_i < total_duration`` (ms).
            Mutually exclusive with ``n_pulses``.

        Examples
        --------
        >>> PulseStimulator.train(130, 0.06, 300, onset=0, n_pulses=10)
        >>> PulseStimulator.train(130, 0.06, 300, onset=0, total_duration=1000)
        """
        if frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {frequency}")
        if (n_pulses is None) == (total_duration is None):
            raise ValueError("Provide exactly one of n_pulses or total_duration")

        period = 1000.0 / frequency  # ms per cycle
        if n_pulses is not None:
            onsets = [onset + i * period for i in range(n_pulses)]
        else:
            onsets = list(np.arange(onset, total_duration, period))

        return cls(onsets, duration, amplitude, **kwargs)

    @classmethod
    def burst(
        cls,
        n_pulses: int,
        intra_freq: float,
        duration: float,
        amplitude: float,
        onset: float = 0.0,
        **kwargs,
    ) -> "PulseStimulator":
        """
        Single burst: ``n_pulses`` pulses at ``intra_freq`` Hz.

        To repeat bursts, compose multiple ``PulseStimulator`` instances
        or pre-compute onsets and call ``from_onsets()``.

        Parameters
        ----------
        n_pulses : int
            Number of pulses in the burst.
        intra_freq : float
            Within-burst pulse frequency in Hz.
        duration : float
            Individual pulse duration in ms.
        amplitude : float
            Pulse amplitude in µA/cm².
        onset : float
            Burst start time in ms.

        Example
        -------
        >>> PulseStimulator.burst(5, intra_freq=300, duration=0.06,
        ...                        amplitude=200, onset=500.0)
        """
        period = 1000.0 / intra_freq
        onsets = [onset + i * period for i in range(n_pulses)]
        return cls(onsets, duration, amplitude, **kwargs)

    # ------------------------------------------------------------------
    # Current generation
    # ------------------------------------------------------------------

    def generate(self, total_duration: float, dt: float) -> NDArray[np.float64]:
        """
        Generate the pulse current trace as a 1-D numpy array.

        The returned array has ``int(total_duration / dt)`` elements,
        consistent with the ``I_ext`` arrays expected by
        ``Network.simulate()`` and ``RegionalNetwork.simulate()``.

        Parameters
        ----------
        total_duration : float
            Total waveform duration in ms.
        dt : float
            Simulation time step in ms.

        Returns
        -------
        ndarray, shape (n_steps,)
            Current in µA/cm² at each time step.
        """
        n_steps = int(total_duration / dt)
        I = np.zeros(n_steps, dtype=np.float64)

        for onset in self._onsets:
            # Cathodic (or only) phase
            c_start = int(round(onset / dt))
            c_end   = int(round((onset + self._duration) / dt))
            c_start = max(0, min(c_start, n_steps))
            c_end   = max(0, min(c_end, n_steps))
            I[c_start:c_end] += self._amplitude

            if self._waveform == "biphasic":
                # Anodic phase (opposite sign, subtracts from I)
                a_t0    = onset + self._duration + self._interphase_gap
                a_start = int(round(a_t0 / dt))
                a_end   = int(round((a_t0 + self._anodic_duration) / dt))
                a_start = max(0, min(a_start, n_steps))
                a_end   = max(0, min(a_end, n_steps))
                I[a_start:a_end] -= self._anodic_amplitude

        return I

    def apply_to(
        self,
        base_current: float | NDArray[np.float64],
        total_duration: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Add pulse to a base (background) current and return the sum.

        Parameters
        ----------
        base_current : float or ndarray
            Tonic background current.  A scalar is broadcast to all time
            steps; a 1-D array is summed element-wise.
        total_duration : float
            Duration in ms.
        dt : float
            Time step in ms.

        Returns
        -------
        ndarray, shape (n_steps,)
        """
        pulse = self.generate(total_duration, dt)
        base  = np.asarray(base_current, dtype=np.float64)
        if base.ndim == 0:
            base = np.full(pulse.size, base.item(), dtype=np.float64)
        return base + pulse

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def onsets(self) -> list[float]:
        """Sorted pulse onset times in ms."""
        return list(self._onsets)

    @property
    def duration(self) -> float:
        """Cathodic-phase duration in ms."""
        return self._duration

    @property
    def amplitude(self) -> float:
        """Cathodic-phase amplitude in µA/cm²."""
        return self._amplitude

    @property
    def waveform(self) -> str:
        """Pulse waveform type (``'rectangular'`` or ``'biphasic'``)."""
        return self._waveform

    @property
    def n_pulses(self) -> int:
        """Number of pulses."""
        return len(self._onsets)

    @property
    def charge_per_phase(self) -> float:
        """
        Charge delivered per cathodic phase (amplitude × duration,
        in µA·ms/cm²).
        """
        return self._amplitude * self._duration

    @property
    def is_charge_balanced(self) -> bool:
        """
        True when cathodic and anodic charges are equal.

        Always True for rectangular waveforms (no anodic phase).
        For biphasic, requires amplitude×duration == anodic_amplitude×anodic_duration.
        """
        if self._waveform == "rectangular":
            return True
        cathodic = abs(self._amplitude) * self._duration
        anodic   = self._anodic_amplitude * self._anodic_duration
        return abs(cathodic - anodic) < 1e-10

    def __repr__(self) -> str:
        return (
            f"<PulseStimulator {self._waveform} "
            f"n_pulses={self.n_pulses} "
            f"amp={self._amplitude} µA/cm² "
            f"dur={self._duration} ms>"
        )
