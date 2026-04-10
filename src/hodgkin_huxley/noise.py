"""
Noise injection adapter for neural simulations.

This module does **not** generate noise — it adapts noise produced by
external packages (numpy, colorednoise, scipy, stochastic, …) into the
array shapes expected by ``Network.simulate()`` and
``RegionalNetwork.simulate()``.

Quick reference
---------------
::

    import numpy as np
    from hodgkin_huxley import NoiseInjector

    # --- From a pre-generated 2D array (e.g. numpy) ---
    rng  = np.random.default_rng(42)
    arr  = rng.normal(0, 1.5, size=(50, n_steps))   # (n_neurons, n_steps)
    inj  = NoiseInjector(arr, dt=0.1)
    rnet.simulate(duration, 0.1, {"STN": inj.apply_to(25.0, 50, duration, 0.1)})

    # --- 1-D broadcast (same trace for every neuron) ---
    trace = rng.normal(0, 1.5, n_steps)              # (n_steps,)
    inj   = NoiseInjector(trace, dt=0.1)             # broadcasts automatically

    # --- From any package that generates one trace at a time ---
    import colorednoise as cn
    inj = NoiseInjector.from_generator(
        lambda n, rng: cn.powerlaw_psd_gaussian(1, n) * 1.5,   # pink noise, std≈1.5
        n_neurons=50, n_steps=n_steps, dt=0.1
    )

    # --- scipy filtered noise ---
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 0.05)   # low-pass
    inj = NoiseInjector.from_generator(
        lambda n, rng: filtfilt(b, a, rng.standard_normal(n)),
        n_neurons=50, n_steps=n_steps, dt=0.1, seed=0
    )
"""

from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import ArrayLike, NDArray


class NoiseInjector:
    """
    Adapts externally-generated noise for injection into simulations.

    Accepts pre-generated noise arrays in any of these shapes:

    ``(n_steps,)``
        A single trace that is **broadcast** to every neuron.
    ``(n_neurons, n_steps)``
        Per-neuron traces — one row per neuron.

    The two main output methods mirror ``PulseStimulator``:

    * ``get(n_neurons, total_duration, dt)`` — returns a 2-D array ready
      to assign directly to an ``I_ext`` population key.
    * ``apply_to(base, n_neurons, total_duration, dt)`` — adds noise on
      top of a scalar or existing array base current.

    Parameters
    ----------
    noise : array_like
        Pre-generated noise.  Shape ``(n_steps,)`` or ``(n_neurons, n_steps)``.
    dt : float
        Time step at which the noise was generated (ms).  Used to verify
        alignment when the simulation is run at a different dt.
    """

    def __init__(self, noise: ArrayLike, dt: float):
        arr = np.asarray(noise, dtype=np.float64)

        if arr.ndim == 1:
            # Store as (1, n_steps); broadcast in get()
            self._noise = arr[np.newaxis, :]
            self._broadcast = True
        elif arr.ndim == 2:
            self._noise = arr
            self._broadcast = False
        else:
            raise ValueError(
                f"noise must be 1-D (n_steps,) or 2-D (n_neurons, n_steps), "
                f"got shape {arr.shape}"
            )

        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self._dt = float(dt)

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_generator(
        cls,
        fn,
        n_neurons: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> "NoiseInjector":
        """
        Build noise by calling ``fn(n_steps, rng)`` independently for each neuron.

        This is compatible with any package that produces one 1-D trace per
        call, including ``colorednoise``, filtered scipy signals, custom
        processes, etc.

        Parameters
        ----------
        fn : callable
            Signature: ``fn(n_steps: int, rng: np.random.Generator) -> array_like``

            The ``rng`` argument is a seeded ``numpy.random.Generator``.
            The function may use it (for reproducibility) or ignore it (if
            the package has its own random state).  Must return an array-like
            of length ``n_steps``.
        n_neurons : int
            Number of independent traces to generate (one per neuron).
        n_steps : int
            Length of each trace in samples.
        dt : float
            Time step in ms — label only; does not affect generation.
        seed : int, optional
            Seed for the numpy RNG passed to ``fn``.

        Examples
        --------
        **colorednoise** (pink noise, α=1):

        .. code-block:: python

            import colorednoise as cn
            inj = NoiseInjector.from_generator(
                lambda n, rng: cn.powerlaw_psd_gaussian(1, n) * std,
                n_neurons=50, n_steps=n_steps, dt=dt
            )

        **scipy low-pass filtered white noise** (approximate OU-like):

        .. code-block:: python

            from scipy.signal import butter, filtfilt
            b, a = butter(2, cutoff / (0.5 / dt * 1e3))  # cutoff in Hz
            inj = NoiseInjector.from_generator(
                lambda n, rng: filtfilt(b, a, rng.standard_normal(n)) * std,
                n_neurons=50, n_steps=n_steps, dt=dt, seed=42
            )

        **Pure numpy white noise** (simpler to do directly, but works here too):

        .. code-block:: python

            inj = NoiseInjector.from_generator(
                lambda n, rng: rng.normal(0, std, n),
                n_neurons=50, n_steps=n_steps, dt=dt, seed=0
            )
        """
        rng = np.random.default_rng(seed)
        rows = np.array([np.asarray(fn(n_steps, rng), dtype=np.float64)
                         for _ in range(n_neurons)])
        return cls(rows, dt)

    @classmethod
    def from_array(
        cls,
        noise: ArrayLike,
        dt: float,
    ) -> "NoiseInjector":
        """
        Alias for the constructor — explicit for readability.

        Accepts any shape supported by the constructor:
        ``(n_steps,)`` or ``(n_neurons, n_steps)``.
        """
        return cls(noise, dt)

    # ------------------------------------------------------------------
    # Core output methods
    # ------------------------------------------------------------------

    def get(
        self,
        n_neurons: int,
        total_duration: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Return a ``(n_neurons, n_steps)`` noise array for use as ``I_ext``.

        Handles shape broadcasting and duration trimming.  A warning is
        issued if ``dt`` does not match the stored generation dt.

        Parameters
        ----------
        n_neurons : int
            Number of neurons the noise should cover.
        total_duration : float
            Simulation duration in ms.
        dt : float
            Simulation time step in ms.

        Returns
        -------
        ndarray, shape ``(n_neurons, n_steps)``
            Ready to pass as a population I_ext value.

        Raises
        ------
        ValueError
            If the stored noise has fewer neurons than requested (and is not
            in broadcast mode), or if the stored noise is shorter than needed.
        """
        self._check_dt(dt)
        n_steps = int(total_duration / dt)
        self._check_length(n_steps, total_duration, dt)

        if self._broadcast:
            # 1-D source → same trace for every neuron
            row = self._noise[0, :n_steps]
            return np.broadcast_to(row, (n_neurons, n_steps)).copy()

        stored_n = self._noise.shape[0]
        if stored_n < n_neurons:
            raise ValueError(
                f"Noise has {stored_n} neuron traces but {n_neurons} were "
                f"requested.  Generate more traces or use a 1-D broadcast array."
            )
        if stored_n > n_neurons:
            warnings.warn(
                f"Noise has {stored_n} rows; using the first {n_neurons}.",
                stacklevel=2,
            )
        return self._noise[:n_neurons, :n_steps].copy()

    def apply_to(
        self,
        base: "float | NDArray[np.float64]",
        n_neurons: int,
        total_duration: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Add noise to a base current and return the result.

        Parameters
        ----------
        base : float or ndarray
            Tonic background current.  Accepted shapes:

            * scalar → broadcast to ``(n_neurons, n_steps)``
            * ``(n_steps,)`` → broadcast across neurons
            * ``(n_neurons, n_steps)`` → added element-wise

        n_neurons : int
            Number of neurons.
        total_duration : float
            Simulation duration in ms.
        dt : float
            Simulation time step in ms.

        Returns
        -------
        ndarray, shape ``(n_neurons, n_steps)``
        """
        n_steps = int(total_duration / dt)
        noise   = self.get(n_neurons, total_duration, dt)
        base    = np.asarray(base, dtype=np.float64)

        if base.ndim == 0:
            return np.full((n_neurons, n_steps), base.item()) + noise
        elif base.ndim == 1:
            return base[np.newaxis, :n_steps] + noise
        elif base.ndim == 2:
            return base[:, :n_steps] + noise
        else:
            raise ValueError(f"base must be 0-D, 1-D, or 2-D, got {base.ndim}-D")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def array(self) -> NDArray[np.float64]:
        """Raw stored noise array, shape ``(n_neurons, n_steps)``."""
        return self._noise

    @property
    def n_neurons(self) -> int:
        """Number of stored neuron traces (1 if broadcast mode)."""
        return self._noise.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of time steps in the stored noise."""
        return self._noise.shape[1]

    @property
    def dt(self) -> float:
        """Time step at which the noise was generated (ms)."""
        return self._dt

    @property
    def is_broadcast(self) -> bool:
        """True if a single trace will be broadcast to all neurons."""
        return self._broadcast

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_dt(self, sim_dt: float) -> None:
        if abs(self._dt - sim_dt) / max(self._dt, sim_dt) > 1e-6:
            warnings.warn(
                f"Noise was generated at dt={self._dt} ms but simulation "
                f"dt={sim_dt} ms.  Temporal alignment may be incorrect.  "
                f"Consider regenerating noise at the simulation dt or "
                f"resampling with scipy.signal.resample before injecting.",
                stacklevel=3,
            )

    def _check_length(self, n_steps: int, total_duration: float, dt: float) -> None:
        available = self._noise.shape[1]
        if available < n_steps:
            raise ValueError(
                f"Noise has {available} steps ({available * self._dt:.1f} ms) "
                f"but {n_steps} steps ({total_duration:.1f} ms) are needed.  "
                f"Generate a longer noise trace."
            )

    def __repr__(self) -> str:
        mode = "broadcast" if self._broadcast else f"{self.n_neurons} neurons"
        return (
            f"<NoiseInjector {mode} × {self.n_steps} steps "
            f"@ dt={self._dt} ms>"
        )
