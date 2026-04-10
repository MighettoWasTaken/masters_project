"""
hodgkin_huxley._dbs — Deep Brain Stimulation stimulator wrapper.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

from ._core import DBSParameters
from ._core import DBSStimulator as _DBSStimulator


class DBSStimulator:
    """
    Deep Brain Stimulation current generator.

    Produces a periodic rectangular pulse train:

        I(t) = amplitude   if  t mod (1000/frequency) < pulse_width
                0           otherwise

    Pulses begin at t=0 and are spaced by the inter-stimulus interval
    (ISI = 1000/frequency ms), exactly matching the benchmark creatdbs().

    Parameters
    ----------
    frequency : float
        Stimulation frequency in Hz. Use 0 to disable (all zeros). Default: 130.
    amplitude : float
        Pulse amplitude in uA/cm^2. Default: 0.
    pulse_width : float
        Pulse width in ms. Default: 0.06.

    Raises
    ------
    ValueError
        If frequency < 0, pulse_width <= 0, or pulse_width >= ISI.

    Examples
    --------
    >>> dbs = DBSStimulator(frequency=130, amplitude=300, pulse_width=0.06)
    >>> trace = dbs.generate(duration=1000, dt=0.01)  # numpy array, length ~100001
    >>> rn.attach_stimulator("STN", dbs)
    >>> traces = rn.simulate(1000, 0.01, {})
    """

    @overload
    def __init__(self, params: DBSParameters) -> None: ...
    @overload
    def __init__(self, frequency: float = ..., amplitude: float = ..., pulse_width: float = ...) -> None: ...

    def __init__(  # type: ignore[override]
        self,
        frequency: float | DBSParameters = 130.0,
        amplitude: float = 0.0,
        pulse_width: float = 0.06,
    ):
        if isinstance(frequency, DBSParameters):
            p = frequency
        else:
            p = DBSParameters()
            p.frequency = frequency
            p.amplitude = amplitude
            p.pulse_width = pulse_width
        self._dbs = _DBSStimulator(p)

    def generate(self, duration: float, dt: float) -> "NDArray[np.float64]":
        """
        Generate the full DBS current trace.

        Returns a numpy array of length ceil((duration + dt) / dt),
        matching np.arange(0, duration+dt, dt).

        Parameters
        ----------
        duration : float
            Total duration in ms.
        dt : float
            Time step in ms.
        """
        return np.array(self._dbs.generate(duration, dt), dtype=np.float64)

    def current_at(self, step_index: int, dt: float) -> float:
        """
        Get the DBS current at a specific simulation step index.

        Parameters
        ----------
        step_index : int
            Zero-based step index.
        dt : float
            Time step in ms.
        """
        return self._dbs.current_at(step_index, dt)

    @property
    def parameters(self) -> DBSParameters:
        """Current stimulator parameters."""
        return self._dbs.parameters

    @property
    def frequency(self) -> float:
        """Stimulation frequency in Hz."""
        return self._dbs.parameters.frequency

    @property
    def amplitude(self) -> float:
        """Pulse amplitude in uA/cm^2."""
        return self._dbs.parameters.amplitude

    @property
    def pulse_width(self) -> float:
        """Pulse width in ms."""
        return self._dbs.parameters.pulse_width

    def set_parameters(
        self,
        frequency: float | None = None,
        amplitude: float | None = None,
        pulse_width: float | None = None,
    ) -> None:
        """
        Update stimulator parameters. Validates on assignment.

        Only provided keyword arguments are changed; omitted ones keep their
        current value.
        """
        p = DBSParameters()
        p.frequency = frequency if frequency is not None else self.frequency
        p.amplitude = amplitude if amplitude is not None else self.amplitude
        p.pulse_width = pulse_width if pulse_width is not None else self.pulse_width
        self._dbs.set_parameters(p)

    def __repr__(self) -> str:
        return (
            f"<DBSStimulator freq={self.frequency}Hz "
            f"amp={self.amplitude} PW={self.pulse_width}ms>"
        )
