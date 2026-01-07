"""
Hodgkin-Huxley Neuron Simulation Library

A fast C++ implementation of Hodgkin-Huxley neurons with Python bindings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import (
    HHNeuron as _HHNeuron,
    Network as _Network,
    Parameters,
    State,
    IntegrationMethod,
    __version__,
)

__all__ = [
    "HHNeuron",
    "Network",
    "Parameters",
    "State",
    "IntegrationMethod",
    "__version__",
]


class HHNeuron:
    """
    Hodgkin-Huxley neuron model.

    Implements the classic Hodgkin-Huxley model with Na+, K+, and leak channels.

    Parameters
    ----------
    parameters : Parameters, optional
        Custom parameters for the neuron. If not provided, uses default
        squid giant axon parameters.

    Examples
    --------
    >>> neuron = HHNeuron()
    >>> trace = neuron.simulate(duration=100, dt=0.01, I_ext=10)
    >>> print(f"Max voltage: {max(trace):.1f} mV")
    """

    def __init__(self, parameters: Parameters | None = None):
        if parameters is not None:
            self._neuron = _HHNeuron(parameters)
        else:
            self._neuron = _HHNeuron()

    @property
    def V(self) -> float:
        """Membrane potential in mV."""
        return self._neuron.V

    @V.setter
    def V(self, value: float) -> None:
        self._neuron.V = value

    @property
    def state(self) -> State:
        """Current state of the neuron (V, m, h, n)."""
        return self._neuron.state

    @property
    def parameters(self) -> Parameters:
        """Neuron parameters."""
        return self._neuron.parameters

    def reset(self) -> None:
        """Reset the neuron to resting state."""
        self._neuron.reset()

    def step(self, dt: float, I_ext: float) -> None:
        """
        Advance the simulation by dt milliseconds.

        Parameters
        ----------
        dt : float
            Time step in milliseconds.
        I_ext : float
            External current in uA/cm^2.
        """
        self._neuron.step(dt, I_ext)

    def simulate(
        self,
        duration: float,
        dt: float = 0.01,
        I_ext: float | ArrayLike = 0.0,
    ) -> NDArray[np.float64]:
        """
        Run a simulation and return the voltage trace.

        Parameters
        ----------
        duration : float
            Simulation duration in milliseconds.
        dt : float, optional
            Time step in milliseconds. Default is 0.01 ms.
        I_ext : float or array-like, optional
            External current in uA/cm^2. Can be a constant value or
            a time series. Default is 0.

        Returns
        -------
        NDArray[np.float64]
            Membrane potential trace in mV.
        """
        if np.isscalar(I_ext):
            trace = self._neuron.simulate(duration, dt, float(I_ext))
        else:
            I_ext = np.asarray(I_ext, dtype=np.float64)
            trace = self._neuron.simulate(duration, dt, I_ext.tolist())

        return np.array(trace, dtype=np.float64)

    def __repr__(self) -> str:
        return f"<HHNeuron V={self.V:.2f} mV>"


class Network:
    """
    Network of interconnected Hodgkin-Huxley neurons.

    Parameters
    ----------
    num_neurons : int, optional
        Number of neurons to create. Default is 0.

    Examples
    --------
    >>> net = Network(2)
    >>> net.add_synapse(0, 1, weight=0.5)
    >>> traces = net.simulate(duration=100, dt=0.01, I_ext=[[10]*10000, [0]*10000])
    """

    def __init__(self, num_neurons: int = 0):
        self._network = _Network(num_neurons)

    def add_neuron(self, parameters: Parameters | None = None) -> int:
        """
        Add a neuron to the network.

        Parameters
        ----------
        parameters : Parameters, optional
            Custom parameters for the neuron.

        Returns
        -------
        int
            Index of the added neuron.
        """
        if parameters is not None:
            return self._network.add_neuron(parameters)
        return self._network.add_neuron()

    def add_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        E_syn: float = 0.0,
        tau: float = 2.0,
    ) -> None:
        """
        Add a synaptic connection between two neurons.

        Parameters
        ----------
        pre_idx : int
            Index of the pre-synaptic neuron.
        post_idx : int
            Index of the post-synaptic neuron.
        weight : float
            Synaptic weight (conductance).
        E_syn : float, optional
            Synaptic reversal potential in mV. Default is 0.
        tau : float, optional
            Synaptic time constant in ms. Default is 2.
        """
        self._network.add_synapse(pre_idx, post_idx, weight, E_syn, tau)

    @property
    def num_neurons(self) -> int:
        """Number of neurons in the network."""
        return self._network.num_neurons

    @property
    def num_synapses(self) -> int:
        """Number of synaptic connections."""
        return self._network.num_synapses

    def neuron(self, idx: int) -> _HHNeuron:
        """Get a neuron by index."""
        return self._network.neuron(idx)

    def get_potentials(self) -> NDArray[np.float64]:
        """Get membrane potentials of all neurons."""
        return np.array(self._network.get_potentials(), dtype=np.float64)

    def reset(self) -> None:
        """Reset all neurons to resting state."""
        self._network.reset()

    def step(self, dt: float, I_ext: ArrayLike) -> None:
        """
        Advance the simulation by dt milliseconds.

        Parameters
        ----------
        dt : float
            Time step in milliseconds.
        I_ext : array-like
            External currents for each neuron in uA/cm^2.
        """
        I_ext = np.asarray(I_ext, dtype=np.float64).tolist()
        self._network.step(dt, I_ext)

    def simulate(
        self,
        duration: float,
        dt: float,
        I_ext: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Run a network simulation.

        Parameters
        ----------
        duration : float
            Simulation duration in milliseconds.
        dt : float
            Time step in milliseconds.
        I_ext : array-like
            External currents, shape (num_neurons, num_timesteps).

        Returns
        -------
        NDArray[np.float64]
            Voltage traces, shape (num_neurons, num_timesteps).
        """
        I_ext = np.asarray(I_ext, dtype=np.float64)
        traces = self._network.simulate(duration, dt, I_ext.tolist())
        return np.array(traces, dtype=np.float64)

    def __len__(self) -> int:
        return self.num_neurons

    def __repr__(self) -> str:
        return f"<Network neurons={self.num_neurons} synapses={self.num_synapses}>"
