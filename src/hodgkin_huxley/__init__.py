"""
Neural Simulation Library

A fast C++ implementation of various neuron models with Python bindings.
Supports Hodgkin-Huxley, Izhikevich, and extensible to other models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Union

from ._core import (
    # Base class
    NeuronBase as _NeuronBase,
    # HH neuron
    HHNeuron as _HHNeuron,
    HHParameters,
    HHState,
    # Izhikevich neuron
    IzhikevichNeuron as _IzhikevichNeuron,
    IzhikevichParameters,
    IzhikevichState,
    IzhikevichType,
    # Synapse classes
    SynapseBase,
    ExponentialSynapse,
    AlphaSynapse,
    DoubleExponentialSynapse,
    # Network
    Network as _Network,
    NetworkNeuronType,
    ReceptorType,
    # Enums
    IntegrationMethod,
    # Version
    __version__,
    # Backwards compatibility
    Parameters,
    State,
)

__all__ = [
    # Neuron classes
    "HHNeuron",
    "IzhikevichNeuron",
    # Parameter/State classes
    "HHParameters",
    "HHState",
    "IzhikevichParameters",
    "IzhikevichState",
    "IzhikevichType",
    # Synapse classes
    "SynapseBase",
    "ExponentialSynapse",
    "AlphaSynapse",
    "DoubleExponentialSynapse",
    # Network
    "Network",
    "NetworkNeuronType",
    "ReceptorType",
    # Enums
    "IntegrationMethod",
    # Version
    "__version__",
    # Backwards compatibility
    "Parameters",
    "State",
]


class HHNeuron:
    """
    Hodgkin-Huxley neuron model.

    Implements the classic Hodgkin-Huxley model with Na+, K+, and leak channels.

    Parameters
    ----------
    parameters : HHParameters, optional
        Custom parameters for the neuron. If not provided, uses default
        squid giant axon parameters.
    method : IntegrationMethod, optional
        Integration method (EULER, RK4, RK45_ADAPTIVE). Default is RK4.

    Examples
    --------
    >>> neuron = HHNeuron()
    >>> trace = neuron.simulate(duration=100, dt=0.01, I_ext=10)
    >>> print(f"Max voltage: {max(trace):.1f} mV")
    """

    def __init__(
        self,
        parameters: HHParameters | None = None,
        method: IntegrationMethod | None = None,
    ):
        if parameters is not None and method is not None:
            self._neuron = _HHNeuron(parameters, method)
        elif parameters is not None:
            self._neuron = _HHNeuron(parameters)
        else:
            self._neuron = _HHNeuron()
            if method is not None:
                self._neuron.integration_method = method

    @property
    def V(self) -> float:
        """Membrane potential in mV."""
        return self._neuron.V

    @V.setter
    def V(self, value: float) -> None:
        self._neuron.V = value

    @property
    def state(self) -> HHState:
        """Current state of the neuron (V, m, h, n)."""
        return self._neuron.state

    @property
    def parameters(self) -> HHParameters:
        """Neuron parameters."""
        return self._neuron.parameters

    @property
    def integration_method(self) -> IntegrationMethod:
        """Integration method (EULER, RK4, or RK45_ADAPTIVE)."""
        return self._neuron.integration_method

    @integration_method.setter
    def integration_method(self, method: IntegrationMethod) -> None:
        self._neuron.integration_method = method

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


class IzhikevichNeuron:
    """
    Izhikevich neuron model.

    A computationally efficient model that can reproduce many biologically
    realistic spiking patterns with only 2 state variables.

    Parameters
    ----------
    neuron_type : IzhikevichType, optional
        Preset neuron type (REGULAR_SPIKING, FAST_SPIKING, etc.).
    parameters : IzhikevichParameters, optional
        Custom parameters. Overrides neuron_type if both provided.

    Examples
    --------
    >>> # Regular spiking cortical neuron
    >>> neuron = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
    >>> trace = neuron.simulate(duration=100, dt=0.1, I_ext=10)

    >>> # Fast spiking interneuron
    >>> neuron = IzhikevichNeuron(IzhikevichType.FAST_SPIKING)
    """

    def __init__(
        self,
        neuron_type: IzhikevichType | None = None,
        parameters: IzhikevichParameters | None = None,
    ):
        if parameters is not None:
            self._neuron = _IzhikevichNeuron(parameters)
        elif neuron_type is not None:
            self._neuron = _IzhikevichNeuron(neuron_type)
        else:
            self._neuron = _IzhikevichNeuron()

    @property
    def V(self) -> float:
        """Membrane potential in mV."""
        return self._neuron.V

    @V.setter
    def V(self, value: float) -> None:
        self._neuron.V = value

    @property
    def u(self) -> float:
        """Recovery variable."""
        return self._neuron.u

    @property
    def state(self) -> IzhikevichState:
        """Current state of the neuron (v, u)."""
        return self._neuron.state

    @property
    def parameters(self) -> IzhikevichParameters:
        """Neuron parameters (a, b, c, d)."""
        return self._neuron.parameters

    @property
    def spiked(self) -> bool:
        """True if neuron spiked in the last step."""
        return self._neuron.spiked

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
            External current.
        """
        self._neuron.step(dt, I_ext)

    def simulate(
        self,
        duration: float,
        dt: float = 0.1,
        I_ext: float | ArrayLike = 0.0,
    ) -> NDArray[np.float64]:
        """
        Run a simulation and return the voltage trace.

        Parameters
        ----------
        duration : float
            Simulation duration in milliseconds.
        dt : float, optional
            Time step in milliseconds. Default is 0.1 ms.
        I_ext : float or array-like, optional
            External current. Can be a constant value or a time series.

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

    @staticmethod
    def get_preset(neuron_type: IzhikevichType) -> IzhikevichParameters:
        """Get parameters for a preset neuron type."""
        return _IzhikevichNeuron.get_preset(neuron_type)

    def __repr__(self) -> str:
        return f"<IzhikevichNeuron v={self.V:.2f} mV>"


class Network:
    """
    Network of interconnected neurons.

    Supports mixed networks with HH and Izhikevich neurons.

    Parameters
    ----------
    num_neurons : int, optional
        Number of HH neurons to create. Default is 0.
    neuron_type : NetworkNeuronType, optional
        Type of neurons to create when using num_neurons.

    Examples
    --------
    >>> # Create HH network
    >>> net = Network(2)
    >>> net.add_synapse(0, 1, weight=0.5)
    >>> traces = net.simulate(duration=100, dt=0.01, I_ext=[[10]*10000, [0]*10000])

    >>> # Create mixed network
    >>> net = Network()
    >>> net.add_hh_neuron()
    >>> net.add_izhikevich_neuron(IzhikevichType.FAST_SPIKING)
    >>> net.add_synapse(0, 1, weight=1.0)
    """

    def __init__(
        self,
        num_neurons: int = 0,
        neuron_type: NetworkNeuronType | None = None,
    ):
        if neuron_type is not None:
            self._network = _Network(num_neurons, neuron_type)
        else:
            self._network = _Network(num_neurons)

    def add_neuron(
        self,
        parameters: Union[HHParameters, IzhikevichParameters, None] = None,
        neuron_type: NetworkNeuronType | None = None,
    ) -> int:
        """
        Add a neuron to the network.

        Parameters
        ----------
        parameters : HHParameters or IzhikevichParameters, optional
            Custom parameters for the neuron.
        neuron_type : NetworkNeuronType, optional
            Type of neuron to add.

        Returns
        -------
        int
            Index of the added neuron.
        """
        if neuron_type is not None:
            return self._network.add_neuron(neuron_type)
        elif parameters is not None:
            return self._network.add_neuron(parameters)
        return self._network.add_neuron()

    def add_hh_neuron(self, parameters: HHParameters | None = None) -> int:
        """
        Add an HH neuron to the network.

        Parameters
        ----------
        parameters : HHParameters, optional
            Custom parameters for the neuron.

        Returns
        -------
        int
            Index of the added neuron.
        """
        if parameters is not None:
            return self._network.add_hh_neuron(parameters)
        return self._network.add_hh_neuron()

    def add_izhikevich_neuron(
        self,
        neuron_type: IzhikevichType = IzhikevichType.REGULAR_SPIKING,
        parameters: IzhikevichParameters | None = None,
    ) -> int:
        """
        Add an Izhikevich neuron to the network.

        Parameters
        ----------
        neuron_type : IzhikevichType, optional
            Preset neuron type. Default is REGULAR_SPIKING.
        parameters : IzhikevichParameters, optional
            Custom parameters. Overrides neuron_type if provided.

        Returns
        -------
        int
            Index of the added neuron.
        """
        if parameters is not None:
            return self._network.add_izhikevich_neuron(parameters)
        return self._network.add_izhikevich_neuron(neuron_type)

    def add_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        E_syn: float = 0.0,
        tau: float = 2.0,
        delay: float = 0.0,
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
            Synaptic reversal potential in mV. Default is 0 (excitatory).
            Use -80 for inhibitory synapses.
        tau : float, optional
            Synaptic time constant in ms. Default is 2.
        delay : float, optional
            Axonal conduction delay in ms. Default is 0.
        """
        self._network.add_synapse(pre_idx, post_idx, weight, E_syn, tau, delay)

    def add_alpha_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        E_syn: float = 0.0,
        tau: float = 2.0,
        delay: float = 0.0,
    ) -> None:
        """
        Add an alpha-function synapse between two neurons.

        Conductance follows g(t) = g_peak * (t/tau) * exp(1 - t/tau),
        producing a smooth rise and fall that peaks at t = tau.

        Parameters
        ----------
        pre_idx : int
            Index of the pre-synaptic neuron.
        post_idx : int
            Index of the post-synaptic neuron.
        weight : float
            Peak synaptic conductance.
        E_syn : float, optional
            Synaptic reversal potential in mV. Default is 0 (excitatory).
        tau : float, optional
            Time to peak in ms. Default is 2.
        delay : float, optional
            Axonal conduction delay in ms. Default is 0.
        """
        self._network.add_alpha_synapse(pre_idx, post_idx, weight, E_syn, tau, delay)

    def add_double_exp_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        E_syn: float = 0.0,
        tau_rise: float = 0.4,
        tau_decay: float = 2.5,
        delay: float = 0.0,
    ) -> None:
        """
        Add a double-exponential synapse between two neurons.

        Conductance follows g(t) = g_peak * f * (exp(-t/tau_d) - exp(-t/tau_r)),
        with separate rise and decay time constants. Used for AMPA/NMDA
        receptor kinetics. Requires tau_rise < tau_decay.

        Parameters
        ----------
        pre_idx : int
            Index of the pre-synaptic neuron.
        post_idx : int
            Index of the post-synaptic neuron.
        weight : float
            Peak synaptic conductance.
        E_syn : float, optional
            Synaptic reversal potential in mV. Default is 0 (excitatory).
        tau_rise : float, optional
            Rise time constant in ms. Default is 0.4.
        tau_decay : float, optional
            Decay time constant in ms. Default is 2.5.
        delay : float, optional
            Axonal conduction delay in ms. Default is 0.
        """
        self._network.add_double_exp_synapse(
            pre_idx, post_idx, weight, E_syn, tau_rise, tau_decay, delay)

    def add_ampa_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        delay: float = 0.0,
    ) -> None:
        """
        Add an AMPA synapse (fast excitatory).

        Uses double-exponential kinetics with E_syn=0 mV,
        tau_rise=0.5 ms, tau_decay=2.5 ms.

        Parameters
        ----------
        pre_idx : int
            Index of the pre-synaptic neuron.
        post_idx : int
            Index of the post-synaptic neuron.
        weight : float
            Peak synaptic conductance.
        delay : float, optional
            Axonal conduction delay in ms. Default is 0.
        """
        self._network.add_ampa_synapse(pre_idx, post_idx, weight, delay)

    def add_nmda_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        delay: float = 0.0,
    ) -> None:
        """
        Add an NMDA synapse (slow excitatory).

        Uses double-exponential kinetics with E_syn=0 mV,
        tau_rise=2.0 ms, tau_decay=67.0 ms.

        Parameters
        ----------
        pre_idx : int
            Index of the pre-synaptic neuron.
        post_idx : int
            Index of the post-synaptic neuron.
        weight : float
            Peak synaptic conductance.
        delay : float, optional
            Axonal conduction delay in ms. Default is 0.
        """
        self._network.add_nmda_synapse(pre_idx, post_idx, weight, delay)

    def add_gaba_a_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        delay: float = 0.0,
    ) -> None:
        """
        Add a GABA_A synapse (inhibitory).

        Uses double-exponential kinetics with E_syn=-80 mV,
        tau_rise=0.4 ms, tau_decay=7.7 ms.

        Parameters
        ----------
        pre_idx : int
            Index of the pre-synaptic neuron.
        post_idx : int
            Index of the post-synaptic neuron.
        weight : float
            Peak synaptic conductance.
        delay : float, optional
            Axonal conduction delay in ms. Default is 0.
        """
        self._network.add_gaba_a_synapse(pre_idx, post_idx, weight, delay)

    def add_receptor_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        receptor: "ReceptorType",
        delay: float = 0.0,
    ) -> None:
        """
        Add a synapse by receptor type.

        Parameters
        ----------
        pre_idx : int
            Index of the pre-synaptic neuron.
        post_idx : int
            Index of the post-synaptic neuron.
        weight : float
            Peak synaptic conductance.
        receptor : ReceptorType
            Receptor type (AMPA, NMDA, or GABA_A).
        delay : float, optional
            Axonal conduction delay in ms. Default is 0.
        """
        self._network.add_receptor_synapse(pre_idx, post_idx, weight, receptor, delay)

    def synapse(self, idx: int) -> SynapseBase:
        """Get a synapse by index (polymorphic access)."""
        return self._network.synapse(idx)

    @property
    def num_neurons(self) -> int:
        """Number of neurons in the network."""
        return self._network.num_neurons

    @property
    def num_synapses(self) -> int:
        """Number of synaptic connections."""
        return self._network.num_synapses

    def neuron(self, idx: int) -> _NeuronBase:
        """Get a neuron by index (polymorphic access)."""
        return self._network.neuron(idx)

    def hh_neuron(self, idx: int) -> _HHNeuron:
        """Get an HH neuron by index. Throws if wrong type."""
        return self._network.hh_neuron(idx)

    def iz_neuron(self, idx: int) -> _IzhikevichNeuron:
        """Get an Izhikevich neuron by index. Throws if wrong type."""
        return self._network.iz_neuron(idx)

    def neuron_type(self, idx: int) -> str:
        """Get the type name of a neuron at given index."""
        return self._network.neuron_type(idx)

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
            External currents for each neuron.
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
