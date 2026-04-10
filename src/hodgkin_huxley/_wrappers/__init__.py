"""
hodgkin_huxley._wrappers — deprecated neuron/network wrapper classes.

These are internal implementation classes exposed only via
``hodgkin_huxley.legacy`` with DeprecationWarning.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._core import (
    ConnectivityPattern,
    HHNeuron as _HHNeuron,
    IzhikevichNeuron as _IzhikevichNeuron,
    IzhikevichType,
    NeuronBase as _NeuronBase,
    NeuronModelSpec,
    ReceptorType,
    SynapseBase,
    _Network as _Network,
    _NetworkNeuronType as _NetworkNeuronType,
)
from ..recording import MetricsResult, RecordingConfig, _run_recording


class _HHNeuronWrapper:
    """
    Hodgkin-Huxley neuron model (deprecated wrapper).

    .. deprecated::
        Use :class:`NeuronModelSpec.hh_default()` with :class:`RegionalNetwork` instead.
    """

    def __init__(
        self,
        parameters=None,
        method=None,
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
    def state(self):
        """Current state of the neuron (V, m, h, n)."""
        return self._neuron.state

    @property
    def parameters(self):
        """Neuron parameters."""
        return self._neuron.parameters

    @property
    def integration_method(self):
        """Integration method (EULER, RK4, or RK45_ADAPTIVE)."""
        return self._neuron.integration_method

    @integration_method.setter
    def integration_method(self, method) -> None:
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
        I_ext: "float | ArrayLike" = 0.0,
    ) -> "NDArray[np.float64]":
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
            trace = self._neuron.simulate(duration, dt, float(I_ext))  # type: ignore[arg-type]
        else:
            I_ext = np.asarray(I_ext, dtype=np.float64)
            trace = self._neuron.simulate(duration, dt, I_ext.tolist())

        return np.array(trace, dtype=np.float64)

    def __repr__(self) -> str:
        return f"<HHNeuron V={self.V:.2f} mV>"


class _IzhikevichNeuronWrapper:
    """
    Izhikevich neuron model (deprecated wrapper).

    .. deprecated::
        Use :class:`NeuronModelSpec.izhikevich()` with :class:`RegionalNetwork` instead.
    """

    def __init__(
        self,
        neuron_type=None,
        parameters=None,
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
    def state(self):
        """Current state of the neuron (v, u)."""
        return self._neuron.state

    @property
    def parameters(self):
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
        I_ext: "float | ArrayLike" = 0.0,
    ) -> "NDArray[np.float64]":
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
            trace = self._neuron.simulate(duration, dt, float(I_ext))  # type: ignore[arg-type]
        else:
            I_ext = np.asarray(I_ext, dtype=np.float64)
            trace = self._neuron.simulate(duration, dt, I_ext.tolist())

        return np.array(trace, dtype=np.float64)

    @staticmethod
    def get_preset(neuron_type: "IzhikevichType"):
        """Get parameters for a preset neuron type."""
        return _IzhikevichNeuron.get_preset(neuron_type)

    def __repr__(self) -> str:
        return f"<IzhikevichNeuron v={self.V:.2f} mV>"


class _NetworkWrapper:
    """
    Network of interconnected neurons.

    Supports mixed networks with HH and Izhikevich neurons.

    .. deprecated::
        Use :class:`RegionalNetwork` instead.

    Parameters
    ----------
    num_neurons : int, optional
        Number of HH neurons to create. Default is 0.
    neuron_type : _NetworkNeuronType, optional
        Type of neurons to create when using num_neurons.
    """

    def __init__(
        self,
        num_neurons: int = 0,
        neuron_type=None,
    ):
        if neuron_type is not None:
            self._network = _Network(num_neurons, neuron_type)
        else:
            self._network = _Network(num_neurons)

    def add_neuron(
        self,
        parameters=None,
        neuron_type=None,
        model=None,
    ) -> int:
        """
        Add a neuron to the network.

        Parameters
        ----------
        parameters : HHParameters, IzhikevichParameters, or NeuronModelSpec, optional
            Custom parameters for the neuron.
        neuron_type : _NetworkNeuronType, optional
            Type of neuron to add.
        model : NeuronModelSpec or NeuronModel, optional
            Composable neuron model specification.

        Returns
        -------
        int
            Index of the added neuron.
        """
        if model is not None:
            from .._equations import NeuronModel  # lazy — avoids import-time cycle
            spec = model.to_spec() if isinstance(model, NeuronModel) else model
            return self._network.add_neuron(spec)
        if neuron_type is not None:
            return self._network.add_neuron(neuron_type)
        elif parameters is not None:
            return self._network.add_neuron(parameters)
        return self._network.add_neuron()

    def add_hh_neuron(self, parameters=None) -> int:
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
            return self._network.add_neuron(parameters)
        return self._network.add_neuron()

    def add_izhikevich_neuron(
        self,
        neuron_type: IzhikevichType = IzhikevichType.REGULAR_SPIKING,
        parameters=None,
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
            return self._network.add_neuron(parameters)
        # IzhikevichType is not a Network::NeuronType — convert via get_preset
        params = _IzhikevichNeuron.get_preset(neuron_type)
        return self._network.add_neuron(params)

    def add_synapse(
        self,
        pre_idx: int,
        post_idx: int,
        weight: float,
        E_syn: float = 0.0,
        tau: float = 2.0,
        delay: float = 0.0,
    ) -> None:
        """Add a synaptic connection between two neurons."""
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
        """Add an alpha-function synapse between two neurons."""
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
        """Add a double-exponential synapse between two neurons."""
        self._network.add_double_exp_synapse(
            pre_idx, post_idx, weight, E_syn, tau_rise, tau_decay, delay
        )

    def add_ampa_synapse(self, pre_idx: int, post_idx: int, weight: float,
                         delay: float = 0.0) -> None:
        """Add an AMPA synapse (fast excitatory)."""
        self._network.add_ampa_synapse(pre_idx, post_idx, weight, delay)

    def add_nmda_synapse(self, pre_idx: int, post_idx: int, weight: float,
                         delay: float = 0.0) -> None:
        """Add an NMDA synapse (slow excitatory)."""
        self._network.add_nmda_synapse(pre_idx, post_idx, weight, delay)

    def add_gaba_a_synapse(self, pre_idx: int, post_idx: int, weight: float,
                           delay: float = 0.0) -> None:
        """Add a GABA_A synapse (inhibitory)."""
        self._network.add_gaba_a_synapse(pre_idx, post_idx, weight, delay)

    def add_receptor_synapse(self, pre_idx: int, post_idx: int, weight: float,
                             receptor: "ReceptorType", delay: float = 0.0) -> None:
        """Add a synapse by receptor type."""
        self._network.add_receptor_synapse(pre_idx, post_idx, weight, receptor, delay)

    def add_kinetic_synapse(self, pre: int, post: int, weight: float,
                            spec, delay: float = 0.0) -> int:
        """Add a kinetic (continuous V_pre-dependent) synapse. Returns synapse index."""
        return self._network.add_kinetic_synapse(pre, post, weight, spec, delay)

    def get_kin_S(self, synapse_idx: int) -> float:
        """Get the current kinetic gating variable S for a synapse."""
        return self._network.get_kin_S(synapse_idx)

    def get_kin_g(self, synapse_idx: int) -> float:
        """Get the current effective conductance g for a synapse."""
        return self._network.get_kin_g(synapse_idx)

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

    @property
    def fast_math(self) -> bool:
        """Use fast polynomial exp (~8 digits) vs full precision. Default: True."""
        return self._network.fast_math

    @fast_math.setter
    def fast_math(self, enabled: bool) -> None:
        self._network.fast_math = enabled

    def neuron(self, idx: int) -> "_NeuronBase":
        """Get a neuron by index (polymorphic access)."""
        return self._network.neuron(idx)

    def hh_neuron(self, idx: int):
        """Get an HH neuron by index. Throws if wrong type."""
        return self._network.hh_neuron(idx)

    def iz_neuron(self, idx: int):
        """Get an Izhikevich neuron by index. Throws if wrong type."""
        return self._network.iz_neuron(idx)

    def neuron_type(self, idx: int) -> str:
        """Get the type name of a neuron at given index."""
        return self._network.neuron_type(idx)

    def get_potentials(self) -> "NDArray[np.float64]":
        """Get membrane potentials of all neurons."""
        return np.array(self._network.get_potentials(), dtype=np.float64)

    def reset(self) -> None:
        """Reset all neurons to resting state."""
        self._network.reset()

    def step(self, dt: float, I_ext: "ArrayLike") -> None:
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
        I_ext: "ArrayLike",
        record: "RecordingConfig | None" = None,
        detection_threshold: "float | None" = None,
    ) -> "NDArray[np.float64] | MetricsResult":
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
        record : RecordingConfig, optional
            Recording configuration. If None (default), returns voltage
            traces as a plain ndarray (backward compatible). If provided,
            returns a MetricsResult with the requested metrics.
        detection_threshold : float, optional
            Voltage threshold (mV) used by the C++ hot loop to detect
            pre-synaptic spikes for synapse updates.

        Returns
        -------
        NDArray[np.float64]
            Voltage traces (num_neurons, num_timesteps) when record=None.
        MetricsResult
            When record is a RecordingConfig.
        """
        I_ext_list = np.ascontiguousarray(I_ext, dtype=np.float64)
        cfg = record if record is not None else RecordingConfig(["V"])
        result = _run_recording(
            self._network,
            duration,
            dt,
            I_ext_list,
            cfg,
            detection_threshold=detection_threshold,
        )
        if record is None:
            return result["V"]
        return result

    def __len__(self) -> int:
        return self.num_neurons

    def __repr__(self) -> str:
        return f"<Network neurons={self.num_neurons} synapses={self.num_synapses}>"
