"""
Neural Simulation Library

A fast C++ implementation of various neuron models with Python bindings.
Supports Hodgkin-Huxley, Izhikevich, and extensible to other models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Union

from .recording import RecordingConfig, MetricsResult, PopulationMetricsResult
from .spectral import mtspectrumpt, beta_band_power, analyze_beta_power
from .pulse import PulseStimulator
from .noise import NoiseInjector

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
    # Regional network
    RegionalNetwork as _RegionalNetwork,
    ConnectivityPattern,
    SynapseSpec,
    SynapseSpecType,
    WeightDistribution,
    WeightDistType,
    # Enums
    IntegrationMethod,
    # Composable neuron types
    BoltzmannParams,
    TauParams,
    TauForm,
    RateFuncParams,
    RateFuncForm,
    GateSpec,
    GateUpdateForm,
    GateDependency,
    ChannelSpec,
    CalciumSpec,
    NeuronModelSpec,
    ComposableNeuron as _ComposableNeuron,
    # Kinetic synapse
    KineticSynapseSpec,
    KineticUpdateForm,
    KineticCurrentForm,
    # DBS stimulator
    DBSStimulator as _DBSStimulator,
    DBSParameters,
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
    # Regional network
    "RegionalNetwork",
    "ConnectivityPattern",
    "SynapseSpec",
    "SynapseSpecType",
    "WeightDistribution",
    "WeightDistType",
    # Enums
    "IntegrationMethod",
    # Composable neuron types
    "BoltzmannParams",
    "TauParams",
    "TauForm",
    "RateFuncParams",
    "RateFuncForm",
    "GateSpec",
    "GateUpdateForm",
    "GateDependency",
    "ChannelSpec",
    "CalciumSpec",
    "NeuronModelSpec",
    "NeuronModel",
    "Boltzmann",
    "Tau",
    "RateFunc",
    # Kinetic synapse
    "KineticSynapseSpec",
    "KineticUpdateForm",
    "KineticCurrentForm",
    "KineticSynapseModel",
    # DBS stimulator
    "DBSStimulator",
    "DBSParameters",
    # Recording system
    "RecordingConfig",
    "MetricsResult",
    "PopulationMetricsResult",
    # Spectral analysis
    "mtspectrumpt",
    "beta_band_power",
    "analyze_beta_power",
    # Pulse stimulator
    "PulseStimulator",
    # Noise injection adapter
    "NoiseInjector",
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
        parameters: "Union[HHParameters, IzhikevichParameters, NeuronModelSpec, None]" = None,
        neuron_type: NetworkNeuronType | None = None,
        model: "NeuronModelSpec | NeuronModel | None" = None,
    ) -> int:
        """
        Add a neuron to the network.

        Parameters
        ----------
        parameters : HHParameters, IzhikevichParameters, or NeuronModelSpec, optional
            Custom parameters for the neuron.
        neuron_type : NetworkNeuronType, optional
            Type of neuron to add.
        model : NeuronModelSpec or NeuronModel, optional
            Composable neuron model specification.

        Returns
        -------
        int
            Index of the added neuron.
        """
        if model is not None:
            spec = model.to_spec() if isinstance(model, NeuronModel) else model
            return self._network.add_neuron(spec)
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

    def add_kinetic_synapse(
        self,
        pre: int,
        post: int,
        weight: float,
        spec: "KineticSynapseSpec",
        delay: float = 0.0,
    ) -> int:
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
        record: "RecordingConfig | None" = None,
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

        Returns
        -------
        NDArray[np.float64]
            Voltage traces (num_neurons, num_timesteps) when record=None.
        MetricsResult
            When record is a RecordingConfig.
        """
        from .recording import _run_recording, RecordingConfig as _RC
        I_ext_arr = np.asarray(I_ext, dtype=np.float64)
        I_ext_list = I_ext_arr.tolist()
        cfg = record if record is not None else _RC(["V"])
        result = _run_recording(self._network, duration, dt, I_ext_list, cfg)
        if record is None:
            return result["V"]
        return result

    def __len__(self) -> int:
        return self.num_neurons

    def __repr__(self) -> str:
        return f"<Network neurons={self.num_neurons} synapses={self.num_synapses}>"


# String-to-enum mapping for neuron types
_NEURON_TYPE_MAP = {
    "HH": NetworkNeuronType.HH,
    "IZHIKEVICH_RS": NetworkNeuronType.IZHIKEVICH_RS,
    "IZHIKEVICH_FS": NetworkNeuronType.IZHIKEVICH_FS,
    "IZHIKEVICH_IB": NetworkNeuronType.IZHIKEVICH_IB,
    "IZHIKEVICH_CH": NetworkNeuronType.IZHIKEVICH_CH,
    "IZHIKEVICH_LTS": NetworkNeuronType.IZHIKEVICH_LTS,
    "IZHIKEVICH_CUSTOM": NetworkNeuronType.IZHIKEVICH_CUSTOM,
}

# String-to-enum mapping for connectivity patterns
_PATTERN_MAP = {
    "ALL_TO_ALL": ConnectivityPattern.ALL_TO_ALL,
    "ONE_TO_ONE": ConnectivityPattern.ONE_TO_ONE,
    "SHIFTED": ConnectivityPattern.SHIFTED,
    "RANDOM_SPARSE": ConnectivityPattern.RANDOM_SPARSE,
    "RANDOM_PERMUTATION": ConnectivityPattern.RANDOM_PERMUTATION,
}


def _resolve_weight(weight) -> WeightDistribution:
    """Convert weight shorthand to WeightDistribution."""
    if isinstance(weight, WeightDistribution):
        return weight
    if isinstance(weight, (int, float)):
        return WeightDistribution.constant(float(weight))
    if isinstance(weight, tuple) and len(weight) == 2:
        return WeightDistribution.uniform(float(weight[0]), float(weight[1]))
    raise TypeError(
        f"weight must be a float, (min, max) tuple, or WeightDistribution, "
        f"got {type(weight).__name__}"
    )


def _generate_pairs(pattern, src_size, dst_size, shift, probability, allow_self, rng,
                    same_pop=False):
    """Generate (i, j) pairs for a preset connectivity pattern."""
    pairs = []
    if pattern == ConnectivityPattern.ALL_TO_ALL:
        for i in range(src_size):
            for j in range(dst_size):
                if not allow_self and same_pop and i == j:
                    continue
                pairs.append((i, j))
    elif pattern == ConnectivityPattern.ONE_TO_ONE:
        n = min(src_size, dst_size)
        for k in range(n):
            pairs.append((k, k))
    elif pattern == ConnectivityPattern.SHIFTED:
        n = min(src_size, dst_size)
        for k in range(n):
            j = (k + shift) % n
            pairs.append((k, j))
    elif pattern == ConnectivityPattern.RANDOM_SPARSE:
        for i in range(src_size):
            for j in range(dst_size):
                if not allow_self and same_pop and i == j:
                    continue
                if rng.random() < probability:
                    pairs.append((i, j))
    elif pattern == ConnectivityPattern.RANDOM_PERMUTATION:
        n = min(src_size, dst_size)
        perm = rng.permutation(n)
        for k in range(n):
            pairs.append((k, int(perm[k])))
    return pairs


# ---------------------------------------------------------------------------
# Heterogeneity helpers (used by RegionalNetwork.add_population)
# ---------------------------------------------------------------------------

def _het_sample(dist: str, p1: float, p2: float, rng) -> float:
    """Sample a multiplicative scale factor from the given distribution."""
    if dist == "normal":
        return float(rng.normal(p1, p2))
    elif dist == "uniform":
        return float(rng.uniform(p1, p2))
    else:
        raise ValueError(
            f"Unknown heterogeneity distribution '{dist}'; "
            f"expected 'normal' or 'uniform'"
        )


def _het_get(spec, path: str):
    """Read a field from a NeuronModelSpec via a dotted path."""
    import re
    m = re.match(r'^(channels|gates)\[(\d+)\]\.(\w+)$', path)
    if m:
        col, idx, attr = m.group(1), int(m.group(2)), m.group(3)
        return getattr(getattr(spec, col)[idx], attr)
    if hasattr(spec, path):
        return getattr(spec, path)
    raise ValueError(
        f"Unknown heterogeneity field path '{path}'.  "
        f"Supported: 'channels[i].g', 'gates[i].initial_value', 'C_m', 'V_init'"
    )


def _het_set(spec, path: str, value: float) -> None:
    """Write a field on a NeuronModelSpec via a dotted path."""
    import re
    m = re.match(r'^(channels|gates)\[(\d+)\]\.(\w+)$', path)
    if m:
        col, idx, attr = m.group(1), int(m.group(2)), m.group(3)
        items = getattr(spec, col)   # get as Python list
        setattr(items[idx], attr, value)
        setattr(spec, col, items)    # write back (safe for copy or reference semantics)
        return
    if hasattr(spec, path):
        setattr(spec, path, value)
        return
    raise ValueError(
        f"Unknown heterogeneity field path '{path}'.  "
        f"Supported: 'channels[i].g', 'gates[i].initial_value', 'C_m', 'V_init'"
    )


def _build_heterogeneous_specs(base_spec, count: int, heterogeneity: dict, seed) -> list:
    """
    Return a list of *count* NeuronModelSpec copies, each with independent
    parameter samples drawn from *heterogeneity*.
    """
    import copy
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(count):
        s = copy.copy(base_spec)   # C++ copy constructor via __copy__
        for path, (dist, p1, p2) in heterogeneity.items():
            base_val = _het_get(s, path)
            scale    = _het_sample(dist, p1, p2, rng)
            _het_set(s, path, base_val * scale)
        specs.append(s)
    return specs


class RegionalNetwork:
    """
    Population-based network abstraction for multi-region neural simulations.

    Wraps a single Network. All neurons flow into the same HHPool/IzPool
    during simulate(). Populations are bookkeeping: {name, start_idx, count}.

    Examples
    --------
    >>> rn = RegionalNetwork()
    >>> rn.add_population("STN", 10, neuron_type="HH")
    >>> rn.add_population("GPe", 10, neuron_type="HH")
    >>> rn.connect("STN", "GPe", "all_to_all", weight=0.3, delay=2.0,
    ...            synapse=SynapseSpec.ampa())
    >>> traces = rn.simulate(100, 0.01, {"STN": 10.0})
    """

    def __init__(self):
        self._rnet = _RegionalNetwork()
        self._stimulators: dict = {}  # {pop_name: DBSStimulator}

    def add_population(
        self,
        name: str,
        count: int,
        neuron_type: "str | NetworkNeuronType | None" = None,
        parameters: "HHParameters | IzhikevichParameters | None" = None,
        model: "NeuronModelSpec | NeuronModel | None" = None,
        heterogeneity: "dict | None" = None,
        seed: "int | None" = None,
    ) -> None:
        """
        Add a population of neurons.

        Parameters
        ----------
        name : str
            Unique name for the population.
        count : int
            Number of neurons.
        neuron_type : str or NetworkNeuronType, optional
            Neuron type preset ('HH', 'IZHIKEVICH_RS', etc.).
        parameters : HHParameters or IzhikevichParameters, optional
            Custom neuron parameters (overrides neuron_type).
        model : NeuronModelSpec or NeuronModel, optional
            Composable neuron model specification.
        heterogeneity : dict, optional
            Per-neuron parameter scatter.  Maps dotted field paths to
            ``(distribution, param1, param2)`` triples, where:

            * ``"normal"``  → scale factor ~ N(param1, param2)
            * ``"uniform"`` → scale factor ~ Uniform(param1, param2)

            Supported paths: ``"channels[i].g"``, ``"C_m"``, ``"V_init"``,
            ``"gates[i].initial_value"``.  Requires ``model=`` to be set.
        seed : int, optional
            RNG seed for reproducible heterogeneity.
        """
        if heterogeneity is not None:
            if model is None:
                raise ValueError(
                    "heterogeneity requires 'model' (NeuronModelSpec or NeuronModel)"
                )
            spec = model.to_spec() if isinstance(model, NeuronModel) else model
            specs = _build_heterogeneous_specs(spec, count, heterogeneity, seed)
            self._rnet.add_population(name, specs)
        elif model is not None:
            spec = model.to_spec() if isinstance(model, NeuronModel) else model
            self._rnet.add_population(name, count, spec)
        elif parameters is not None:
            self._rnet.add_population(name, count, parameters)
        elif neuron_type is not None:
            if isinstance(neuron_type, str):
                neuron_type = _NEURON_TYPE_MAP[neuron_type.upper()]
            self._rnet.add_population(name, count, neuron_type)
        else:
            # Default to HH
            self._rnet.add_population(name, count, NetworkNeuronType.HH)

    def connect(
        self,
        src: str,
        dst: str,
        pattern,
        weight=0.0,
        delay: float = 0.0,
        synapse: "SynapseSpec | None" = None,
        shift: int = 1,
        probability: float = 0.1,
        allow_self: bool = False,
        seed: int = 0,
        kinetic_spec: "KineticSynapseSpec | None" = None,
    ) -> None:
        """
        Connect two populations.

        Parameters
        ----------
        src : str
            Source population name.
        dst : str
            Destination population name.
        pattern : str, ConnectivityPattern, or callable
            Connectivity pattern. String presets: 'all_to_all', 'one_to_one',
            'shifted', 'random_sparse', 'random_permutation'.
            Or a callable: f(src_size, dst_size) -> list of (i, j) tuples.
        weight : float, tuple, or WeightDistribution
            Synaptic weight. Float for constant, (min, max) for uniform.
        delay : float
            Axonal delay in ms.
        synapse : SynapseSpec, optional
            Synapse type/kinetics. Default: SynapseSpec.ampa().
        shift : int
            Shift offset for SHIFTED pattern.
        probability : float
            Connection probability for RANDOM_SPARSE pattern.
        allow_self : bool
            Allow self-connections within same population.
        seed : int
            RNG seed (0 = random).
        """
        wdist = _resolve_weight(weight)

        if kinetic_spec is not None:
            # Kinetic synapse path — generate pairs and call add_kinetic_connection
            rng = np.random.default_rng(seed if seed != 0 else None)
            src_size = self._rnet.population_size(src)
            dst_size = self._rnet.population_size(dst)
            same_pop = (src == dst)
            if callable(pattern):
                pairs = pattern(src_size, dst_size)
            elif isinstance(pattern, str):
                pattern = _PATTERN_MAP[pattern.upper()]
                pairs = _generate_pairs(pattern, src_size, dst_size,
                                        shift, probability, allow_self, rng,
                                        same_pop=same_pop)
            else:
                pairs = _generate_pairs(pattern, src_size, dst_size,
                                        shift, probability, allow_self, rng,
                                        same_pop=same_pop)
            for (i, j) in pairs:
                if wdist.type == WeightDistType.CONSTANT:
                    w = wdist.param1
                elif wdist.type == WeightDistType.UNIFORM:
                    w = rng.uniform(wdist.param1, wdist.param2)
                else:
                    w = rng.normal(wdist.param1, wdist.param2)
                self._rnet.add_kinetic_connection(src, int(i), dst, int(j),
                                                  float(w), kinetic_spec, delay)
            return

        synapse = synapse or SynapseSpec.ampa()

        if callable(pattern):
            # Custom pattern: call Python function, add connections individually
            src_size = self._rnet.population_size(src)
            dst_size = self._rnet.population_size(dst)
            pairs = pattern(src_size, dst_size)
            rng = np.random.default_rng(seed if seed != 0 else None)
            for (i, j) in pairs:
                if wdist.type == WeightDistType.CONSTANT:
                    w = wdist.param1
                elif wdist.type == WeightDistType.UNIFORM:
                    w = rng.uniform(wdist.param1, wdist.param2)
                else:  # NORMAL
                    w = rng.normal(wdist.param1, wdist.param2)
                self._rnet.add_connection(src, int(i), dst, int(j),
                                          float(w), synapse, delay)
        else:
            # Preset pattern: delegate to C++
            if isinstance(pattern, str):
                pattern = _PATTERN_MAP[pattern.upper()]
            self._rnet.connect(src, dst, pattern, synapse, wdist, delay,
                               shift, probability, allow_self, seed)

    def add_connection(
        self,
        src: str,
        src_local: int,
        dst: str,
        dst_local: int,
        weight: float,
        synapse: "SynapseSpec | None" = None,
        delay: float = 0.0,
    ) -> None:
        """Add a single synapse between two populations using local indices."""
        synapse = synapse or SynapseSpec.ampa()
        self._rnet.add_connection(src, src_local, dst, dst_local, weight,
                                  synapse, delay)

    def add_kinetic_connection(
        self,
        src: str,
        i: int,
        dst: str,
        j: int,
        weight: float,
        spec: "KineticSynapseSpec",
        delay: float = 0.0,
    ) -> None:
        """Add a kinetic synapse between two populations using local indices."""
        self._rnet.add_kinetic_connection(src, i, dst, j, weight, spec, delay)

    def connect_from_matrix(
        self,
        src: str,
        dst: str,
        matrix,
        synapse: "SynapseSpec | None" = None,
        delay: float = 0.0,
        weight_scale: float = 1.0,
    ) -> None:
        """Connect two populations using a weight matrix.

        Each non-zero entry ``matrix[i][j]`` creates a synapse from neuron
        ``i`` in *src* to neuron ``j`` in *dst* with that weight (multiplied
        by *weight_scale*).  Zero entries are skipped.

        Parameters
        ----------
        src : str
            Source population name.
        dst : str
            Destination population name.
        matrix : 2-D array-like
            Weight matrix of shape ``(src_size, dst_size)``.  Accepts numpy
            arrays, nested Python lists, or any object that supports
            ``matrix[i][j]`` indexing.  A zero value means no connection.
        synapse : SynapseSpec, optional
            Synapse type for all created connections.  Defaults to AMPA.
        delay : float
            Synaptic delay in ms applied to all connections.
        weight_scale : float
            Scalar multiplier applied to every weight before adding the
            synapse.  Useful for unit conversions or gain tuning without
            modifying the matrix.
        """
        synapse = synapse or SynapseSpec.ampa()
        src_size = self._rnet.population_size(src)
        dst_size = self._rnet.population_size(dst)

        # Support numpy arrays via the buffer protocol, plain lists, or any
        # other 2-D indexable.  We avoid importing numpy at the top level so
        # the library stays usable without it as a hard dep at import time.
        try:
            import numpy as np
            if isinstance(matrix, np.ndarray):
                if matrix.ndim != 2:
                    raise ValueError(
                        f"connect_from_matrix: matrix must be 2-D, got shape {matrix.shape}"
                    )
                rows, cols = matrix.shape
                if rows != src_size or cols != dst_size:
                    raise ValueError(
                        f"connect_from_matrix: matrix shape ({rows}, {cols}) does not match "
                        f"population sizes src='{src}'({src_size}), dst='{dst}'({dst_size})"
                    )
                # Iterate only non-zero entries for speed
                nz_i, nz_j = matrix.nonzero()
                for i, j in zip(nz_i, nz_j):
                    w = float(matrix[i, j]) * weight_scale
                    self._rnet.add_connection(src, int(i), dst, int(j), w, synapse, delay)
                return
        except ImportError:
            pass

        # Plain Python list path
        rows = len(matrix)
        if rows != src_size:
            raise ValueError(
                f"connect_from_matrix: matrix has {rows} rows but src='{src}' has {src_size} neurons"
            )
        for i, row in enumerate(matrix):
            if len(row) != dst_size:
                raise ValueError(
                    f"connect_from_matrix: row {i} has {len(row)} entries but dst='{dst}' has {dst_size} neurons"
                )
            for j, w in enumerate(row):
                if w:
                    self._rnet.add_connection(src, i, dst, j, float(w) * weight_scale, synapse, delay)

    def randomize_membrane_potentials(
        self, name: str, V_mean: float, V_std: float, seed: int = 0,
        reset_gates: bool = False
    ) -> None:
        """Randomize membrane potentials for a population."""
        self._rnet.randomize_membrane_potentials(name, V_mean, V_std, seed, reset_gates)

    # ---- Population queries ----

    def population_names(self) -> list:
        """Get names of all populations in order."""
        return self._rnet.population_names()

    def population_size(self, name: str) -> int:
        """Get the number of neurons in a population."""
        return self._rnet.population_size(name)

    def population_start(self, name: str) -> int:
        """Get the global start index of a population."""
        return self._rnet.population_start(name)

    @property
    def num_populations(self) -> int:
        return self._rnet.num_populations()

    @property
    def num_neurons(self) -> int:
        return self._rnet.num_neurons

    @property
    def num_synapses(self) -> int:
        return self._rnet.num_synapses

    @property
    def fast_math(self) -> bool:
        return self._rnet.fast_math

    @fast_math.setter
    def fast_math(self, enabled: bool) -> None:
        self._rnet.fast_math = enabled

    def reset(self) -> None:
        """Reset all neurons to resting state."""
        self._rnet.reset()

    def attach_stimulator(self, pop_name: str, stimulator: "DBSStimulator") -> None:
        """
        Attach a DBS stimulator to a population.

        The stimulator current is automatically added to that population's
        I_ext on every call to simulate(). If an I_ext entry for the same
        population is also supplied to simulate(), the two are summed.

        Parameters
        ----------
        pop_name : str
            Name of the target population.
        stimulator : DBSStimulator
            Stimulator to attach.
        """
        if pop_name not in self._rnet.population_names():
            raise ValueError(f"Unknown population: '{pop_name}'")
        self._stimulators[pop_name] = stimulator

    def detach_stimulator(self, pop_name: str) -> None:
        """Remove any attached stimulator from a population."""
        self._stimulators.pop(pop_name, None)

    @property
    def network(self) -> _Network:
        """Access the underlying Network object."""
        return self._rnet.network()

    def simulate(
        self,
        duration: float,
        dt: float,
        I_ext: "dict | ArrayLike" = None,
        record: "RecordingConfig | None" = None,
    ) -> "dict[str, NDArray[np.float64]] | PopulationMetricsResult":
        """
        Run a network simulation.

        Parameters
        ----------
        duration : float
            Simulation duration in milliseconds.
        dt : float
            Time step in milliseconds.
        I_ext : dict or array-like, optional
            External currents. Can be:
            - dict {pop_name: value} where value is:
              - scalar float: constant current for all neurons/timesteps
              - 1D array (num_steps,): broadcast to all neurons in population
              - 2D array (pop_size, num_steps): per-neuron current
            - Missing populations get zero current
            - Or a flat 2D array (num_neurons, num_steps) for raw access
        record : RecordingConfig, optional
            Recording configuration. If None (default), returns a dict of
            voltage traces keyed by population (backward compatible).

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Voltage traces by population when record=None.
        PopulationMetricsResult
            When record is a RecordingConfig.
        """
        from .recording import _run_recording, RecordingConfig as _RC, PopulationMetricsResult as _PMR

        num_neurons = self._rnet.num_neurons
        num_steps = int(duration / dt)

        if I_ext is None:
            I_ext = {}

        if isinstance(I_ext, dict):
            flat = np.zeros((num_neurons, num_steps), dtype=np.float64)
            pop_names = self._rnet.population_names()
            for name in pop_names:
                if name not in I_ext:
                    continue
                val = I_ext[name]
                start = self._rnet.population_start(name)
                size = self._rnet.population_size(name)
                val = np.asarray(val, dtype=np.float64)
                if val.ndim == 0:
                    flat[start:start + size, :] = val.item()
                elif val.ndim == 1:
                    flat[start:start + size, :] = val[np.newaxis, :num_steps]
                elif val.ndim == 2:
                    flat[start:start + size, :] = val[:, :num_steps]
                else:
                    raise ValueError(
                        f"I_ext['{name}'] must be scalar, 1D, or 2D, "
                        f"got {val.ndim}D"
                    )
            for pop_name, stim in self._stimulators.items():
                stim_trace = stim.generate(duration, dt)
                start = self._rnet.population_start(pop_name)
                size  = self._rnet.population_size(pop_name)
                flat[start:start + size, :] += np.array(
                    stim_trace[:num_steps], dtype=np.float64
                )[np.newaxis, :]
            I_ext_list = flat.tolist()
        else:
            I_ext_arr = np.asarray(I_ext, dtype=np.float64)
            I_ext_list = I_ext_arr.tolist()

        pop_info = {
            name: (self._rnet.population_start(name),
                   self._rnet.population_size(name))
            for name in self._rnet.population_names()
        }
        cfg = record if record is not None else _RC(["V"])
        result = _run_recording(
            self._rnet.network(), duration, dt, I_ext_list, cfg, pop_info)

        if record is None:
            # Backward compat: return {pop_name: ndarray}
            V = result["V"]  # shape (n_neurons, n_steps) since neurons="all"
            return {name: V[start:start + size, :]
                    for name, (start, size) in pop_info.items()}

        return _PMR(result, pop_info, record)

    def __len__(self) -> int:
        return self.num_neurons

    def __repr__(self) -> str:
        return (
            f"<RegionalNetwork populations={self.num_populations} "
            f"neurons={self.num_neurons} synapses={self.num_synapses}>"
        )


# =============================================================================
# Python helper classes for composable neuron model building
# =============================================================================

class Boltzmann:
    """Helper to create BoltzmannParams."""
    def __init__(self, v_half: float, k: float):
        self.v_half = v_half
        self.k = k

    def to_params(self) -> BoltzmannParams:
        p = BoltzmannParams()
        p.v_half = self.v_half
        p.k = self.k
        return p


class Tau:
    """Helper to create TauParams."""

    @staticmethod
    def constant(value: float) -> TauParams:
        t = TauParams()
        t.form = TauForm.CONSTANT
        t.set_param(0, value)
        return t

    @staticmethod
    def boltzmann(base: float, amp: float, v_half: float, k: float) -> TauParams:
        t = TauParams()
        t.form = TauForm.BOLTZMANN
        t.set_param(0, base)
        t.set_param(1, amp)
        t.set_param(2, v_half)
        t.set_param(3, k)
        return t

    @staticmethod
    def scaled_exp(scale: float, v_half: float, k: float) -> TauParams:
        t = TauParams()
        t.form = TauForm.SCALED_EXP
        t.set_param(0, scale)
        t.set_param(1, v_half)
        t.set_param(2, k)
        return t

    @staticmethod
    def double_exp_sum(base, amp, v1, s1, v2, s2) -> TauParams:
        t = TauParams()
        t.form = TauForm.DOUBLE_EXP_SUM
        t.set_param(0, base)
        t.set_param(1, amp)
        t.set_param(2, v1)
        t.set_param(3, s1)
        t.set_param(5, v2)
        t.set_param(6, s2)
        return t


class RateFunc:
    """Helper to create RateFuncParams."""

    @staticmethod
    def linear_over_exp(A: float, B: float, C: float) -> RateFuncParams:
        r = RateFuncParams()
        r.form = RateFuncForm.LINEAR_OVER_EXP
        r.A = A
        r.B = B
        r.C = C
        return r

    @staticmethod
    def exp_decay(A: float, B: float, C: float) -> RateFuncParams:
        r = RateFuncParams()
        r.form = RateFuncForm.EXP_DECAY
        r.A = A
        r.B = B
        r.C = C
        return r

    @staticmethod
    def linear_over_expm1(A: float, B: float, C: float) -> RateFuncParams:
        r = RateFuncParams()
        r.form = RateFuncForm.LINEAR_OVER_EXPM1
        r.A = A
        r.B = B
        r.C = C
        return r

    @staticmethod
    def sigmoid(A: float, B: float, C: float) -> RateFuncParams:
        r = RateFuncParams()
        r.form = RateFuncForm.SIGMOID
        r.A = A
        r.B = B
        r.C = C
        return r


class NeuronModel:
    """
    Ergonomic builder for NeuronModelSpec.

    Examples
    --------
    >>> model = NeuronModel("custom", C_m=1.0, V_init=-65.0)
    >>> model.add_gate("m", update_form="instant", inf=Boltzmann(-37, 7))
    >>> model.add_channel("Leak", g=0.3, E_rev=-54.3)
    >>> spec = model.to_spec()
    """

    def __init__(self, name: str = "custom", C_m: float = 1.0, V_init: float = -65.0):
        self._spec = NeuronModelSpec()
        self._spec.name = name
        self._spec.C_m = C_m
        self._spec.V_init = V_init

    def add_gate(
        self,
        name: str,
        update_form: str = "inf_tau",
        dependency: str = "voltage",
        scale: float = 1.0,
        initial_value: float = 0.0,
        inf: "Boltzmann | BoltzmannParams | None" = None,
        tau: "TauParams | None" = None,
        alpha: "RateFuncParams | None" = None,
        beta: "RateFuncParams | None" = None,
        derived_source_gate: int = -1,
        derived_a: float = 1.0,
        derived_b: float = 0.0,
        derived_c: float = 1.0,
    ) -> int:
        """Add a gate and return its index."""
        g = GateSpec()
        g.name = name
        g.update_form = {
            "inf_tau": GateUpdateForm.INF_TAU,
            "alpha_beta": GateUpdateForm.ALPHA_BETA,
            "instant": GateUpdateForm.INSTANT,
            "derived": GateUpdateForm.DERIVED,
        }[update_form.lower()]
        g.dependency = (GateDependency.CALCIUM if dependency.lower() == "calcium"
                        else GateDependency.VOLTAGE)
        g.scale = scale
        g.initial_value = initial_value
        if inf is not None:
            g.inf = inf.to_params() if isinstance(inf, Boltzmann) else inf
        if tau is not None:
            g.tau = tau
        if alpha is not None:
            g.alpha = alpha
        if beta is not None:
            g.beta = beta
        g.derived_source_gate = derived_source_gate
        g.derived_a = derived_a
        g.derived_b = derived_b
        g.derived_c = derived_c
        self._spec.gates.append(g)
        return len(self._spec.gates) - 1

    def add_channel(
        self,
        name: str,
        g: float,
        E_rev: float,
        gates: "list[tuple[int, int]] | None" = None,
        use_calcium_nernst: bool = False,
        is_ahp: bool = False,
        ahp_k1: float = 0.0,
    ) -> int:
        """Add a channel and return its index."""
        ch = ChannelSpec()
        ch.name = name
        ch.g = g
        ch.E_rev = E_rev
        ch.gates = gates or []
        ch.use_calcium_nernst = use_calcium_nernst
        ch.is_ahp = is_ahp
        ch.ahp_k1 = ahp_k1
        self._spec.channels.append(ch)
        return len(self._spec.channels) - 1

    def add_leak(self, g: float, E_rev: float) -> int:
        """Add a leak channel (no gates)."""
        return self.add_channel("Leak", g, E_rev)

    def set_calcium(
        self,
        epsilon: float = 1e-4,
        K_Ca: float = 15.0,
        Ca_init: float = 0.1,
        use_nernst: bool = False,
        Ca_o: float = 2000.0,
        source_channels: "list[int] | None" = None,
    ) -> None:
        """Configure calcium dynamics."""
        ca = self._spec.calcium
        ca.enabled = True
        ca.epsilon = epsilon
        ca.K_Ca = K_Ca
        ca.Ca_init = Ca_init
        ca.use_nernst = use_nernst
        ca.Ca_o = Ca_o
        ca.source_channels = source_channels or []

    def to_spec(self) -> NeuronModelSpec:
        """Build and return the NeuronModelSpec."""
        return self._spec

    @staticmethod
    def thalamic() -> "NeuronModel":
        m = NeuronModel.__new__(NeuronModel)
        m._spec = NeuronModelSpec.thalamic()
        return m

    @staticmethod
    def stn() -> "NeuronModel":
        m = NeuronModel.__new__(NeuronModel)
        m._spec = NeuronModelSpec.stn()
        return m

    @staticmethod
    def gpe() -> "NeuronModel":
        m = NeuronModel.__new__(NeuronModel)
        m._spec = NeuronModelSpec.gpe()
        return m

    @staticmethod
    def gpi() -> "NeuronModel":
        m = NeuronModel.__new__(NeuronModel)
        m._spec = NeuronModelSpec.gpi()
        return m

    @staticmethod
    def striatum(pd: float = 0.0) -> "NeuronModel":
        m = NeuronModel.__new__(NeuronModel)
        m._spec = NeuronModelSpec.striatum(pd)
        return m

    def __repr__(self) -> str:
        return (
            f"<NeuronModel '{self._spec.name}' "
            f"gates={len(self._spec.gates)} "
            f"channels={len(self._spec.channels)}>"
        )


class KineticSynapseModel:
    """
    Fluent builder for KineticSynapseSpec.

    Examples
    --------
    >>> spec = (KineticSynapseModel("GABA_kin")
    ...         .tanh_gate(amp=2.0, v_half=0.0, k=4.0, tau_decay=13.0)
    ...         .linear_current(g=0.1, E_syn=-80.0)
    ...         .to_spec())
    """

    def __init__(self, name: str):
        self._spec = KineticSynapseSpec()
        self._spec.name = name

    def tanh_gate(
        self,
        amp: float = 2.0,
        v_half: float = 0.0,
        k: float = 4.0,
        tau_decay: float = 13.0,
    ) -> "KineticSynapseModel":
        self._spec.update_form = KineticUpdateForm.TANH_GATE
        self._spec.tanh_amp = amp
        self._spec.tanh_vh = v_half
        self._spec.tanh_k = k
        self._spec.tau_decay = tau_decay
        return self

    def boltzmann_gate(
        self,
        v_half: float,
        k: float,
        tau: float,
    ) -> "KineticSynapseModel":
        self._spec.update_form = KineticUpdateForm.BOLTZMANN_GATE
        s_inf = BoltzmannParams()
        s_inf.v_half = v_half
        s_inf.k = k
        self._spec.s_inf = s_inf
        t = TauParams()
        t.form = TauForm.CONSTANT
        t.set_param(0, tau)
        self._spec.tau = t
        return self

    def alpha_beta(
        self,
        alpha: "RateFuncParams",
        beta: "RateFuncParams",
    ) -> "KineticSynapseModel":
        self._spec.update_form = KineticUpdateForm.ALPHA_BETA
        self._spec.alpha = alpha
        self._spec.beta = beta
        return self

    def linear_current(
        self,
        g: float,
        E_syn: float,
        power: int = 1,
    ) -> "KineticSynapseModel":
        self._spec.current_form = KineticCurrentForm.LINEAR
        self._spec.g = g
        self._spec.E_syn = E_syn
        self._spec.power = power
        return self

    def mg_block_current(
        self,
        g: float,
        E_syn: float,
        mg_conc: float = 1.0,
        mg_scale: float = 0.062,
        mg_denom: float = 3.57,
    ) -> "KineticSynapseModel":
        self._spec.current_form = KineticCurrentForm.MG_BLOCK
        self._spec.g = g
        self._spec.E_syn = E_syn
        self._spec.mg_conc = mg_conc
        self._spec.mg_scale = mg_scale
        self._spec.mg_denom = mg_denom
        return self

    def to_spec(self) -> KineticSynapseSpec:
        """Return the built KineticSynapseSpec."""
        return self._spec

    def __repr__(self) -> str:
        return f"<KineticSynapseModel '{self._spec.name}'>"


# =============================================================================
# DBSStimulator — Deep Brain Stimulation pulse-train generator
# =============================================================================

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

    def __init__(
        self,
        frequency: float = 130.0,
        amplitude: float = 0.0,
        pulse_width: float = 0.06,
    ):
        p = DBSParameters()
        p.frequency   = frequency
        p.amplitude   = amplitude
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
        p.frequency   = frequency   if frequency   is not None else self.frequency
        p.amplitude   = amplitude   if amplitude   is not None else self.amplitude
        p.pulse_width = pulse_width if pulse_width is not None else self.pulse_width
        self._dbs.set_parameters(p)

    def __repr__(self) -> str:
        return (
            f"<DBSStimulator freq={self.frequency}Hz "
            f"amp={self.amplitude} PW={self.pulse_width}ms>"
        )
