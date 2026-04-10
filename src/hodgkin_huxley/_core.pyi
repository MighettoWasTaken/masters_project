"""
Neural simulation library - C++ backend
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['ABS', 'ADD', 'ALL_TO_ALL', 'ALPHA', 'ALPHA_BETA', 'AMPA', 'AlphaSynapse', 'BOLTZMANN', 'BOLTZMANN_GATE', 'BoltzmannParams', 'CALCIUM', 'CHATTERING', 'COMPOSABLE', 'COMPOUND_AB', 'CONSTANT', 'COS', 'CUSTOM', 'CUSTOM_EXPR', 'CalciumSpec', 'ChannelSpec', 'ChannelSpecVector', 'ComposableNeuron', 'ConnectivityPattern', 'DBSParameters', 'DBSStimulator', 'DERIVED', 'DOUBLE_EXPONENTIAL', 'DOUBLE_EXP_SUM', 'DoubleExponentialSynapse', 'EULER', 'EXP', 'EXPONENTIAL', 'EXP_DECAY', 'ExponentialSynapse', 'FAST_SPIKING', 'GABA_A', 'GateDependency', 'GateSpec', 'GateSpecVector', 'GateUpdateForm', 'HH', 'HHNeuron', 'HHParameters', 'HHState', 'INF_TAU', 'INSTANT', 'INTRINSICALLY_BURSTING', 'IZHIKEVICH_CH', 'IZHIKEVICH_CUSTOM', 'IZHIKEVICH_FS', 'IZHIKEVICH_IB', 'IZHIKEVICH_LTS', 'IZHIKEVICH_RS', 'IntegrationMethod', 'IzhikevichNeuron', 'IzhikevichParameters', 'IzhikevichState', 'IzhikevichType', 'KineticCurrentForm', 'KineticSynapseSpec', 'KineticUpdateForm', 'LINEAR', 'LINEAR_OVER_EXP', 'LINEAR_OVER_EXPM1', 'LOG', 'LOW_THRESHOLD_SPIKING', 'MG_BLOCK', 'MUL', 'NEG', 'NMDA', 'NORMAL', 'NeuronBase', 'NeuronModelSpec', 'OFFSET_DOUBLE_EXP', 'ONE_TO_ONE', 'POW_GEN', 'POW_HALF', 'POW_INT', 'PUSH_CONST', 'PUSH_DEP', 'PUSH_GATE', 'PUSH_S', 'Parameters', 'RANDOM_PERMUTATION', 'RANDOM_SPARSE', 'RCP', 'REGULAR_SPIKING', 'RK4', 'RK45_ADAPTIVE', 'RateFuncForm', 'RateFuncParams', 'ReceptorType', 'RegionalNetwork', 'SCALED_EXP', 'SHIFTED', 'SIGMOID', 'SIN', 'SQRT', 'State', 'SynapseBase', 'SynapseSpec', 'SynapseSpecType', 'TANH', 'TANH_GATE', 'TauForm', 'TauParams', 'UNIFORM', 'VOLTAGE', 'VmExpr', 'VmInstruction', 'VmInstructionVector', 'VmOp', 'WeightDistType', 'WeightDistribution']
class AlphaSynapse(SynapseBase):
    @property
    def tau(self) -> float:
        """
        Time to peak (ms)
        """
class BoltzmannParams:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def k(self) -> float:
        ...
    @k.setter
    def k(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def v_half(self) -> float:
        ...
    @v_half.setter
    def v_half(self, arg0: typing.SupportsFloat) -> None:
        ...
class CalciumSpec:
    enabled: bool
    use_nernst: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def Ca_init(self) -> float:
        ...
    @Ca_init.setter
    def Ca_init(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def Ca_o(self) -> float:
        ...
    @Ca_o.setter
    def Ca_o(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def F(self) -> float:
        ...
    @F.setter
    def F(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def K_Ca(self) -> float:
        ...
    @K_Ca.setter
    def K_Ca(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def R(self) -> float:
        ...
    @R.setter
    def R(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def T(self) -> float:
        ...
    @T.setter
    def T(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def source_channels(self) -> list[int]:
        ...
    @source_channels.setter
    def source_channels(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat) -> None:
        ...
class ChannelSpec:
    gate_product_vm: VmExpr
    is_ahp: bool
    name: str
    use_calcium_nernst: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def E_rev(self) -> float:
        ...
    @E_rev.setter
    def E_rev(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def ahp_k1(self) -> float:
        ...
    @ahp_k1.setter
    def ahp_k1(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def g(self) -> float:
        ...
    @g.setter
    def g(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def gates(self) -> list[tuple[int, int]]:
        ...
    @gates.setter
    def gates(self, arg0: collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsInt]]) -> None:
        ...
class ChannelSpecVector:
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> ChannelSpecVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> ...:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ChannelSpecVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[typing.Any]:
        ...
    def __len__(self) -> int:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.Any) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: ChannelSpecVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: ...) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: ChannelSpecVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: ...) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> ...:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> ...:
        """
        Remove and return the item at index ``i``
        """
class ComposableNeuron(NeuronBase):
    def __init__(self, spec: NeuronModelSpec) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def calcium(self) -> float:
        ...
    @property
    def gate_states(self) -> list[float]:
        ...
    @property
    def model_spec(self) -> NeuronModelSpec:
        ...
class ConnectivityPattern:
    """
    Members:
    
      ALL_TO_ALL
    
      ONE_TO_ONE
    
      SHIFTED
    
      RANDOM_SPARSE
    
      RANDOM_PERMUTATION
    """
    ALL_TO_ALL: typing.ClassVar[ConnectivityPattern]  # value = <ConnectivityPattern.ALL_TO_ALL: 0>
    ONE_TO_ONE: typing.ClassVar[ConnectivityPattern]  # value = <ConnectivityPattern.ONE_TO_ONE: 1>
    RANDOM_PERMUTATION: typing.ClassVar[ConnectivityPattern]  # value = <ConnectivityPattern.RANDOM_PERMUTATION: 4>
    RANDOM_SPARSE: typing.ClassVar[ConnectivityPattern]  # value = <ConnectivityPattern.RANDOM_SPARSE: 3>
    SHIFTED: typing.ClassVar[ConnectivityPattern]  # value = <ConnectivityPattern.SHIFTED: 2>
    __members__: typing.ClassVar[dict[str, ConnectivityPattern]]  # value = {'ALL_TO_ALL': <ConnectivityPattern.ALL_TO_ALL: 0>, 'ONE_TO_ONE': <ConnectivityPattern.ONE_TO_ONE: 1>, 'SHIFTED': <ConnectivityPattern.SHIFTED: 2>, 'RANDOM_SPARSE': <ConnectivityPattern.RANDOM_SPARSE: 3>, 'RANDOM_PERMUTATION': <ConnectivityPattern.RANDOM_PERMUTATION: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DBSParameters:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def amplitude(self) -> float:
        """
        Pulse amplitude in uA/cm^2
        """
    @amplitude.setter
    def amplitude(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frequency(self) -> float:
        """
        Stimulation frequency in Hz (0 = off)
        """
    @frequency.setter
    def frequency(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def pulse_width(self) -> float:
        """
        Pulse width in ms
        """
    @pulse_width.setter
    def pulse_width(self, arg0: typing.SupportsFloat) -> None:
        ...
class DBSStimulator:
    @typing.overload
    def __init__(self) -> None:
        """
        Create a DBS stimulator with default parameters
        """
    @typing.overload
    def __init__(self, params: DBSParameters) -> None:
        """
        Create a DBS stimulator with given parameters
        """
    def __repr__(self) -> str:
        ...
    def current_at(self, step_index: typing.SupportsInt, dt: typing.SupportsFloat) -> float:
        """
        Get current value at a specific step index
        """
    def generate(self, duration: typing.SupportsFloat, dt: typing.SupportsFloat) -> list[float]:
        """
        Generate full current trace (length ≈ duration/dt + 1)
        """
    def set_parameters(self, params: DBSParameters) -> None:
        """
        Update stimulator parameters (validates on assignment)
        """
    @property
    def parameters(self) -> DBSParameters:
        """
        Current stimulator parameters
        """
class DoubleExponentialSynapse(SynapseBase):
    @property
    def tau_decay(self) -> float:
        """
        Decay time constant (ms)
        """
    @property
    def tau_rise(self) -> float:
        """
        Rise time constant (ms)
        """
class ExponentialSynapse(SynapseBase):
    @property
    def tau(self) -> float:
        """
        Decay time constant (ms)
        """
class GateDependency:
    """
    Members:
    
      VOLTAGE
    
      CALCIUM
    """
    CALCIUM: typing.ClassVar[GateDependency]  # value = <GateDependency.CALCIUM: 1>
    VOLTAGE: typing.ClassVar[GateDependency]  # value = <GateDependency.VOLTAGE: 0>
    __members__: typing.ClassVar[dict[str, GateDependency]]  # value = {'VOLTAGE': <GateDependency.VOLTAGE: 0>, 'CALCIUM': <GateDependency.CALCIUM: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GateSpec:
    alpha: RateFuncParams
    alpha_vm: VmExpr
    beta: RateFuncParams
    beta_vm: VmExpr
    alpha_vm: VmExpr
    beta_vm: VmExpr
    dependency: GateDependency
    dxdt_vm: VmExpr
    inf: BoltzmannParams
    inf_vm: VmExpr
    name: str
    tau: TauParams
    tau_vm: VmExpr
    update_form: GateUpdateForm
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def derived_a(self) -> float:
        ...
    @derived_a.setter
    def derived_a(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def derived_b(self) -> float:
        ...
    @derived_b.setter
    def derived_b(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def derived_c(self) -> float:
        ...
    @derived_c.setter
    def derived_c(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def derived_source_gate(self) -> int:
        ...
    @derived_source_gate.setter
    def derived_source_gate(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def initial_value(self) -> float:
        ...
    @initial_value.setter
    def initial_value(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def scale(self) -> float:
        ...
    @scale.setter
    def scale(self, arg0: typing.SupportsFloat) -> None:
        ...
class GateSpecVector:
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> GateSpecVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> ...:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: GateSpecVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[typing.Any]:
        ...
    def __len__(self) -> int:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.Any) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: GateSpecVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: ...) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: GateSpecVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: ...) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> ...:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> ...:
        """
        Remove and return the item at index ``i``
        """
class GateUpdateForm:
    """
    Members:
    
      INF_TAU
    
      ALPHA_BETA
    
      INSTANT
    
      DERIVED
    
      CUSTOM_EXPR
    """
    ALPHA_BETA: typing.ClassVar[GateUpdateForm]  # value = <GateUpdateForm.ALPHA_BETA: 1>
    CUSTOM_EXPR: typing.ClassVar[GateUpdateForm]  # value = <GateUpdateForm.CUSTOM_EXPR: 4>
    DERIVED: typing.ClassVar[GateUpdateForm]  # value = <GateUpdateForm.DERIVED: 3>
    INF_TAU: typing.ClassVar[GateUpdateForm]  # value = <GateUpdateForm.INF_TAU: 0>
    INSTANT: typing.ClassVar[GateUpdateForm]  # value = <GateUpdateForm.INSTANT: 2>
    __members__: typing.ClassVar[dict[str, GateUpdateForm]]  # value = {'INF_TAU': <GateUpdateForm.INF_TAU: 0>, 'ALPHA_BETA': <GateUpdateForm.ALPHA_BETA: 1>, 'INSTANT': <GateUpdateForm.INSTANT: 2>, 'DERIVED': <GateUpdateForm.DERIVED: 3>, 'CUSTOM_EXPR': <GateUpdateForm.CUSTOM_EXPR: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class HHNeuron(NeuronBase):
    @typing.overload
    def __init__(self) -> None:
        """
        Create a Hodgkin-Huxley neuron with default parameters
        """
    @typing.overload
    def __init__(self, parameters: HHParameters) -> None:
        """
        Create a neuron with custom parameters
        """
    @typing.overload
    def __init__(self, parameters: HHParameters, method: IntegrationMethod) -> None:
        """
        Create a neuron with custom parameters and integration method
        """
    def __repr__(self) -> str:
        ...
    def set_parameters(self, parameters: HHParameters) -> None:
        """
        Set the neuron parameters
        """
    def set_state(self, state: HHState) -> None:
        """
        Set the neuron state
        """
    @property
    def parameters(self) -> HHParameters:
        """
        Neuron parameters
        """
    @property
    def state(self) -> HHState:
        """
        Current state of the neuron
        """
class HHParameters:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def C_m(self) -> float:
        """
        Membrane capacitance (uF/cm^2)
        """
    @C_m.setter
    def C_m(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def E_K(self) -> float:
        """
        Potassium reversal potential (mV)
        """
    @E_K.setter
    def E_K(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def E_L(self) -> float:
        """
        Leak reversal potential (mV)
        """
    @E_L.setter
    def E_L(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def E_Na(self) -> float:
        """
        Sodium reversal potential (mV)
        """
    @E_Na.setter
    def E_Na(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def g_K(self) -> float:
        """
        Potassium conductance (mS/cm^2)
        """
    @g_K.setter
    def g_K(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def g_L(self) -> float:
        """
        Leak conductance (mS/cm^2)
        """
    @g_L.setter
    def g_L(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def g_Na(self) -> float:
        """
        Sodium conductance (mS/cm^2)
        """
    @g_Na.setter
    def g_Na(self, arg0: typing.SupportsFloat) -> None:
        ...
class HHState:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def V(self) -> float:
        """
        Membrane potential (mV)
        """
    @V.setter
    def V(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def h(self) -> float:
        """
        Na+ inactivation gate
        """
    @h.setter
    def h(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def m(self) -> float:
        """
        Na+ activation gate
        """
    @m.setter
    def m(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def n(self) -> float:
        """
        K+ activation gate
        """
    @n.setter
    def n(self, arg0: typing.SupportsFloat) -> None:
        ...
class IntegrationMethod:
    """
    Members:
    
      EULER
    
      RK4
    
      RK45_ADAPTIVE
    """
    EULER: typing.ClassVar[IntegrationMethod]  # value = <IntegrationMethod.EULER: 0>
    RK4: typing.ClassVar[IntegrationMethod]  # value = <IntegrationMethod.RK4: 1>
    RK45_ADAPTIVE: typing.ClassVar[IntegrationMethod]  # value = <IntegrationMethod.RK45_ADAPTIVE: 2>
    __members__: typing.ClassVar[dict[str, IntegrationMethod]]  # value = {'EULER': <IntegrationMethod.EULER: 0>, 'RK4': <IntegrationMethod.RK4: 1>, 'RK45_ADAPTIVE': <IntegrationMethod.RK45_ADAPTIVE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class IzhikevichNeuron(NeuronBase):
    @staticmethod
    def get_preset(type: IzhikevichType) -> IzhikevichParameters:
        """
        Get parameters for a preset neuron type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Create an Izhikevich neuron with default (Regular Spiking) parameters
        """
    @typing.overload
    def __init__(self, type: IzhikevichType) -> None:
        """
        Create a neuron with preset type
        """
    @typing.overload
    def __init__(self, parameters: IzhikevichParameters) -> None:
        """
        Create a neuron with custom parameters
        """
    def __repr__(self) -> str:
        ...
    def set_parameters(self, parameters: IzhikevichParameters) -> None:
        """
        Set the neuron parameters
        """
    def set_state(self, state: IzhikevichState) -> None:
        """
        Set the neuron state
        """
    @property
    def parameters(self) -> IzhikevichParameters:
        """
        Neuron parameters
        """
    @property
    def spiked(self) -> bool:
        """
        True if neuron spiked in last step
        """
    @property
    def state(self) -> IzhikevichState:
        """
        Current state of the neuron
        """
    @property
    def u(self) -> float:
        """
        Recovery variable u
        """
class IzhikevichParameters:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def a(self) -> float:
        """
        Time scale of recovery variable
        """
    @a.setter
    def a(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def b(self) -> float:
        """
        Sensitivity of u to subthreshold v
        """
    @b.setter
    def b(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def c(self) -> float:
        """
        After-spike reset value of v (mV)
        """
    @c.setter
    def c(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def d(self) -> float:
        """
        After-spike reset increment of u
        """
    @d.setter
    def d(self, arg0: typing.SupportsFloat) -> None:
        ...
class IzhikevichState:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def u(self) -> float:
        """
        Recovery variable
        """
    @u.setter
    def u(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def v(self) -> float:
        """
        Membrane potential (mV)
        """
    @v.setter
    def v(self, arg0: typing.SupportsFloat) -> None:
        ...
class IzhikevichType:
    """
    Members:
    
      REGULAR_SPIKING
    
      FAST_SPIKING
    
      INTRINSICALLY_BURSTING
    
      CHATTERING
    
      LOW_THRESHOLD_SPIKING
    
      CUSTOM
    """
    CHATTERING: typing.ClassVar[IzhikevichType]  # value = <IzhikevichType.CHATTERING: 3>
    CUSTOM: typing.ClassVar[IzhikevichType]  # value = <IzhikevichType.CUSTOM: 5>
    FAST_SPIKING: typing.ClassVar[IzhikevichType]  # value = <IzhikevichType.FAST_SPIKING: 1>
    INTRINSICALLY_BURSTING: typing.ClassVar[IzhikevichType]  # value = <IzhikevichType.INTRINSICALLY_BURSTING: 2>
    LOW_THRESHOLD_SPIKING: typing.ClassVar[IzhikevichType]  # value = <IzhikevichType.LOW_THRESHOLD_SPIKING: 4>
    REGULAR_SPIKING: typing.ClassVar[IzhikevichType]  # value = <IzhikevichType.REGULAR_SPIKING: 0>
    __members__: typing.ClassVar[dict[str, IzhikevichType]]  # value = {'REGULAR_SPIKING': <IzhikevichType.REGULAR_SPIKING: 0>, 'FAST_SPIKING': <IzhikevichType.FAST_SPIKING: 1>, 'INTRINSICALLY_BURSTING': <IzhikevichType.INTRINSICALLY_BURSTING: 2>, 'CHATTERING': <IzhikevichType.CHATTERING: 3>, 'LOW_THRESHOLD_SPIKING': <IzhikevichType.LOW_THRESHOLD_SPIKING: 4>, 'CUSTOM': <IzhikevichType.CUSTOM: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KineticCurrentForm:
    """
    Members:
    
      LINEAR
    
      MG_BLOCK
    
      CUSTOM_EXPR
    """
    CUSTOM_EXPR: typing.ClassVar[KineticCurrentForm]  # value = <KineticCurrentForm.CUSTOM_EXPR: 2>
    LINEAR: typing.ClassVar[KineticCurrentForm]  # value = <KineticCurrentForm.LINEAR: 0>
    MG_BLOCK: typing.ClassVar[KineticCurrentForm]  # value = <KineticCurrentForm.MG_BLOCK: 1>
    __members__: typing.ClassVar[dict[str, KineticCurrentForm]]  # value = {'LINEAR': <KineticCurrentForm.LINEAR: 0>, 'MG_BLOCK': <KineticCurrentForm.MG_BLOCK: 1>, 'CUSTOM_EXPR': <KineticCurrentForm.CUSTOM_EXPR: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KineticSynapseSpec:
    alpha: RateFuncParams
    beta: RateFuncParams
    current_form: KineticCurrentForm
    name: str
    s_inf: BoltzmannParams
    tau: TauParams
    update_form: KineticUpdateForm
    @staticmethod
    def gaba_b() -> KineticSynapseSpec:
        ...
    @staticmethod
    def gaba_kinetic() -> KineticSynapseSpec:
        ...
    @staticmethod
    def nmda_kinetic() -> KineticSynapseSpec:
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def E_syn(self) -> float:
        ...
    @E_syn.setter
    def E_syn(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def S_init(self) -> float:
        ...
    @S_init.setter
    def S_init(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def current_fn_ptr(self) -> int:
        ...
    @current_fn_ptr.setter
    def current_fn_ptr(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def dS_dt_fn_ptr(self) -> int:
        ...
    @dS_dt_fn_ptr.setter
    def dS_dt_fn_ptr(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def g(self) -> float:
        ...
    @g.setter
    def g(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def mg_conc(self) -> float:
        ...
    @mg_conc.setter
    def mg_conc(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def mg_denom(self) -> float:
        ...
    @mg_denom.setter
    def mg_denom(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def mg_scale(self) -> float:
        ...
    @mg_scale.setter
    def mg_scale(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def power(self) -> int:
        ...
    @power.setter
    def power(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def tanh_amp(self) -> float:
        ...
    @tanh_amp.setter
    def tanh_amp(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def tanh_k(self) -> float:
        ...
    @tanh_k.setter
    def tanh_k(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def tanh_vh(self) -> float:
        ...
    @tanh_vh.setter
    def tanh_vh(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def tau_decay(self) -> float:
        ...
    @tau_decay.setter
    def tau_decay(self, arg0: typing.SupportsFloat) -> None:
        ...
class KineticUpdateForm:
    """
    Members:
    
      ALPHA_BETA
    
      TANH_GATE
    
      BOLTZMANN_GATE
    
      CUSTOM_EXPR
    """
    ALPHA_BETA: typing.ClassVar[KineticUpdateForm]  # value = <KineticUpdateForm.ALPHA_BETA: 0>
    BOLTZMANN_GATE: typing.ClassVar[KineticUpdateForm]  # value = <KineticUpdateForm.BOLTZMANN_GATE: 2>
    CUSTOM_EXPR: typing.ClassVar[KineticUpdateForm]  # value = <KineticUpdateForm.CUSTOM_EXPR: 3>
    TANH_GATE: typing.ClassVar[KineticUpdateForm]  # value = <KineticUpdateForm.TANH_GATE: 1>
    __members__: typing.ClassVar[dict[str, KineticUpdateForm]]  # value = {'ALPHA_BETA': <KineticUpdateForm.ALPHA_BETA: 0>, 'TANH_GATE': <KineticUpdateForm.TANH_GATE: 1>, 'BOLTZMANN_GATE': <KineticUpdateForm.BOLTZMANN_GATE: 2>, 'CUSTOM_EXPR': <KineticUpdateForm.CUSTOM_EXPR: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NeuronBase:
    def __repr__(self) -> str:
        ...
    def reset(self) -> None:
        """
        Reset to resting state
        """
    @typing.overload
    def simulate(self, duration: typing.SupportsFloat, dt: typing.SupportsFloat, I_ext: typing.SupportsFloat) -> list[float]:
        """
        Run simulation with constant current
        """
    @typing.overload
    def simulate(self, duration: typing.SupportsFloat, dt: typing.SupportsFloat, I_ext: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
        """
        Run simulation with time-varying current
        """
    def step(self, dt: typing.SupportsFloat, I_ext: typing.SupportsFloat) -> None:
        """
        Advance simulation by dt milliseconds
        """
    def type_name(self) -> str:
        """
        Get neuron type name
        """
    @property
    def V(self) -> float:
        """
        Membrane potential (mV)
        """
    @V.setter
    def V(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def integration_method(self) -> IntegrationMethod:
        """
        Integration method
        """
    @integration_method.setter
    def integration_method(self, arg1: IntegrationMethod) -> None:
        ...
class NeuronModelSpec:
    calcium: CalciumSpec
    channels: ChannelSpecVector
    gates: GateSpecVector
    name: str
    @staticmethod
    def hh_default() -> NeuronModelSpec:
        """
        Classic Hodgkin-Huxley squid axon model (Na, K, Leak channels)
        """
    @staticmethod
    @typing.overload
    def izhikevich(type: IzhikevichType = ...) -> NeuronModelSpec:
        """
        Izhikevich spiking neuron with preset type
        """
    @staticmethod
    @typing.overload
    def izhikevich(parameters: IzhikevichParameters) -> NeuronModelSpec:
        """
        Izhikevich spiking neuron with custom parameters
        """
    def __copy__(self) -> NeuronModelSpec:
        ...
    def __deepcopy__(self, arg0: dict) -> NeuronModelSpec:
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def validate(self) -> None:
        """
        Validate spec — raises ValueError on structural errors
        """
    @property
    def C_m(self) -> float:
        ...
    @C_m.setter
    def C_m(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def V_init(self) -> float:
        ...
    @V_init.setter
    def V_init(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def is_izhikevich(self) -> bool:
        """
        True when this spec represents an Izhikevich neuron
        """
    @is_izhikevich.setter
    def is_izhikevich(self, arg0: bool) -> None:
        ...
    @property
    def iz_params(self) -> IzhikevichParameters:
        """
        Izhikevich parameters (only valid when is_izhikevich=True)
        """
    @iz_params.setter
    def iz_params(self, arg0: IzhikevichParameters) -> None:
        ...
class RateFuncForm:
    """
    Members:
    
      LINEAR_OVER_EXP
    
      EXP_DECAY
    
      LINEAR_OVER_EXPM1
    
      SIGMOID
    """
    EXP_DECAY: typing.ClassVar[RateFuncForm]  # value = <RateFuncForm.EXP_DECAY: 1>
    LINEAR_OVER_EXP: typing.ClassVar[RateFuncForm]  # value = <RateFuncForm.LINEAR_OVER_EXP: 0>
    LINEAR_OVER_EXPM1: typing.ClassVar[RateFuncForm]  # value = <RateFuncForm.LINEAR_OVER_EXPM1: 2>
    SIGMOID: typing.ClassVar[RateFuncForm]  # value = <RateFuncForm.SIGMOID: 3>
    __members__: typing.ClassVar[dict[str, RateFuncForm]]  # value = {'LINEAR_OVER_EXP': <RateFuncForm.LINEAR_OVER_EXP: 0>, 'EXP_DECAY': <RateFuncForm.EXP_DECAY: 1>, 'LINEAR_OVER_EXPM1': <RateFuncForm.LINEAR_OVER_EXPM1: 2>, 'SIGMOID': <RateFuncForm.SIGMOID: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RateFuncParams:
    form: RateFuncForm
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def A(self) -> float:
        ...
    @A.setter
    def A(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def B(self) -> float:
        ...
    @B.setter
    def B(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def C(self) -> float:
        ...
    @C.setter
    def C(self, arg0: typing.SupportsFloat) -> None:
        ...
class ReceptorType:
    """
    Members:
    
      AMPA
    
      NMDA
    
      GABA_A
    """
    AMPA: typing.ClassVar[ReceptorType]  # value = <ReceptorType.AMPA: 0>
    GABA_A: typing.ClassVar[ReceptorType]  # value = <ReceptorType.GABA_A: 2>
    NMDA: typing.ClassVar[ReceptorType]  # value = <ReceptorType.NMDA: 1>
    __members__: typing.ClassVar[dict[str, ReceptorType]]  # value = {'AMPA': <ReceptorType.AMPA: 0>, 'NMDA': <ReceptorType.NMDA: 1>, 'GABA_A': <ReceptorType.GABA_A: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RegionalNetwork:
    fast_math: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def add_connection(self, src: str, src_local: typing.SupportsInt, dst: str, dst_local: typing.SupportsInt, weight: typing.SupportsFloat, synapse: SynapseSpec, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add a single connection using local indices
        """
    def add_kinetic_connection(self, src: str, i: typing.SupportsInt, dst: str, j: typing.SupportsInt, weight: typing.SupportsFloat, spec: KineticSynapseSpec, delay: typing.SupportsFloat = 0.0) -> None:
        ...
    @typing.overload
    def add_population(self, name: str, count: typing.SupportsInt, neuron_type: _NetworkNeuronType) -> None:
        """
        Add a population with neuron type preset
        """
    @typing.overload
    def add_population(self, name: str, count: typing.SupportsInt, parameters: HHParameters) -> None:
        """
        Add a population with custom HH parameters
        """
    @typing.overload
    def add_population(self, name: str, count: typing.SupportsInt, parameters: IzhikevichParameters) -> None:
        """
        Add a population with custom Izhikevich parameters
        """
    @typing.overload
    def add_population(self, name: str, count: typing.SupportsInt, spec: ...) -> None:
        """
        Add a population with a composable neuron model spec
        """
    @typing.overload
    def add_population(self, name: str, specs: collections.abc.Sequence[...]) -> None:
        """
        Add a heterogeneous population from a list of per-neuron specs
        """
    def connect(self, src: str, dst: str, pattern: ConnectivityPattern, synapse: SynapseSpec, weight: WeightDistribution, delay: typing.SupportsFloat = 0.0, shift: typing.SupportsInt = 1, probability: typing.SupportsFloat = 0.1, allow_self: bool = False, seed: typing.SupportsInt = 0) -> None:
        """
        Connect two populations with a preset pattern
        """
    def network(self) -> _Network:
        ...
    def num_populations(self) -> int:
        ...
    def population_names(self) -> list[str]:
        ...
    def population_size(self, name: str) -> int:
        ...
    def population_start(self, name: str) -> int:
        ...
    def randomize_membrane_potentials(self, name: str, mean: typing.SupportsFloat, std_dev: typing.SupportsFloat, seed: typing.SupportsInt = 0, reset_gates: bool = False) -> None:
        """
        Randomize membrane potentials in a population
        """
    def reset(self) -> None:
        ...
    @property
    def num_neurons(self) -> int:
        ...
    @property
    def num_synapses(self) -> int:
        ...
class SynapseBase:
    def __repr__(self) -> str:
        ...
    def type_name(self) -> str:
        """
        Get synapse type name
        """
    @property
    def conductance(self) -> float:
        """
        Current conductance
        """
    @property
    def delay(self) -> float:
        """
        Axonal conduction delay (ms)
        """
    @property
    def post_idx(self) -> int:
        """
        Post-synaptic neuron index
        """
    @property
    def pre_idx(self) -> int:
        """
        Pre-synaptic neuron index
        """
    @property
    def reversal_potential(self) -> float:
        """
        Reversal potential (mV)
        """
    @property
    def weight(self) -> float:
        """
        Synaptic weight
        """
class SynapseSpec:
    type: ...
    @staticmethod
    def alpha(E_syn: typing.SupportsFloat, tau: typing.SupportsFloat) -> SynapseSpec:
        ...
    @staticmethod
    def ampa() -> SynapseSpec:
        ...
    @staticmethod
    def double_exponential(E_syn: typing.SupportsFloat, tau_rise: typing.SupportsFloat, tau_decay: typing.SupportsFloat) -> SynapseSpec:
        ...
    @staticmethod
    def exponential(E_syn: typing.SupportsFloat, tau: typing.SupportsFloat) -> SynapseSpec:
        ...
    @staticmethod
    def gaba_a() -> SynapseSpec:
        ...
    @staticmethod
    def nmda() -> SynapseSpec:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def E_syn(self) -> float:
        ...
    @E_syn.setter
    def E_syn(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def tau(self) -> float:
        ...
    @tau.setter
    def tau(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def tau_decay(self) -> float:
        ...
    @tau_decay.setter
    def tau_decay(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def tau_rise(self) -> float:
        ...
    @tau_rise.setter
    def tau_rise(self, arg0: typing.SupportsFloat) -> None:
        ...
class SynapseSpecType:
    """
    Members:
    
      EXPONENTIAL
    
      ALPHA
    
      DOUBLE_EXPONENTIAL
    """
    ALPHA: typing.ClassVar[SynapseSpecType]  # value = <SynapseSpecType.ALPHA: 1>
    DOUBLE_EXPONENTIAL: typing.ClassVar[SynapseSpecType]  # value = <SynapseSpecType.DOUBLE_EXPONENTIAL: 2>
    EXPONENTIAL: typing.ClassVar[SynapseSpecType]  # value = <SynapseSpecType.EXPONENTIAL: 0>
    __members__: typing.ClassVar[dict[str, SynapseSpecType]]  # value = {'EXPONENTIAL': <SynapseSpecType.EXPONENTIAL: 0>, 'ALPHA': <SynapseSpecType.ALPHA: 1>, 'DOUBLE_EXPONENTIAL': <SynapseSpecType.DOUBLE_EXPONENTIAL: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TauForm:
    """
    Members:
    
      CONSTANT
    
      BOLTZMANN
    
      DOUBLE_EXP_SUM
    
      OFFSET_DOUBLE_EXP
    
      SCALED_EXP
    
      COMPOUND_AB
    """
    BOLTZMANN: typing.ClassVar[TauForm]  # value = <TauForm.BOLTZMANN: 1>
    COMPOUND_AB: typing.ClassVar[TauForm]  # value = <TauForm.COMPOUND_AB: 5>
    CONSTANT: typing.ClassVar[TauForm]  # value = <TauForm.CONSTANT: 0>
    DOUBLE_EXP_SUM: typing.ClassVar[TauForm]  # value = <TauForm.DOUBLE_EXP_SUM: 2>
    OFFSET_DOUBLE_EXP: typing.ClassVar[TauForm]  # value = <TauForm.OFFSET_DOUBLE_EXP: 3>
    SCALED_EXP: typing.ClassVar[TauForm]  # value = <TauForm.SCALED_EXP: 4>
    __members__: typing.ClassVar[dict[str, TauForm]]  # value = {'CONSTANT': <TauForm.CONSTANT: 0>, 'BOLTZMANN': <TauForm.BOLTZMANN: 1>, 'DOUBLE_EXP_SUM': <TauForm.DOUBLE_EXP_SUM: 2>, 'OFFSET_DOUBLE_EXP': <TauForm.OFFSET_DOUBLE_EXP: 3>, 'SCALED_EXP': <TauForm.SCALED_EXP: 4>, 'COMPOUND_AB': <TauForm.COMPOUND_AB: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TauParams:
    form: TauForm
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_param(self, arg0: typing.SupportsInt) -> float:
        ...
    def set_param(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None:
        ...
class VmExpr:
    instructions: VmInstructionVector
    def __init__(self) -> None:
        ...
    def add_constant(self, arg0: typing.SupportsFloat) -> int:
        ...
    def add_instruction(self, op: VmOp, operand: typing.SupportsInt = 0) -> None:
        ...
    def empty(self) -> bool:
        ...
    @property
    def constants(self) -> list[float]:
        ...
    @constants.setter
    def constants(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
class VmInstruction:
    op: VmOp
    def __init__(self) -> None:
        ...
    @property
    def operand(self) -> int:
        ...
    @operand.setter
    def operand(self, arg0: typing.SupportsInt) -> None:
        ...
class VmInstructionVector:
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> VmInstructionVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> ...:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: VmInstructionVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[typing.Any]:
        ...
    def __len__(self) -> int:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.Any) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: VmInstructionVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: ...) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: VmInstructionVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: ...) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> ...:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> ...:
        """
        Remove and return the item at index ``i``
        """
class VmOp:
    """
    Members:
    
      PUSH_DEP
    
      PUSH_CONST
    
      ADD
    
      MUL
    
      NEG
    
      RCP
    
      POW_INT
    
      POW_HALF
    
      POW_GEN
    
      EXP
    
      LOG
    
      TANH
    
      SIN
    
      COS
    
      SQRT
    
      ABS
    
      PUSH_GATE
    """
    ABS: typing.ClassVar[VmOp]  # value = <VmOp.ABS: 15>
    ADD: typing.ClassVar[VmOp]  # value = <VmOp.ADD: 2>
    COS: typing.ClassVar[VmOp]  # value = <VmOp.COS: 13>
    EXP: typing.ClassVar[VmOp]  # value = <VmOp.EXP: 9>
    LOG: typing.ClassVar[VmOp]  # value = <VmOp.LOG: 10>
    MUL: typing.ClassVar[VmOp]  # value = <VmOp.MUL: 3>
    NEG: typing.ClassVar[VmOp]  # value = <VmOp.NEG: 4>
    POW_GEN: typing.ClassVar[VmOp]  # value = <VmOp.POW_GEN: 8>
    POW_HALF: typing.ClassVar[VmOp]  # value = <VmOp.POW_HALF: 7>
    POW_INT: typing.ClassVar[VmOp]  # value = <VmOp.POW_INT: 6>
    PUSH_CONST: typing.ClassVar[VmOp]  # value = <VmOp.PUSH_CONST: 1>
    PUSH_DEP: typing.ClassVar[VmOp]  # value = <VmOp.PUSH_DEP: 0>
    PUSH_GATE: typing.ClassVar[VmOp]  # value = <VmOp.PUSH_GATE: 16>
    PUSH_S: typing.ClassVar[VmOp]  # value = <VmOp.PUSH_S: 17>
    RCP: typing.ClassVar[VmOp]  # value = <VmOp.RCP: 5>
    SIN: typing.ClassVar[VmOp]  # value = <VmOp.SIN: 12>
    SQRT: typing.ClassVar[VmOp]  # value = <VmOp.SQRT: 14>
    TANH: typing.ClassVar[VmOp]  # value = <VmOp.TANH: 11>
    __members__: typing.ClassVar[dict[str, VmOp]]  # value = {'PUSH_DEP': <VmOp.PUSH_DEP: 0>, 'PUSH_CONST': <VmOp.PUSH_CONST: 1>, 'ADD': <VmOp.ADD: 2>, 'MUL': <VmOp.MUL: 3>, 'NEG': <VmOp.NEG: 4>, 'RCP': <VmOp.RCP: 5>, 'POW_INT': <VmOp.POW_INT: 6>, 'POW_HALF': <VmOp.POW_HALF: 7>, 'POW_GEN': <VmOp.POW_GEN: 8>, 'EXP': <VmOp.EXP: 9>, 'LOG': <VmOp.LOG: 10>, 'TANH': <VmOp.TANH: 11>, 'SIN': <VmOp.SIN: 12>, 'COS': <VmOp.COS: 13>, 'SQRT': <VmOp.SQRT: 14>, 'ABS': <VmOp.ABS: 15>, 'PUSH_GATE': <VmOp.PUSH_GATE: 16>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class WeightDistType:
    """
    Members:
    
      CONSTANT
    
      UNIFORM
    
      NORMAL
    """
    CONSTANT: typing.ClassVar[WeightDistType]  # value = <WeightDistType.CONSTANT: 0>
    NORMAL: typing.ClassVar[WeightDistType]  # value = <WeightDistType.NORMAL: 2>
    UNIFORM: typing.ClassVar[WeightDistType]  # value = <WeightDistType.UNIFORM: 1>
    __members__: typing.ClassVar[dict[str, WeightDistType]]  # value = {'CONSTANT': <WeightDistType.CONSTANT: 0>, 'UNIFORM': <WeightDistType.UNIFORM: 1>, 'NORMAL': <WeightDistType.NORMAL: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class WeightDistribution:
    type: WeightDistType
    @staticmethod
    def constant(value: typing.SupportsFloat) -> WeightDistribution:
        ...
    @staticmethod
    def normal(mean: typing.SupportsFloat, std: typing.SupportsFloat) -> WeightDistribution:
        ...
    @staticmethod
    def uniform(min: typing.SupportsFloat, max: typing.SupportsFloat) -> WeightDistribution:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def param1(self) -> float:
        ...
    @param1.setter
    def param1(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def param2(self) -> float:
        ...
    @param2.setter
    def param2(self, arg0: typing.SupportsFloat) -> None:
        ...
class _Network:
    @typing.overload
    def __init__(self) -> None:
        """
        Create an empty network
        """
    @typing.overload
    def __init__(self, num_neurons: typing.SupportsInt) -> None:
        """
        Create a network with n HH neurons
        """
    @typing.overload
    def __init__(self, num_neurons: typing.SupportsInt, neuron_type: _NetworkNeuronType) -> None:
        """
        Create a network with n neurons of specified type
        """
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def _simulate_into_buffers(self, duration: typing.SupportsFloat, dt: typing.SupportsFloat, I_ext: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], V_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], gate_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], calcium_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], u_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], g_syn_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], I_syn_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], spike_event_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], interval: typing.SupportsInt, spike_threshold: typing.SupportsFloat = 0.0) -> None:
        ...
    def _simulate_with_descriptors(self, duration: typing.SupportsFloat, dt: typing.SupportsFloat, I_const: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], pulses: collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsInt, typing.SupportsInt, typing.SupportsInt, typing.SupportsFloat]], dbs_events: collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsInt, typing.SupportsInt, typing.SupportsInt, typing.SupportsFloat]], V_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], gate_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], calcium_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], u_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], g_syn_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], I_syn_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], spike_event_buf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], interval: typing.SupportsInt, spike_threshold: typing.SupportsFloat = 0.0) -> None:
        ...
    def add_alpha_synapse(self, pre_idx: typing.SupportsInt, post_idx: typing.SupportsInt, weight: typing.SupportsFloat, E_syn: typing.SupportsFloat = 0.0, tau: typing.SupportsFloat = 2.0, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add an alpha-function synapse between neurons
        """
    def add_ampa_synapse(self, pre_idx: typing.SupportsInt, post_idx: typing.SupportsInt, weight: typing.SupportsFloat, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add an AMPA synapse (E=0, tau_r=0.5, tau_d=2.5)
        """
    def add_double_exp_synapse(self, pre_idx: typing.SupportsInt, post_idx: typing.SupportsInt, weight: typing.SupportsFloat, E_syn: typing.SupportsFloat = 0.0, tau_rise: typing.SupportsFloat = 0.4, tau_decay: typing.SupportsFloat = 2.5, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add a double-exponential synapse between neurons
        """
    def add_gaba_a_synapse(self, pre_idx: typing.SupportsInt, post_idx: typing.SupportsInt, weight: typing.SupportsFloat, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add a GABA_A synapse (E=-80, tau_r=0.4, tau_d=7.7)
        """
    def add_kinetic_synapse(self, pre: typing.SupportsInt, post: typing.SupportsInt, weight: typing.SupportsFloat, spec: KineticSynapseSpec, delay: typing.SupportsFloat = 0.0) -> int:
        ...
    @typing.overload
    def add_neuron(self) -> int:
        """
        Add a HH neuron with default parameters, returns index
        """
    @typing.overload
    def add_neuron(self, parameters: HHParameters) -> int:
        """
        Add a HH neuron with custom parameters, returns index
        """
    @typing.overload
    def add_neuron(self, neuron_type: _NetworkNeuronType) -> int:
        """
        Add a neuron of specified type, returns index
        """
    @typing.overload
    def add_neuron(self, parameters: IzhikevichParameters) -> int:
        """
        Add an Izhikevich neuron with custom parameters, returns index
        """
    @typing.overload
    def add_neuron(self, spec: ...) -> int:
        """
        Add a composable neuron from a model spec, returns index
        """
    def add_nmda_synapse(self, pre_idx: typing.SupportsInt, post_idx: typing.SupportsInt, weight: typing.SupportsFloat, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add an NMDA synapse (E=0, tau_r=2.0, tau_d=67.0)
        """
    def add_receptor_synapse(self, pre_idx: typing.SupportsInt, post_idx: typing.SupportsInt, weight: typing.SupportsFloat, receptor: ReceptorType, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add a synapse by receptor type (AMPA, NMDA, GABA_A)
        """
    def add_synapse(self, pre_idx: typing.SupportsInt, post_idx: typing.SupportsInt, weight: typing.SupportsFloat, E_syn: typing.SupportsFloat = 0.0, tau: typing.SupportsFloat = 2.0, delay: typing.SupportsFloat = 0.0) -> None:
        """
        Add an exponential synapse between neurons
        """
    def get_kin_S(self, synapse_idx: typing.SupportsInt) -> float:
        """
        Get kinetic gating variable S for a synapse by index
        """
    def get_kin_g(self, synapse_idx: typing.SupportsInt) -> float:
        """
        Get effective conductance g for a synapse by index
        """
    def get_potentials(self) -> list[float]:
        """
        Get membrane potentials of all neurons
        """
    def get_synapse_post_indices(self) -> list[int]:
        """
        Flat postsynaptic neuron index vector
        """
    def get_synapse_pre_indices(self) -> list[int]:
        """
        Flat presynaptic neuron index vector
        """
    def hh_neuron(self, idx: typing.SupportsInt) -> HHNeuron:
        """
        Get HH neuron by index (throws if wrong type)
        """
    def iz_neuron(self, idx: typing.SupportsInt) -> IzhikevichNeuron:
        """
        Get Izhikevich neuron by index (throws if wrong type)
        """
    def max_gate_count(self) -> int:
        """
        Max gate variables across all composable neurons
        """
    def neuron(self, idx: typing.SupportsInt) -> NeuronBase:
        """
        Get neuron by index (polymorphic)
        """
    def neuron_type(self, idx: typing.SupportsInt) -> str:
        """
        Get neuron type name at index
        """
    def reset(self) -> None:
        """
        Reset all neurons to resting state
        """
    def simulate(self, duration: typing.SupportsFloat, dt: typing.SupportsFloat, I_ext: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat]]) -> list[list[float]]:
        """
        Run network simulation (returns voltage traces as nested list)
        """
    def step(self, dt: typing.SupportsFloat, I_ext: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Advance simulation by dt
        """
    def synapse(self, idx: typing.SupportsInt) -> SynapseBase:
        """
        Get synapse by index (polymorphic)
        """
    @property
    def fast_math(self) -> bool:
        """
        Use fast polynomial exp (~8 digits) vs full precision. Default: true.
        """
    @fast_math.setter
    def fast_math(self, arg1: bool) -> None:
        ...
    @property
    def num_neurons(self) -> int:
        ...
    @property
    def num_synapses(self) -> int:
        ...
class _NetworkNeuronType:
    """
    Members:
    
      HH
    
      IZHIKEVICH_RS
    
      IZHIKEVICH_FS
    
      IZHIKEVICH_IB
    
      IZHIKEVICH_CH
    
      IZHIKEVICH_LTS
    
      IZHIKEVICH_CUSTOM
    
      COMPOSABLE
    """
    COMPOSABLE: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.COMPOSABLE: 7>
    HH: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.HH: 0>
    IZHIKEVICH_CH: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.IZHIKEVICH_CH: 4>
    IZHIKEVICH_CUSTOM: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.IZHIKEVICH_CUSTOM: 6>
    IZHIKEVICH_FS: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.IZHIKEVICH_FS: 2>
    IZHIKEVICH_IB: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.IZHIKEVICH_IB: 3>
    IZHIKEVICH_LTS: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.IZHIKEVICH_LTS: 5>
    IZHIKEVICH_RS: typing.ClassVar[_NetworkNeuronType]  # value = <_NetworkNeuronType.IZHIKEVICH_RS: 1>
    __members__: typing.ClassVar[dict[str, _NetworkNeuronType]]  # value = {'HH': <_NetworkNeuronType.HH: 0>, 'IZHIKEVICH_RS': <_NetworkNeuronType.IZHIKEVICH_RS: 1>, 'IZHIKEVICH_FS': <_NetworkNeuronType.IZHIKEVICH_FS: 2>, 'IZHIKEVICH_IB': <_NetworkNeuronType.IZHIKEVICH_IB: 3>, 'IZHIKEVICH_CH': <_NetworkNeuronType.IZHIKEVICH_CH: 4>, 'IZHIKEVICH_LTS': <_NetworkNeuronType.IZHIKEVICH_LTS: 5>, 'IZHIKEVICH_CUSTOM': <_NetworkNeuronType.IZHIKEVICH_CUSTOM: 6>, 'COMPOSABLE': <_NetworkNeuronType.COMPOSABLE: 7>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
ABS: VmOp  # value = <VmOp.ABS: 15>
ADD: VmOp  # value = <VmOp.ADD: 2>
ALL_TO_ALL: ConnectivityPattern  # value = <ConnectivityPattern.ALL_TO_ALL: 0>
ALPHA: SynapseSpecType  # value = <SynapseSpecType.ALPHA: 1>
ALPHA_BETA: KineticUpdateForm  # value = <KineticUpdateForm.ALPHA_BETA: 0>
AMPA: ReceptorType  # value = <ReceptorType.AMPA: 0>
BOLTZMANN: TauForm  # value = <TauForm.BOLTZMANN: 1>
BOLTZMANN_GATE: KineticUpdateForm  # value = <KineticUpdateForm.BOLTZMANN_GATE: 2>
CALCIUM: GateDependency  # value = <GateDependency.CALCIUM: 1>
CHATTERING: IzhikevichType  # value = <IzhikevichType.CHATTERING: 3>
COMPOSABLE: _NetworkNeuronType  # value = <_NetworkNeuronType.COMPOSABLE: 7>
COMPOUND_AB: TauForm  # value = <TauForm.COMPOUND_AB: 5>
CONSTANT: TauForm  # value = <TauForm.CONSTANT: 0>
COS: VmOp  # value = <VmOp.COS: 13>
CUSTOM: IzhikevichType  # value = <IzhikevichType.CUSTOM: 5>
CUSTOM_EXPR: KineticCurrentForm  # value = <KineticCurrentForm.CUSTOM_EXPR: 2>
DERIVED: GateUpdateForm  # value = <GateUpdateForm.DERIVED: 3>
DOUBLE_EXPONENTIAL: SynapseSpecType  # value = <SynapseSpecType.DOUBLE_EXPONENTIAL: 2>
DOUBLE_EXP_SUM: TauForm  # value = <TauForm.DOUBLE_EXP_SUM: 2>
EULER: IntegrationMethod  # value = <IntegrationMethod.EULER: 0>
EXP: VmOp  # value = <VmOp.EXP: 9>
EXPONENTIAL: SynapseSpecType  # value = <SynapseSpecType.EXPONENTIAL: 0>
EXP_DECAY: RateFuncForm  # value = <RateFuncForm.EXP_DECAY: 1>
FAST_SPIKING: IzhikevichType  # value = <IzhikevichType.FAST_SPIKING: 1>
GABA_A: ReceptorType  # value = <ReceptorType.GABA_A: 2>
HH: _NetworkNeuronType  # value = <_NetworkNeuronType.HH: 0>
INF_TAU: GateUpdateForm  # value = <GateUpdateForm.INF_TAU: 0>
INSTANT: GateUpdateForm  # value = <GateUpdateForm.INSTANT: 2>
INTRINSICALLY_BURSTING: IzhikevichType  # value = <IzhikevichType.INTRINSICALLY_BURSTING: 2>
IZHIKEVICH_CH: _NetworkNeuronType  # value = <_NetworkNeuronType.IZHIKEVICH_CH: 4>
IZHIKEVICH_CUSTOM: _NetworkNeuronType  # value = <_NetworkNeuronType.IZHIKEVICH_CUSTOM: 6>
IZHIKEVICH_FS: _NetworkNeuronType  # value = <_NetworkNeuronType.IZHIKEVICH_FS: 2>
IZHIKEVICH_IB: _NetworkNeuronType  # value = <_NetworkNeuronType.IZHIKEVICH_IB: 3>
IZHIKEVICH_LTS: _NetworkNeuronType  # value = <_NetworkNeuronType.IZHIKEVICH_LTS: 5>
IZHIKEVICH_RS: _NetworkNeuronType  # value = <_NetworkNeuronType.IZHIKEVICH_RS: 1>
LINEAR: KineticCurrentForm  # value = <KineticCurrentForm.LINEAR: 0>
LINEAR_OVER_EXP: RateFuncForm  # value = <RateFuncForm.LINEAR_OVER_EXP: 0>
LINEAR_OVER_EXPM1: RateFuncForm  # value = <RateFuncForm.LINEAR_OVER_EXPM1: 2>
LOG: VmOp  # value = <VmOp.LOG: 10>
LOW_THRESHOLD_SPIKING: IzhikevichType  # value = <IzhikevichType.LOW_THRESHOLD_SPIKING: 4>
MG_BLOCK: KineticCurrentForm  # value = <KineticCurrentForm.MG_BLOCK: 1>
MUL: VmOp  # value = <VmOp.MUL: 3>
NEG: VmOp  # value = <VmOp.NEG: 4>
NMDA: ReceptorType  # value = <ReceptorType.NMDA: 1>
NORMAL: WeightDistType  # value = <WeightDistType.NORMAL: 2>
OFFSET_DOUBLE_EXP: TauForm  # value = <TauForm.OFFSET_DOUBLE_EXP: 3>
ONE_TO_ONE: ConnectivityPattern  # value = <ConnectivityPattern.ONE_TO_ONE: 1>
POW_GEN: VmOp  # value = <VmOp.POW_GEN: 8>
POW_HALF: VmOp  # value = <VmOp.POW_HALF: 7>
POW_INT: VmOp  # value = <VmOp.POW_INT: 6>
PUSH_CONST: VmOp  # value = <VmOp.PUSH_CONST: 1>
PUSH_DEP: VmOp  # value = <VmOp.PUSH_DEP: 0>
PUSH_GATE: VmOp  # value = <VmOp.PUSH_GATE: 16>
PUSH_S: VmOp  # value = <VmOp.PUSH_S: 17>
RANDOM_PERMUTATION: ConnectivityPattern  # value = <ConnectivityPattern.RANDOM_PERMUTATION: 4>
RANDOM_SPARSE: ConnectivityPattern  # value = <ConnectivityPattern.RANDOM_SPARSE: 3>
RCP: VmOp  # value = <VmOp.RCP: 5>
REGULAR_SPIKING: IzhikevichType  # value = <IzhikevichType.REGULAR_SPIKING: 0>
RK4: IntegrationMethod  # value = <IntegrationMethod.RK4: 1>
RK45_ADAPTIVE: IntegrationMethod  # value = <IntegrationMethod.RK45_ADAPTIVE: 2>
SCALED_EXP: TauForm  # value = <TauForm.SCALED_EXP: 4>
SHIFTED: ConnectivityPattern  # value = <ConnectivityPattern.SHIFTED: 2>
SIGMOID: RateFuncForm  # value = <RateFuncForm.SIGMOID: 3>
SIN: VmOp  # value = <VmOp.SIN: 12>
SQRT: VmOp  # value = <VmOp.SQRT: 14>
TANH: VmOp  # value = <VmOp.TANH: 11>
TANH_GATE: KineticUpdateForm  # value = <KineticUpdateForm.TANH_GATE: 1>
UNIFORM: WeightDistType  # value = <WeightDistType.UNIFORM: 1>
VOLTAGE: GateDependency  # value = <GateDependency.VOLTAGE: 0>
__version__: str = '0.7.0'
Parameters = HHParameters
State = HHState
