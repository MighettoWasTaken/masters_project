"""
Neural Simulation Library

A fast C++ implementation of various neuron models with Python bindings.
Supports Hodgkin-Huxley, Izhikevich, and extensible to other models.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public types from C++ extension
# ---------------------------------------------------------------------------
from ._core import (
    CalciumSpec,
    ChannelSpec,
    ConnectivityPattern,
    DBSParameters,
    IntegrationMethod,
    IzhikevichType,
    NeuronModelSpec,
    Parameters,
    ReceptorType,
    State,
    SynapseCurrentForm,
    SynapseSpec,
    SynapseUpdateForm,
    WeightDistType,
    WeightDistribution,
    __version__,
)

# Backward-compat aliases for old enum names
KineticSynapseSpec = SynapseSpec  # C++ alias (same object)
KineticUpdateForm = SynapseUpdateForm
KineticCurrentForm = SynapseCurrentForm

# ---------------------------------------------------------------------------
# Sub-package exports
# ---------------------------------------------------------------------------
from ._dbs import DBSStimulator
from ._equations import (
    Boltzmann,
    GateRef,
    GateSpec,
    IntracellularDynamics,
    KineticSynapseModel,
    KineticSynapseSpec,
    Modulation,
    NeuronModel,
    RateFunc,
    SynapseModel,
    Tau,
)
from ._network import RegionalNetwork

# ---------------------------------------------------------------------------
# Already-separate flat modules
# ---------------------------------------------------------------------------
from ._codegen import (
    A,
    Ca,
    DA,
    CUDAPrinter,
    EigenPrinter,
    HHEquationError,
    I_source,
    IP3,
    NO,
    S,
    TaggedExpr,
    V,
    V_post,
    V_pre,
    X_ic,
    cAMP,
    compile_gate_product_vm,
    compile_to_vm_bytecode,
    gate,
    jit_compile,
    substance,
    try_pattern_match,
    w,
    x,
    x_post,
    x_pre,
)
from .noise import NoiseInjector
from .pulse import PulseStimulator
from .recording import MetricsResult, PopulationMetricsResult, RecordingConfig
from .spectral import analyze_beta_power, beta_band_power, mtspectrumpt

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__ = [
    # Neuron type enum
    "IzhikevichType",
    "ReceptorType",
    # Regional network (primary public API)
    "RegionalNetwork",
    "ConnectivityPattern",
    "SynapseSpec",
    "SynapseUpdateForm",
    "SynapseCurrentForm",
    "WeightDistribution",
    "WeightDistType",
    # Enums
    "IntegrationMethod",
    # Composable neuron types
    "GateRef",
    "GateSpec",
    "ChannelSpec",
    "CalciumSpec",
    "NeuronModelSpec",
    "NeuronModel",
    "Boltzmann",
    "Tau",
    "RateFunc",
    # Intracellular dynamics (task14)
    "IntracellularDynamics",
    "Modulation",
    # Synapse builders
    "SynapseModel",
    "KineticSynapseSpec",
    "KineticSynapseModel",
    # SymPy equation symbols
    "A", "V", "Ca", "V_pre", "V_post", "S",
    "x", "x_pre", "x_post", "w",
    "I_source", "DA", "cAMP", "IP3", "NO", "X_ic",
    "gate", "substance",
    "compile_gate_product_vm",
    "TaggedExpr",
    "EigenPrinter",
    "CUDAPrinter",
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

# ---------------------------------------------------------------------------
# Deprecated names — accessed via __getattr__ to emit DeprecationWarning
# ---------------------------------------------------------------------------
_DEPRECATED_NAMES = {
    # Legacy neuron/network types (task12)
    "Network",
    "NetworkNeuronType",
    "HHNeuron",
    "IzhikevichNeuron",
    "HHParameters",
    "HHState",
    "IzhikevichParameters",
    "IzhikevichState",
    "SynapseBase",
    "ExponentialSynapse",
    "AlphaSynapse",
    "DoubleExponentialSynapse",
    # Legacy equation struct types (task13)
    "BoltzmannParams",
    "TauParams",
    "TauForm",
    "RateFuncParams",
    "RateFuncForm",
    "GateUpdateForm",
    "GateDependency",
    "KineticUpdateForm",
    "KineticCurrentForm",
}


def __getattr__(name: str) -> object:
    if name in _DEPRECATED_NAMES:
        from hodgkin_huxley import legacy as _legacy
        return getattr(_legacy, name)  # legacy.__getattr__ emits DeprecationWarning
    raise AttributeError(f"module 'hodgkin_huxley' has no attribute {name!r}")
