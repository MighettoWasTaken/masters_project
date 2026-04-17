"""
hodgkin_huxley.legacy — deprecated public API

Classes and types moved here are still importable for backwards compatibility,
but emit a DeprecationWarning on access. They will be removed in a future version.

Preferred migration:
  - ``Network`` / ``NetworkNeuronType``  → use ``RegionalNetwork`` with populations
  - ``HHNeuron`` / ``HHParameters`` / ``HHState`` → use ``NeuronModelSpec.hh_default()``
  - ``IzhikevichNeuron`` / ``IzhikevichParameters`` / ``IzhikevichState``
      → use ``NeuronModelSpec.izhikevich(IzhikevichType.<variant>)``
  - ``SynapseBase`` / ``ExponentialSynapse`` / ``AlphaSynapse``
    / ``DoubleExponentialSynapse`` → use ``SynapseSpec`` factories
  - ``BoltzmannParams`` / ``TauParams`` / ``TauForm`` / ``RateFuncParams``
    / ``RateFuncForm`` → use ``Boltzmann`` / ``Tau.*`` / ``RateFunc.*`` helpers
  - ``GateUpdateForm`` / ``GateDependency`` / ``KineticUpdateForm``
    / ``KineticCurrentForm`` → use ``GateSpec`` / ``KineticSynapseModel``
"""

import warnings

# ---------------------------------------------------------------------------
# All underlying objects imported under private aliases so that nothing
# lands in the module's __dict__ under a public name.  Every access to a
# public name therefore goes through __getattr__, which emits the warning.
# ---------------------------------------------------------------------------
from ._core import (
    HHNeuron as _HHNeuron,
    HHParameters as _HHParameters,
    HHState as _HHState,
    IzhikevichNeuron as _IzhikevichNeuron,
    IzhikevichParameters as _IzhikevichParameters,
    IzhikevichState as _IzhikevichState,
    SynapseBase as _SynapseBase,
    SynapseSpec as _SynapseSpec,
    _NetworkNeuronType as _NetworkNeuronType,
    # Equation struct types (task13)
    BoltzmannParams as _BoltzmannParams,
    TauParams as _TauParams,
    TauForm as _TauForm,
    RateFuncParams as _RateFuncParams,
    RateFuncForm as _RateFuncForm,
    GateUpdateForm as _GateUpdateForm,
    GateDependency as _GateDependency,
    KineticUpdateForm as _KineticUpdateForm,
    KineticCurrentForm as _KineticCurrentForm,
)

# ExponentialSynapse / AlphaSynapse / DoubleExponentialSynapse were removed in
# the unified synapse rewrite (task 12). Provide factory shims so legacy access
# still works (returns a SynapseSpec) and emits the deprecation warning.
_ExponentialSynapse = _SynapseSpec.exponential
_AlphaSynapse = _SynapseSpec.alpha_function
_DoubleExponentialSynapse = _SynapseSpec.double_exponential

_DEPRECATION_MESSAGES: dict[str, str] = {
    "Network": (
        "hodgkin_huxley.Network is deprecated. "
        "Use hodgkin_huxley.RegionalNetwork instead."
    ),
    "NetworkNeuronType": (
        "hodgkin_huxley.NetworkNeuronType is deprecated. "
        "Use NeuronModelSpec.hh_default() or NeuronModelSpec.izhikevich() instead."
    ),
    "HHNeuron": (
        "hodgkin_huxley.HHNeuron is deprecated. "
        "Use NeuronModelSpec.hh_default() with RegionalNetwork instead."
    ),
    "HHParameters": (
        "hodgkin_huxley.HHParameters is deprecated. "
        "Use NeuronModelSpec.hh_default() with RegionalNetwork instead."
    ),
    "HHState": (
        "hodgkin_huxley.HHState is deprecated."
    ),
    "IzhikevichNeuron": (
        "hodgkin_huxley.IzhikevichNeuron is deprecated. "
        "Use NeuronModelSpec.izhikevich(IzhikevichType.<variant>) with RegionalNetwork instead."
    ),
    "IzhikevichParameters": (
        "hodgkin_huxley.IzhikevichParameters is deprecated. "
        "Use NeuronModelSpec.izhikevich(params) with RegionalNetwork instead."
    ),
    "IzhikevichState": (
        "hodgkin_huxley.IzhikevichState is deprecated."
    ),
    "SynapseBase": (
        "hodgkin_huxley.SynapseBase is deprecated. Use SynapseSpec instead."
    ),
    "ExponentialSynapse": (
        "hodgkin_huxley.ExponentialSynapse is deprecated. "
        "Use SynapseSpec.exponential() instead."
    ),
    "AlphaSynapse": (
        "hodgkin_huxley.AlphaSynapse is deprecated. "
        "Use SynapseSpec.alpha_function() instead."
    ),
    "DoubleExponentialSynapse": (
        "hodgkin_huxley.DoubleExponentialSynapse is deprecated. "
        "Use SynapseSpec.ampa() / SynapseSpec.nmda() / SynapseSpec.gaba_a() instead."
    ),
    # Equation struct types (task13)
    "BoltzmannParams": (
        "hodgkin_huxley.BoltzmannParams is deprecated. "
        "Use hodgkin_huxley.Boltzmann(v_half, k) instead."
    ),
    "TauParams": (
        "hodgkin_huxley.TauParams is deprecated. "
        "Use hodgkin_huxley.Tau.constant() / Tau.boltzmann() etc. instead."
    ),
    "TauForm": (
        "hodgkin_huxley.TauForm is deprecated. "
        "Use hodgkin_huxley.Tau.*() helpers instead."
    ),
    "RateFuncParams": (
        "hodgkin_huxley.RateFuncParams is deprecated. "
        "Use hodgkin_huxley.RateFunc.*() helpers instead."
    ),
    "RateFuncForm": (
        "hodgkin_huxley.RateFuncForm is deprecated. "
        "Use hodgkin_huxley.RateFunc.*() helpers instead."
    ),
    "GateUpdateForm": (
        "hodgkin_huxley.GateUpdateForm is deprecated. "
        "Use hodgkin_huxley.GateSpec(update_form=...) instead."
    ),
    "GateDependency": (
        "hodgkin_huxley.GateDependency is deprecated. "
        "Use hodgkin_huxley.GateSpec(dependency=...) instead."
    ),
    "KineticUpdateForm": (
        "hodgkin_huxley.KineticUpdateForm is deprecated. "
        "Use hodgkin_huxley.KineticSynapseModel builder instead."
    ),
    "KineticCurrentForm": (
        "hodgkin_huxley.KineticCurrentForm is deprecated. "
        "Use hodgkin_huxley.KineticSynapseModel builder instead."
    ),
}

_LEGACY_OBJECTS: dict[str, object] = {
    "NetworkNeuronType": _NetworkNeuronType,
    "HHParameters": _HHParameters,
    "HHState": _HHState,
    "IzhikevichParameters": _IzhikevichParameters,
    "IzhikevichState": _IzhikevichState,
    "SynapseBase": _SynapseBase,
    "ExponentialSynapse": _ExponentialSynapse,
    "AlphaSynapse": _AlphaSynapse,
    "DoubleExponentialSynapse": _DoubleExponentialSynapse,
    # Equation struct types (task13)
    "BoltzmannParams": _BoltzmannParams,
    "TauParams": _TauParams,
    "TauForm": _TauForm,
    "RateFuncParams": _RateFuncParams,
    "RateFuncForm": _RateFuncForm,
    "GateUpdateForm": _GateUpdateForm,
    "GateDependency": _GateDependency,
    "KineticUpdateForm": _KineticUpdateForm,
    "KineticCurrentForm": _KineticCurrentForm,
}

__all__ = list(_DEPRECATION_MESSAGES.keys())


def __getattr__(name: str) -> object:
    if name not in _DEPRECATION_MESSAGES:
        raise AttributeError(f"module 'hodgkin_huxley.legacy' has no attribute {name!r}")
    warnings.warn(_DEPRECATION_MESSAGES[name], DeprecationWarning, stacklevel=2)
    if name in ("Network", "HHNeuron", "IzhikevichNeuron"):
        from hodgkin_huxley._wrappers import (
            _HHNeuronWrapper,
            _IzhikevichNeuronWrapper,
            _NetworkWrapper,
        )
        return {
            "Network": _NetworkWrapper,
            "HHNeuron": _HHNeuronWrapper,
            "IzhikevichNeuron": _IzhikevichNeuronWrapper,
        }[name]
    return _LEGACY_OBJECTS[name]
