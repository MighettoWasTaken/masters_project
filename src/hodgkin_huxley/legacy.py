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
    ExponentialSynapse as _ExponentialSynapse,
    AlphaSynapse as _AlphaSynapse,
    DoubleExponentialSynapse as _DoubleExponentialSynapse,
    _NetworkNeuronType as _NetworkNeuronType,
)

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
        "Use SynapseSpec.alpha() instead."
    ),
    "DoubleExponentialSynapse": (
        "hodgkin_huxley.DoubleExponentialSynapse is deprecated. "
        "Use SynapseSpec.ampa() / SynapseSpec.nmda() / SynapseSpec.gaba_a() instead."
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
}

__all__ = list(_DEPRECATION_MESSAGES.keys())


def __getattr__(name: str) -> object:
    if name not in _DEPRECATION_MESSAGES:
        raise AttributeError(f"module 'hodgkin_huxley.legacy' has no attribute {name!r}")
    warnings.warn(_DEPRECATION_MESSAGES[name], DeprecationWarning, stacklevel=2)
    if name in ("Network", "HHNeuron", "IzhikevichNeuron"):
        # Lazily grab wrapper classes from the main package to avoid circular import
        import hodgkin_huxley as _hh
        return {
            "Network": _hh._NetworkWrapper,
            "HHNeuron": _hh._HHNeuronWrapper,
            "IzhikevichNeuron": _hh._IzhikevichNeuronWrapper,
        }[name]
    return _LEGACY_OBJECTS[name]
