"""
Metrics & Recording System for Hodgkin-Huxley neural simulations.

Provides RecordingConfig, MetricsResult, PopulationMetricsResult,
and the internal _run_recording() function that drives simulate_into_buffers().
"""

from __future__ import annotations

import numpy as np
from typing import Any
from dataclasses import dataclass, field

from ._core import _Network as _Network


# =============================================================================
# StimPlan — compact stimulation descriptor (no dense I_ext matrix)
# =============================================================================

@dataclass
class _StimPlan:
    """
    Compact external-stimulus representation passed to _simulate_with_descriptors.

    Avoids the large dense (n_neurons × n_steps) I_ext matrix for inputs that
    can be expressed as constant baselines, rectangular pulse events, or DBS
    trains.

    Fields
    ------
    I_const : ndarray, shape (n_neurons,)
        Per-neuron constant baseline current (µA/cm²).
    pulses  : list of (neuron_start, neuron_end, onset_step, end_step, amplitude)
        Rectangular pulse intervals.  Multiple entries allowed per population.
    dbs     : list of (neuron_start, neuron_end, isi_steps, pw_steps, amplitude)
        Periodic DBS pulse trains evaluated via modulo arithmetic.
    """
    I_const: np.ndarray                            # shape (N,), float64
    pulses:  list = field(default_factory=list)    # list of 5-tuples
    dbs:     list = field(default_factory=list)    # list of 5-tuples


# =============================================================================
# RecordingConfig
# =============================================================================

class RecordingConfig:
    """Configuration for what metrics to record during a simulation."""

    SLOW_METRICS = {"gates", "calcium", "g_syn", "I_syn"}
    DERIVED_FROM_V = {"spikes", "spike_count", "firing_rate", "mean_V",
                      "ISI_mean", "ISI_cv"}

    def __init__(self,
                 metrics=("V",),
                 neurons="all",
                 interval: int = 1,
                 spike_threshold: float = 0.0):
        self.metrics = list(metrics)
        self.neurons = neurons
        self.interval = interval
        self.spike_threshold = spike_threshold

    # ------------------------------------------------------------------
    # Preset constructors
    # ------------------------------------------------------------------

    @classmethod
    def voltage_only(cls, interval=1, neurons="all"):
        return cls(["V"], neurons, interval)

    @classmethod
    def spikes_only(cls, threshold=0.0, neurons="all"):
        # interval=1 is critical: subsampled V can miss narrow spikes
        return cls(["spikes", "spike_count", "firing_rate"], neurons,
                   interval=1, spike_threshold=threshold)

    @classmethod
    def spike_stats(cls, threshold=0.0, neurons="all"):
        return cls(["spikes", "spike_count", "firing_rate", "ISI_mean", "ISI_cv"],
                   neurons, interval=1, spike_threshold=threshold)

    @classmethod
    def summary_metrics(cls, threshold=0.0, neurons="all"):
        """No full traces → minimal memory."""
        return cls(["spike_count", "firing_rate", "mean_V"], neurons,
                   interval=1, spike_threshold=threshold)

    @classmethod
    def all_neuron_metrics(cls, interval=1, neurons="all"):
        return cls(["V", "gates", "calcium", "u", "spikes", "firing_rate", "mean_V",
                    "ISI_mean", "ISI_cv"], neurons, interval)

    @classmethod
    def all_synapse_metrics(cls, interval=1, neurons="all"):
        return cls(["V", "spikes", "firing_rate", "g_syn", "I_syn",
                    "spike_count_per_synapse"], neurons, interval)

    @classmethod
    def for_population(cls, pop_name, metrics=("V", "spikes"),
                       interval=1, threshold=0.0):
        return cls(list(metrics), neurons={pop_name: "all"},
                   interval=interval, spike_threshold=threshold)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def needs_V(self):
        return "V" in self.metrics or bool(self.DERIVED_FROM_V & set(self.metrics))

    def needs_slow_path(self):
        return bool(self.SLOW_METRICS & set(self.metrics))


# =============================================================================
# MetricsResult
# =============================================================================

class MetricsResult:
    """Dict-like container for recorded simulation metrics."""

    def __init__(self, data: dict, time: np.ndarray, duration: float,
                 dt: float, interval: int, neuron_indices: list,
                 gate_names: list | None = None):
        self._data = data
        self.time = time
        self.duration = duration
        self.dt = dt
        self.interval = interval
        self.neuron_indices = neuron_indices
        self.gate_names = gate_names

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def to_dict(self):
        return dict(self._data)

    def summary(self):
        """Print a table of per-neuron scalar metrics."""
        scalar_keys = ["spike_count", "firing_rate", "mean_V", "ISI_mean", "ISI_cv"]
        available = [k for k in scalar_keys if k in self._data]
        if not available:
            print("No scalar metrics available.")
            return

        header = f"{'Neuron':>8}" + "".join(f"  {k:>12}" for k in available)
        print(header)
        print("-" * len(header))
        for row_idx, global_idx in enumerate(self.neuron_indices):
            row = f"{global_idx:>8}"
            for k in available:
                val = self._data[k]
                row += f"  {val[row_idx]:>12.4f}"
            print(row)

    def __repr__(self):
        keys = list(self._data.keys())
        n = len(self.neuron_indices)
        return f"<MetricsResult metrics={keys} neurons={n} T={len(self.time)}>"


# =============================================================================
# PopulationMetricsResult
# =============================================================================

class PopulationMetricsResult:
    """Dict-like by population name; each value is a MetricsResult."""

    _PER_NEURON_KEYS = frozenset(["V", "gates", "calcium", "u", "I_syn", "spikes",
                                   "spike_count", "firing_rate", "mean_V",
                                   "ISI_mean", "ISI_cv", "spike_events"])

    def __init__(self, result: MetricsResult, pop_info: dict,
                 config: RecordingConfig):
        self._result = result
        self._pop_info = pop_info  # {name: (start, count)}
        self._config = config
        self._neuron_indices = list(result.neuron_indices)

        # Build index into result arrays for each population
        idx_set = {gi: pos for pos, gi in enumerate(self._neuron_indices)}
        self._pop_positions = {}
        for name, (start, count) in pop_info.items():
            positions = [idx_set[i] for i in range(start, start + count)
                         if i in idx_set]
            if positions:
                self._pop_positions[name] = positions

    def __getitem__(self, pop_name: str) -> MetricsResult:
        if pop_name not in self._pop_positions:
            raise KeyError(pop_name)
        positions = self._pop_positions[pop_name]
        start, count = self._pop_info[pop_name]
        pop_global = [self._neuron_indices[p] for p in positions]

        data = {}
        for key, val in self._result._data.items():
            if key in self._PER_NEURON_KEYS:
                if isinstance(val, list):  # spikes
                    data[key] = [val[p] for p in positions]
                else:
                    data[key] = val[positions]
            else:  # g_syn, spike_count_per_synapse: network-wide
                data[key] = val

        return MetricsResult(
            data, self._result.time, self._result.duration,
            self._result.dt, self._result.interval,
            pop_global, self._result.gate_names
        )

    def keys(self):
        return list(self._pop_positions.keys())

    def __contains__(self, key):
        return key in self._pop_positions

    @property
    def populations(self):
        return list(self._pop_positions.keys())

    def __repr__(self):
        return f"<PopulationMetricsResult populations={self.populations}>"


# =============================================================================
# Spike detection helpers
# =============================================================================

def _detect_spikes(V: np.ndarray, dt: float, threshold: float = -20.0) -> list:
    """
    Detect spike times from voltage traces.

    Parameters
    ----------
    V : ndarray, shape (n_neurons, n_steps)
    dt : float, time step in ms (may be interval * sim_dt)

    Returns
    -------
    list of ndarray : spike times in ms per neuron
    """
    result = []
    for i in range(len(V)):
        diff = np.diff((V[i] > threshold).astype(np.int8), prepend=0)
        result.append(np.where(diff == 1)[0].astype(np.float64) * dt)
    return result


def _isi_mean(spikes: list) -> np.ndarray:
    out = np.zeros(len(spikes))
    for i, sp in enumerate(spikes):
        if len(sp) >= 2:
            out[i] = np.mean(np.diff(sp))
    return out


def _isi_cv(spikes: list) -> np.ndarray:
    out = np.zeros(len(spikes))
    for i, sp in enumerate(spikes):
        if len(sp) >= 2:
            isis = np.diff(sp)
            m = np.mean(isis)
            if m > 0:
                out[i] = np.std(isis) / m
    return out


# =============================================================================
# Neuron index resolution
# =============================================================================

def _resolve_neuron_indices(neuron_spec, n_neurons: int, pop_info: dict | None) -> list:
    if neuron_spec == "all":
        return list(range(n_neurons))
    elif isinstance(neuron_spec, (list, np.ndarray)):
        return [int(x) for x in neuron_spec]
    elif isinstance(neuron_spec, dict):
        if pop_info is None:
            raise ValueError("neuron dict requires pop_info (RegionalNetwork only)")
        indices = []
        for pop_name, sel in neuron_spec.items():
            start, count = pop_info[pop_name]
            if sel == "all":
                indices.extend(range(start, start + count))
            else:
                indices.extend(start + int(i) for i in sel)
        return sorted(set(indices))
    else:
        return list(range(n_neurons))


# =============================================================================
# Gate name extraction
# =============================================================================

def _extract_gate_names(network_core, selected: list) -> list | None:
    """Return gate names from the first composable neuron in selected."""
    try:
        from ._core import ComposableNeuron as _ComposableNeuron
    except ImportError:
        return None

    for i in selected:
        try:
            n = network_core.neuron(i)
            if isinstance(n, _ComposableNeuron):
                spec = n.model_spec
                return [spec.gates[g].name for g in range(len(spec.gates))]
        except Exception:
            continue
    return None


# =============================================================================
# Core recording function
# =============================================================================

def _run_recording(network_core: _Network, duration: float, dt: float,
                   I_ext_list, config: RecordingConfig,
                   pop_info: dict | None = None,
                   detection_threshold: float | None = None,
                   stim_plan: "_StimPlan | None" = None) -> MetricsResult:
    """
    Allocate output buffers, run simulation, build MetricsResult.

    Parameters
    ----------
    network_core : _Network
        The C++ Network object.  Always pass ``net._network`` (from Network)
        or ``rnet.network()`` (from RegionalNetwork) — never a RegionalNetwork
        directly.
    duration     : simulation duration in ms
    dt           : time step in ms
    I_ext_list   : ndarray (n_neurons, n_steps) or None when stim_plan is given
    config       : RecordingConfig
    pop_info     : {pop_name: (start, count)} for RegionalNetwork slicing
    detection_threshold : float, optional
        Voltage threshold (mV) for C++ hot-loop spike detection.
    stim_plan    : _StimPlan, optional
        When provided, calls _simulate_with_descriptors() instead of
        _simulate_into_buffers(), avoiding the dense I_ext matrix allocation.
    """
    det_thresh = detection_threshold if detection_threshold is not None \
                 else config.spike_threshold
    n_neurons  = network_core.num_neurons
    n_synapses = network_core.num_synapses
    n_steps    = int(duration / dt)
    interval   = config.interval
    n_rec      = (n_steps + interval - 1) // interval

    metrics      = set(config.metrics)
    need_V       = config.needs_V()
    need_gates   = "gates"        in metrics
    need_calcium = "calcium"      in metrics
    need_u       = "u"            in metrics
    need_g_syn   = "g_syn"        in metrics
    need_I_syn   = "I_syn"        in metrics
    need_sevents = "spike_events" in metrics

    max_gates = network_core.max_gate_count() if need_gates else 0

    # Allocate buffers (empty array for skipped metrics)
    def alloc(shape):
        return np.zeros(shape, dtype=np.float64) if shape else np.empty(0, dtype=np.float64)

    V_buf            = alloc((n_neurons, n_rec)            if need_V       else ())
    gate_buf         = alloc((n_neurons, max_gates, n_rec) if need_gates   else ())
    calcium_buf      = alloc((n_neurons, n_rec)            if need_calcium else ())
    u_buf            = alloc((n_neurons, n_rec)            if need_u       else ())
    g_syn_buf        = alloc((n_synapses, n_rec)           if need_g_syn   else ())
    I_syn_buf        = alloc((n_neurons, n_rec)            if need_I_syn   else ())
    spike_event_buf  = alloc((n_neurons, n_rec)            if need_sevents else ())

    # Run C++ hot loop with recording
    if stim_plan is not None:
        # Compact descriptor path — no dense I_ext matrix
        network_core._simulate_with_descriptors(
            duration, dt, stim_plan.I_const,
            stim_plan.pulses, stim_plan.dbs,
            V_buf, gate_buf, calcium_buf, u_buf, g_syn_buf, I_syn_buf,
            spike_event_buf, interval, det_thresh)
    else:
        network_core._simulate_into_buffers(
            duration, dt, I_ext_list,
            V_buf, gate_buf, calcium_buf, u_buf, g_syn_buf, I_syn_buf,
            spike_event_buf, interval, det_thresh)

    # Neuron selection
    selected = _resolve_neuron_indices(config.neurons, n_neurons, pop_info)
    time_axis = np.arange(n_rec, dtype=np.float64) * interval * dt

    data = {}
    if "V" in metrics:        data["V"]            = V_buf[selected]
    if need_gates:            data["gates"]        = gate_buf[selected]
    if need_calcium:          data["calcium"]      = calcium_buf[selected]
    if need_u:                data["u"]            = u_buf[selected]
    if need_g_syn:            data["g_syn"]        = g_syn_buf
    if need_I_syn:            data["I_syn"]        = I_syn_buf[selected]
    if need_sevents:          data["spike_events"] = spike_event_buf[selected]

    # Spike-derived metrics
    derived_requested = metrics & (RecordingConfig.DERIVED_FROM_V |
                                   {"spike_count_per_synapse"})
    if derived_requested:
        V_sel = V_buf[selected]
        spikes = _detect_spikes(V_sel, interval * dt, config.spike_threshold)

        if "spikes"      in metrics: data["spikes"]      = spikes
        if "spike_count" in metrics: data["spike_count"] = np.array([len(s) for s in spikes])
        if "firing_rate" in metrics: data["firing_rate"] = np.array(
            [len(s) / (duration * 1e-3) for s in spikes])
        if "mean_V"      in metrics: data["mean_V"]      = V_sel.mean(axis=1)
        if "ISI_mean"    in metrics: data["ISI_mean"]    = _isi_mean(spikes)
        if "ISI_cv"      in metrics: data["ISI_cv"]      = _isi_cv(spikes)

        if "spike_count_per_synapse" in metrics:
            pre_idx = np.array(network_core.get_synapse_pre_indices(), dtype=np.int64)
            # Need spike counts for ALL neurons
            all_spikes = _detect_spikes(V_buf, interval * dt, config.spike_threshold)
            all_sc = np.array([len(s) for s in all_spikes])
            data["spike_count_per_synapse"] = all_sc[pre_idx]

    gate_names = _extract_gate_names(network_core, selected) if need_gates else None

    return MetricsResult(data, time_axis, duration, dt, interval, selected, gate_names)
