"""
hodgkin_huxley._network — RegionalNetwork and supporting helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from .._equations import SynapseModel

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._core import (
    ConnectivityPattern,
    NeuronModelSpec,
    RegionalNetwork as _RegionalNetwork,
    SynapseSpec,
    WeightDistType,
    WeightDistribution,
    _Network as _Network,
    _NetworkNeuronType as _NetworkNeuronType,
)
from ..recording import PopulationMetricsResult, RecordingConfig, _StimPlan, _run_recording

# ---------------------------------------------------------------------------
# String-to-enum maps
# ---------------------------------------------------------------------------

_NEURON_TYPE_MAP = {
    "HH": _NetworkNeuronType.HH,
    "IZHIKEVICH_RS": _NetworkNeuronType.IZHIKEVICH_RS,
    "IZHIKEVICH_FS": _NetworkNeuronType.IZHIKEVICH_FS,
    "IZHIKEVICH_IB": _NetworkNeuronType.IZHIKEVICH_IB,
    "IZHIKEVICH_CH": _NetworkNeuronType.IZHIKEVICH_CH,
    "IZHIKEVICH_LTS": _NetworkNeuronType.IZHIKEVICH_LTS,
    "IZHIKEVICH_CUSTOM": _NetworkNeuronType.IZHIKEVICH_CUSTOM,
}

_PATTERN_MAP = {
    "ALL_TO_ALL": ConnectivityPattern.ALL_TO_ALL,
    "ONE_TO_ONE": ConnectivityPattern.ONE_TO_ONE,
    "SHIFTED": ConnectivityPattern.SHIFTED,
    "RANDOM_SPARSE": ConnectivityPattern.RANDOM_SPARSE,
    "RANDOM_PERMUTATION": ConnectivityPattern.RANDOM_PERMUTATION,
}


# ---------------------------------------------------------------------------
# Weight / connectivity helpers
# ---------------------------------------------------------------------------


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


def _generate_pairs(
    pattern, src_size, dst_size, shift, probability, allow_self, rng, same_pop=False
):
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
# Heterogeneity helpers
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

    m = re.match(r"^(channels|gates)\[(\d+)\]\.(\w+)$", path)
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

    m = re.match(r"^(channels|gates)\[(\d+)\]\.(\w+)$", path)
    if m:
        col, idx, attr = m.group(1), int(m.group(2)), m.group(3)
        items = getattr(spec, col)  # get as Python list
        setattr(items[idx], attr, value)
        setattr(spec, col, items)  # write back (safe for copy or reference semantics)
        return
    if hasattr(spec, path):
        setattr(spec, path, value)
        return
    raise ValueError(
        f"Unknown heterogeneity field path '{path}'.  "
        f"Supported: 'channels[i].g', 'gates[i].initial_value', 'C_m', 'V_init'"
    )


def _build_heterogeneous_specs(
    base_spec, count: int, heterogeneity: dict, seed
) -> list:
    """
    Return a list of *count* NeuronModelSpec copies, each with independent
    parameter samples drawn from *heterogeneity*.
    """
    import copy

    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(count):
        s = copy.copy(base_spec)  # C++ copy constructor via __copy__
        for path, (dist, p1, p2) in heterogeneity.items():
            base_val = _het_get(s, path)
            scale = _het_sample(dist, p1, p2, rng)
            _het_set(s, path, base_val * scale)
        specs.append(s)
    return specs


# ---------------------------------------------------------------------------
# RegionalNetwork
# ---------------------------------------------------------------------------


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
        self._pop_specs: dict = {}    # {pop_name: NeuronModelSpec} (task14)

    def add_population(
        self,
        name: str,
        count: int,
        neuron_type: "str | _NetworkNeuronType | None" = None,
        parameters=None,
        model: "NeuronModelSpec | None" = None,
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
        neuron_type : str, optional
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
            from .._equations import NeuronModel  # lazy — avoids import-time cycle
            spec = model.to_spec() if isinstance(model, NeuronModel) else model
            specs = _build_heterogeneous_specs(spec, count, heterogeneity, seed)
            self._rnet.add_population(name, specs)
            self._pop_specs[name] = spec
        elif model is not None:
            from .._equations import NeuronModel  # lazy — avoids import-time cycle
            spec = model.to_spec() if isinstance(model, NeuronModel) else model
            self._rnet.add_population(name, count, spec)
            self._pop_specs[name] = spec
        elif parameters is not None:
            self._rnet.add_population(name, count, parameters)
        elif neuron_type is not None:
            if isinstance(neuron_type, str):
                neuron_type = _NEURON_TYPE_MAP[neuron_type.upper()]
            self._rnet.add_population(name, count, neuron_type)
        else:
            # Default to HH
            self._rnet.add_population(name, count, _NetworkNeuronType.HH)

    def add_intracellular(
        self,
        dynamics,
        populations=None,
    ) -> None:
        """Attach intracellular dynamics to one or more populations.

        Parameters
        ----------
        dynamics : IntracellularDynamics
            The dynamics object (built with IntracellularDynamics(...)).
        populations : str | list[str] | None
            Population name(s) to attach to.  None = all Composable populations.
        """
        # Resolve target population names
        if populations is None:
            targets = list(self._pop_specs.keys())
        elif isinstance(populations, str):
            targets = [populations]
        else:
            targets = list(populations)

        for pop_name in targets:
            if pop_name not in self._pop_specs:
                raise ValueError(
                    f"add_intracellular: population {pop_name!r} not found or "
                    f"is not a Composable population (no stored spec)"
                )
            spec = self._pop_specs[pop_name]

            # Build name → index maps for channels and gates
            channel_names = [ch.name for ch in spec.channels]
            gate_names = [g.name for g in spec.gates]

            # Build substance_map for cross-substance references
            substance_map = {
                ic.name: i for i, ic in enumerate(spec.intracellular)
            }

            # Check for duplicate substance name
            if dynamics.name in substance_map:
                raise ValueError(
                    f"Substance {dynamics.name!r} already exists in population {pop_name!r}"
                )

            # Compile dynamics to IntracellularSpec
            ic_spec = dynamics.to_spec(channel_names, gate_names, substance_map)

            # Append to the Python-side spec
            spec.intracellular.append(ic_spec)

            # Give each population a unique model name so build_from_neurons
            # creates separate pools for populations with different dynamics.
            if spec.name != pop_name:
                spec.name = pop_name

            # Push updated spec back to C++
            self._rnet.update_population_spec(pop_name, spec)

    def connect(
        self,
        src: str,
        dst: str,
        pattern,
        weight: float | tuple[float, float] | WeightDistribution | None = None,
        delay: float = 0.0,
        synapse: SynapseSpec | SynapseModel | None = None,
        shift: int = 1,
        probability: float = 0.1,
        allow_self: bool = False,
        seed: int = 0,
        kinetic_spec=None,
        g_syn: float | None = None,
        g_pre: float | None = None,
        plasticity=None,
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
            Synaptic weight (= g_pre * g_syn). Mutually exclusive with g_syn/g_pre.
        g_syn : float, optional
            Peak synaptic conductance. Multiplied by g_pre to give weight.
            Must be provided together with g_pre; mutually exclusive with weight.
        g_pre : float, optional
            Presynaptic scaling factor. Multiplied by g_syn to give weight.
            Must be provided together with g_syn; mutually exclusive with weight.
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
        # Resolve weight from either weight or g_syn/g_pre
        using_gsyn = g_syn is not None or g_pre is not None
        using_weight = weight is not None
        if using_gsyn and using_weight:
            raise ValueError(
                "Provide either 'weight' or 'g_syn'/'g_pre', not both."
            )
        if using_gsyn:
            if g_syn is None or g_pre is None:
                raise ValueError(
                    "Both 'g_syn' and 'g_pre' must be provided together "
                    "(weight = g_pre * g_syn)."
                )
            weight = g_pre * g_syn
        elif not using_weight:
            weight = 0.0

        wdist = _resolve_weight(weight)

        # Normalize SynapseModel to SynapseSpec
        from .._equations import SynapseModel as _SynapseModel
        if isinstance(synapse, _SynapseModel):
            synapse = synapse.to_spec()
        if isinstance(kinetic_spec, _SynapseModel):
            kinetic_spec = kinetic_spec.to_spec()
        # With the unified system, kinetic_spec is just a SynapseSpec — merge into synapse
        if kinetic_spec is not None and synapse is None:
            synapse = kinetic_spec
            kinetic_spec = None

        # Resolve plasticity rule: STDPRule/STPRule → PlasticitySpec
        import copy as _copy
        plast_spec = None
        if plasticity is not None:
            pl = _copy.copy(plasticity)
            if hasattr(pl, 'modulator_population') and pl.modulator_population:
                pl._resolved_pop_start = int(self._rnet.population_start(pl.modulator_population))
            if hasattr(pl, 'modulator_substance') and pl.modulator_substance is not None:
                pop_name = pl.modulator_population or dst
                pop_model = self._pop_specs.get(pop_name)
                if pop_model is not None:
                    names = [s.name for s in pop_model.intracellular]
                    if pl.modulator_substance in names:
                        pl._resolved_subst_idx = names.index(pl.modulator_substance)
            plast_spec = pl.to_spec()

        if kinetic_spec is not None:
            # Kinetic synapse path — generate pairs and call add_kinetic_connection
            rng = np.random.default_rng(seed if seed != 0 else None)
            src_size = self._rnet.population_size(src)
            dst_size = self._rnet.population_size(dst)
            same_pop = src == dst
            if callable(pattern):
                pairs = pattern(src_size, dst_size)
            elif isinstance(pattern, str):
                pattern = _PATTERN_MAP[pattern.upper()]
                pairs = _generate_pairs(
                    pattern, src_size, dst_size, shift, probability,
                    allow_self, rng, same_pop=same_pop,
                )
            else:
                pairs = _generate_pairs(
                    pattern, src_size, dst_size, shift, probability,
                    allow_self, rng, same_pop=same_pop,
                )
            for i, j in pairs:  # type: ignore[union-attr]
                if wdist.type == WeightDistType.CONSTANT:
                    w = wdist.param1
                elif wdist.type == WeightDistType.UNIFORM:
                    w = rng.uniform(wdist.param1, wdist.param2)
                else:
                    w = rng.normal(wdist.param1, wdist.param2)
                self._rnet.add_kinetic_connection(
                    src, int(i), dst, int(j), float(w), kinetic_spec, delay
                )
            return

        synapse = synapse or SynapseSpec.ampa()

        if callable(pattern) or plast_spec is not None:
            # Per-connection path: custom pattern OR plasticity (C++ connect() doesn't thread plasticity)
            src_size = self._rnet.population_size(src)
            dst_size = self._rnet.population_size(dst)
            same_pop = src == dst
            rng = np.random.default_rng(seed if seed != 0 else None)
            if callable(pattern):
                pairs = pattern(src_size, dst_size)
            else:
                if isinstance(pattern, str):
                    pattern = _PATTERN_MAP[pattern.upper()]
                pairs = _generate_pairs(
                    pattern, src_size, dst_size, shift, probability,
                    allow_self, rng, same_pop=same_pop,
                )
            for i, j in pairs:  # type: ignore[union-attr]
                if wdist.type == WeightDistType.CONSTANT:
                    w = wdist.param1
                elif wdist.type == WeightDistType.UNIFORM:
                    w = rng.uniform(wdist.param1, wdist.param2)
                else:  # NORMAL
                    w = rng.normal(wdist.param1, wdist.param2)
                if plast_spec is not None:
                    self._rnet.add_connection(
                        src, int(i), dst, int(j), float(w), synapse, delay, plast_spec
                    )
                else:
                    self._rnet.add_connection(
                        src, int(i), dst, int(j), float(w), synapse, delay
                    )
        else:
            # Preset pattern, no plasticity: delegate to C++
            if isinstance(pattern, str):
                pattern = _PATTERN_MAP[pattern.upper()]
            self._rnet.connect(
                src, dst, pattern, synapse, wdist, delay, shift,
                probability, allow_self, seed,
            )

    def add_connection(
        self,
        src: str,
        src_local: int,
        dst: str,
        dst_local: int,
        weight: float,
        synapse: SynapseSpec | SynapseModel | None = None,
        delay: float = 0.0,
    ) -> None:
        """Add a single synapse between two populations using local indices."""
        from .._equations import SynapseModel as _SynapseModel
        if isinstance(synapse, _SynapseModel):
            synapse = synapse.to_spec()
        synapse = synapse or SynapseSpec.ampa()
        self._rnet.add_connection(
            src, src_local, dst, dst_local, weight, synapse, delay
        )

    def add_kinetic_connection(
        self,
        src: str,
        i: int,
        dst: str,
        j: int,
        weight: float,
        spec,
        delay: float = 0.0,
    ) -> None:
        """Add a kinetic synapse between two populations using local indices."""
        self._rnet.add_kinetic_connection(src, i, dst, j, weight, spec, delay)

    def connect_from_matrix(
        self,
        src: str,
        dst: str,
        matrix,
        synapse: SynapseSpec | SynapseModel | None = None,
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
        from .._equations import SynapseModel as _SynapseModel
        if isinstance(synapse, _SynapseModel):
            synapse = synapse.to_spec()
        synapse = synapse or SynapseSpec.ampa()
        src_size = self._rnet.population_size(src)
        dst_size = self._rnet.population_size(dst)

        try:
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
                nz_i, nz_j = matrix.nonzero()
                for i, j in zip(nz_i, nz_j):
                    w = float(matrix[i, j]) * weight_scale
                    self._rnet.add_connection(
                        src, int(i), dst, int(j), w, synapse, delay
                    )
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
                    self._rnet.add_connection(
                        src, i, dst, j, float(w) * weight_scale, synapse, delay
                    )

    def randomize_membrane_potentials(
        self,
        name: str,
        V_mean: float,
        V_std: float,
        seed: int = 0,
        reset_gates: bool = False,
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

    def attach_stimulator(self, pop_name: str, stimulator) -> None:
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

    def get_synapse_weights(self) -> "np.ndarray":
        """Return current weights of all synapses as a numpy array."""
        return np.array(self._rnet.network().get_synapse_weights())

    @property
    def network(self) -> "_Network":
        """Access the underlying Network object."""
        return self._rnet.network()

    @overload
    def simulate(
        self,
        duration: float,
        dt: float,
        I_ext: dict | ArrayLike = ...,
        record: None = ...,
        detection_threshold: float | None = ...,
    ) -> dict[str, NDArray[np.float64]]: ...
    @overload
    def simulate(
        self,
        duration: float,
        dt: float,
        I_ext: dict | ArrayLike | None = ...,
        *,
        record: RecordingConfig,
        detection_threshold: float | None = ...,
    ) -> PopulationMetricsResult: ...

    def simulate(
        self,
        duration: float,
        dt: float,
        I_ext: "dict | ArrayLike | None" = None,
        record: "RecordingConfig | None" = None,
        detection_threshold: "float | None" = None,
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
        detection_threshold : float, optional
            Voltage threshold (mV) used by the C++ hot loop to detect
            pre-synaptic spikes for synapse updates.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Voltage traces by population when record=None.
        PopulationMetricsResult
            When record is a RecordingConfig.
        """
        from .._dbs import DBSStimulator as _DBSStimulatorPy  # lazy — no circular import

        num_neurons = self._rnet.num_neurons
        num_steps = int(duration / dt)

        if I_ext is None:
            I_ext = {}

        pop_info = {
            name: (self._rnet.population_start(name), self._rnet.population_size(name))
            for name in self._rnet.population_names()
        }

        stim_plan = None
        I_ext_list = None

        if isinstance(I_ext, dict):
            # Try to build a compact StimPlan (avoids 128 MB dense matrix).
            # Falls back to dense only when a population receives a pre-computed
            # 1D/2D numpy array that cannot be expressed as descriptors.
            can_use_descriptors = all(
                np.asarray(v, dtype=np.float64).ndim == 0 for v in I_ext.values()
            ) and all(
                isinstance(s, _DBSStimulatorPy) for s in self._stimulators.values()
            )

            if can_use_descriptors:
                I_const = np.zeros(num_neurons, dtype=np.float64)
                for name, val in I_ext.items():
                    start, size = pop_info[name]
                    I_const[start : start + size] = float(np.asarray(val))

                dbs_events = []
                for pop_name, stim in self._stimulators.items():
                    start, size = pop_info[pop_name]
                    freq = stim.frequency
                    if freq > 0:
                        isi_ms = 1000.0 / freq
                        isi_steps = max(1, round(isi_ms / dt))
                        pw_steps = max(1, round(stim.pulse_width / dt))
                        dbs_events.append(
                            (start, start + size, isi_steps, pw_steps, stim.amplitude)
                        )

                stim_plan = _StimPlan(
                    I_const=I_const,
                    pulses=[],
                    dbs=dbs_events,
                )
            else:
                # Dense fallback for populations with array-valued I_ext
                flat = np.zeros((num_neurons, num_steps), dtype=np.float64)
                for name, val in I_ext.items():
                    start, size = pop_info[name]
                    val = np.asarray(val, dtype=np.float64)
                    if val.ndim == 0:
                        flat[start : start + size, :] = val.item()
                    elif val.ndim == 1:
                        flat[start : start + size, :] = val[np.newaxis, :num_steps]
                    elif val.ndim == 2:
                        flat[start : start + size, :] = val[:, :num_steps]
                    else:
                        raise ValueError(
                            f"I_ext['{name}'] must be scalar, 1D, or 2D, "
                            f"got {val.ndim}D"
                        )
                for pop_name, stim in self._stimulators.items():
                    stim_trace = stim.generate(duration, dt)
                    start, size = pop_info[pop_name]
                    flat[start : start + size, :] += np.array(
                        stim_trace[:num_steps], dtype=np.float64
                    )[np.newaxis, :]
                I_ext_list = flat
        else:
            I_ext_list = np.ascontiguousarray(I_ext, dtype=np.float64)

        cfg = record if record is not None else RecordingConfig(["V"])
        result = _run_recording(
            self._rnet.network(),
            duration,
            dt,
            I_ext_list,
            cfg,
            pop_info,
            detection_threshold=detection_threshold,
            stim_plan=stim_plan,
        )

        if record is None:
            # Backward compat: return {pop_name: ndarray}
            V = result["V"]  # shape (n_neurons, n_steps) since neurons="all"
            return {
                name: V[start : start + size, :]
                for name, (start, size) in pop_info.items()
            }

        return PopulationMetricsResult(result, pop_info, record)

    def __len__(self) -> int:
        return self.num_neurons

    def __repr__(self) -> str:
        return (
            f"<RegionalNetwork populations={self.num_populations} "
            f"neurons={self.num_neurons} synapses={self.num_synapses}>"
        )
