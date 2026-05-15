# Task 24: Python API + `RegionalNetwork` Integration + Recording

**Depends on:** task23 (`MCPool` C++ class), task21 (`CompartmentSpec`, `MorphologySpec` Python types)  
**Unlocks:** task25 (SWC import produces a `MorphologySpec` consumed by this API)

---

## What to implement

Exposes multi-compartment models at the Python level via a `Morphology` builder and extends `RegionalNetwork` to route morphology-bearing specs to `MCPool`. Recording is extended to return per-compartment voltage traces.

### `Morphology` builder — `src/hodgkin_huxley/_equations/__init__.py`

```python
class Morphology:
    """Factory for common multi-compartment morphology patterns."""

    @staticmethod
    def linear(
        n_dendrite_comps: int,
        soma: "CompartmentSpec",
        dendrite: "CompartmentSpec",
    ) -> "MorphologySpec":
        """Unbranched soma + dendrite chain.

        parent_idx = [-1, 0, 1, 2, ..., n_dendrite_comps - 1]
        All dendrite compartments are copies of the dendrite spec.
        """
        comps = [soma] + [dendrite] * n_dendrite_comps
        parents = [-1] + list(range(n_dendrite_comps))
        return MorphologySpec(comps, parents)

    @staticmethod
    def branched(
        soma: "CompartmentSpec",
        branches: list[list["CompartmentSpec"]],
    ) -> "MorphologySpec":
        """Soma + arbitrary number of unbranched branches.

        Each branch is a list of CompartmentSpecs from proximal to distal.
        All branches attach to the soma (compartment 0).

        Example: branched(soma, [[d1a, d1b], [d2a, d2b]])
          → compartments: [soma, d1a, d1b, d2a, d2b]
          → parent_idx:   [-1,   0,   1,   0,   3  ]
        """
        comps: list[CompartmentSpec] = [soma]
        parents: list[int] = [-1]
        for branch in branches:
            root_of_branch = 0  # always attaches to soma
            for comp in branch:
                parents.append(root_of_branch)
                comps.append(comp)
                root_of_branch = len(comps) - 1
        return MorphologySpec(comps, parents)

    @staticmethod
    def ball_and_stick(
        soma_diameter_um: float = 20.0,
        dend_length_um: float = 500.0,
        n_dend_comps: int = 10,
        soma_channels: list | None = None,
        dend_channels: list | None = None,
    ) -> "MorphologySpec":
        """Classic two-compartment-family model: spherical soma + unbranched dendrite."""
        soma_comp = CompartmentSpec("soma",
                                   length_um=soma_diameter_um,
                                   diameter_um=soma_diameter_um)
        if soma_channels:
            soma_comp.channels = soma_channels

        seg_len = dend_length_um / n_dend_comps
        dend_comp = CompartmentSpec("dend",
                                    length_um=seg_len,
                                    diameter_um=2.0)
        if dend_channels:
            dend_comp.channels = dend_channels

        return Morphology.linear(n_dend_comps, soma_comp, dend_comp)
```

Export `Morphology` from `src/hodgkin_huxley/__init__.py` and add to `__all__`.

### `NeuronModel.multicompartment()` classmethod

Add to `NeuronModel` in `src/hodgkin_huxley/_equations/__init__.py`:

```python
@classmethod
def multicompartment(cls, morphology: "MorphologySpec") -> "NeuronModel":
    """Build a NeuronModel from a MorphologySpec.

    The resulting model's NeuronModelSpec carries the morphology; gates and
    channels per compartment are taken from each CompartmentSpec directly.
    Use this with RegionalNetwork.add_population() — the network will
    automatically dispatch to MCPool.
    """
    spec = NeuronModelSpec()
    spec.morphology = morphology
    model = cls.__new__(cls)
    model._spec = spec
    return model
```

### `RegionalNetwork.add_population()` dispatch — `src/hodgkin_huxley/_network/__init__.py`

In the pool-construction path of `add_population()`, add the MCPool branch:

```python
if spec.has_morphology():
    from hodgkin_huxley._core import MCPool as _MCPool
    pool = _MCPool(spec)
else:
    # existing ComposablePool / HHPool / IzPool dispatch
    ...
```

### `RegionalNetwork.add_intracellular()` — compartment targeting

Extend the signature:

```python
def add_intracellular(
    self,
    dynamics: "IntracellularDynamics",
    populations: str | list[str] | None = None,
    compartments: list[int] | None = None,
) -> None:
```

When `compartments` is not `None` and the population uses an `MCPool`:
- Attach the `IntracellularSpec` only to `spec.morphology.compartments[c]` for `c in compartments`
- Validate: each index in `compartments` is `< spec.morphology.n_comps()`
- Raise `ValueError` for point-neuron populations when `compartments` is non-None (compartment targeting requires a morphology)

### `RecordingConfig` — compartment selection

`src/hodgkin_huxley/recording.py` — add field to `RecordingConfig`:

```python
@dataclass
class RecordingConfig:
    record_V: bool = True
    record_spikes: bool = True
    record_weights: bool = False
    intracellular: bool = False
    compartments: list[int] | None = None
    # None  → record soma only (compartment 0) — backward-compatible default
    # [0,2] → record compartments 0 and 2; result.V_compartments shape (N, 2, T)
    # []    → record no voltage (same as record_V=False for MC populations)
```

### Recording shape change

For multi-compartment populations:

```python
# Default (compartments=None):
result["STN"].V               # shape (N, T) — soma voltage, backward compatible

# With compartments=[0, 2, 5]:
result["STN"].V               # shape (N, T) — soma voltage (compartment 0)
result["STN"].V_compartments  # shape (N, 3, T) — compartments [0, 2, 5]
```

`PopulationMetricsResult` gains `V_compartments: np.ndarray | None = None`.

Recording hot loop extension in `RegionalNetwork.simulate()`: when `cfg.compartments` is not None and the pool is an `MCPool`, call `pool.scatter_V_comp_into(c, buf_c, ...)` for each `c in cfg.compartments` and stack into `V_compartments` at the end.

### `I_ext` layout for multi-compartment populations

`RegionalNetwork.simulate()` must supply `I_ext` to `MCPool.step()` as a flat `[N * C]` array, where index `n * C + c` is the current to neuron `n` compartment `c`. External stimulators (`PulseStimulator`, `DBSStimulator`) apply to compartment 0 (soma) by default. Future work: `PulseStimulator(target_compartment=2)` to inject into a specific compartment.

---

## Key files

| File | Change |
|---|---|
| `src/hodgkin_huxley/_equations/__init__.py` | Add `Morphology` builder, `NeuronModel.multicompartment()` |
| `src/hodgkin_huxley/_network/__init__.py` | MCPool dispatch in `add_population()`, compartment kwarg in `add_intracellular()` |
| `src/hodgkin_huxley/recording.py` | `RecordingConfig.compartments`, `PopulationMetricsResult.V_compartments` |
| `src/hodgkin_huxley/__init__.py` | Export `Morphology`; add to `__all__` |
| `tests/python/test_mc_network.py` | New |

---

## Baseline tests (before PR to testing branch)

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] `Morphology.linear(5, soma, dend)` produces `MorphologySpec` with 6 compartments and correct `parent_idx`
- [ ] `Morphology.branched(soma, [[d1a, d1b], [d2a]])` produces 4 compartments, `parent_idx = [-1, 0, 1, 0]`
- [ ] `Morphology.ball_and_stick(n_dend_comps=10)` produces 11 compartments; resulting spec passes `validate()`
- [ ] `add_population("MC", 20, NeuronModel.multicompartment(morph))` — pool is `MCPool`, not `ComposablePool`
- [ ] `rn.simulate(500, 0.01)` on a 2-pop network (1 MC pop + 1 point-neuron pop) completes without error; `result["MC"].V` has shape `(20, T)`
- [ ] `RecordingConfig(compartments=[0, 2])` on MC population → `result["MC"].V_compartments.shape == (N, 2, T)`, `result["MC"].V.shape == (N, T)` (soma unchanged)
- [ ] `add_intracellular(calcium, populations="MC", compartments=[1,2,3])` attaches substance only to compartments 1–3; compartment 0 has empty `intracellular`
- [ ] `add_intracellular(dopamine, populations="point_pop", compartments=[0])` raises `ValueError` for point-neuron population

---

## Contract for downstream tasks

- task25's `load_swc()` returns a `MorphologySpec` — it is passed directly to `NeuronModel.multicompartment(morph)` and then to `add_population()`. No additional API changes needed.
- `result["pop"].V_compartments` is `None` for point-neuron populations — downstream code must check `is not None` before using it.
- `Morphology.ball_and_stick()` is the canonical quick-start example for the documentation (task18).
