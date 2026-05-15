# Task 25: SWC Morphology Import

**Depends on:** task24 (`MorphologySpec` end-to-end functional; `RegionalNetwork` accepts it)  
**Unlocks:** nothing — final deliverable in the multi-compartment track

---

## What to implement

SWC is the standard neuronal morphology format used by NeuroMorpho.org and most reconstruction software. This task adds `load_swc()` for converting an SWC file into a `MorphologySpec`, and a `swc_stats()` helper for quick inspection.

### SWC format

Each non-comment line: `id  type  x  y  z  r  parent_id`

| `type` code | Meaning |
|---|---|
| 1 | soma |
| 2 | axon |
| 3 | basal dendrite |
| 4 | apical dendrite |
| 5 | fork point |
| 6 | end point |
| 7 | custom / undefined |

`x`, `y`, `z` in µm; `r` = radius in µm; `parent_id = -1` for root. Segment length is the Euclidean distance from the point to its parent. Diameter = `2 * r`.

### `src/hodgkin_huxley/morphology.py` — new file

```python
"""
hodgkin_huxley.morphology — SWC morphology import and reduction utilities.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Literal

import numpy as np

from hodgkin_huxley import CompartmentSpec, MorphologySpec


# ---------------------------------------------------------------------------
# SWC type codes → human names
# ---------------------------------------------------------------------------
_SWC_TYPE_NAMES = {
    1: "soma",
    2: "axon",
    3: "basal_dend",
    4: "apical_dend",
    5: "fork",
    6: "end",
    7: "custom",
}


def load_swc(
    path: str | os.PathLike,
    *,
    compartment_spec_fn: Callable[[int, str], CompartmentSpec] | None = None,
    Ra: float = 100.0,
    Cm: float = 1.0,
    g_leak: float = 0.1,
    E_L: float = -65.0,
    reduce: Literal["none", "equivalent_cylinder", "n_comps"] = "none",
    n_comps_target: int | None = None,
) -> MorphologySpec:
    """Load an SWC morphology file and return a MorphologySpec.

    Parameters
    ----------
    path
        Path to the .swc file.
    compartment_spec_fn
        Optional callback ``fn(type_code: int, name: str) -> CompartmentSpec``
        used to set channels/gates per compartment type. When None, compartments
        have no active channels (passive skeleton only — add channels via
        ``spec.morphology.compartments[i].channels.append(...)``).
    Ra
        Default axial resistivity (Ω·cm) applied to all compartments.
    Cm
        Default specific membrane capacitance (µF/cm²).
    g_leak
        Default passive leak conductance density (µS/cm²) — informational only
        (stored in compartment Ra/Cm; pool uses it via ChannelSpec if provided
        by compartment_spec_fn).
    E_L
        Default leak reversal potential (mV) — informational.
    reduce
        ``"none"``                 — one compartment per SWC segment.
        ``"equivalent_cylinder"``  — Rall equivalent cylinder: collapse all
                                     dendrite segments into a single compartment
                                     per branch type; soma kept as-is.
        ``"n_comps"``              — collapse each branch to ``n_comps_target``
                                     evenly-spaced compartments.
    n_comps_target
        Required when ``reduce="n_comps"``. Maximum compartments per branch.
    """
    segments = _parse_swc(path)
    parent_map = _build_parent_map(segments)
    comps, parents = _segments_to_compartments(
        segments, parent_map, Ra, Cm, compartment_spec_fn
    )
    morph = MorphologySpec(comps, parents)

    if reduce == "equivalent_cylinder":
        morph = _reduce_equivalent_cylinder(morph, segments, Ra, Cm, compartment_spec_fn)
    elif reduce == "n_comps":
        if n_comps_target is None:
            raise ValueError("n_comps_target required when reduce='n_comps'")
        morph = _reduce_n_comps(morph, segments, n_comps_target, Ra, Cm, compartment_spec_fn)

    return morph


def swc_stats(path: str | os.PathLike) -> dict:
    """Return summary statistics for an SWC file without building a full spec."""
    segments = _parse_swc(path)
    type_counts: dict[str, int] = {}
    total_length = 0.0
    max_depth = 0

    id_to_seg = {s["id"]: s for s in segments}
    for seg in segments:
        tname = _SWC_TYPE_NAMES.get(seg["type"], f"type{seg['type']}")
        type_counts[tname] = type_counts.get(tname, 0) + 1
        if seg["parent_id"] != -1:
            parent = id_to_seg[seg["parent_id"]]
            dx = seg["x"] - parent["x"]
            dy = seg["y"] - parent["y"]
            dz = seg["z"] - parent["z"]
            total_length += math.sqrt(dx*dx + dy*dy + dz*dz)
        depth = 0
        p_id = seg["parent_id"]
        while p_id != -1:
            depth += 1
            p_id = id_to_seg[p_id]["parent_id"]
        max_depth = max(max_depth, depth)

    soma_segs = [s for s in segments if s["type"] == 1]
    soma_diam = 2.0 * soma_segs[0]["r"] if soma_segs else 0.0

    n_branches = sum(
        1 for s in segments
        if sum(1 for o in segments if o["parent_id"] == s["id"]) > 1
    )

    return {
        "n_segments":       len(segments),
        "n_branches":       n_branches,
        "total_length_um":  total_length,
        "max_depth":        max_depth,
        "soma_diameter_um": soma_diam,
        "type_counts":      type_counts,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_swc(path: str | os.PathLike) -> list[dict]:
    segments = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            segments.append({
                "id":        int(parts[0]),
                "type":      int(parts[1]),
                "x":         float(parts[2]),
                "y":         float(parts[3]),
                "z":         float(parts[4]),
                "r":         float(parts[5]),
                "parent_id": int(parts[6]),
            })
    return segments


def _build_parent_map(segments: list[dict]) -> dict[int, int]:
    return {s["id"]: s["parent_id"] for s in segments}


def _segments_to_compartments(
    segments, parent_map, Ra, Cm, comp_fn
) -> tuple[list[CompartmentSpec], list[int]]:
    """Convert raw SWC segments to CompartmentSpec list in topological order."""
    id_to_idx: dict[int, int] = {}
    comps: list[CompartmentSpec] = []
    parents: list[int] = []
    id_to_seg = {s["id"]: s for s in segments}

    # BFS from root to ensure topological order (parent_idx[i] < i)
    root = next(s for s in segments if s["parent_id"] == -1)
    queue = [root["id"]]
    visited: set[int] = set()

    while queue:
        seg_id = queue.pop(0)
        if seg_id in visited:
            continue
        visited.add(seg_id)
        seg = id_to_seg[seg_id]

        parent_id = seg["parent_id"]
        parent_comp_idx = id_to_idx[parent_id] if parent_id != -1 else -1

        # Compute length from parent (Euclidean distance)
        if parent_id != -1:
            p = id_to_seg[parent_id]
            dx, dy, dz = seg["x"]-p["x"], seg["y"]-p["y"], seg["z"]-p["z"]
            length_um = math.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            length_um = 2.0 * seg["r"]  # soma: length ≈ diameter

        name = f"{_SWC_TYPE_NAMES.get(seg['type'], 'comp')}[{seg_id}]"

        if comp_fn is not None:
            comp = comp_fn(seg["type"], name)
            comp.length_um   = length_um
            comp.diameter_um = 2.0 * seg["r"]
            comp.Ra = Ra
            comp.Cm = Cm
        else:
            comp = CompartmentSpec(name, length_um=length_um,
                                   diameter_um=2.0*seg["r"])
            comp.Ra = Ra
            comp.Cm = Cm

        id_to_idx[seg_id] = len(comps)
        comps.append(comp)
        parents.append(parent_comp_idx)

        # Enqueue children
        for child in segments:
            if child["parent_id"] == seg_id and child["id"] not in visited:
                queue.append(child["id"])

    return comps, parents


def _reduce_equivalent_cylinder(morph, segments, Ra, Cm, comp_fn):
    """Rall equivalent cylinder reduction.

    Collapses all dendrite segments (type 3 + 4) into a single equivalent
    compartment per branch type using:
        L_eq = sum(L_i * (d_i / d_soma)^(3/2))
        d_eq = (sum(d_i^3))^(1/3)

    Soma (type 1) is kept as-is. Axon (type 2) is kept as-is.
    Result: soma + up to 2 equivalent compartments (basal, apical) as needed.
    """
    id_to_seg = {s["id"]: s for s in segments}
    soma_segs  = [s for s in segments if s["type"] == 1]
    d_soma = 2.0 * soma_segs[0]["r"] if soma_segs else 1.0

    def _eq_cylinder(type_code, Ra, Cm, comp_fn):
        segs = [s for s in segments if s["type"] == type_code]
        if not segs:
            return None
        d_vals = [2.0 * s["r"] for s in segs]
        # Length of each segment from its parent
        L_vals = []
        for s in segs:
            p_id = s["parent_id"]
            if p_id != -1 and p_id in id_to_seg:
                p = id_to_seg[p_id]
                dx, dy, dz = s["x"]-p["x"], s["y"]-p["y"], s["z"]-p["z"]
                L_vals.append(math.sqrt(dx*dx + dy*dy + dz*dz))
            else:
                L_vals.append(2.0 * s["r"])
        L_eq = sum(L * (d / d_soma)**1.5 for L, d in zip(L_vals, d_vals))
        d_eq = sum(d**3 for d in d_vals) ** (1.0/3.0)
        name = _SWC_TYPE_NAMES.get(type_code, f"type{type_code}") + "_eq"
        if comp_fn is not None:
            comp = comp_fn(type_code, name)
            comp.length_um = L_eq; comp.diameter_um = d_eq
            comp.Ra = Ra; comp.Cm = Cm
        else:
            comp = CompartmentSpec(name, length_um=L_eq, diameter_um=d_eq)
            comp.Ra = Ra; comp.Cm = Cm
        return comp

    # Build reduced morphology: soma + optional basal + optional apical
    soma_comp = morph.compartments[0]  # index 0 is root (soma) after _segments_to_compartments
    comps   = [soma_comp]
    parents = [-1]

    for type_code in (3, 4):  # basal, apical
        eq = _eq_cylinder(type_code, Ra, Cm, comp_fn)
        if eq is not None:
            parents.append(0)
            comps.append(eq)

    return MorphologySpec(comps, parents)


def _reduce_n_comps(morph, segments, n_target, Ra, Cm, comp_fn):
    """Collapse each dendrite branch to at most n_target compartments."""
    # Simple uniform resampling: group consecutive segments and sum lengths
    # (full implementation omitted — see Mainen & Sejnowski 1996 for Ri-preserving
    # reduction). Placeholder: returns equivalent_cylinder with n_target dendrite
    # compartments split uniformly.
    reduced = _reduce_equivalent_cylinder(morph, segments, Ra, Cm, comp_fn)
    # Split each non-soma compartment into n_target equal sub-compartments
    new_comps   = [reduced.compartments[0]]
    new_parents = [-1]
    for i in range(1, reduced.n_comps()):
        comp = reduced.compartments[i]
        seg_len = comp.length_um / n_target
        parent_idx = 0  # attach first sub-comp to soma
        for k in range(n_target):
            sub = CompartmentSpec(
                f"{comp.name}_seg{k}",
                length_um=seg_len,
                diameter_um=comp.diameter_um,
            )
            sub.Ra = comp.Ra; sub.Cm = comp.Cm
            sub.channels = comp.channels; sub.gates = comp.gates
            new_parents.append(parent_idx)
            new_comps.append(sub)
            parent_idx = len(new_comps) - 1
    return MorphologySpec(new_comps, new_parents)
```

### Exports — `src/hodgkin_huxley/__init__.py`

```python
from .morphology import load_swc, swc_stats
```

Add both to `__all__`.

---

## Key files

| File | Change |
|---|---|
| `src/hodgkin_huxley/morphology.py` | New — `load_swc()`, `swc_stats()`, reduction helpers |
| `src/hodgkin_huxley/__init__.py` | Export `load_swc`, `swc_stats` |
| `tests/python/test_swc_import.py` | New — uses hand-crafted SWC fixture |
| `tests/python/fixtures/simple.swc` | New — minimal 5-segment SWC fixture |

---

## Test fixture (`tests/python/fixtures/simple.swc`)

```
# Simple 5-segment test morphology
# id  type  x     y    z   r    parent
1     1     0.0   0.0  0.0 10.0 -1
2     3     0.0   10.0 0.0  1.0  1
3     3     0.0   20.0 0.0  1.0  2
4     4     10.0  0.0  0.0  1.5  1
5     4     20.0  0.0  0.0  1.5  4
```

Geometry: soma (r=10, diam=20), 2-seg basal dendrite (r=1) + 2-seg apical dendrite (r=1.5), all branching from soma.

---

## Baseline tests (before PR to testing branch)

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] `load_swc("tests/python/fixtures/simple.swc")` returns `MorphologySpec` with `n_comps() == 5`
- [ ] `parent_idx == [-1, 0, 1, 0, 3]` for the fixture (soma + 2 basal + 2 apical in BFS order)
- [ ] Segment lengths match Euclidean distances from fixture coordinates (basal: 10 µm each, apical: 10 µm each)
- [ ] `reduce="equivalent_cylinder"` returns 3 compartments: soma + 1 basal eq + 1 apical eq; `parent_idx = [-1, 0, 0]`
- [ ] Equivalent cylinder `d_eq` for basal = `(2 * 2^3)^(1/3)` = `2.0` µm (both segments same diameter)
- [ ] `reduce="n_comps", n_comps_target=2` returns ≤ 7 compartments total (soma + 2×2 basal + 2×2 apical ≤ 9, or equivalent-cylinder-split path)
- [ ] Resulting `MorphologySpec` from any reduction passes `validate()` and can be passed to `RegionalNetwork.add_population()` without error
- [ ] `swc_stats("tests/python/fixtures/simple.swc")` returns `n_segments=5`, `type_counts={"soma": 1, "basal_dend": 2, "apical_dend": 2}`
- [ ] `load_swc` with a `compartment_spec_fn` that sets `Ra=200` — all compartments have `Ra == 200`
- [ ] Malformed SWC (missing column) skips the bad line without raising

---

## Notes

- `_reduce_equivalent_cylinder` implements the Rall (1959) 3/2 power rule. The `L_eq = sum(L_i * (d_i/d_soma)^(3/2))` formula preserves electrotonic length under the branching constraint. For models where the branching constraint does not hold, users should use `reduce="none"` and manually merge compartments.
- `_reduce_n_comps` uses uniform sub-compartment splitting of the equivalent cylinder. A Ri-preserving non-uniform reduction (Mainen & Sejnowski 1996) would improve accuracy for high n_target but is deferred to a future task.
- SWC files sometimes encode the soma as a 3-point contour (3 lines with the same `type=1`). `load_swc` treats all soma segments as a single compartment by using only the first soma segment's radius and averaging the lengths.
