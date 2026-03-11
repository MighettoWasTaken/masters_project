"""
flexibility_metric.py — API configuration space metric.

Measures library flexibility as the number of distinct choices available at
each independent decision point in the public API.  Each enum class represents
one axis of configurability; each member represents one variant the user can
select.

Two scores are reported:
  Breadth  = sum of variants across all axes  (additive — "how many options")
  Space    = product of variants across all axes  (multiplicative — "how many
             distinct model configurations are theoretically expressible")

The benchmark (simulate_network_model.py) scores 1 on every axis — all
choices are hardcoded with no variation points.

Usage
-----
    python benchmarks/flexibility_metric.py
"""

from __future__ import annotations

import enum
import importlib
import math


# ---------------------------------------------------------------------------
# Axes of configurability
# Each entry is (human label, enum class or list of named options).
# Add new rows here as the library grows — the metric updates automatically.
# ---------------------------------------------------------------------------

def _get_axes() -> list[tuple[str, list[str]]]:
    """
    Import the library and collect all configurable axes.
    Returns a list of (axis_label, [variant_name, ...]) pairs.
    """
    lib = importlib.import_module("hodgkin_huxley")

    def variants(enum_cls) -> list[str]:
        # pybind11 enums expose members via __members__, not via iteration
        return list(enum_cls.__members__.keys())

    axes: list[tuple[str, list[str]]] = []

    # --- Gate dynamics ---
    axes.append(("Gate update form",    variants(lib.GateUpdateForm)))
    axes.append(("Gate dependency",     variants(lib.GateDependency)))

    # --- Time-constant parameterisation ---
    axes.append(("Tau form",            variants(lib.TauForm)))

    # --- Alpha/beta rate functions ---
    axes.append(("Rate function form",  variants(lib.RateFuncForm)))

    # --- Synapse kinetics ---
    axes.append(("Synapse type",        variants(lib.SynapseSpecType)))
    axes.append(("Kinetic update form", variants(lib.KineticUpdateForm)))
    axes.append(("Kinetic current form",variants(lib.KineticCurrentForm)))

    # --- Network connectivity ---
    axes.append(("Connectivity pattern",variants(lib.ConnectivityPattern)))
    axes.append(("Weight distribution", variants(lib.WeightDistType)))

    # --- Neuron model types ---
    # HH and Izhikevich are fixed-form models; Composable is the open-ended builder
    axes.append(("Neuron model",        ["HodgkinHuxley", "Izhikevich", "Composable"]))

    return axes


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores(axes: list[tuple[str, list[str]]]) -> dict:
    counts = [(label, len(variants)) for label, variants in axes]
    breadth = sum(c for _, c in counts)
    space   = math.prod(c for _, c in counts)
    return {"axes": counts, "breadth": breadth, "space": space}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(scores: dict) -> None:
    axes    = scores["axes"]
    label_w = max(len(label) for label, _ in axes) + 2

    print("\nLibrary Configuration Space")
    print("===========================")
    print(f"{'Axis':<{label_w}} {'Variants':>8}")
    print("-" * (label_w + 10))
    for label, count in axes:
        print(f"{label:<{label_w}} {count:>8}")

    print("-" * (label_w + 10))
    print(f"{'BREADTH (sum)':<{label_w}} {scores['breadth']:>8}")
    print(f"{'SPACE (product)':<{label_w}} {scores['space']:>8,}")
    print(
        f"\n{scores['space']:,} theoretically distinct model configurations "
        f"across {len(axes)} structural axes of configurability."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(_here)
    sys.path.insert(0, os.path.join(_repo, "src"))

    try:
        axes = _get_axes()
    except ImportError as e:
        print(f"Could not import hodgkin_huxley: {e}")
        print("Run from the project root after installing the package.")
        return

    scores = compute_scores(axes)
    _print_report(scores)



if __name__ == "__main__":
    main()
