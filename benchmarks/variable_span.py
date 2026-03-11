"""
variable_span.py — Variable span readability metric.

For each variable in a Python source file, computes the distance (in lines)
between its definition site and each subsequent use site.  Reports mean span,
max span, and the number of variables with span > 0, for each file analysed.

Variable span is a surrogate for readability: a short span means a reader can
understand any given line without having to scroll far to find where its
operands were defined.  Lower mean/max span = more readable.

Usage
-----
    python benchmarks/variable_span.py [file1.py file2.py ...]

    # Default: compare library model against benchmark
    python benchmarks/variable_span.py

Algorithm
---------
1. Parse each file with ast.parse().
2. Walk all Name nodes, recording each Store context as a definition and each
   Load context as a use.
3. For each (file, name) pair, collect all definition lines and use lines.
4. For each use, find the most recent definition at or before that line
   (nearest enclosing definition) and compute span = use_line - def_line.
5. Spans of 0 (same-line use) are excluded — they contribute nothing to
   cognitive distance.
6. Report per-variable stats and file-level summary.
"""

from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# AST walk
# ---------------------------------------------------------------------------

def _collect_name_lines(tree: ast.AST) -> tuple[dict, dict]:
    """
    Returns:
        defs  : {name: sorted list of line numbers where name is assigned}
        uses  : {name: sorted list of line numbers where name is loaded}
    """
    defs: dict[str, list[int]] = defaultdict(list)
    uses: dict[str, list[int]] = defaultdict(list)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defs[node.id].append(node.lineno)
            elif isinstance(node.ctx, ast.Load):
                uses[node.id].append(node.lineno)

        # Also capture augmented assignment targets (x += 1)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            defs[node.target.id].append(node.lineno)

    # Sort
    for name in defs:
        defs[name].sort()
    for name in uses:
        uses[name].sort()

    return dict(defs), dict(uses)


def _build_code_line_map(source: str) -> dict[int, int]:
    """
    Build a mapping {raw_line_number: code_line_rank} that assigns a
    sequential index (1-based) only to non-blank, non-comment lines.
    Lines that are blank or pure comments are absent from the map.
    """
    mapping: dict[int, int] = {}
    code_rank = 0
    for raw_lineno, line in enumerate(source.splitlines(), start=1):
        s = line.strip()
        if s and not s.startswith("#"):
            code_rank += 1
            mapping[raw_lineno] = code_rank
    return mapping


def _spans_for_file(source: str) -> dict[str, list[int]]:
    """
    For each variable name that has at least one definition, compute the span
    (code_line_of_use - code_line_of_def) for every use that follows a
    definition.  Spans are measured in lines of code (blank lines and comments
    excluded), so they are directly comparable to LOC.

    Returns {name: [span, span, ...]} — only spans > 0 are included.
    """
    tree = ast.parse(source)
    defs, uses = _collect_name_lines(tree)
    code_line = _build_code_line_map(source)

    spans: dict[str, list[int]] = {}

    all_names = set(defs) & set(uses)   # only names with both def and use
    for name in all_names:
        def_lines = defs[name]
        use_lines = uses[name]

        name_spans: list[int] = []
        for use_line in use_lines:
            # Nearest definition at or before this use
            prior = [d for d in def_lines if d <= use_line]
            if not prior:
                continue
            nearest_def = max(prior)
            # Map both endpoints to code-line ranks; skip if either falls on
            # a blank/comment line (shouldn't happen for Name nodes, but safe)
            cl_def = code_line.get(nearest_def)
            cl_use = code_line.get(use_line)
            if cl_def is None or cl_use is None:
                continue
            span = cl_use - cl_def
            if span > 0:
                name_spans.append(span)

        if name_spans:
            spans[name] = name_spans

    return spans


# ---------------------------------------------------------------------------
# Per-file statistics
# ---------------------------------------------------------------------------

def _count_code_lines(source: str) -> int:
    """Count non-blank, non-comment lines (logical lines of code)."""
    count = 0
    for line in source.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            count += 1
    return count


def analyse_file(path: str) -> dict:
    source = Path(path).read_text(encoding="utf-8")
    spans = _spans_for_file(source)
    loc = _count_code_lines(source)

    all_spans: list[int] = [s for lst in spans.values() for s in lst]

    if not all_spans:
        return {
            "path": path,
            "loc": loc,
            "n_vars_with_span": 0,
            "n_span_observations": 0,
            "mean_span": 0.0,
            "median_span": 0.0,
            "max_span": 0,
            "p90_span": 0.0,
            "per_var": {},
        }

    all_spans_sorted = sorted(all_spans)
    n = len(all_spans_sorted)

    def _percentile(data: list[int], pct: float) -> float:
        idx = (len(data) - 1) * pct / 100.0
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        return data[lo] + (data[hi] - data[lo]) * (idx - lo)

    per_var = {
        name: {
            "n": len(lst),
            "mean": sum(lst) / len(lst),
            "max": max(lst),
        }
        for name, lst in spans.items()
    }

    mean_span   = sum(all_spans) / n
    median_span = _percentile(all_spans_sorted, 50)
    max_span    = max(all_spans)
    p90_span    = _percentile(all_spans_sorted, 90)

    return {
        "path": path,
        "loc": loc,
        "n_vars_with_span": len(spans),
        "n_span_observations": n,
        "mean_span":   mean_span,
        "median_span": median_span,
        "max_span":    max_span,
        "p90_span":    p90_span,
        # Normalised by LOC — guaranteed in [0, 1] because spans are now
        # measured in code lines and the max possible span is LOC - 1.
        "mean_span_norm":   mean_span   / loc if loc else 0.0,
        "median_span_norm": median_span / loc if loc else 0.0,
        "max_span_norm":    max_span    / loc if loc else 0.0,
        "p90_span_norm":    p90_span    / loc if loc else 0.0,
        "per_var": per_var,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    col_w = max(len(os.path.basename(r["path"])) for r in results) + 2

    # --- Absolute spans ---
    header = (
        f"{'File':<{col_w}} {'LOC':>6} {'Vars':>6} {'Obs':>7} "
        f"{'Mean':>8} {'Median':>8} {'P90':>8} {'Max':>7}"
    )
    print("Absolute span (lines)")
    print(header)
    print("-" * len(header))
    for r in results:
        name = os.path.basename(r["path"])
        print(
            f"{name:<{col_w}} {r['loc']:>6} {r['n_vars_with_span']:>6} "
            f"{r['n_span_observations']:>7} "
            f"{r['mean_span']:>8.1f} {r['median_span']:>8.1f} "
            f"{r['p90_span']:>8.1f} {r['max_span']:>7}"
        )

    # --- Normalised spans (span / LOC) ---
    print()
    header_n = (
        f"{'File':<{col_w}} {'LOC':>6} {'Vars':>6} {'Obs':>7} "
        f"{'Mean':>8} {'Median':>8} {'P90':>8} {'Max':>7}"
    )
    print("Normalised span (span / LOC)  — spans measured in code lines, so always in [0, 1]")
    print(header_n)
    print("-" * len(header_n))
    for r in results:
        name = os.path.basename(r["path"])
        print(
            f"{name:<{col_w}} {r['loc']:>6} {r['n_vars_with_span']:>6} "
            f"{r['n_span_observations']:>7} "
            f"{r['mean_span_norm']:>8.3f} {r['median_span_norm']:>8.3f} "
            f"{r['p90_span_norm']:>8.3f} {r['max_span_norm']:>7.3f}"
        )


def _print_top_vars(result: dict, top_n: int = 15) -> None:
    per_var = result["per_var"]
    if not per_var:
        return
    ranked = sorted(per_var.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:top_n]
    print(f"\n  Top {top_n} highest mean-span variables in {os.path.basename(result['path'])}:")
    print(f"  {'Name':<30} {'Uses':>5} {'Mean span':>10} {'Max span':>10}")
    print(f"  {'-'*30} {'-'*5} {'-'*10} {'-'*10}")
    for name, s in ranked:
        print(f"  {name:<30} {s['n']:>5} {s['mean']:>10.1f} {s['max']:>10}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

_DEFAULTS = [
    os.path.join(_HERE, "ctxbgth_model.py"),
    os.path.join(_HERE, "simulate_network_model.py"),
]


def main(paths: list[str]) -> None:
    results = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"Warning: {p} not found, skipping.", file=sys.stderr)
            continue
        results.append(analyse_file(p))

    if not results:
        print("No files to analyse.")
        return

    print("\nVariable Span Summary")
    print("=====================")
    print("Cols: Vars=variables with span>0, Obs=total (var, use) pairs measured\n")
    _print_summary(results)

    if len(results) == 2:
        a, b = results
        if a["mean_span"] > 0 and b["mean_span"] > 0:
            ratio      = a["mean_span"]      / b["mean_span"]
            ratio_norm = a["mean_span_norm"] / b["mean_span_norm"]
            na, nb = os.path.basename(a["path"]), os.path.basename(b["path"])
            print(
                f"\nMean span ratio          ({na} / {nb}): {ratio:.2f}x"
            )
            print(
                f"Mean span ratio (normed) ({na} / {nb}): {ratio_norm:.2f}x"
            )
            winner      = nb if ratio      > 1 else na
            winner_norm = nb if ratio_norm > 1 else na
            print(f"Lower absolute mean span:   {winner}")
            print(f"Lower normalised mean span: {winner_norm}")

    for r in results:
        _print_top_vars(r)

    print()


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else _DEFAULTS
    main(paths)
