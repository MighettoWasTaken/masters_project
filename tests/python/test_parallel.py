"""
test_parallel.py — test suite for task16 parallelism.

Sections:
  A — TestPhase1OpenMP:               set_num_threads API, no-crash, correctness
  B — TestPhase2ThreadGroups:         set_thread_groups API stubs (Phase 2)
  C — TestDelayDecompositionCorrectness: serial vs parallel trace comparison (Phase 2)
  D — TestPhase1Speedup:              soft timing test (does not assert speedup)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import hodgkin_huxley as hh
from hodgkin_huxley import (
    NeuronModelSpec,
    RegionalNetwork,
    RecordingConfig,
    SynapseSpec,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DT = 0.05       # ms — coarser for speed, still resolves spikes
DURATION = 100.0  # ms — enough to establish firing patterns


def _two_pop_net(delay_ms: float = 5.0, n: int = 10, seed: int = 42) -> RegionalNetwork:
    """CTX→B-style: pop A drives pop B with given delay."""
    rn = RegionalNetwork()
    rn.add_population("A", n, model=NeuronModelSpec.hh_default())
    rn.add_population("B", n, model=NeuronModelSpec.hh_default())
    rn.connect(
        "A", "B", "random_sparse",
        weight=0.5, synapse=SynapseSpec.ampa(), delay=delay_ms,
        probability=0.5, seed=seed,
    )
    rn.connect(
        "B", "B", "random_sparse",
        weight=0.3, synapse=SynapseSpec.ampa(), delay=delay_ms,
        probability=0.3, seed=seed + 1,
    )
    return rn


def _run(rn: RegionalNetwork, I: float = 10.0) -> dict:
    return rn.simulate(
        DURATION, DT,
        {"A": I, "B": 0.0},
        record=RecordingConfig(["V"]),
    )


# ===========================================================================
# Section A — Phase 1 OpenMP
# ===========================================================================

class TestPhase1OpenMP:

    def test_set_num_threads_no_crash(self):
        rn = _two_pop_net()
        rn.set_num_threads(2)
        result = _run(rn)
        V = result["A"]["V"]
        assert np.all(np.isfinite(V)), "V trace contains non-finite values"

    def test_num_threads_getter(self):
        rn = _two_pop_net()
        rn.set_num_threads(4)
        assert rn.num_threads() == 4

    def test_num_threads_default_zero(self):
        rn = _two_pop_net()
        assert rn.num_threads() == 0

    def test_set_num_threads_zero_resets_to_auto(self):
        rn = _two_pop_net()
        rn.set_num_threads(4)
        rn.set_num_threads(0)
        assert rn.num_threads() == 0

    def test_serial_vs_openmp_identical(self):
        """Bit-identical traces: serial (threads=1) vs OpenMP (threads=4)."""
        rn_serial = _two_pop_net(delay_ms=5.0, n=8)
        rn_serial.set_num_threads(1)
        res_serial = _run(rn_serial)

        rn_omp = _two_pop_net(delay_ms=5.0, n=8)
        rn_omp.set_num_threads(4)
        res_omp = _run(rn_omp)

        for pop in ("A", "B"):
            np.testing.assert_array_equal(
                res_serial[pop]["V"], res_omp[pop]["V"],
                err_msg=f"Population {pop}: serial vs OpenMP traces diverged",
            )

    def test_serial_vs_openmp_repeated(self):
        """Run 10 times; all OpenMP traces must be identical to serial (race check)."""
        rn_ref = _two_pop_net(delay_ms=5.0, n=8)
        rn_ref.set_num_threads(1)
        ref = _run(rn_ref)["A"]["V"].copy()

        for _ in range(10):
            rn = _two_pop_net(delay_ms=5.0, n=8)
            rn.set_num_threads(4)
            trial = _run(rn)["A"]["V"]
            np.testing.assert_array_equal(
                ref, trial,
                err_msg="OpenMP trace diverged from serial reference (possible race)",
            )

    def test_finite_traces_multi_thread(self):
        """Sanity: all recorded voltages are finite with 4 threads."""
        rn = _two_pop_net(n=20)
        rn.set_num_threads(4)
        result = _run(rn, I=12.0)
        for pop in ("A", "B"):
            assert np.all(np.isfinite(result[pop]["V"])), \
                f"Non-finite V in population {pop}"

    def test_set_num_threads_on_network_directly(self):
        """set_num_threads and num_threads exposed on the underlying _Network binding."""
        net = hh._core._Network()
        net.set_num_threads(2)
        assert net.num_threads == 2  # property, not method


# ===========================================================================
# Section B — Phase 2 thread group API stubs
# ===========================================================================

class TestPhase2ThreadGroups:
    """API smoke tests — Phase 2 not yet implemented; verify raise / no-op."""

    def test_set_thread_groups_attribute_exists(self):
        rn = RegionalNetwork()
        assert hasattr(rn, "set_thread_groups"), \
            "RegionalNetwork.set_thread_groups() not found"

    def test_clear_thread_groups_attribute_exists(self):
        rn = RegionalNetwork()
        assert hasattr(rn, "clear_thread_groups"), \
            "RegionalNetwork.clear_thread_groups() not found"

    def test_has_thread_groups_attribute_exists(self):
        rn = RegionalNetwork()
        assert hasattr(rn, "has_thread_groups"), \
            "RegionalNetwork.has_thread_groups() not found"

    def test_has_thread_groups_default_false(self):
        rn = _two_pop_net()
        assert rn.has_thread_groups() is False

    def test_set_then_has_thread_groups(self):
        rn = _two_pop_net(delay_ms=5.0)
        rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})
        assert rn.has_thread_groups() is True

    def test_clear_thread_groups(self):
        rn = _two_pop_net(delay_ms=5.0)
        rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})
        rn.clear_thread_groups()
        assert rn.has_thread_groups() is False

    def test_set_none_clears(self):
        rn = _two_pop_net(delay_ms=5.0)
        rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})
        rn.set_thread_groups(None)
        assert rn.has_thread_groups() is False

    def test_invalid_population_raises(self):
        rn = _two_pop_net()
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            rn.set_thread_groups({"g0": ["nonexistent_population"]})

    def test_zero_delay_different_groups_raises(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5, model=NeuronModelSpec.hh_default())
        rn.add_population("B", 5, model=NeuronModelSpec.hh_default())
        rn.connect("A", "B", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.ampa(), delay=0.0)
        with pytest.raises((ValueError, RuntimeError)):
            rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})


# ===========================================================================
# Section C — Delay decomposition correctness (Phase 2)
# ===========================================================================

class TestDelayDecompositionCorrectness:
    """Serial vs parallel trace comparison — skipped until Phase 2 is implemented."""

    @staticmethod
    def _skip_if_not_implemented(rn: RegionalNetwork) -> None:
        """Skip test gracefully if set_thread_groups isn't functional yet."""
        try:
            rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})
        except NotImplementedError:
            pytest.skip("Phase 2 delay decomposition not yet implemented")

    def test_two_pop_serial_vs_parallel(self):
        rn_s = _two_pop_net(delay_ms=5.0, n=8)
        res_s = _run(rn_s)

        rn_p = _two_pop_net(delay_ms=5.0, n=8)
        self._skip_if_not_implemented(rn_p)
        res_p = _run(rn_p)

        for pop in ("A", "B"):
            np.testing.assert_allclose(
                res_s[pop]["V"], res_p[pop]["V"], atol=1e-10,
                err_msg=f"Population {pop}: serial vs parallel diverged",
            )

    def test_two_pop_repeated_n_times(self):
        """Run parallel 20 times; all must match serial (race condition probe)."""
        rn_s = _two_pop_net(delay_ms=5.0, n=8)
        ref = _run(rn_s)["A"]["V"].copy()

        for _ in range(20):
            rn = _two_pop_net(delay_ms=5.0, n=8)
            try:
                rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})
            except NotImplementedError:
                pytest.skip("Phase 2 not yet implemented")
            trial = _run(rn)["A"]["V"]
            np.testing.assert_allclose(
                ref, trial, atol=1e-10,
                err_msg="Parallel trace diverged from serial (race condition)",
            )

    def test_three_pop_chain(self):
        def _make_chain():
            rn = RegionalNetwork()
            for name in ("A", "B", "C"):
                rn.add_population(name, 8, model=NeuronModelSpec.hh_default())
            rn.connect("A", "B", "random_sparse", weight=0.5,
                       synapse=SynapseSpec.ampa(), delay=5.0,
                       probability=0.6, seed=10)
            rn.connect("B", "C", "random_sparse", weight=0.5,
                       synapse=SynapseSpec.ampa(), delay=5.0,
                       probability=0.6, seed=11)
            return rn

        rn_s = _make_chain()
        res_s = rn_s.simulate(DURATION, DT, {"A": 10.0, "B": 0.0, "C": 0.0},
                               record=RecordingConfig(["V"]))

        rn_p = _make_chain()
        try:
            rn_p.set_thread_groups({"g0": ["A"], "g1": ["B"], "g2": ["C"]})
        except NotImplementedError:
            pytest.skip("Phase 2 not yet implemented")
        res_p = rn_p.simulate(DURATION, DT, {"A": 10.0, "B": 0.0, "C": 0.0},
                               record=RecordingConfig(["V"]))

        for pop in ("A", "B", "C"):
            np.testing.assert_allclose(
                res_s[pop]["V"], res_p[pop]["V"], atol=1e-10,
                err_msg=f"Chain: population {pop} diverged",
            )

    def test_recording_completeness(self):
        rn = _two_pop_net(delay_ms=5.0, n=8)
        try:
            rn.set_thread_groups({"g0": ["A"], "g1": ["B"]})
        except NotImplementedError:
            pytest.skip("Phase 2 not yet implemented")
        result = _run(rn)
        assert "A" in result and "B" in result
        n_steps = int(DURATION / DT) + 1
        for pop in ("A", "B"):
            assert result[pop]["V"].ndim == 2
            assert result[pop]["V"].shape[0] == 8
            assert result[pop]["V"].shape[1] == pytest.approx(n_steps, abs=2)

    def test_large_network_no_crash(self):
        rn = RegionalNetwork()
        pops = [f"P{i}" for i in range(6)]
        for name in pops:
            rn.add_population(name, 20, model=NeuronModelSpec.hh_default())
        for i in range(len(pops) - 1):
            rn.connect(pops[i], pops[i + 1], "random_sparse",
                       weight=0.4, synapse=SynapseSpec.ampa(),
                       delay=5.0, probability=0.3, seed=100 + i)
        try:
            rn.set_thread_groups({f"g{i}": [p] for i, p in enumerate(pops)})
        except NotImplementedError:
            pytest.skip("Phase 2 not yet implemented")
        I_ext = {pops[0]: 10.0, **{p: 0.0 for p in pops[1:]}}
        result = rn.simulate(500.0, DT, I_ext, record=RecordingConfig(["V"]))
        for pop in pops:
            assert np.all(np.isfinite(result[pop]["V"])), \
                f"Non-finite V in population {pop}"


# ===========================================================================
# Section D — Phase 1 speedup (soft — timing only, no strict assertions)
# ===========================================================================

class TestPhase1Speedup:

    def test_time_serial_vs_openmp(self):
        """OpenMP should not be more than 2× slower than serial."""
        N, DUR = 30, 200.0

        rn_s = _two_pop_net(n=N)
        rn_s.set_num_threads(1)
        t0 = time.perf_counter()
        _run_dur(rn_s, DUR)
        t_serial = time.perf_counter() - t0

        rn_p = _two_pop_net(n=N)
        rn_p.set_num_threads(4)
        t0 = time.perf_counter()
        _run_dur(rn_p, DUR)
        t_omp = time.perf_counter() - t0

        # Safety valve: parallel should not be more than 2× slower than serial
        # (on most machines it will be faster, but CI cores vary)
        assert t_omp < t_serial * 2.0, (
            f"OpenMP ({t_omp:.3f}s) more than 2× slower than serial ({t_serial:.3f}s)"
        )


def _run_dur(rn: RegionalNetwork, duration: float) -> dict:
    return rn.simulate(
        duration, DT,
        {"A": 10.0, "B": 0.0},
        record=RecordingConfig(["V"]),
    )
