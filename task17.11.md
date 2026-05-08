# Task 17.11: CUDA Correctness Tests

**Role:** Test engineer  
**Status:** Not started  
**Depends on:** 17.4 (Python API), 17.6, 17.7, 17.8, 17.9, 17.10 (full CUDA stack)  
**Unlocks:** 17.12 (benchmarks run after correctness confirmed)

---

## What to implement

### `tests/python/test_cuda.py` (new file)

All tests are guarded with:
```python
import pytest
import hodgkin_huxley as hh
cuda = pytest.importorskip("hodgkin_huxley", reason="cuda tests")
pytestmark = pytest.mark.skipif(
    not hh.cuda_is_available(), reason="No CUDA device available"
)
```

---

### Section A — Device API (`TestDeviceAPI`)

These tests require task 17.4 (Python API) merged. The basic `Device` struct tests already live in `tests/python/test_device.py` (task 17.1). This section extends them with network-level CUDA dispatch tests.

| Test | What it checks |
|---|---|
| `test_device_cpu_str` | `hh.Device.cpu().__repr__() == "cpu"` |
| `test_device_cuda_str` | `hh.Device.cuda(0).__repr__() == "cuda:0"` |
| `test_device_parse` | `hh.device("cuda:0") == hh.Device.cuda(0)` |
| `test_cuda_device_count_positive` | `hh.cuda_device_count() >= 1` |
| `test_no_cuda_build_count` | (non-CUDA build only) `cuda_device_count() == 0` |
| `test_to_cuda_no_crash` | `rn.to(hh.Device.cuda(0))` on a minimal network; no exception |
| `test_to_cpu_roundtrip` | `.to(cuda).to(cpu)` → `current_device() == Device.cpu()` |

---

### Section B — HH + Iz Pool Correctness (`TestCudaPoolCorrectness`)

For each test, build a small single-population network, simulate 200ms, compare CUDA vs CPU V trace.

| Test | Network | Tolerance |
|---|---|---|
| `test_hh_single_pop_vs_cpu` | 20 HH neurons, no synapses, I=10 µA/cm² | `np.allclose(atol=1e-8)` |
| `test_iz_single_pop_vs_cpu` | 20 Izhikevich neurons, RS type, no synapses | `np.allclose(atol=1e-8)` |
| `test_hh_with_synapses_vs_cpu` | 2×20 HH pops, AMPA connections, delay=5ms | `np.allclose(atol=1e-6)` |
| `test_iz_with_synapses_vs_cpu` | 2×20 Iz pops, GABA connections, delay=3ms | `np.allclose(atol=1e-6)` |

Tolerance is relaxed for networks with synapses due to order-of-operations differences in float accumulation.

---

### Section C — ComposablePool Correctness (`TestCudaComposablePool`)

| Test | Network | Notes |
|---|---|---|
| `test_composable_standard_gates_vs_cpu` | Single composable pop with Na+K+leak channels (Boltzmann inf, pattern-matched tau) | `atol=1e-8` |
| `test_composable_custom_gate_vs_cpu` | Gate with non-standard SymPy expression (if CUSTOM_EXPR supported; skip if not) | `atol=1e-6` |
| `test_intracellular_calcium_vs_cpu` | Composable pop with calcium dynamics, driven by I_Ca | `atol=1e-7` |
| `test_modulation_synapse_g_vs_cpu` | SYNAPSE_G modulation from calcium; check g_syn_buf matches | `atol=1e-6` |

---

### Section D — Recording Correctness (`TestCudaRecording`)

| Test | What it checks |
|---|---|
| `test_V_recording_shape` | `result["pop"]["V"].shape == (n_neurons, n_rec_steps)` |
| `test_V_recording_values_vs_cpu` | V traces from CUDA recording match CPU traces (`atol=1e-6`) |
| `test_gate_recording_vs_cpu` | Gate state recordings match |
| `test_calcium_recording_vs_cpu` | Calcium recordings match |
| `test_recording_at_intervals` | `record_every=10` produces correct step count and correct timestep values |

---

### Section E — Robustness (`TestCudaRobustness`)

| Test | What it checks |
|---|---|
| `test_migrate_to_device` | `.to(cuda)` then re-simulate; no state corruption |
| `test_to_cpu_after_cuda_simulate` | Simulate on CUDA, `.to(cpu)`, re-simulate on CPU; traces match |
| `test_no_cuda_raises_clear_error` | (non-CUDA build) `rn.to(Device.cuda(0))` raises `RuntimeError` |
| `test_large_network_no_oom` | 500 HH + 500 Iz neurons, 500 synapses each, 1000ms; no CUDA OOM |
| `test_plasticity_cuda_fallback` | Network with STDP synapses, `.to(cuda)`; either works or raises `NotImplementedError` (no silent wrong answer) |

---

## Baseline tests (before PR to testing branch)

Requires: all of 17.4–17.10 merged.

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] `pytest tests/python/test_cuda.py -v` on a CUDA machine — all sections pass with no skip beyond `pytestmark`
- [ ] On non-CUDA build: `pytest tests/python/test_cuda.py -v` — all tests correctly skip (no errors)

---

## Running

```bash
# Skip on CPU-only CI automatically
pytest tests/python/test_cuda.py -v

# Force run if CUDA present
CUDA_VISIBLE_DEVICES=0 pytest tests/python/test_cuda.py -v

# Full regression must still pass
pytest tests/python/ -v --ignore=tests/python/test_cuda.py
```

---

## Key files

| File | Change |
|---|---|
| `tests/python/test_cuda.py` | New — all CUDA correctness tests |
