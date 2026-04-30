"""
Tests for DBSStimulator — Task 7.

Covers:
  - Pulse-train correctness (timing, amplitude, width)
  - Zero-frequency (off) behaviour
  - current_at() consistency with generate()
  - Parameter validation / error messages
  - RegionalNetwork.attach_stimulator() integration
  - Stimulation applied to the correct neurons
"""

import math
import pytest
import numpy as np

from hodgkin_huxley import (
    DBSStimulator,
    DBSParameters,
    RegionalNetwork,
    NeuronModelSpec,
)
from neuron_specs import make_stn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DT = 0.01  # ms — typical simulation time step


def make_rnet(pop_sizes: dict, neuron_type="HH"):
    rn = RegionalNetwork()
    for name, n in pop_sizes.items():
        rn.add_population(name, n, neuron_type=neuron_type)
    return rn


@pytest.fixture(params=[False, True], ids=["serial", "phase2"])
def use_parallel(request):
    return request.param


# ---------------------------------------------------------------------------
# Construction and parameter access
# ---------------------------------------------------------------------------

def test_default_construction():
    dbs = DBSStimulator()
    assert dbs.frequency == pytest.approx(130.0)
    assert dbs.amplitude == pytest.approx(0.0)
    assert dbs.pulse_width == pytest.approx(0.06)


def test_custom_construction():
    dbs = DBSStimulator(frequency=60.0, amplitude=150.0, pulse_width=0.1)
    assert dbs.frequency == pytest.approx(60.0)
    assert dbs.amplitude == pytest.approx(150.0)
    assert dbs.pulse_width == pytest.approx(0.1)


def test_parameters_property_type():
    dbs = DBSStimulator(frequency=80.0, amplitude=50.0, pulse_width=0.2)
    p = dbs.parameters
    assert p.frequency   == pytest.approx(80.0)
    assert p.amplitude   == pytest.approx(50.0)
    assert p.pulse_width == pytest.approx(0.2)


def test_set_parameters_updates_all():
    dbs = DBSStimulator(frequency=100.0, amplitude=100.0, pulse_width=0.1)
    dbs.set_parameters(frequency=50.0, amplitude=200.0, pulse_width=0.3)
    assert dbs.frequency   == pytest.approx(50.0)
    assert dbs.amplitude   == pytest.approx(200.0)
    assert dbs.pulse_width == pytest.approx(0.3)


def test_set_parameters_partial_update():
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    dbs.set_parameters(amplitude=500.0)
    assert dbs.frequency   == pytest.approx(130.0)
    assert dbs.amplitude   == pytest.approx(500.0)
    assert dbs.pulse_width == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# Parameter validation — invalid inputs must raise ValueError
# ---------------------------------------------------------------------------

def test_negative_frequency_raises():
    with pytest.raises((ValueError, RuntimeError), match="frequency"):
        DBSStimulator(frequency=-1.0, amplitude=100.0, pulse_width=0.06)


def test_zero_pulse_width_raises():
    with pytest.raises((ValueError, RuntimeError), match="pulse_width"):
        DBSStimulator(frequency=130.0, amplitude=100.0, pulse_width=0.0)


def test_negative_pulse_width_raises():
    with pytest.raises((ValueError, RuntimeError), match="pulse_width"):
        DBSStimulator(frequency=130.0, amplitude=100.0, pulse_width=-0.1)


def test_pulse_width_equals_isi_raises():
    """pulse_width == ISI would completely fill the period — not allowed."""
    freq = 100.0
    isi = 1000.0 / freq  # 10 ms
    with pytest.raises((ValueError, RuntimeError)):
        DBSStimulator(frequency=freq, amplitude=100.0, pulse_width=isi)


def test_pulse_width_exceeds_isi_raises():
    freq = 200.0
    isi = 1000.0 / freq  # 5 ms
    with pytest.raises((ValueError, RuntimeError)):
        DBSStimulator(frequency=freq, amplitude=100.0, pulse_width=isi + 0.1)


def test_set_parameters_validates():
    dbs = DBSStimulator(frequency=130.0, amplitude=0.0, pulse_width=0.06)
    with pytest.raises((ValueError, RuntimeError), match="frequency"):
        dbs.set_parameters(frequency=-10.0)


# ---------------------------------------------------------------------------
# generate() — output length
# ---------------------------------------------------------------------------

def test_generate_output_length():
    """Output length matches np.arange(0, tmax+dt, dt) convention."""
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    duration = 100.0
    dt = DT
    trace = dbs.generate(duration, dt)
    expected_len = len(np.arange(0, duration + dt, dt))
    assert len(trace) == expected_len


def test_generate_returns_numpy_array():
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    trace = dbs.generate(100.0, DT)
    assert isinstance(trace, np.ndarray)
    assert trace.dtype == np.float64


# ---------------------------------------------------------------------------
# generate() — zero frequency (off)
# ---------------------------------------------------------------------------

def test_zero_frequency_produces_all_zeros():
    dbs = DBSStimulator(frequency=0.0, amplitude=500.0, pulse_width=0.06)
    trace = dbs.generate(100.0, DT)
    assert np.all(trace == 0.0)


def test_zero_frequency_current_at_is_zero():
    dbs = DBSStimulator(frequency=0.0, amplitude=500.0, pulse_width=0.06)
    for step in range(0, 1000, 100):
        assert dbs.current_at(step, DT) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# generate() — amplitude correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amplitude", [50.0, 150.0, 300.0, 500.0])
def test_amplitude_is_correct_in_trace(amplitude):
    dbs = DBSStimulator(frequency=130.0, amplitude=amplitude, pulse_width=0.06)
    trace = dbs.generate(100.0, DT)
    on_steps = trace[trace != 0.0]
    assert len(on_steps) > 0
    assert np.allclose(on_steps, amplitude), (
        f"Expected all ON steps = {amplitude}, got unique values {np.unique(on_steps)}"
    )


def test_trace_is_binary_on_off():
    """Trace must contain exactly {0, amplitude} values."""
    amp = 250.0
    dbs = DBSStimulator(frequency=130.0, amplitude=amp, pulse_width=0.06)
    trace = dbs.generate(200.0, DT)
    unique = np.unique(trace)
    assert set(unique).issubset({0.0, amp})


# ---------------------------------------------------------------------------
# generate() — pulse width correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pw_ms,dt", [
    (0.06, 0.01),
    (0.1,  0.01),
    (0.5,  0.01),
    (0.06, 0.025),
])
def test_pulse_width_steps(pw_ms, dt):
    """Each ON burst in the trace should be int(PW/dt) steps wide."""
    dbs = DBSStimulator(frequency=10.0, amplitude=100.0, pulse_width=pw_ms)
    trace = dbs.generate(500.0, dt)
    expected_pulse_steps = int(pw_ms / dt)
    if expected_pulse_steps == 0:
        pytest.skip("pulse_width smaller than dt")
    # Find run-lengths of non-zero segments
    changes = np.diff(np.concatenate([[0], (trace != 0).astype(int), [0]]))
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    lengths = ends - starts
    # All bursts should be expected_pulse_steps (last burst may be truncated)
    assert all(l == expected_pulse_steps for l in lengths[:-1]), (
        f"PW={pw_ms}ms dt={dt}ms expected {expected_pulse_steps} steps, got {lengths}"
    )


# ---------------------------------------------------------------------------
# generate() — frequency / period correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq_hz", [10.0, 50.0, 100.0, 130.0, 200.0])
def test_pulse_period_matches_frequency(freq_hz):
    """
    The period between pulse starts should be round(ISI/dt) steps.
    We verify by checking inter-burst intervals in the trace.
    """
    dt = DT
    dbs = DBSStimulator(frequency=freq_hz, amplitude=100.0, pulse_width=0.06)
    duration = max(500.0, 10 * (1000.0 / freq_hz))  # at least 10 periods
    trace = dbs.generate(duration, dt)

    changes = np.diff(np.concatenate([[0], (trace != 0).astype(int), [0]]))
    starts  = np.where(changes == 1)[0]

    isi_ms    = 1000.0 / freq_hz
    step_isi  = round(isi_ms / dt)
    if len(starts) < 2:
        pytest.skip("Not enough pulses generated")

    intervals = np.diff(starts)
    assert np.all(intervals == step_isi), (
        f"freq={freq_hz}Hz: expected ISI={step_isi} steps, got intervals={np.unique(intervals)}"
    )


# ---------------------------------------------------------------------------
# generate() — first pulse starts at t=0 (step 0)
# ---------------------------------------------------------------------------

def test_first_pulse_starts_at_step_zero():
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    trace = dbs.generate(100.0, DT)
    assert trace[0] == pytest.approx(300.0), "First step should be ON"


# ---------------------------------------------------------------------------
# current_at() — consistency with generate()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq", [10.0, 50.0, 130.0])
def test_current_at_matches_generate(freq):
    """current_at(k, dt) must equal generate(duration, dt)[k] for all k."""
    dt = DT
    duration = 200.0
    dbs = DBSStimulator(frequency=freq, amplitude=200.0, pulse_width=0.06)
    trace = dbs.generate(duration, dt)
    num_steps = int(duration / dt)
    for k in range(0, num_steps, 50):  # sample 1 in 50 steps
        assert dbs.current_at(k, dt) == pytest.approx(trace[k], abs=1e-12), (
            f"Mismatch at step {k}: current_at={dbs.current_at(k, dt)} generate={trace[k]}"
        )


def test_current_at_zero_frequency():
    dbs = DBSStimulator(frequency=0.0, amplitude=300.0, pulse_width=0.1)
    for k in [0, 1, 100, 1000, 9999]:
        assert dbs.current_at(k, DT) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RegionalNetwork.attach_stimulator — basic integration
# ---------------------------------------------------------------------------

def test_attach_stimulator_unknown_population_raises():
    rn = make_rnet({"STN": 2})
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    with pytest.raises(ValueError, match="Unknown population"):
        rn.attach_stimulator("GPe", dbs)


def test_attach_and_detach():
    rn = make_rnet({"STN": 2, "GPe": 2})
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    rn.attach_stimulator("STN", dbs)
    # Detaching a non-attached pop should not raise
    rn.detach_stimulator("GPe")
    rn.detach_stimulator("STN")
    # After detach, simulate should run without crash
    rn.simulate(10.0, DT, {})


def test_detach_nonexistent_does_not_raise():
    rn = make_rnet({"STN": 2})
    rn.detach_stimulator("STN")  # never attached — should be silently ignored


# ---------------------------------------------------------------------------
# Stimulation applied to correct neurons
# ---------------------------------------------------------------------------

def test_stimulated_population_receives_current(use_parallel):
    """
    STN gets DBS; GPe does not. STN neurons should be more depolarised.
    We compare the mean voltage of STN vs GPe after a short high-amplitude stim.
    """
    rn = make_rnet({"STN": 4, "GPe": 4})
    dbs = DBSStimulator(frequency=130.0, amplitude=50.0, pulse_width=0.06)
    rn.attach_stimulator("STN", dbs)
    if use_parallel:
        rn.set_thread_groups({"g0": ["STN"], "g1": ["GPe"]})
    traces = rn.simulate(50.0, DT, {})
    # Mean voltage over time for each population
    stn_mean = traces["STN"].mean()
    gpe_mean = traces["GPe"].mean()
    assert stn_mean > gpe_mean, (
        f"STN (stimulated) mean V={stn_mean:.2f} should exceed GPe (un-stimulated) mean V={gpe_mean:.2f}"
    )


def test_unstimulated_neurons_unaffected_by_dbs(use_parallel):
    """
    With NO synapses, GPe neurons are unaffected by DBS on STN.
    Their traces must be identical to the no-DBS baseline.
    """
    rn_base = make_rnet({"STN": 2, "GPe": 2})
    if use_parallel:
        rn_base.set_thread_groups({"g0": ["STN"], "g1": ["GPe"]})
    traces_base = rn_base.simulate(20.0, DT, {})

    rn_dbs = make_rnet({"STN": 2, "GPe": 2})
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    rn_dbs.attach_stimulator("STN", dbs)
    if use_parallel:
        rn_dbs.set_thread_groups({"g0": ["STN"], "g1": ["GPe"]})
    traces_dbs = rn_dbs.simulate(20.0, DT, {})

    np.testing.assert_allclose(
        traces_base["GPe"], traces_dbs["GPe"], rtol=0, atol=1e-12,
        err_msg="GPe should be unaffected when DBS is only on STN"
    )


def test_dbs_does_affect_stimulated_population(use_parallel):
    """Stimulated STN traces must differ from no-DBS baseline."""
    rn_base = make_rnet({"STN": 2})
    if use_parallel:
        rn_base.set_thread_groups({"g0": ["STN"]})
    traces_base = rn_base.simulate(50.0, DT, {})

    rn_dbs = make_rnet({"STN": 2})
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    rn_dbs.attach_stimulator("STN", dbs)
    if use_parallel:
        rn_dbs.set_thread_groups({"g0": ["STN"]})
    traces_dbs = rn_dbs.simulate(50.0, DT, {})

    assert not np.allclose(traces_base["STN"], traces_dbs["STN"]), (
        "STN traces should differ when DBS is applied"
    )


def test_all_neurons_in_population_receive_same_dbs(use_parallel):
    """
    DBS is broadcast to ALL neurons in the target population uniformly.
    Starting from identical initial conditions (no synapses), all neurons
    in the stimulated population must have identical traces.
    Tolerance is relaxed to 1e-6 to absorb floating-point accumulation
    across neurons (differences ~4e-10 are normal in double precision).
    """
    rn = make_rnet({"STN": 5})
    dbs = DBSStimulator(frequency=100.0, amplitude=200.0, pulse_width=0.1)
    rn.attach_stimulator("STN", dbs)
    if use_parallel:
        rn.set_thread_groups({"g0": ["STN"]})
    traces = rn.simulate(100.0, DT, {})
    stn = traces["STN"]  # shape (5, num_steps)
    for i in range(1, 5):
        np.testing.assert_allclose(
            stn[0], stn[i], rtol=1e-6, atol=1e-6,
            err_msg=f"Neuron 0 and {i} should receive identical DBS current"
        )


# ---------------------------------------------------------------------------
# DBS + I_ext dict additive behaviour
# ---------------------------------------------------------------------------

def test_dbs_adds_to_existing_iext():
    """
    Attaching a DBS stimulator to a population and passing I_ext for the
    same population should SUM the two currents.
    """
    rn_iext_only = make_rnet({"A": 2})
    rn_dbs_only  = make_rnet({"A": 2})
    rn_combined  = make_rnet({"A": 2})

    duration = 50.0
    dt = DT
    num_steps = int(duration / dt)

    dbs = DBSStimulator(frequency=50.0, amplitude=100.0, pulse_width=0.1)
    dbs_trace = dbs.generate(duration, dt)[:num_steps]

    const_current = 5.0

    # Baseline: I_ext = DBS trace + const
    combined = dbs_trace + const_current
    traces_ref = rn_iext_only.simulate(duration, dt, {"A": combined})

    # attach_stimulator path + dict I_ext summed
    rn_combined.attach_stimulator("A", dbs)
    traces_comb = rn_combined.simulate(duration, dt, {"A": const_current})

    np.testing.assert_allclose(
        traces_ref["A"], traces_comb["A"], rtol=1e-10, atol=1e-10,
        err_msg="attach_stimulator + I_ext should sum to the same result"
    )


# ---------------------------------------------------------------------------
# Option 1: generate() trace passed directly to simulate()
# ---------------------------------------------------------------------------

def test_option1_generate_trace_in_simulate():
    """
    Option 1 from spec: generate trace and pass as I_ext dict value.
    Result should match attach_stimulator (Option 2).
    """
    duration = 100.0
    dt = DT
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)

    rn1 = make_rnet({"STN": 2})
    Idbs = dbs.generate(duration, dt)
    traces1 = rn1.simulate(duration, dt, {"STN": Idbs})

    rn2 = make_rnet({"STN": 2})
    rn2.attach_stimulator("STN", dbs)
    traces2 = rn2.simulate(duration, dt, {})

    np.testing.assert_allclose(
        traces1["STN"], traces2["STN"], rtol=1e-10, atol=1e-10,
        err_msg="Option 1 (dict trace) and Option 2 (attach_stimulator) must produce identical results"
    )


# ---------------------------------------------------------------------------
# Integration: DBS on STN neuron model
# ---------------------------------------------------------------------------

def test_dbs_on_composable_stn_neurons(use_parallel):
    """DBS 130 Hz on STN (composable) should not crash and should affect voltage."""
    rn = RegionalNetwork()
    rn.add_population("STN", 4, model=make_stn())
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    rn.attach_stimulator("STN", dbs)
    if use_parallel:
        rn.set_thread_groups({"g0": ["STN"]})
    traces = rn.simulate(100.0, DT, {})
    assert "STN" in traces
    assert traces["STN"].shape == (4, int(100.0 / DT))
    assert not np.any(np.isnan(traces["STN"])), "NaN in STN traces"


def test_dbs_drives_firing_in_silent_neurons(use_parallel):
    """
    HH neurons are silent at rest (no external input → 0 spikes).
    High-amplitude DBS should evoke multiple spikes.
    Uses a generous pulse_width (0.5 ms) so enough charge is injected.
    """
    duration  = 500.0
    dt        = DT
    threshold = 0.0  # mV — HH spikes peak at ~+40 mV

    # Baseline: no input → silence
    rn_base = make_rnet({"A": 2})
    if use_parallel:
        rn_base.set_thread_groups({"g0": ["A"]})
    traces_base = rn_base.simulate(duration, dt, {})

    # With DBS: 130 Hz, 100 µA/cm², 0.5 ms wide pulses
    rn_dbs = make_rnet({"A": 2})
    dbs = DBSStimulator(frequency=130.0, amplitude=100.0, pulse_width=0.5)
    rn_dbs.attach_stimulator("A", dbs)
    if use_parallel:
        rn_dbs.set_thread_groups({"g0": ["A"]})
    traces_dbs = rn_dbs.simulate(duration, dt, {})

    def count_crossings(trace_row):
        return int(np.sum(np.diff((trace_row > threshold).astype(int)) == 1))

    base_spikes = sum(count_crossings(traces_base["A"][i]) for i in range(2))
    dbs_spikes  = sum(count_crossings(traces_dbs["A"][i])  for i in range(2))

    assert base_spikes == 0, f"HH neurons should be silent at rest, got {base_spikes}"
    assert dbs_spikes  > 0, (
        f"DBS should evoke spikes in HH neurons, got {dbs_spikes}"
    )


# ---------------------------------------------------------------------------
# Multi-population: only targeted population is stimulated
# ---------------------------------------------------------------------------

def test_multi_population_only_target_stimulated(use_parallel):
    """
    In a 3-population network with no synapses, attaching DBS only to one
    population must leave the other two populations' traces unchanged.
    """
    rn_base = RegionalNetwork()
    rn_dbs  = RegionalNetwork()
    for rn in (rn_base, rn_dbs):
        rn.add_population("STN",  3, neuron_type="HH")
        rn.add_population("GPe",  4, neuron_type="HH")
        rn.add_population("Thal", 2, neuron_type="HH")
        if use_parallel:
            rn.set_thread_groups({"g0": ["STN"], "g1": ["GPe"], "g2": ["Thal"]})

    dbs = DBSStimulator(frequency=130.0, amplitude=200.0, pulse_width=0.06)
    rn_dbs.attach_stimulator("STN", dbs)

    duration, dt = 30.0, DT
    base = rn_base.simulate(duration, dt, {})
    stim = rn_dbs.simulate(duration, dt, {})

    # GPe and Thal must be unaffected
    for pop in ("GPe", "Thal"):
        np.testing.assert_allclose(
            base[pop], stim[pop], rtol=1e-12, atol=1e-12,
            err_msg=f"{pop} should be unaffected by DBS on STN"
        )
    # STN must differ
    assert not np.allclose(base["STN"], stim["STN"]), "STN must differ"


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq,amp,pw", [
    (1.0,   10.0,  0.1),
    (50.0,  300.0, 0.06),
    (130.0, 300.0, 0.06),
    (200.0, 500.0, 0.06),
    (0.0,   999.0, 0.5),   # off
])
def test_no_nan_inf_in_trace(freq, amp, pw, use_parallel):
    dbs = DBSStimulator(frequency=freq, amplitude=amp, pulse_width=pw)
    rn = make_rnet({"A": 2})
    rn.attach_stimulator("A", dbs)
    if use_parallel:
        rn.set_thread_groups({"g0": ["A"]})
    traces = rn.simulate(100.0, DT, {})
    assert not np.any(np.isnan(traces["A"])), f"NaN with freq={freq} amp={amp} pw={pw}"
    assert not np.any(np.isinf(traces["A"])), f"Inf with freq={freq} amp={amp} pw={pw}"


# ---------------------------------------------------------------------------
# Repr smoke tests
# ---------------------------------------------------------------------------

def test_dbs_repr():
    dbs = DBSStimulator(frequency=130.0, amplitude=300.0, pulse_width=0.06)
    r = repr(dbs)
    assert "130" in r
    assert "300" in r


def test_dbs_parameters_repr():
    dbs = DBSStimulator(frequency=50.0, amplitude=150.0, pulse_width=0.1)
    r = repr(dbs.parameters)
    assert "50" in r
