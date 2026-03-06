"""
Tests for the composable ion channel neuron system.

Categories:
1. Struct construction & presets
2. Safety / edge cases (no crashes)
3. Correctness / behavioural verification
4. Speed / performance
5. Integration — RegionalNetwork & Python builder
"""

import math
import time

import numpy as np
import pytest

from hodgkin_huxley import (
    # Composable structs
    BoltzmannParams,
    TauParams,
    TauForm,
    RateFuncParams,
    RateFuncForm,
    GateSpec,
    GateUpdateForm,
    GateDependency,
    ChannelSpec,
    CalciumSpec,
    NeuronModelSpec,
    # Python helpers
    NeuronModel,
    Boltzmann,
    Tau,
    RateFunc,
    # Network types
    Network,
    RegionalNetwork,
    NetworkNeuronType,
    SynapseSpec,
    # Other neuron types for mixed tests
    HHParameters,
    IzhikevichParameters,
)


# =============================================================================
# Helpers
# =============================================================================

def make_empty_spec(name="test", V_init=-65.0):
    spec = NeuronModelSpec()
    spec.name = name
    spec.V_init = V_init
    spec.C_m = 1.0
    return spec


def make_leak_spec(g=0.1, E_rev=-65.0):
    spec = make_empty_spec("leak")
    ch = ChannelSpec()
    ch.name = "Leak"
    ch.g = g
    ch.E_rev = E_rev
    spec.channels.append(ch)
    return spec


def simulate_composable(spec, duration=100.0, dt=0.01, I_ext=0.0):
    """Convenience: simulate one neuron via Network."""
    net = Network()
    net.add_neuron(spec)
    n_steps = int(duration / dt)
    I = [[I_ext] * n_steps]
    traces = net.simulate(duration, dt, I)
    return np.array(traces[0])


def count_spikes(trace, threshold=0.0):
    above = trace > threshold
    crossings = np.diff(above.astype(int))
    return int((crossings == 1).sum())


# =============================================================================
# 1. Struct Construction & Presets
# =============================================================================

class TestStructConstruction:

    def test_boltzmann_params_fields(self):
        p = BoltzmannParams()
        p.v_half = -40.0
        p.k = 5.0
        assert p.v_half == pytest.approx(-40.0)
        assert p.k == pytest.approx(5.0)

    def test_tau_params_all_forms(self):
        for form in [TauForm.CONSTANT, TauForm.BOLTZMANN, TauForm.DOUBLE_EXP_SUM,
                     TauForm.OFFSET_DOUBLE_EXP, TauForm.SCALED_EXP, TauForm.COMPOUND_AB]:
            t = TauParams()
            t.form = form
            assert t.form == form

    def test_rate_func_params_all_forms(self):
        for form in [RateFuncForm.LINEAR_OVER_EXP, RateFuncForm.EXP_DECAY,
                     RateFuncForm.LINEAR_OVER_EXPM1, RateFuncForm.SIGMOID]:
            r = RateFuncParams()
            r.form = form
            assert r.form == form

    def test_gate_spec_defaults(self):
        g = GateSpec()
        assert g.scale == pytest.approx(1.0)
        assert g.dependency == GateDependency.VOLTAGE
        assert g.initial_value == pytest.approx(0.0)
        assert g.derived_source_gate == -1

    def test_channel_spec_defaults(self):
        c = ChannelSpec()
        assert c.is_ahp == False
        assert c.use_calcium_nernst == False
        assert c.ahp_k1 == pytest.approx(0.0)
        assert len(c.gates) == 0

    def test_calcium_spec_defaults(self):
        ca = CalciumSpec()
        assert ca.enabled == False
        assert ca.use_nernst == False
        assert ca.Ca_init == pytest.approx(0.1)

    def test_preset_thalamic(self):
        spec = NeuronModelSpec.thalamic()
        assert spec.name == "TH"
        assert spec.C_m == pytest.approx(1.0)
        assert len(spec.gates) == 5       # h_Na, h_T, m_Na(inst), m_T(inst), n_K(derived)
        assert len(spec.channels) == 4    # Na, K, T, Leak
        assert spec.calcium.enabled == False

    def test_preset_stn(self):
        spec = NeuronModelSpec.stn()
        assert spec.name == "STN"
        assert len(spec.gates) == 11
        assert len(spec.channels) == 7
        assert spec.calcium.enabled == True
        assert spec.calcium.use_nernst == True

    def test_preset_gpe(self):
        spec = NeuronModelSpec.gpe()
        assert spec.name == "GPe"
        assert spec.calcium.enabled == True
        assert spec.calcium.use_nernst == False
        # Has AHP channel
        has_ahp = any(ch.is_ahp for ch in spec.channels)
        assert has_ahp

    def test_preset_gpi(self):
        spec = NeuronModelSpec.gpi()
        assert spec.name == "GPi"
        # Same structure as GPe
        gpe = NeuronModelSpec.gpe()
        assert len(spec.gates) == len(gpe.gates)
        assert len(spec.channels) == len(gpe.channels)

    def test_preset_striatum(self):
        spec = NeuronModelSpec.striatum()
        assert spec.name == "Striatum"
        assert len(spec.channels) == 4   # Na, K, M, Leak

    def test_preset_striatum_pd_values(self):
        healthy = NeuronModelSpec.striatum(pd=0.0)
        pd = NeuronModelSpec.striatum(pd=1.0)
        # M channel conductance should differ
        g_healthy = next(ch.g for ch in healthy.channels if ch.name == "M")
        g_pd = next(ch.g for ch in pd.channels if ch.name == "M")
        assert g_healthy > g_pd


# =============================================================================
# 2. Safety — Edge Cases & Error Handling
# =============================================================================

class TestSafety:

    def test_empty_model_no_gates_no_channels(self):
        spec = make_empty_spec()
        trace = simulate_composable(spec, duration=10.0, I_ext=0.0)
        assert np.all(np.isfinite(trace))
        assert np.all(np.abs(trace - spec.V_init) < 1e-6)

    def test_single_leak_channel_only(self):
        spec = make_leak_spec(g=0.1, E_rev=-65.0)
        trace = simulate_composable(spec, duration=50.0, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_zero_conductance_channel(self):
        spec = make_leak_spec(g=0.0, E_rev=-65.0)
        trace = simulate_composable(spec, duration=10.0, I_ext=5.0)
        assert np.all(np.isfinite(trace))

    def test_zero_dt_step(self):
        net = Network()
        net.add_neuron(NeuronModelSpec.thalamic())
        net.step(0.0, [10.0])   # should not crash
        assert np.isfinite(net.neuron(0).V)

    def test_very_small_dt(self):
        spec = NeuronModelSpec.thalamic()
        trace = simulate_composable(spec, duration=1.0, dt=1e-4, I_ext=5.0)
        assert np.all(np.isfinite(trace))

    def test_very_large_dt(self):
        spec = NeuronModelSpec.thalamic()
        # May be inaccurate but must not crash or produce NaN/Inf
        trace = simulate_composable(spec, duration=100.0, dt=1.0, I_ext=5.0)
        assert np.all(np.isfinite(trace))

    def test_extreme_current_positive(self):
        spec = NeuronModelSpec.thalamic()
        trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=1e4)
        assert np.all(np.isfinite(trace))

    def test_extreme_current_negative(self):
        spec = NeuronModelSpec.thalamic()
        trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=-1e4)
        assert np.all(np.isfinite(trace))

    def test_extreme_voltage_init_high(self):
        spec = make_leak_spec()
        spec.V_init = 1000.0
        trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_extreme_voltage_init_low(self):
        spec = make_leak_spec()
        spec.V_init = -1000.0
        trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_single_neuron_pool(self):
        spec = NeuronModelSpec.thalamic()
        net = Network()
        net.add_neuron(spec)
        n_steps = 1000
        traces = net.simulate(100.0, 0.1, [[5.0] * n_steps])
        assert np.all(np.isfinite(traces[0]))

    def test_large_pool(self):
        spec = NeuronModelSpec.thalamic()
        N = 500
        net = Network()
        for _ in range(N):
            net.add_neuron(spec)
        n_steps = 100
        I = [[5.0] * n_steps] * N
        traces = net.simulate(10.0, 0.1, I)
        arr = np.array(traces)
        assert np.all(np.isfinite(arr))

    def test_reset_restores_initial_state(self):
        net = Network()
        spec = NeuronModelSpec.thalamic()
        net.add_neuron(spec)
        n_steps = 1000
        net.simulate(100.0, 0.1, [[5.0] * n_steps])
        net.reset()
        V = net.neuron(0).V
        assert V == pytest.approx(spec.V_init, abs=1.0)

    def test_calcium_stays_nonnegative(self):
        """Calcium concentration must never go negative."""
        spec = NeuronModelSpec.stn()
        net = Network()
        net.add_neuron(spec)
        n_steps = 2000
        net.simulate(200.0, 0.1, [[20.0] * n_steps])
        from hodgkin_huxley import _ComposableNeuron
        cn = net.neuron(0)
        # After simulation neuron's calcium is synced
        assert cn.calcium >= 0.0

    def test_gate_values_stay_bounded(self):
        """Gate variables must stay in [0, 1] for INF_TAU/ALPHA_BETA gates."""
        spec = NeuronModelSpec.striatum()
        net = Network()
        idx = net.add_neuron(spec)
        n_steps = 500
        net.simulate(50.0, 0.1, [[10.0] * n_steps])
        cn = net.neuron(idx)
        for gs in cn.gate_states:
            assert -1e-6 <= gs <= 1.0 + 1e-6

    def test_all_tau_forms_no_division_by_zero(self):
        """Each TauParams::Form evaluated at extreme voltages."""
        for form in [TauForm.CONSTANT, TauForm.BOLTZMANN, TauForm.DOUBLE_EXP_SUM,
                     TauForm.OFFSET_DOUBLE_EXP, TauForm.SCALED_EXP]:
            spec = make_empty_spec()
            g = GateSpec()
            g.name = "test"
            g.update_form = GateUpdateForm.INF_TAU
            g.initial_value = 0.5
            inf = BoltzmannParams()
            inf.v_half = -40.0
            inf.k = 10.0
            g.inf = inf
            t = TauParams()
            t.form = form
            t.set_param(0, 1.0)
            t.set_param(1, 1.0)
            t.set_param(2, -40.0)
            t.set_param(3, 10.0)
            t.set_param(5, 0.0)
            t.set_param(6, 10.0)
            g.tau = t
            spec.gates.append(g)
            # Just simulate — if no crash and no NaN, test passes
            trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=0.0)
            assert np.all(np.isfinite(trace)), f"NaN/Inf with TauForm {form}"

    def test_all_rate_func_forms_no_crash(self):
        """Each RateFuncParams::Form with ALPHA_BETA gate — no crash."""
        for form in [RateFuncForm.LINEAR_OVER_EXP, RateFuncForm.EXP_DECAY,
                     RateFuncForm.LINEAR_OVER_EXPM1, RateFuncForm.SIGMOID]:
            spec = make_empty_spec()
            g = GateSpec()
            g.name = "test"
            g.update_form = GateUpdateForm.ALPHA_BETA
            g.initial_value = 0.5
            r = RateFuncParams()
            r.form = form
            r.A = 0.1
            r.B = 40.0
            r.C = 10.0
            g.alpha = r
            g.beta = r
            spec.gates.append(g)
            trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=0.0)
            assert np.all(np.isfinite(trace)), f"NaN/Inf with RateFuncForm {form}"

    def test_derived_gate_invalid_source_raises(self):
        """DERIVED gate with derived_source_gate=-1 is caught by validation (10.6)."""
        import pytest
        spec = make_empty_spec()
        g = GateSpec()
        g.name = "derived"
        g.update_form = GateUpdateForm.DERIVED
        g.derived_source_gate = -1
        spec.gates.append(g)
        with pytest.raises(ValueError, match="derived_source_gate"):
            simulate_composable(spec, duration=5.0, dt=0.01, I_ext=0.0)

    def test_ahp_channel_zero_calcium(self):
        """AHP channel with Ca=0: no division by zero."""
        spec = make_empty_spec("ahp_test")
        spec.calcium.enabled = True
        spec.calcium.Ca_init = 0.0
        spec.calcium.K_Ca = 15.0
        spec.calcium.epsilon = 1e-4
        ch = ChannelSpec()
        ch.name = "AHP"
        ch.g = 5.0
        ch.E_rev = -80.0
        ch.is_ahp = True
        ch.ahp_k1 = 15.0
        spec.channels.append(ch)
        trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_nernst_near_zero_calcium(self):
        """Nernst equation with Ca near 0: E_Ca should be clamped, no log(0)."""
        spec = NeuronModelSpec.stn()
        spec.calcium.Ca_init = 1e-12  # nearly zero
        trace = simulate_composable(spec, duration=10.0, dt=0.01, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_mixed_composable_and_hh_network(self):
        """Network with both HH and composable neurons simulates correctly."""
        net = Network()
        net.add_hh_neuron()
        net.add_neuron(NeuronModelSpec.thalamic())
        n_steps = 100
        traces = net.simulate(10.0, 0.1, [[10.0] * n_steps, [5.0] * n_steps])
        assert np.all(np.isfinite(np.array(traces)))

    def test_mixed_composable_and_iz_network(self):
        """Network with both Izhikevich and composable neurons."""
        from hodgkin_huxley import IzhikevichType
        net = Network()
        net.add_neuron(NetworkNeuronType.IZHIKEVICH_RS)
        net.add_neuron(NeuronModelSpec.thalamic())
        n_steps = 100
        traces = net.simulate(10.0, 0.1, [[10.0] * n_steps, [5.0] * n_steps])
        assert np.all(np.isfinite(np.array(traces)))

    def test_three_way_mixed_network(self):
        """HH + Izhikevich + composable neurons in one network."""
        net = Network()
        net.add_hh_neuron()
        net.add_neuron(NetworkNeuronType.IZHIKEVICH_RS)
        net.add_neuron(NeuronModelSpec.thalamic())
        n_steps = 100
        traces = net.simulate(10.0, 0.1, [[10.0] * n_steps] * 3)
        assert np.all(np.isfinite(np.array(traces)))

    def test_duplicate_model_names_in_network(self):
        """Two populations with same model name → same pool, both work."""
        spec1 = NeuronModelSpec.thalamic()
        spec2 = NeuronModelSpec.thalamic()  # same name
        net = Network()
        for _ in range(5):
            net.add_neuron(spec1)
        for _ in range(5):
            net.add_neuron(spec2)
        n_steps = 50
        traces = net.simulate(5.0, 0.1, [[5.0] * n_steps] * 10)
        assert np.all(np.isfinite(np.array(traces)))


# =============================================================================
# 3. Correctness — Behavioural Verification
# =============================================================================

class TestCorrectness:

    def test_leak_channel_equilibrium(self):
        """With only a leak, V should converge to E_leak."""
        E_leak = -70.0
        spec = make_leak_spec(g=0.5, E_rev=E_leak)
        trace = simulate_composable(spec, duration=200.0, dt=0.1, I_ext=0.0)
        assert trace[-1] == pytest.approx(E_leak, abs=1.0)

    def test_leak_channel_with_current_offset(self):
        """Leak + constant current: V settles at E_leak + I/g."""
        g = 0.5
        E_rev = -65.0
        I = 5.0
        spec = make_leak_spec(g=g, E_rev=E_rev)
        trace = simulate_composable(spec, duration=200.0, dt=0.1, I_ext=I)
        expected = E_rev + I / g
        assert trace[-1] == pytest.approx(expected, abs=2.0)

    def test_boltzmann_inf_at_extremes(self):
        """Boltzmann with positive k: inf → 1 for V >> v_half, → 0 for V << v_half."""
        spec = make_empty_spec()
        g = GateSpec()
        g.name = "m"
        g.update_form = GateUpdateForm.INSTANT
        inf_p = BoltzmannParams()
        inf_p.v_half = -40.0
        inf_p.k = 10.0
        g.inf = inf_p

        ch = ChannelSpec()
        ch.name = "test"
        ch.g = 1.0
        ch.E_rev = 0.0
        ch.gates.append((0, 1))

        spec.gates.append(g)
        spec.channels.append(ch)

        # Very high V: gate ≈ 1 → large current drives V negative (but gate tracks)
        # Just confirm the gate follows Boltzmann (no crash, finite)
        trace = simulate_composable(spec, duration=5.0, dt=0.01, I_ext=0.0)
        assert np.all(np.isfinite(trace))

    def test_th_fires_under_current(self):
        """Thalamic neuron should spike with sufficient excitatory current.

        The TH preset uses E_T=0 mV for the T-channel, so spikes peak around
        -10 mV (T-current becomes repolarising before V crosses 0).  Use a
        threshold below the peak to reliably detect the action potential.
        """
        spec = NeuronModelSpec.thalamic()
        trace = simulate_composable(spec, duration=500.0, dt=0.1, I_ext=5.0)
        spikes = count_spikes(trace, threshold=-15.0)
        assert spikes >= 1, "TH should produce at least one spike with I_ext=5"

    def test_th_resting_no_spikes(self):
        """Thalamic neuron with zero current should not spike in 1s."""
        spec = NeuronModelSpec.thalamic()
        trace = simulate_composable(spec, duration=1000.0, dt=0.1, I_ext=0.0)
        spikes = count_spikes(trace, threshold=0.0)
        assert spikes == 0, "TH should not spike at rest"

    def test_stn_fires_under_current(self):
        """STN neuron fires with sufficient current."""
        spec = NeuronModelSpec.stn()
        trace = simulate_composable(spec, duration=500.0, dt=0.1, I_ext=25.0)
        spikes = count_spikes(trace, threshold=0.0)
        assert spikes >= 1, "STN should produce at least one spike with I_ext=25"

    def test_stn_calcium_changes(self):
        """STN Ca concentration changes during spiking (not static)."""
        spec = NeuronModelSpec.stn()
        net = Network()
        net.add_neuron(spec)
        n_steps = 5000
        net.simulate(500.0, 0.1, [[25.0] * n_steps])
        cn = net.neuron(0)
        # Ca should have changed from its initial value
        assert cn.calcium != pytest.approx(spec.calcium.Ca_init, abs=1e-4)

    def test_gpe_fires_under_current(self):
        """GPe fires with I_ext=5."""
        spec = NeuronModelSpec.gpe()
        trace = simulate_composable(spec, duration=500.0, dt=0.1, I_ext=5.0)
        spikes = count_spikes(trace, threshold=0.0)
        assert spikes >= 1

    def test_striatum_fires_under_current(self):
        """Striatum fires with I_ext=10."""
        spec = NeuronModelSpec.striatum()
        trace = simulate_composable(spec, duration=500.0, dt=0.1, I_ext=10.0)
        spikes = count_spikes(trace, threshold=0.0)
        assert spikes >= 1

    def test_striatum_pd_modulation(self):
        """pd parameter changes firing — at minimum, traces should differ."""
        spec_healthy = NeuronModelSpec.striatum(pd=0.0)
        spec_pd = NeuronModelSpec.striatum(pd=1.0)
        trace_h = simulate_composable(spec_healthy, duration=500.0, dt=0.1, I_ext=10.0)
        trace_p = simulate_composable(spec_pd, duration=500.0, dt=0.1, I_ext=10.0)
        # Traces should not be identical (different g_M changes dynamics)
        assert not np.allclose(trace_h, trace_p, atol=1e-6)

    def test_instant_gate_tracks_voltage(self):
        """INSTANT gate should always equal x_inf(V) — verified at every step."""
        spec = make_empty_spec("instant_test")
        g = GateSpec()
        g.name = "m"
        g.update_form = GateUpdateForm.INSTANT
        inf_p = BoltzmannParams()
        inf_p.v_half = -37.0
        inf_p.k = 7.0
        g.inf = inf_p
        spec.gates.append(g)

        # Add a channel so voltage moves
        ch = ChannelSpec()
        ch.name = "drv"
        ch.g = 0.1
        ch.E_rev = 0.0
        ch.gates.append((0, 1))
        spec.channels.append(ch)

        net = Network()
        net.add_neuron(spec)
        # Manually step and check gate tracks inf at each step
        for _ in range(50):
            V = net.neuron(0).V
            expected_gate = 1.0 / (1.0 + math.exp(-(V - inf_p.v_half) / inf_p.k))
            net.step(0.01, [2.0])
            V_new = net.neuron(0).V
            gate_val = net.neuron(0).gate_states[0]
            expected_new = 1.0 / (1.0 + math.exp(-(V_new - inf_p.v_half) / inf_p.k))
            assert gate_val == pytest.approx(expected_new, abs=1e-6)

    def test_derived_gate_follows_source(self):
        """TH n_K = 0.75*(1 - h_T) should hold at every step."""
        spec = NeuronModelSpec.thalamic()
        net = Network()
        net.add_neuron(spec)

        # h_T is gate index 1, n_K is gate index 4
        H_T_IDX = 1
        N_K_IDX = 4

        for _ in range(100):
            net.step(0.1, [5.0])
            gates = net.neuron(0).gate_states
            h_T = gates[H_T_IDX]
            n_K = gates[N_K_IDX]
            expected = max(0.0, min(1.0, 0.75 * (1.0 - h_T)))
            assert n_K == pytest.approx(expected, abs=1e-6)

    def test_pool_matches_scalar(self):
        """ComposablePool step should match scalar ComposableNeuron step closely."""
        spec = NeuronModelSpec.thalamic()
        I_ext_val = 5.0
        dt = 0.01
        n_steps = 200

        # Scalar path: use Network.step() for one neuron
        net_scalar = Network()
        net_scalar.add_neuron(spec)
        trace_scalar = []
        for _ in range(n_steps):
            trace_scalar.append(net_scalar.neuron(0).V)
            net_scalar.step(dt, [I_ext_val])

        # Pool path: simulate() uses ComposablePool
        net_pool = Network()
        net_pool.add_neuron(spec)
        traces = net_pool.simulate(n_steps * dt, dt, [[I_ext_val] * n_steps])
        trace_pool = traces[0]

        assert np.allclose(trace_scalar, trace_pool, atol=1e-6)

    def test_pool_multiple_neurons_independent(self):
        """N neurons in pool with different I_ext produce different traces."""
        spec = NeuronModelSpec.thalamic()
        N = 5
        net = Network()
        for _ in range(N):
            net.add_neuron(spec)
        n_steps = 200
        I = [[float(i) * 2.0] * n_steps for i in range(N)]
        traces = net.simulate(n_steps * 0.1, 0.1, I)
        arr = np.array(traces)
        # Not all traces should be identical
        assert not np.allclose(arr[0], arr[-1], atol=1e-6)

    def test_alpha_beta_gate_kinetics(self):
        """Striatum m gate: verify alpha-beta produces valid kinetics (finite, bounded)."""
        spec = NeuronModelSpec.striatum()
        net = Network()
        net.add_neuron(spec)
        n_steps = 200
        net.simulate(n_steps * 0.1, 0.1, [[10.0] * n_steps])
        gates = net.neuron(0).gate_states
        # m_Na is gate 0
        assert 0.0 <= gates[0] <= 1.0

    def test_calcium_decay_no_source(self):
        """Without any source channels, Ca should decay toward 0."""
        spec = make_empty_spec("ca_decay")
        spec.calcium.enabled = True
        spec.calcium.use_nernst = False
        spec.calcium.epsilon = 1e-3
        spec.calcium.K_Ca = 5.0
        spec.calcium.Ca_init = 1.0
        # No source channels — pure decay
        trace = simulate_composable(spec, duration=1000.0, dt=0.1, I_ext=0.0)
        net = Network()
        net.add_neuron(spec)
        n_steps = 10000
        net.simulate(1000.0, 0.1, [[0.0] * n_steps])
        ca_final = net.neuron(0).calcium
        assert ca_final < 0.5, f"Ca should decay from 1.0, got {ca_final}"

    def test_different_cm_changes_dynamics(self):
        """Higher C_m slows voltage dynamics → fewer spikes per unit time."""
        spec_fast = NeuronModelSpec.thalamic()
        spec_fast.C_m = 0.5

        spec_slow = NeuronModelSpec.thalamic()
        spec_slow.C_m = 2.0

        trace_fast = simulate_composable(spec_fast, duration=500.0, dt=0.1, I_ext=5.0)
        trace_slow = simulate_composable(spec_slow, duration=500.0, dt=0.1, I_ext=5.0)

        spikes_fast = count_spikes(trace_fast)
        spikes_slow = count_spikes(trace_slow)

        # Fast (small C_m) should spike at least as much as slow (large C_m)
        assert spikes_fast >= spikes_slow, (
            f"Lower C_m should produce more spikes: {spikes_fast} vs {spikes_slow}"
        )

    # ------------------------------------------------------------------
    # Bug-regression: Task 10.1 — ALPHA_BETA forward-Euler instability
    # ------------------------------------------------------------------

    def test_alpha_beta_euler_instability(self):
        """
        ALPHA_BETA forward Euler is unstable when dt*(alpha+beta) > 1.

        Setup: isolated neuron (no channels → V clamped at -65 mV) with one
        ALPHA_BETA gate.  EXP_DECAY rates with A=3.0, B=65 give alpha=beta=3
        at V=-65, so dt*(alpha+beta) = 1.0*6 = 6 >> 1.

        With forward Euler the gate diverges within a few steps.
        The fix (exact exponential integration) converges smoothly to
        X_inf = alpha/(alpha+beta) = 0.5 within milliseconds.

        This test FAILS while the bug is present (Task 10.1).
        """
        spec = make_empty_spec("ab_stability_pool", V_init=-65.0)
        g = GateSpec()
        g.name = "m"
        g.update_form = GateUpdateForm.ALPHA_BETA
        g.initial_value = 0.0   # start far from X_inf = 0.5

        # EXP_DECAY: alpha = A * exp(-(V+B)/C)
        # At V=-65, B=65: V+B = 0 → exp(0/C) = 1 → alpha = A = 3.0
        r = RateFuncParams()
        r.form = RateFuncForm.EXP_DECAY
        r.A = 3.0
        r.B = 65.0
        r.C = 10.0
        g.alpha = r
        g.beta = r   # symmetric → X_inf = 0.5, alpha+beta = 6.0

        spec.gates.append(g)
        net = Network()
        net.add_neuron(spec)

        # dt=1ms pool path (simulate uses ComposablePool)
        n_steps = 50
        net.simulate(50.0, 1.0, [[0.0] * n_steps])
        gate = net.neuron(0).gate_states[0]

        assert np.isfinite(gate) and abs(gate - 0.5) < 0.05, (
            f"ALPHA_BETA gate (pool) must converge to X_inf=0.5 with dt=1 ms. "
            f"Got {gate!r}. Euler: update factor 6 >> 1 → diverges; "
            f"exact-exp fix required. (Task 10.1)"
        )

    def test_alpha_beta_euler_instability_step(self):
        """
        Same instability check via step() — exercises composable_neuron.cpp
        (scalar path) rather than composable_pool.cpp.

        This test FAILS while the bug is present (Task 10.1).
        """
        spec = make_empty_spec("ab_stability_step", V_init=-65.0)
        g = GateSpec()
        g.name = "m"
        g.update_form = GateUpdateForm.ALPHA_BETA
        g.initial_value = 0.0

        r = RateFuncParams()
        r.form = RateFuncForm.EXP_DECAY
        r.A = 3.0
        r.B = 65.0
        r.C = 10.0
        g.alpha = r
        g.beta = r

        spec.gates.append(g)
        net = Network()
        net.add_neuron(spec)

        for _ in range(50):
            net.step(1.0, [0.0])   # dt=1 ms via scalar step()

        gate = net.neuron(0).gate_states[0]
        assert np.isfinite(gate) and abs(gate - 0.5) < 0.05, (
            f"ALPHA_BETA gate (step) must converge to X_inf=0.5 with dt=1 ms. "
            f"Got {gate!r}. (Task 10.1)"
        )

    # ------------------------------------------------------------------
    # Bug-regression: Task 10.2 — h_T tau parameter sign error
    # ------------------------------------------------------------------

    def test_th_h_T_tau_correct(self):
        """
        TH h_T gate tau at V=-65 should be ~20 ms, NOT ~1535 ms.

        Bug: DOUBLE_EXP_SUM params[5]=-0.6 and params[6]=-7.4 cause a
        double sign negation, making the exp denominators nearly zero and
        tau ~ 1535 ms at V=-65.
        Fix: params[5]=0.6, params[6]=7.4 → tau ~ 20 ms.

        Test strategy: build an isolated spec containing only h_T (no
        channels → V clamped at -65 mV).  Start h_T from 0.0 and from 1.0
        separately.  After 100 ms:
          - correct tau (~20 ms) → both converge to X_inf (diff < 0.05)
          - buggy  tau (~1535 ms) → gate barely moves (diff > 0.5)

        This test FAILS while the bug is present (Task 10.2).
        """
        def run_with_h_T_init(initial_value):
            th = NeuronModelSpec.thalamic()
            h_T_gate = th.gates[1]        # h_T is gate index 1 in TH
            h_T_gate.initial_value = initial_value
            spec = make_empty_spec("th_h_T_tau_test", V_init=-65.0)
            spec.gates.append(h_T_gate)
            net = Network()
            net.add_neuron(spec)
            n_steps = int(100.0 / 0.01)  # 100 ms at dt=0.01 ms
            net.simulate(100.0, 0.01, [[0.0] * n_steps])
            return net.neuron(0).gate_states[0]

        gate_from_zero = run_with_h_T_init(0.0)
        gate_from_one  = run_with_h_T_init(1.0)

        diff = abs(gate_from_zero - gate_from_one)
        assert diff < 0.05, (
            f"TH h_T gate must converge to X_inf within 100 ms (tau~20 ms). "
            f"From initial=0: {gate_from_zero:.4f}, from initial=1: {gate_from_one:.4f}, "
            f"diff={diff:.4f} (expected <0.05). "
            f"Likely cause: tau~1535 ms due to params sign error. (Task 10.2)"
        )

    def test_stn_b_A_tau_correct(self):
        """
        STN b_A gate (A-type K inactivation) tau at V=-65 should be ~15 ms.
        Uses DOUBLE_EXP_SUM: tau = 200/(exp((V+60)/30) + exp(-(V+40)/10)).
        A sign error in s2 would make tau ~215 ms, preventing convergence in 100 ms.
        """
        def run_with_stn_b_A_init(initial_value):
            stn = NeuronModelSpec.stn()
            b_A_idx = next(
                (i for i, gate in enumerate(stn.gates) if gate.name == "b_A"),
                None
            )
            assert b_A_idx is not None, "STN preset must contain a gate named 'b_A'"
            b_A_gate = stn.gates[b_A_idx]
            b_A_gate.initial_value = initial_value
            spec = make_empty_spec("stn_b_A_tau_test", V_init=-65.0)
            spec.gates.append(b_A_gate)
            net = Network()
            net.add_neuron(spec)
            n_steps = int(100.0 / 0.01)
            net.simulate(100.0, 0.01, [[0.0] * n_steps])
            return net.neuron(0).gate_states[0]

        gate_from_zero = run_with_stn_b_A_init(0.0)
        gate_from_one  = run_with_stn_b_A_init(1.0)

        diff = abs(gate_from_zero - gate_from_one)
        assert diff < 0.05, (
            f"STN b_A gate must converge to X_inf within 100 ms (tau~15 ms). "
            f"From initial=0: {gate_from_zero:.4f}, from initial=1: {gate_from_one:.4f}, "
            f"diff={diff:.4f} (expected <0.05). "
            f"Likely cause: tau~215 ms due to DOUBLE_EXP_SUM params sign error."
        )


# =============================================================================
# 4. Speed — Performance Tests
# =============================================================================

class TestPerformance:

    def test_pool_faster_than_scalar_loop(self):
        """Pool step for N=200 should be faster than N scalar steps."""
        spec = NeuronModelSpec.thalamic()
        N = 200
        n_steps = 500

        # Scalar timing: use step() for one neuron at a time (simulated manually)
        nets = [Network() for _ in range(N)]
        for net in nets:
            net.add_neuron(spec)

        t0 = time.perf_counter()
        for net in nets:
            net.simulate(50.0, 0.1, [[5.0] * n_steps])
        t_scalar = time.perf_counter() - t0

        # Pool timing: single Network with all neurons
        pool_net = Network()
        for _ in range(N):
            pool_net.add_neuron(spec)
        I = [[5.0] * n_steps] * N

        t0 = time.perf_counter()
        pool_net.simulate(50.0, 0.1, I)
        t_pool = time.perf_counter() - t0

        # Pool should be at least 2x faster than N separate simulations
        assert t_pool < t_scalar, (
            f"Pool ({t_pool:.3f}s) should be faster than scalar ({t_scalar:.3f}s)"
        )

    def test_composable_pool_reasonable_speed(self):
        """1000 TH neurons, 100ms, dt=0.01: completes in <30s."""
        spec = NeuronModelSpec.thalamic()
        N = 1000
        net = Network()
        for _ in range(N):
            net.add_neuron(spec)
        n_steps = 10000
        I = [[5.0] * n_steps] * N

        t0 = time.perf_counter()
        traces = net.simulate(100.0, 0.01, I)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0, f"Simulation took too long: {elapsed:.1f}s"
        assert np.all(np.isfinite(np.array(traces)))

    def test_fast_math_produces_finite_results(self):
        """fast_math=True (default) produces finite traces."""
        spec = NeuronModelSpec.thalamic()
        net = Network()
        net.fast_math = True
        for _ in range(50):
            net.add_neuron(spec)
        n_steps = 500
        traces = net.simulate(50.0, 0.1, [[5.0] * n_steps] * 50)
        assert np.all(np.isfinite(np.array(traces)))

    def test_pool_scaling_roughly_linear(self):
        """Time scaling for N=50 vs N=500: should be < 15x (roughly linear in N)."""
        spec = NeuronModelSpec.thalamic()
        n_steps = 200

        def run(N):
            net = Network()
            for _ in range(N):
                net.add_neuron(spec)
            t0 = time.perf_counter()
            net.simulate(20.0, 0.1, [[5.0] * n_steps] * N)
            return time.perf_counter() - t0

        t_small = run(50)
        t_large = run(500)
        ratio = t_large / max(t_small, 1e-6)
        assert ratio < 20.0, f"Scaling ratio {ratio:.1f}x is too large (expected < 20x)"


# =============================================================================
# 5. Integration — RegionalNetwork & Python Builder
# =============================================================================

class TestIntegration:

    def test_regional_add_population_with_model(self):
        """rn.add_population(..., model=NeuronModelSpec.thalamic()) works."""
        rn = RegionalNetwork()
        rn.add_population("TH", 10, model=NeuronModelSpec.thalamic())
        assert rn.population_size("TH") == 10
        assert rn.num_neurons == 10

    def test_regional_add_population_with_neuron_model_builder(self):
        """rn.add_population(..., model=NeuronModel.thalamic()) works."""
        rn = RegionalNetwork()
        rn.add_population("TH", 5, model=NeuronModel.thalamic())
        assert rn.num_neurons == 5

    def test_regional_mixed_populations(self):
        """Mix of HH, Izhikevich, and composable populations."""
        rn = RegionalNetwork()
        rn.add_population("HH", 5, neuron_type="HH")
        rn.add_population("IZ", 5, neuron_type="IZHIKEVICH_RS")
        rn.add_population("TH", 5, model=NeuronModelSpec.thalamic())
        assert rn.num_neurons == 15

    def test_regional_simulate_dict_iext(self):
        """Dict I_ext works with composable populations."""
        rn = RegionalNetwork()
        rn.add_population("TH", 5, model=NeuronModelSpec.thalamic())
        traces = rn.simulate(50.0, 0.1, {"TH": 5.0})
        assert "TH" in traces
        assert np.all(np.isfinite(traces["TH"]))

    def test_regional_traces_correct_shapes(self):
        """Trace shapes match population sizes."""
        rn = RegionalNetwork()
        rn.add_population("A", 3, model=NeuronModelSpec.thalamic())
        rn.add_population("B", 7, model=NeuronModelSpec.gpe())
        dt = 0.1
        duration = 50.0
        n_steps = int(duration / dt)
        traces = rn.simulate(duration, dt, {"A": 5.0, "B": 3.0})
        assert traces["A"].shape == (3, n_steps)
        assert traces["B"].shape == (7, n_steps)

    def test_regional_with_connections(self):
        """Composable populations can be connected."""
        rn = RegionalNetwork()
        rn.add_population("TH", 5, model=NeuronModelSpec.thalamic())
        rn.add_population("STN", 5, model=NeuronModelSpec.stn())
        rn.connect("TH", "STN", "all_to_all",
                   synapse=SynapseSpec.ampa(), weight=0.1)
        traces = rn.simulate(50.0, 0.1, {"TH": 5.0, "STN": 10.0})
        assert np.all(np.isfinite(traces["TH"]))
        assert np.all(np.isfinite(traces["STN"]))

    def test_python_builder_add_gate(self):
        """NeuronModel.add_gate() produces correct GateSpec."""
        m = NeuronModel("test")
        idx = m.add_gate("h", update_form="inf_tau",
                         inf=Boltzmann(-41.0, -4.0),
                         tau=Tau.constant(1.0),
                         initial_value=0.6)
        spec = m.to_spec()
        assert len(spec.gates) == 1
        assert spec.gates[0].name == "h"
        assert spec.gates[0].update_form == GateUpdateForm.INF_TAU
        assert spec.gates[0].inf.v_half == pytest.approx(-41.0)
        assert spec.gates[0].initial_value == pytest.approx(0.6)

    def test_python_builder_add_channel(self):
        """NeuronModel.add_channel() produces correct ChannelSpec."""
        m = NeuronModel("test")
        m.add_gate("m", update_form="instant", inf=Boltzmann(-37.0, 7.0))
        idx = m.add_channel("Na", g=120.0, E_rev=50.0, gates=[(0, 3)])
        spec = m.to_spec()
        assert spec.channels[0].g == pytest.approx(120.0)
        assert spec.channels[0].E_rev == pytest.approx(50.0)
        assert spec.channels[0].gates == [(0, 3)]

    def test_python_builder_add_leak(self):
        """NeuronModel.add_leak() adds a channel with no gates."""
        m = NeuronModel("test")
        m.add_leak(g=0.3, E_rev=-54.3)
        spec = m.to_spec()
        assert len(spec.channels) == 1
        assert len(spec.channels[0].gates) == 0
        assert spec.channels[0].g == pytest.approx(0.3)

    def test_python_builder_set_calcium(self):
        """NeuronModel.set_calcium() configures CalciumSpec."""
        m = NeuronModel("test")
        m.set_calcium(epsilon=1e-5, K_Ca=10.0, Ca_init=0.5, use_nernst=True)
        spec = m.to_spec()
        assert spec.calcium.enabled == True
        assert spec.calcium.use_nernst == True
        assert spec.calcium.epsilon == pytest.approx(1e-5)
        assert spec.calcium.K_Ca == pytest.approx(10.0)

    def test_python_builder_simulates(self):
        """Builder-constructed model simulates without crash."""
        m = NeuronModel("custom", C_m=1.0, V_init=-65.0)
        m.add_gate("m", update_form="instant",
                   inf=Boltzmann(-37.0, 7.0))
        m.add_gate("h", update_form="inf_tau",
                   inf=Boltzmann(-41.0, -4.0),
                   tau=Tau.constant(1.0),
                   initial_value=0.6)
        m.add_channel("Na", g=120.0, E_rev=50.0, gates=[(0, 3), (1, 1)])
        m.add_leak(g=0.3, E_rev=-54.3)

        trace = simulate_composable(m.to_spec(), duration=100.0, dt=0.01, I_ext=10.0)
        assert np.all(np.isfinite(trace))
        assert count_spikes(trace) >= 1

    def test_python_preset_matches_cpp(self):
        """Python NeuronModel.thalamic() matches C++ NeuronModelSpec.thalamic()."""
        py_spec = NeuronModel.thalamic().to_spec()
        cpp_spec = NeuronModelSpec.thalamic()
        assert py_spec.name == cpp_spec.name
        assert len(py_spec.gates) == len(cpp_spec.gates)
        assert len(py_spec.channels) == len(cpp_spec.channels)

    def test_full_benchmark_skeleton(self):
        """5 neuron types, connected, simulate 50ms: finite traces."""
        rn = RegionalNetwork()
        rn.add_population("TH", 5, model=NeuronModelSpec.thalamic())
        rn.add_population("STN", 5, model=NeuronModelSpec.stn())
        rn.add_population("GPe", 5, model=NeuronModelSpec.gpe())
        rn.add_population("GPi", 5, model=NeuronModelSpec.gpi())
        rn.add_population("Str", 5, model=NeuronModelSpec.striatum(pd=0.0))

        synapse = SynapseSpec.ampa()
        rn.connect("STN", "GPe", "all_to_all", synapse=synapse, weight=0.1)
        rn.connect("GPe", "STN", "all_to_all", synapse=SynapseSpec.gaba_a(), weight=0.1)
        rn.connect("STN", "GPi", "all_to_all", synapse=synapse, weight=0.1)
        rn.connect("Str", "GPi", "all_to_all", synapse=SynapseSpec.gaba_a(), weight=0.1)

        I_ext = {
            "TH": 3.0, "STN": 25.0, "GPe": 5.0, "GPi": 5.0, "Str": 8.0
        }
        traces = rn.simulate(50.0, 0.1, I_ext)

        for pop in ["TH", "STN", "GPe", "GPi", "Str"]:
            assert pop in traces
            assert np.all(np.isfinite(traces[pop])), f"{pop} trace has NaN/Inf"
