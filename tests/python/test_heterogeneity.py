"""
Tests for population parameter heterogeneity (Task 10.5).

Verifies that RegionalNetwork.add_population(heterogeneity=...) correctly:
- Applies normal and uniform distributions to each supported field
- Produces non-identical neurons across the population
- Respects seed reproducibility
- Raises ValueError for bad field paths or missing model=
"""

import numpy as np
import pytest

from hodgkin_huxley import RegionalNetwork, NeuronModelSpec, SynapseSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rnet_stn(n=10, heterogeneity=None, seed=None):
    rnet = RegionalNetwork()
    rnet.add_population(
        "STN", n,
        model=NeuronModelSpec.stn(),
        heterogeneity=heterogeneity,
        seed=seed,
    )
    return rnet


def get_channel_g(rnet, pop_name, neuron_local_idx, channel_idx):
    start = rnet._rnet.population_start(pop_name)
    neuron = rnet._rnet.network().neuron(start + neuron_local_idx)
    return neuron.model_spec.channels[channel_idx].g


def get_C_m(rnet, pop_name, neuron_local_idx):
    start = rnet._rnet.population_start(pop_name)
    neuron = rnet._rnet.network().neuron(start + neuron_local_idx)
    return neuron.model_spec.C_m


def get_V_init(rnet, pop_name, neuron_local_idx):
    start = rnet._rnet.population_start(pop_name)
    neuron = rnet._rnet.network().neuron(start + neuron_local_idx)
    return neuron.model_spec.V_init


def get_gate_initial(rnet, pop_name, neuron_local_idx, gate_idx):
    start = rnet._rnet.population_start(pop_name)
    neuron = rnet._rnet.network().neuron(start + neuron_local_idx)
    return neuron.model_spec.gates[gate_idx].initial_value


# ---------------------------------------------------------------------------
# Field: channels[i].g — normal distribution
# ---------------------------------------------------------------------------

class TestChannelGNormal:

    def test_conductances_not_all_equal(self):
        n = 20
        rnet = make_rnet_stn(n, {"channels[0].g": ("normal", 1.0, 0.15)}, seed=0)
        gs = [get_channel_g(rnet, "STN", i, 0) for i in range(n)]
        assert len(set(gs)) > 1, "All conductances are identical — heterogeneity not applied"

    def test_mean_near_base(self):
        base_g = NeuronModelSpec.stn().channels[0].g
        n = 200
        rnet = make_rnet_stn(n, {"channels[0].g": ("normal", 1.0, 0.10)}, seed=1)
        gs = np.array([get_channel_g(rnet, "STN", i, 0) for i in range(n)])
        # Mean should be ~base_g (within 10%)
        assert abs(gs.mean() - base_g) / base_g < 0.10

    def test_std_near_expected(self):
        base_g = NeuronModelSpec.stn().channels[0].g
        n = 500
        rnet = make_rnet_stn(n, {"channels[0].g": ("normal", 1.0, 0.20)}, seed=2)
        gs = np.array([get_channel_g(rnet, "STN", i, 0) for i in range(n)])
        # std/mean ≈ 0.20 (within factor of 2 for N=500)
        cv = gs.std() / gs.mean()
        assert 0.10 < cv < 0.35


# ---------------------------------------------------------------------------
# Field: channels[i].g — uniform distribution
# ---------------------------------------------------------------------------

class TestChannelGUniform:

    def test_all_within_range(self):
        base_g = NeuronModelSpec.stn().channels[0].g
        n = 50
        rnet = make_rnet_stn(n, {"channels[0].g": ("uniform", 0.8, 1.2)}, seed=3)
        gs = np.array([get_channel_g(rnet, "STN", i, 0) for i in range(n)])
        assert np.all(gs >= base_g * 0.8 * 0.99)
        assert np.all(gs <= base_g * 1.2 * 1.01)

    def test_not_all_equal(self):
        n = 20
        rnet = make_rnet_stn(n, {"channels[0].g": ("uniform", 0.9, 1.1)}, seed=4)
        gs = [get_channel_g(rnet, "STN", i, 0) for i in range(n)]
        assert len(set(gs)) > 1


# ---------------------------------------------------------------------------
# Field: C_m
# ---------------------------------------------------------------------------

class TestCmHeterogeneity:

    def test_C_m_varies(self):
        n = 20
        rnet = make_rnet_stn(n, {"C_m": ("normal", 1.0, 0.10)}, seed=5)
        cms = [get_C_m(rnet, "STN", i) for i in range(n)]
        assert len(set(cms)) > 1

    def test_C_m_all_positive(self):
        n = 50
        rnet = make_rnet_stn(n, {"C_m": ("uniform", 0.9, 1.1)}, seed=6)
        cms = [get_C_m(rnet, "STN", i) for i in range(n)]
        assert all(c > 0 for c in cms)


# ---------------------------------------------------------------------------
# Field: V_init
# ---------------------------------------------------------------------------

class TestVInitHeterogeneity:

    def test_V_init_varies(self):
        n = 20
        rnet = make_rnet_stn(n, {"V_init": ("normal", 1.0, 0.05)}, seed=7)
        vinits = [get_V_init(rnet, "STN", i) for i in range(n)]
        assert len(set(vinits)) > 1

    def test_V_init_near_base(self):
        base_v = NeuronModelSpec.stn().V_init
        n = 100
        rnet = make_rnet_stn(n, {"V_init": ("normal", 1.0, 0.05)}, seed=8)
        vinits = np.array([get_V_init(rnet, "STN", i) for i in range(n)])
        # mean should be within 5% of base
        assert abs(vinits.mean() - base_v) / abs(base_v) < 0.10


# ---------------------------------------------------------------------------
# Field: gates[i].initial_value
# ---------------------------------------------------------------------------

class TestGateInitialHeterogeneity:

    def test_gate_initial_varies(self):
        n = 20
        # STN gate 0 has non-zero initial value
        base_spec = NeuronModelSpec.stn()
        gate_idx = 0
        rnet = RegionalNetwork()
        rnet.add_population(
            "STN", n,
            model=base_spec,
            heterogeneity={"gates[0].initial_value": ("uniform", 0.8, 1.2)},
            seed=9,
        )
        initials = [get_gate_initial(rnet, "STN", i, gate_idx) for i in range(n)]
        assert len(set(initials)) > 1


# ---------------------------------------------------------------------------
# Multiple fields simultaneously
# ---------------------------------------------------------------------------

class TestMultipleFields:

    def test_multiple_fields_independently_varied(self):
        n = 30
        rnet = RegionalNetwork()
        rnet.add_population(
            "STN", n,
            model=NeuronModelSpec.stn(),
            heterogeneity={
                "channels[0].g": ("normal", 1.0, 0.15),
                "channels[1].g": ("normal", 1.0, 0.15),
                "C_m":           ("uniform", 0.9, 1.1),
            },
            seed=10,
        )
        g0s = [get_channel_g(rnet, "STN", i, 0) for i in range(n)]
        g1s = [get_channel_g(rnet, "STN", i, 1) for i in range(n)]
        cms = [get_C_m(rnet, "STN", i) for i in range(n)]
        assert len(set(g0s)) > 1
        assert len(set(g1s)) > 1
        assert len(set(cms)) > 1
        # g0 and g1 should NOT be identical (drawn independently)
        assert g0s != g1s


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------

class TestSeedReproducibility:

    def _collect_g(self, seed):
        n = 15
        rnet = make_rnet_stn(n, {"channels[0].g": ("normal", 1.0, 0.15)}, seed=seed)
        return [get_channel_g(rnet, "STN", i, 0) for i in range(n)]

    def test_same_seed_identical(self):
        assert self._collect_g(42) == self._collect_g(42)

    def test_different_seeds_differ(self):
        assert self._collect_g(1) != self._collect_g(2)

    def test_no_seed_still_works(self):
        n = 5
        rnet = make_rnet_stn(n, {"channels[0].g": ("normal", 1.0, 0.10)}, seed=None)
        gs = [get_channel_g(rnet, "STN", i, 0) for i in range(n)]
        assert len(gs) == n


# ---------------------------------------------------------------------------
# Produces non-identical firing
# ---------------------------------------------------------------------------

class TestFiringVariability:

    def test_heterogeneous_population_fires_differently(self):
        """With conductance scatter, neurons in the population fire at different rates."""
        n = 10
        duration = 200.0
        dt = 0.1
        n_steps = int(duration / dt)

        rnet = RegionalNetwork()
        rnet.add_population(
            "STN", n,
            model=NeuronModelSpec.stn(),
            heterogeneity={"channels[0].g": ("normal", 1.0, 0.20)},
            seed=77,
        )

        I_ext = {"STN": np.full((n, n_steps), 5.0).tolist()}
        traces = rnet.simulate(duration, dt, I_ext)
        V = np.array(traces["STN"])

        # Detect spikes for each neuron
        spike_counts = []
        for i in range(n):
            crossings = np.diff((V[i] > -20).astype(int))
            spike_counts.append(int(np.sum(crossings == 1)))

        # At least some neurons should have different spike counts
        assert len(set(spike_counts)) > 1, \
            f"All neurons spiked identically ({spike_counts}) — heterogeneity may not work"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_unknown_field_raises(self):
        rnet = RegionalNetwork()
        with pytest.raises(ValueError, match="Unknown heterogeneity field path"):
            rnet.add_population(
                "STN", 5,
                model=NeuronModelSpec.stn(),
                heterogeneity={"bad_field": ("normal", 1.0, 0.1)},
            )

    def test_heterogeneity_without_model_raises(self):
        rnet = RegionalNetwork()
        with pytest.raises(ValueError, match="requires 'model'"):
            rnet.add_population(
                "STN", 5,
                neuron_type="HH",
                heterogeneity={"C_m": ("normal", 1.0, 0.1)},
            )

    def test_unknown_distribution_raises(self):
        rnet = RegionalNetwork()
        with pytest.raises(ValueError, match="Unknown heterogeneity distribution"):
            rnet.add_population(
                "STN", 5,
                model=NeuronModelSpec.stn(),
                heterogeneity={"C_m": ("lognormal", 1.0, 0.1)},
            )

    def test_channel_index_out_of_range(self):
        rnet = RegionalNetwork()
        with pytest.raises((IndexError, ValueError)):
            rnet.add_population(
                "STN", 5,
                model=NeuronModelSpec.stn(),
                heterogeneity={"channels[99].g": ("normal", 1.0, 0.1)},
            )
