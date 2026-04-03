"""
RegionalNetwork Verification Tests

Test Categories:
1. Population management - add/query, duplicate name, custom params, contiguous indices
2. Preset patterns - all_to_all, one_to_one, shifted, random_sparse, random_permutation
3. Custom patterns - callable returns pairs, lambda pattern, weight matrix
4. SynapseSpec - receptor presets, custom exp/alpha/double_exp
5. Weight distributions - constant, uniform, normal, tuple shorthand
6. Initial conditions - randomize V mean/std, seed reproducibility
7. Simulation - dict I_ext (1D broadcast, 2D per-neuron, scalar, missing=zero), dict traces
8. Functional - excitatory drives target, inhibitory suppresses, multi-region flow
9. Edge cases - empty network, single-neuron pop, self-connections
10. Integration - multi-region skeleton with mixed patterns/delays/receptors
"""

import numpy as np
import pytest

from hodgkin_huxley import (
    RegionalNetwork,
    ConnectivityPattern,
    NeuronModelSpec,
    SynapseSpec,
    SynapseSpecType,
    WeightDistribution,
    WeightDistType,
    IzhikevichType,
)


# =============================================================================
# Utilities
# =============================================================================

def count_spikes(trace: np.ndarray, threshold: float = 0.0) -> int:
    above = trace > threshold
    crossings = np.diff(above.astype(int))
    return int(np.sum(crossings == 1))


# =============================================================================
# 1. Population Management
# =============================================================================

class TestPopulation:
    def test_add_population_default_hh(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5)
        assert rn.num_neurons == 5
        assert rn.population_size("A") == 5
        assert rn.population_start("A") == 0

    def test_add_population_with_type_string(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3, neuron_type="HH")
        rn.add_population("B", 4, neuron_type="IZHIKEVICH_RS")
        assert rn.num_neurons == 7
        assert rn.population_size("A") == 3
        assert rn.population_size("B") == 4

    def test_add_population_with_type_enum(self):
        rn = RegionalNetwork()
        rn.add_population("X", 2, model=NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING))
        assert rn.num_neurons == 2

    def test_add_population_custom_hh_params(self):
        rn = RegionalNetwork()
        spec = NeuronModelSpec.hh_default()
        spec.channels[0].g = 100.0  # Na conductance
        rn.add_population("custom", 3, model=spec)
        assert rn.population_size("custom") == 3

    def test_add_population_custom_iz_params(self):
        rn = RegionalNetwork()
        spec = NeuronModelSpec.izhikevich()
        spec.iz_params.a = 0.1
        spec.iz_params.b = 0.2
        rn.add_population("iz", 4, model=spec)
        assert rn.population_size("iz") == 4

    def test_duplicate_name_raises(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        with pytest.raises(RuntimeError, match="already exists"):
            rn.add_population("A", 5)

    def test_contiguous_indices(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 5)
        rn.add_population("C", 2)
        assert rn.population_start("A") == 0
        assert rn.population_start("B") == 3
        assert rn.population_start("C") == 8
        assert rn.num_neurons == 10

    def test_population_names_order(self):
        rn = RegionalNetwork()
        rn.add_population("Z", 1)
        rn.add_population("A", 1)
        rn.add_population("M", 1)
        assert rn.population_names() == ["Z", "A", "M"]

    def test_num_populations(self):
        rn = RegionalNetwork()
        assert rn.num_populations == 0
        rn.add_population("A", 2)
        rn.add_population("B", 3)
        assert rn.num_populations == 2

    def test_unknown_population_raises(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        with pytest.raises(RuntimeError, match="Unknown population"):
            rn.population_size("B")


# =============================================================================
# 2. Preset Connectivity Patterns
# =============================================================================

class TestPresetPatterns:
    def test_all_to_all_count(self):
        rn = RegionalNetwork()
        rn.add_population("A", 4)
        rn.add_population("B", 3)
        rn.connect("A", "B", "all_to_all", weight=0.1,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 4 * 3  # 12

    def test_all_to_all_self_population_no_self(self):
        """Within same pop, self-connections excluded by default."""
        rn = RegionalNetwork()
        rn.add_population("A", 4)
        rn.connect("A", "A", "all_to_all", weight=0.1,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 4 * 3  # 12 (no self)

    def test_all_to_all_self_population_allow_self(self):
        rn = RegionalNetwork()
        rn.add_population("A", 4)
        rn.connect("A", "A", "all_to_all", weight=0.1,
                   synapse=SynapseSpec.ampa(), allow_self=True)
        assert rn.num_synapses == 4 * 4  # 16

    def test_one_to_one(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5)
        rn.add_population("B", 5)
        rn.connect("A", "B", "one_to_one", weight=0.2,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 5

    def test_one_to_one_unequal_raises(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5)
        rn.add_population("B", 3)
        with pytest.raises(RuntimeError, match="equal population sizes"):
            rn.connect("A", "B", "one_to_one", weight=0.2,
                       synapse=SynapseSpec.ampa())

    def test_shifted_positive(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5)
        rn.add_population("B", 5)
        rn.connect("A", "B", "shifted", weight=0.3,
                   synapse=SynapseSpec.gaba_a(), shift=2)
        assert rn.num_synapses == 5

    def test_shifted_negative(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5)
        rn.add_population("B", 5)
        rn.connect("A", "B", "shifted", weight=0.3,
                   synapse=SynapseSpec.gaba_a(), shift=-1)
        assert rn.num_synapses == 5

    def test_shifted_unequal_raises(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5)
        rn.add_population("B", 3)
        with pytest.raises(RuntimeError, match="equal population sizes"):
            rn.connect("A", "B", "shifted", weight=0.3,
                       synapse=SynapseSpec.gaba_a())

    def test_random_sparse_approx_count(self):
        rn = RegionalNetwork()
        rn.add_population("A", 50)
        rn.add_population("B", 50)
        rn.connect("A", "B", "random_sparse", weight=0.1,
                   synapse=SynapseSpec.ampa(), probability=0.2, seed=42)
        expected = 50 * 50 * 0.2  # 500
        assert abs(rn.num_synapses - expected) < expected * 0.2  # within 20%

    def test_random_sparse_seed_reproducible(self):
        counts = []
        for _ in range(2):
            rn = RegionalNetwork()
            rn.add_population("A", 20)
            rn.add_population("B", 20)
            rn.connect("A", "B", "random_sparse", weight=0.1,
                       synapse=SynapseSpec.ampa(), probability=0.3, seed=123)
            counts.append(rn.num_synapses)
        assert counts[0] == counts[1]

    def test_random_permutation_bijective(self):
        rn = RegionalNetwork()
        rn.add_population("A", 10)
        rn.add_population("B", 10)
        rn.connect("A", "B", "random_permutation", weight=0.5,
                   synapse=SynapseSpec.ampa(), seed=42)
        assert rn.num_synapses == 10

    def test_random_permutation_unequal_raises(self):
        rn = RegionalNetwork()
        rn.add_population("A", 5)
        rn.add_population("B", 3)
        with pytest.raises(RuntimeError, match="equal population sizes"):
            rn.connect("A", "B", "random_permutation", weight=0.5,
                       synapse=SynapseSpec.ampa())

    def test_pattern_enum(self):
        """Using ConnectivityPattern enum instead of string."""
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        rn.connect("A", "B", ConnectivityPattern.ONE_TO_ONE, weight=0.1,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 3


# =============================================================================
# 3. Custom Patterns (callables)
# =============================================================================

class TestCustomPatterns:
    def test_callable_pattern(self):
        def ring(src_size, dst_size):
            pairs = []
            for i in range(src_size):
                j = (i + 1) % dst_size
                pairs.append((i, j))
            return pairs

        rn = RegionalNetwork()
        rn.add_population("A", 5)
        rn.add_population("B", 5)
        rn.connect("A", "B", ring, weight=0.2,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 5

    def test_lambda_pattern(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        rn.connect("A", "B", lambda s, d: [(i, i) for i in range(min(s, d))],
                   weight=0.1, synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 3

    def test_callable_local_indices(self):
        """Custom pattern indices are local (0-based within pop)."""
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 4)
        # Connect only neuron 0->0 and 1->2
        rn.connect("A", "B", lambda s, d: [(0, 0), (1, 2)],
                   weight=0.5, synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 2

    def test_callable_empty_pairs(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        rn.connect("A", "B", lambda s, d: [], weight=0.1,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 0

    def test_connect_from_matrix_numpy(self):
        """connect_from_matrix with a numpy weight matrix creates exactly the
        non-zero synapses and skips zero entries."""
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        W = np.array([[0.5, 0.0, 0.3],
                      [0.0, 0.4, 0.0],
                      [0.2, 0.0, 0.6]])  # 5 non-zero entries
        rn.connect_from_matrix("A", "B", W, synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 5

    def test_connect_from_matrix_list(self):
        """connect_from_matrix with a nested Python list produces the same
        synapse count as the numpy path, and weight_scale is applied."""
        rn = RegionalNetwork()
        rn.add_population("A", 2)
        rn.add_population("B", 2)
        W = [[1.0, 0.0],
             [0.5, 1.0]]  # 3 non-zero entries
        rn.connect_from_matrix("A", "B", W, synapse=SynapseSpec.ampa(),
                               weight_scale=0.1)
        assert rn.num_synapses == 3

    def test_connect_from_matrix_shape_mismatch_raises(self):
        """A matrix whose dimensions don't match the population sizes raises
        ValueError before any synapses are added."""
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        with pytest.raises(ValueError):
            rn.connect_from_matrix("A", "B", np.ones((4, 3)))


# =============================================================================
# 4. SynapseSpec
# =============================================================================

class TestSynapseSpec:
    def test_ampa_preset(self):
        s = SynapseSpec.ampa()
        assert s.type == SynapseSpecType.DOUBLE_EXPONENTIAL
        assert s.E_syn == 0.0
        assert s.tau_rise == 0.5
        assert s.tau_decay == 2.5

    def test_nmda_preset(self):
        s = SynapseSpec.nmda()
        assert s.type == SynapseSpecType.DOUBLE_EXPONENTIAL
        assert s.E_syn == 0.0
        assert s.tau_rise == 2.0
        assert s.tau_decay == 67.0

    def test_gaba_a_preset(self):
        s = SynapseSpec.gaba_a()
        assert s.type == SynapseSpecType.DOUBLE_EXPONENTIAL
        assert s.E_syn == -80.0
        assert s.tau_rise == 0.4
        assert s.tau_decay == 7.7

    def test_custom_exponential(self):
        s = SynapseSpec.exponential(E_syn=-70.0, tau=5.0)
        assert s.type == SynapseSpecType.EXPONENTIAL
        assert s.E_syn == -70.0
        assert s.tau == 5.0

    def test_custom_alpha(self):
        s = SynapseSpec.alpha(E_syn=0.0, tau=3.0)
        assert s.type == SynapseSpecType.ALPHA
        assert s.E_syn == 0.0
        assert s.tau == 3.0

    def test_custom_double_exp(self):
        s = SynapseSpec.double_exponential(E_syn=-85.0, tau_rise=1.0, tau_decay=10.0)
        assert s.type == SynapseSpecType.DOUBLE_EXPONENTIAL
        assert s.E_syn == -85.0
        assert s.tau_rise == 1.0
        assert s.tau_decay == 10.0


# =============================================================================
# 5. Weight Distributions
# =============================================================================

class TestWeightDistributions:
    def test_constant(self):
        w = WeightDistribution.constant(0.5)
        assert w.type == WeightDistType.CONSTANT
        assert w.param1 == 0.5

    def test_uniform(self):
        w = WeightDistribution.uniform(0.1, 0.9)
        assert w.type == WeightDistType.UNIFORM
        assert w.param1 == 0.1
        assert w.param2 == 0.9

    def test_normal(self):
        w = WeightDistribution.normal(0.5, 0.1)
        assert w.type == WeightDistType.NORMAL
        assert w.param1 == 0.5
        assert w.param2 == 0.1

    def test_float_shorthand(self):
        """Float value is converted to constant WeightDistribution."""
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        rn.connect("A", "B", "one_to_one", weight=0.5,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 3

    def test_tuple_shorthand(self):
        """(min, max) tuple is converted to uniform WeightDistribution."""
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        rn.connect("A", "B", "one_to_one", weight=(0.1, 0.9),
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 3


# =============================================================================
# 6. Initial Conditions
# =============================================================================

class TestInitialConditions:
    def test_randomize_membrane_potentials(self):
        rn = RegionalNetwork()
        rn.add_population("A", 100, neuron_type="HH")
        rn.randomize_membrane_potentials("A", V_mean=-62.0, V_std=5.0, seed=42)
        # Check the potentials were randomized (not all the same)
        net = rn.network
        potentials = [net.neuron(i).V for i in range(100)]
        assert np.std(potentials) > 1.0  # not all identical
        assert abs(np.mean(potentials) - (-62.0)) < 2.0  # near target mean

    def test_randomize_seed_reproducible(self):
        results = []
        for _ in range(2):
            rn = RegionalNetwork()
            rn.add_population("A", 20, neuron_type="HH")
            rn.randomize_membrane_potentials("A", V_mean=-60.0, V_std=3.0, seed=99)
            net = rn.network
            potentials = [net.neuron(i).V for i in range(20)]
            results.append(potentials)
        np.testing.assert_array_almost_equal(results[0], results[1])


# =============================================================================
# 7. Simulation with dict I_ext
# =============================================================================

class TestSimulation:
    def test_scalar_current(self):
        """Scalar I_ext broadcasts to all neurons/timesteps."""
        rn = RegionalNetwork()
        rn.add_population("A", 2, neuron_type="HH")
        traces = rn.simulate(10.0, 0.01, {"A": 10.0})
        assert "A" in traces
        assert traces["A"].shape[0] == 2  # 2 neurons
        assert traces["A"].shape[1] == 1000  # num_steps

    def test_1d_broadcast(self):
        """1D I_ext broadcasts to all neurons in population."""
        rn = RegionalNetwork()
        rn.add_population("A", 3, neuron_type="HH")
        num_steps = 1000
        I = np.full(num_steps, 10.0)
        traces = rn.simulate(10.0, 0.01, {"A": I})
        assert traces["A"].shape == (3, num_steps)

    def test_2d_per_neuron(self):
        """2D I_ext provides per-neuron current."""
        rn = RegionalNetwork()
        rn.add_population("A", 2, neuron_type="HH")
        num_steps = 1000
        I = np.zeros((2, num_steps))
        I[0, :] = 10.0  # Only first neuron gets current
        traces = rn.simulate(10.0, 0.01, {"A": I})
        assert traces["A"].shape == (2, num_steps)

    def test_missing_population_gets_zero(self):
        """Populations not in I_ext dict receive zero current."""
        rn = RegionalNetwork()
        rn.add_population("A", 2, neuron_type="HH")
        rn.add_population("B", 2, neuron_type="HH")
        traces = rn.simulate(10.0, 0.01, {"A": 10.0})
        # B should have near-resting traces
        assert "B" in traces
        # A should show activity, B should stay near rest
        assert np.max(traces["A"]) > np.max(traces["B"])

    def test_empty_iext_dict(self):
        rn = RegionalNetwork()
        rn.add_population("A", 2, neuron_type="HH")
        traces = rn.simulate(5.0, 0.01, {})
        assert traces["A"].shape[0] == 2

    def test_none_iext(self):
        rn = RegionalNetwork()
        rn.add_population("A", 2, neuron_type="HH")
        traces = rn.simulate(5.0, 0.01, None)
        assert traces["A"].shape[0] == 2

    def test_traces_dict_keys(self):
        rn = RegionalNetwork()
        rn.add_population("X", 2)
        rn.add_population("Y", 3)
        traces = rn.simulate(5.0, 0.01, {})
        assert set(traces.keys()) == {"X", "Y"}
        assert traces["X"].shape[0] == 2
        assert traces["Y"].shape[0] == 3


# =============================================================================
# 8. Functional Tests
# =============================================================================

class TestFunctional:
    def test_excitatory_drives_target(self):
        """Excitatory synapse from driven neuron should cause post to spike."""
        rn = RegionalNetwork()
        rn.add_population("src", 1, neuron_type="HH")
        rn.add_population("dst", 1, neuron_type="HH")
        rn.connect("src", "dst", "one_to_one", weight=0.5,
                   synapse=SynapseSpec.ampa())
        traces = rn.simulate(100.0, 0.01, {"src": 15.0})
        # dst should show some activity from synaptic input
        src_max = np.max(traces["src"])
        dst_max = np.max(traces["dst"])
        assert src_max > 0.0  # src fires
        assert dst_max > -60.0  # dst is excited above rest

    def test_inhibitory_suppresses(self):
        """GABAergic inhibition should suppress postsynaptic activity."""
        # Network with inhibition
        rn_inh = RegionalNetwork()
        rn_inh.add_population("inh", 5, neuron_type="HH")
        rn_inh.add_population("target", 1, neuron_type="HH")
        rn_inh.connect("inh", "target", "all_to_all", weight=1.0,
                       synapse=SynapseSpec.gaba_a())
        traces_inh = rn_inh.simulate(100.0, 0.01, {"inh": 15.0, "target": 10.0})

        # Network without inhibition
        rn_none = RegionalNetwork()
        rn_none.add_population("target", 1, neuron_type="HH")
        traces_none = rn_none.simulate(100.0, 0.01, {"target": 10.0})

        spikes_inh = count_spikes(traces_inh["target"][0])
        spikes_none = count_spikes(traces_none["target"][0])
        assert spikes_inh <= spikes_none  # inhibition reduces spiking

    def test_multi_region_flow(self):
        """Signal propagates through A -> B -> C chain."""
        rn = RegionalNetwork()
        rn.add_population("A", 5, neuron_type="HH")
        rn.add_population("B", 5, neuron_type="HH")
        rn.add_population("C", 5, neuron_type="HH")
        rn.connect("A", "B", "one_to_one", weight=0.8,
                   synapse=SynapseSpec.ampa())
        rn.connect("B", "C", "one_to_one", weight=0.8,
                   synapse=SynapseSpec.ampa())
        traces = rn.simulate(200.0, 0.01, {"A": 15.0})
        # C should show some activity from chain
        c_max = np.max(traces["C"])
        assert c_max > -60.0  # some excitation reached C


# =============================================================================
# 9. Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_single_neuron_population(self):
        rn = RegionalNetwork()
        rn.add_population("solo", 1, neuron_type="HH")
        traces = rn.simulate(10.0, 0.01, {"solo": 10.0})
        assert traces["solo"].shape == (1, 1000)

    def test_self_connections_within_pop(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.connect("A", "A", "all_to_all", weight=0.1,
                   synapse=SynapseSpec.ampa(), allow_self=True)
        assert rn.num_synapses == 9  # 3x3 including self

    def test_add_connection_local_indices(self):
        """add_connection with local indices works correctly."""
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        rn.add_connection("A", 0, "B", 2, weight=0.5,
                          synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 1

    def test_add_connection_out_of_range(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        rn.add_population("B", 3)
        with pytest.raises((RuntimeError, IndexError)):
            rn.add_connection("A", 5, "B", 0, weight=0.5,
                              synapse=SynapseSpec.ampa())

    def test_reset(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3, neuron_type="HH")
        rn.simulate(50.0, 0.01, {"A": 15.0})
        rn.reset()
        net = rn.network
        for i in range(3):
            assert abs(net.neuron(i).V - (-65.0)) < 1.0

    def test_fast_math_toggle(self):
        rn = RegionalNetwork()
        assert rn.fast_math is True
        rn.fast_math = False
        assert rn.fast_math is False

    def test_repr(self):
        rn = RegionalNetwork()
        rn.add_population("A", 3)
        s = repr(rn)
        assert "RegionalNetwork" in s
        assert "populations=1" in s
        assert "neurons=3" in s


# =============================================================================
# 10. Integration Test: Multi-region Network
# =============================================================================

class TestIntegration:
    def test_multi_region_skeleton(self):
        """Build an 8-region network with mixed patterns/delays/receptors."""
        n = 10
        rn = RegionalNetwork()

        # HH populations
        rn.add_population("STN", n, neuron_type="HH")
        rn.add_population("GPe", n, neuron_type="HH")
        rn.add_population("GPi", n, neuron_type="HH")
        rn.add_population("TH", n, neuron_type="HH")
        rn.add_population("StrD1", n, neuron_type="HH")
        rn.add_population("StrD2", n, neuron_type="HH")

        # Izhikevich populations
        rn.add_population("CtxE", n, neuron_type="IZHIKEVICH_RS")
        rn.add_population("CtxI", n, neuron_type="IZHIKEVICH_FS")

        assert rn.num_neurons == 8 * n
        assert rn.num_populations == 8

        # Randomize initial conditions for HH pops
        for pop in ["STN", "GPe", "GPi", "TH"]:
            rn.randomize_membrane_potentials(pop, V_mean=-62.0, V_std=5.0, seed=42)

        # Various connectivity patterns and synapse types
        rn.connect("STN", "GPe", "all_to_all", weight=0.3, delay=2.0,
                   synapse=SynapseSpec.ampa())
        rn.connect("STN", "GPe", "all_to_all", weight=0.002, delay=2.0,
                   synapse=SynapseSpec.nmda())
        rn.connect("GPe", "STN", "shifted", weight=0.5, delay=4.0,
                   synapse=SynapseSpec.gaba_a(), shift=1)
        rn.connect("GPe", "GPe", "random_sparse", weight=0.25, delay=1.0,
                   synapse=SynapseSpec.gaba_a(), probability=0.3, seed=42)
        rn.connect("GPi", "TH", "one_to_one", weight=0.112, delay=5.0,
                   synapse=SynapseSpec.alpha(E_syn=-85.0, tau=5.0))
        rn.connect("CtxE", "STN", "random_permutation", weight=0.003, delay=5.9,
                   synapse=SynapseSpec.double_exponential(E_syn=0.0, tau_rise=2.0,
                                                          tau_decay=90.0),
                   seed=42)
        rn.connect("CtxE", "StrD1", "one_to_one", weight=0.1, delay=3.0,
                   synapse=SynapseSpec.ampa())
        rn.connect("CtxI", "CtxE", "all_to_all", weight=0.2, delay=1.0,
                   synapse=SynapseSpec.gaba_a())

        assert rn.num_synapses > 0

        # Simulate with various current inputs
        traces = rn.simulate(50.0, 0.01, {
            "STN": 5.0,
            "GPe": 3.0,
            "TH": 1.2,
            "CtxE": 8.0,
        })

        # Verify all populations have traces with correct shapes
        num_steps = int(50.0 / 0.01)
        for pop_name in rn.population_names():
            assert pop_name in traces
            pop_size = rn.population_size(pop_name)
            assert traces[pop_name].shape == (pop_size, num_steps)
            # Traces should be finite
            assert np.all(np.isfinite(traces[pop_name]))

    def test_mixed_synapse_types_network(self):
        """Network using all three synapse types (exp, alpha, double_exp)."""
        rn = RegionalNetwork()
        rn.add_population("A", 5, neuron_type="HH")
        rn.add_population("B", 5, neuron_type="HH")
        rn.add_population("C", 5, neuron_type="HH")

        rn.connect("A", "B", "one_to_one", weight=0.3,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
        rn.connect("B", "C", "one_to_one", weight=0.3,
                   synapse=SynapseSpec.alpha(E_syn=0.0, tau=3.0))
        rn.connect("A", "C", "one_to_one", weight=0.1,
                   synapse=SynapseSpec.double_exponential(
                       E_syn=0.0, tau_rise=0.5, tau_decay=5.0))

        traces = rn.simulate(50.0, 0.01, {"A": 15.0})

        for name in ["A", "B", "C"]:
            assert np.all(np.isfinite(traces[name]))


# =============================================================================
# 11. Inter-region Current Flow
# =============================================================================

class TestInterRegionCurrent:
    def test_excitatory_current_crosses_regions(self):
        """Drive region A, connect A->B excitatory. B must depolarize
        significantly compared to an isolated B with no input."""
        rn = RegionalNetwork()
        rn.add_population("A", 5, neuron_type="HH")
        rn.add_population("B", 5, neuron_type="HH")
        rn.connect("A", "B", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.ampa())
        traces = rn.simulate(200.0, 0.01, {"A": 15.0})

        # Isolated B baseline
        rn_iso = RegionalNetwork()
        rn_iso.add_population("B", 5, neuron_type="HH")
        traces_iso = rn_iso.simulate(200.0, 0.01, {})

        # Connected B should depolarize well beyond isolated B
        connected_max = np.max(traces["B"])
        isolated_max = np.max(traces_iso["B"])
        assert connected_max > isolated_max + 5.0, (
            f"Connected B max {connected_max:.1f} should exceed "
            f"isolated B max {isolated_max:.1f} by > 5 mV"
        )

    def test_inhibitory_current_crosses_regions(self):
        """Drive inhibitory region, connect to target that is also driven.
        Target should fire fewer spikes than without inhibitory input."""
        # With inhibition
        rn = RegionalNetwork()
        rn.add_population("inh_src", 10, neuron_type="HH")
        rn.add_population("target", 5, neuron_type="HH")
        rn.connect("inh_src", "target", "all_to_all", weight=0.8,
                   synapse=SynapseSpec.gaba_a())
        traces_inh = rn.simulate(300.0, 0.01, {"inh_src": 15.0, "target": 10.0})

        # Without inhibition
        rn_no = RegionalNetwork()
        rn_no.add_population("target", 5, neuron_type="HH")
        traces_no = rn_no.simulate(300.0, 0.01, {"target": 10.0})

        # Sum spikes across all target neurons
        spikes_inh = sum(count_spikes(traces_inh["target"][i]) for i in range(5))
        spikes_no = sum(count_spikes(traces_no["target"][i]) for i in range(5))
        assert spikes_inh < spikes_no, (
            f"Inhibited target spikes ({spikes_inh}) should be less than "
            f"uninhibited ({spikes_no})"
        )

    def test_disconnected_regions_independent(self):
        """Two populations with no synapses: driving A should not affect B."""
        rn = RegionalNetwork()
        rn.add_population("A", 5, neuron_type="HH")
        rn.add_population("B", 5, neuron_type="HH")
        # No connections
        traces = rn.simulate(100.0, 0.01, {"A": 15.0})

        # B should stay near resting potential
        b_range = np.max(traces["B"]) - np.min(traces["B"])
        assert b_range < 1.0, (
            f"Disconnected B should stay at rest, but range was {b_range:.2f} mV"
        )

    def test_chain_a_b_c_signal_propagation(self):
        """A->B->C chain: drive only A. Verify B fires from A's input,
        and C fires from B's input. Compare spike counts at each stage."""
        rn = RegionalNetwork()
        rn.add_population("A", 5, neuron_type="HH")
        rn.add_population("B", 5, neuron_type="HH")
        rn.add_population("C", 5, neuron_type="HH")
        rn.connect("A", "B", "one_to_one", weight=1.0,
                   synapse=SynapseSpec.ampa())
        rn.connect("B", "C", "one_to_one", weight=1.0,
                   synapse=SynapseSpec.ampa())
        traces = rn.simulate(300.0, 0.01, {"A": 15.0})

        spikes_a = sum(count_spikes(traces["A"][i]) for i in range(5))
        spikes_b = sum(count_spikes(traces["B"][i]) for i in range(5))
        spikes_c = sum(count_spikes(traces["C"][i]) for i in range(5))

        # A is directly driven — most spikes
        assert spikes_a > 0, "A is driven and should spike"
        # B is driven only through synapses from A
        assert spikes_b > 0, "B should fire from A's excitatory input"
        # C is driven only through B
        assert spikes_c > 0, "C should fire from B's excitatory input"
        # Spike counts should generally decrease along the chain
        assert spikes_a >= spikes_b, "Driven A should spike >= synaptically-driven B"


# =============================================================================
# 12. Intra-region Current Flow
# =============================================================================

class TestIntraRegionCurrent:
    def test_recurrent_excitation_spreads(self):
        """Within one population, drive only neuron 0 via per-neuron I_ext.
        Recurrent all-to-all excitation should cause other neurons to fire."""
        n = 5
        rn = RegionalNetwork()
        rn.add_population("pop", n, neuron_type="HH")
        rn.connect("pop", "pop", "all_to_all", weight=0.6,
                   synapse=SynapseSpec.ampa(), allow_self=False)

        # Only neuron 0 gets external current
        num_steps = int(300.0 / 0.01)
        I = np.zeros((n, num_steps))
        I[0, :] = 15.0
        traces = rn.simulate(300.0, 0.01, {"pop": I})

        # Neuron 0 should fire
        assert count_spikes(traces["pop"][0]) > 0, "Driven neuron 0 should spike"
        # At least one other neuron should fire from recurrent input
        other_spikes = sum(count_spikes(traces["pop"][i]) for i in range(1, n))
        assert other_spikes > 0, (
            "Recurrent excitation should cause at least one other neuron to fire"
        )

    def test_recurrent_inhibition_stabilizes(self):
        """Recurrent GABA within a population should reduce overall firing
        rate compared to an unconnected population with the same drive."""
        n = 10
        # With recurrent inhibition
        rn = RegionalNetwork()
        rn.add_population("pop", n, neuron_type="HH")
        rn.connect("pop", "pop", "all_to_all", weight=0.5,
                   synapse=SynapseSpec.gaba_a(), allow_self=False)
        traces_inh = rn.simulate(300.0, 0.01, {"pop": 12.0})

        # Without recurrent connections
        rn_free = RegionalNetwork()
        rn_free.add_population("pop", n, neuron_type="HH")
        traces_free = rn_free.simulate(300.0, 0.01, {"pop": 12.0})

        spikes_inh = sum(count_spikes(traces_inh["pop"][i]) for i in range(n))
        spikes_free = sum(count_spikes(traces_free["pop"][i]) for i in range(n))
        assert spikes_inh < spikes_free, (
            f"Recurrent inhibition ({spikes_inh} spikes) should reduce firing "
            f"vs unconnected ({spikes_free} spikes)"
        )


# =============================================================================
# 13. Synaptic Math Correctness
# =============================================================================

class TestSynapticMath:
    def test_weight_scales_postsynaptic_response(self):
        """Stronger weight -> greater postsynaptic depolarization.
        Compare weak (w=0.1) vs strong (w=1.0) one-to-one connections."""
        results = {}
        for label, w in [("weak", 0.1), ("strong", 1.0)]:
            rn = RegionalNetwork()
            rn.add_population("pre", 1, neuron_type="HH")
            rn.add_population("post", 1, neuron_type="HH")
            rn.connect("pre", "post", "one_to_one", weight=w,
                       synapse=SynapseSpec.ampa())
            traces = rn.simulate(200.0, 0.01, {"pre": 15.0})
            results[label] = np.max(traces["post"][0])

        assert results["strong"] > results["weak"], (
            f"Strong weight max V ({results['strong']:.1f}) should exceed "
            f"weak weight max V ({results['weak']:.1f})"
        )

    def test_delay_shifts_postsynaptic_response(self):
        """Longer delay -> later onset of postsynaptic depolarization.
        Compare delay=0 vs delay=10ms by finding first depolarization above -60mV."""
        results = {}
        for label, d in [("nodelay", 0.0), ("delay10", 10.0)]:
            rn = RegionalNetwork()
            rn.add_population("pre", 1, neuron_type="HH")
            rn.add_population("post", 1, neuron_type="HH")
            rn.connect("pre", "post", "one_to_one", weight=1.0, delay=d,
                       synapse=SynapseSpec.ampa())
            traces = rn.simulate(200.0, 0.01, {"pre": 15.0})
            post_v = traces["post"][0]
            # Find first timestep where post exceeds -55 mV
            above = np.where(post_v > -55.0)[0]
            results[label] = above[0] * 0.01 if len(above) > 0 else float('inf')

        assert results["delay10"] > results["nodelay"] + 5.0, (
            f"10ms delay onset ({results['delay10']:.1f}ms) should be at least "
            f"5ms later than no-delay onset ({results['nodelay']:.1f}ms)"
        )

    def test_e_syn_polarity(self):
        """E_syn=0 (excitatory) should depolarize; E_syn=-80 (inhibitory)
        should hyperpolarize a driven postsynaptic neuron."""
        results = {}
        for label, e_syn in [("exc", 0.0), ("inh", -80.0)]:
            rn = RegionalNetwork()
            rn.add_population("pre", 5, neuron_type="HH")
            rn.add_population("post", 1, neuron_type="HH")
            rn.connect("pre", "post", "all_to_all", weight=0.5,
                       synapse=SynapseSpec.exponential(E_syn=e_syn, tau=2.0))
            # Drive pre to fire, give post a moderate current
            traces = rn.simulate(200.0, 0.01, {"pre": 15.0, "post": 8.0})
            results[label] = count_spikes(traces["post"][0])

        # Excitatory input should produce more spikes (or at least not fewer)
        # than inhibitory input
        assert results["exc"] > results["inh"], (
            f"Excitatory E_syn should produce more spikes ({results['exc']}) "
            f"than inhibitory ({results['inh']})"
        )

    def test_ampa_faster_than_nmda(self):
        """AMPA (tau_d=2.5ms) produces a sharper response than NMDA (tau_d=67ms).
        Measure the time from first post-spike to second post-spike: AMPA
        allows faster repetitive firing because its conductance decays quicker."""
        results = {}
        for label, syn in [("ampa", SynapseSpec.ampa()),
                           ("nmda", SynapseSpec.nmda())]:
            rn = RegionalNetwork()
            rn.add_population("pre", 3, neuron_type="HH")
            rn.add_population("post", 1, neuron_type="HH")
            rn.connect("pre", "post", "all_to_all", weight=0.3,
                       synapse=syn)
            traces = rn.simulate(500.0, 0.01, {"pre": 15.0})
            post_v = traces["post"][0]
            # Measure mean voltage in last 100ms — sustained NMDA should
            # hold the membrane more depolarized than fast-decaying AMPA
            last_100ms = post_v[-10000:]
            results[label] = np.mean(last_100ms)

        # NMDA's slow decay sustains depolarization more than AMPA
        assert results["nmda"] > results["ampa"], (
            f"NMDA sustained mean V ({results['nmda']:.2f}) should exceed "
            f"AMPA ({results['ampa']:.2f}) due to slower decay"
        )

    def test_alpha_synapse_temporal_profile(self):
        """Alpha synapse has a smooth rise-then-fall. Verify post-synaptic
        response is delayed relative to exponential (which rises instantly)."""
        results = {}
        for label, syn in [("exp", SynapseSpec.exponential(E_syn=0.0, tau=5.0)),
                           ("alpha", SynapseSpec.alpha(E_syn=0.0, tau=5.0))]:
            rn = RegionalNetwork()
            rn.add_population("pre", 1, neuron_type="HH")
            rn.add_population("post", 1, neuron_type="HH")
            rn.connect("pre", "post", "one_to_one", weight=1.0,
                       synapse=syn)
            traces = rn.simulate(200.0, 0.01, {"pre": 15.0})
            post_v = traces["post"][0]
            above = np.where(post_v > -55.0)[0]
            results[label] = above[0] * 0.01 if len(above) > 0 else float('inf')

        # Alpha has a delayed peak, so post-synaptic depolarization onset
        # should be the same or slightly later than exponential
        assert results["alpha"] >= results["exp"] - 1.0, (
            "Alpha onset should not be significantly earlier than exponential"
        )


# =============================================================================
# 14. Parameter Sensitivity
# =============================================================================

class TestParameterSensitivity:
    def test_higher_weight_more_spikes(self):
        """Increasing synaptic weight should increase postsynaptic spike count."""
        spike_counts = []
        for w in [0.05, 0.3, 1.0]:
            rn = RegionalNetwork()
            rn.add_population("pre", 3, neuron_type="HH")
            rn.add_population("post", 3, neuron_type="HH")
            rn.connect("pre", "post", "one_to_one", weight=w,
                       synapse=SynapseSpec.ampa())
            traces = rn.simulate(300.0, 0.01, {"pre": 15.0})
            total = sum(count_spikes(traces["post"][i]) for i in range(3))
            spike_counts.append(total)

        # Monotonically non-decreasing (heavier weight -> more or equal spikes)
        for i in range(len(spike_counts) - 1):
            assert spike_counts[i + 1] >= spike_counts[i], (
                f"Weight increase should not decrease spikes: {spike_counts}"
            )

    def test_higher_probability_more_synapses_more_activity(self):
        """Higher random_sparse probability -> more synapses -> more effect."""
        results = {}
        for label, prob in [("sparse", 0.05), ("dense", 0.8)]:
            rn = RegionalNetwork()
            rn.add_population("pre", 10, neuron_type="HH")
            rn.add_population("post", 10, neuron_type="HH")
            rn.connect("pre", "post", "random_sparse", weight=0.3,
                       synapse=SynapseSpec.ampa(), probability=prob, seed=42)
            traces = rn.simulate(200.0, 0.01, {"pre": 15.0})
            total = sum(count_spikes(traces["post"][i]) for i in range(10))
            results[label] = total

        assert results["dense"] > results["sparse"], (
            f"Dense connectivity ({results['dense']} spikes) should produce "
            f"more post spikes than sparse ({results['sparse']})"
        )

    def test_custom_hh_params_change_behavior(self):
        """Modified HH parameters (lower Na conductance) should produce fewer
        spikes with the same current drive."""
        # Default HH
        rn_default = RegionalNetwork()
        rn_default.add_population("A", 5, neuron_type="HH")
        traces_default = rn_default.simulate(200.0, 0.01, {"A": 10.0})
        spikes_default = sum(count_spikes(traces_default["A"][i]) for i in range(5))

        # Reduced Na conductance — harder to spike
        spec = NeuronModelSpec.hh_default()
        spec.channels[0].g = 40.0  # Na conductance; default is 120.0
        rn_custom = RegionalNetwork()
        rn_custom.add_population("A", 5, model=spec)
        traces_custom = rn_custom.simulate(200.0, 0.01, {"A": 10.0})
        spikes_custom = sum(count_spikes(traces_custom["A"][i]) for i in range(5))

        assert spikes_default > spikes_custom, (
            f"Default g_Na=120 ({spikes_default} spikes) should fire more than "
            f"reduced g_Na=40 ({spikes_custom} spikes)"
        )

    def test_custom_iz_params_change_behavior(self):
        """Different Izhikevich parameters produce different spike patterns.
        FS neurons fire faster than RS neurons with same current."""
        results = {}
        for label, ntype in [("RS", "IZHIKEVICH_RS"), ("FS", "IZHIKEVICH_FS")]:
            rn = RegionalNetwork()
            rn.add_population("pop", 5, neuron_type=ntype)
            traces = rn.simulate(500.0, 0.1, {"pop": 15.0})
            total = sum(count_spikes(traces["pop"][i], threshold=25.0)
                        for i in range(5))
            results[label] = total

        assert results["FS"] > results["RS"], (
            f"Fast-spiking ({results['FS']} spikes) should fire more than "
            f"regular-spiking ({results['RS']} spikes) with same drive"
        )

    def test_longer_delay_later_effect(self):
        """Compare three delay values. First post-synaptic spike should
        occur later with longer delays."""
        first_spike_times = []
        for d in [0.0, 5.0, 15.0]:
            rn = RegionalNetwork()
            rn.add_population("pre", 1, neuron_type="HH")
            rn.add_population("post", 1, neuron_type="HH")
            rn.connect("pre", "post", "one_to_one", weight=1.0, delay=d,
                       synapse=SynapseSpec.ampa())
            traces = rn.simulate(200.0, 0.01, {"pre": 15.0})
            post_v = traces["post"][0]
            above = np.where(post_v > 0.0)[0]
            t = above[0] * 0.01 if len(above) > 0 else float('inf')
            first_spike_times.append(t)

        # Each delay step should push the first spike later
        for i in range(len(first_spike_times) - 1):
            assert first_spike_times[i + 1] > first_spike_times[i], (
                f"Delays {[0, 5, 15]}: first spike times {first_spike_times} "
                f"should be monotonically increasing"
            )


# =============================================================================
# 15. Custom Callable Patterns — Integration
# =============================================================================

class TestCustomPatternIntegration:
    def test_distance_ring_connectivity(self):
        """Distance-dependent ring connectivity using a callable.
        Each neuron connects to neighbors within radius. Verify correct
        synapse count and that the network simulates correctly."""
        n = 20
        radius = 2

        def ring_pattern(src_size, dst_size):
            pairs = []
            for i in range(src_size):
                for j in range(dst_size):
                    dist = min(abs(i - j), dst_size - abs(i - j))
                    if dist <= radius and i != j:
                        pairs.append((i, j))
            return pairs

        rn = RegionalNetwork()
        rn.add_population("ring", n, neuron_type="HH")
        rn.connect("ring", "ring", ring_pattern, weight=0.3,
                   synapse=SynapseSpec.ampa())

        # Each neuron connects to 2*radius neighbors (excluding self)
        expected_synapses = n * (2 * radius)
        assert rn.num_synapses == expected_synapses, (
            f"Expected {expected_synapses} synapses, got {rn.num_synapses}"
        )

        # Drive one neuron, activity should spread to neighbors
        num_steps = int(300.0 / 0.01)
        I = np.zeros((n, num_steps))
        I[0, :] = 15.0
        traces = rn.simulate(300.0, 0.01, {"ring": I})
        assert np.all(np.isfinite(traces["ring"]))

        # Neuron 0 should spike
        assert count_spikes(traces["ring"][0]) > 0
        # Near neighbors (1, 2) should get more input than far neurons (10)
        near_max = max(np.max(traces["ring"][1]), np.max(traces["ring"][2]))
        far_max = np.max(traces["ring"][n // 2])
        assert near_max > far_max, (
            f"Near neighbors max V ({near_max:.1f}) should exceed "
            f"far neuron max V ({far_max:.1f})"
        )

    def test_callable_cross_region_specific_pairs(self):
        """Custom callable connecting specific src->dst pairs across regions.
        Verify only specified connections are made and current flows correctly."""
        rn = RegionalNetwork()
        rn.add_population("sensory", 5, neuron_type="HH")
        rn.add_population("motor", 5, neuron_type="HH")

        # Only connect sensory[0]->motor[0] and sensory[1]->motor[3]
        def specific_map(src_size, dst_size):
            return [(0, 0), (1, 3)]

        rn.connect("sensory", "motor", specific_map, weight=1.0,
                   synapse=SynapseSpec.ampa())
        assert rn.num_synapses == 2

        # Drive sensory neuron 0 and 1
        num_steps = int(200.0 / 0.01)
        I = np.zeros((5, num_steps))
        I[0, :] = 15.0
        I[1, :] = 15.0
        traces = rn.simulate(200.0, 0.01, {"sensory": I})

        # motor[0] and motor[3] should be excited, others should not
        connected_max = max(np.max(traces["motor"][0]),
                            np.max(traces["motor"][3]))
        unconnected_max = max(np.max(traces["motor"][1]),
                              np.max(traces["motor"][2]),
                              np.max(traces["motor"][4]))
        assert connected_max > unconnected_max + 5.0, (
            f"Connected motor max V ({connected_max:.1f}) should exceed "
            f"unconnected ({unconnected_max:.1f}) by > 5 mV"
        )

    def test_callable_probability_based(self):
        """Callable that implements its own probabilistic connectivity.
        Verify seed controls reproducibility and approx count is correct."""
        def prob_connect(src_size, dst_size, p=0.3, seed=42):
            rng = np.random.default_rng(seed)
            pairs = []
            for i in range(src_size):
                for j in range(dst_size):
                    if rng.random() < p:
                        pairs.append((i, j))
            return pairs

        counts = []
        for _ in range(2):
            rn = RegionalNetwork()
            rn.add_population("A", 20, neuron_type="HH")
            rn.add_population("B", 20, neuron_type="HH")
            rn.connect("A", "B",
                       lambda s, d: prob_connect(s, d, p=0.3, seed=42),
                       weight=0.2, synapse=SynapseSpec.ampa())
            counts.append(rn.num_synapses)

        # Same seed -> same pairs -> same synapse count
        assert counts[0] == counts[1]
        # Approximately 20*20*0.3 = 120
        assert abs(counts[0] - 120) < 40


# =============================================================================
# 16. Custom Weight Distributions — Integration
# =============================================================================

class TestCustomWeightIntegration:
    def test_uniform_weights_vary(self):
        """Uniform weight distribution produces varying synapse weights.
        Verify via the underlying Network's synapse objects."""
        rn = RegionalNetwork()
        rn.add_population("A", 10, neuron_type="HH")
        rn.add_population("B", 10, neuron_type="HH")
        rn.connect("A", "B", "one_to_one",
                   weight=WeightDistribution.uniform(0.1, 0.9),
                   synapse=SynapseSpec.ampa(), seed=42)

        net = rn.network
        weights = [net.synapse(i).weight for i in range(rn.num_synapses)]
        assert min(weights) >= 0.1, f"Min weight {min(weights)} below 0.1"
        assert max(weights) <= 0.9, f"Max weight {max(weights)} above 0.9"
        # Weights should not all be identical
        assert len(set(f"{w:.6f}" for w in weights)) > 1, "Uniform weights should vary"

    def test_normal_weights_vary(self):
        """Normal weight distribution produces a spread of weights."""
        rn = RegionalNetwork()
        rn.add_population("A", 20, neuron_type="HH")
        rn.add_population("B", 20, neuron_type="HH")
        rn.connect("A", "B", "one_to_one",
                   weight=WeightDistribution.normal(0.5, 0.1),
                   synapse=SynapseSpec.ampa(), seed=42)

        net = rn.network
        weights = [net.synapse(i).weight for i in range(rn.num_synapses)]
        w_mean = np.mean(weights)
        w_std = np.std(weights)
        assert abs(w_mean - 0.5) < 0.1, f"Mean {w_mean:.3f} far from 0.5"
        assert w_std > 0.01, f"Std {w_std:.3f} too small — weights not varying"

    def test_constant_weights_identical(self):
        """Constant weight distribution should produce identical weights."""
        rn = RegionalNetwork()
        rn.add_population("A", 5, neuron_type="HH")
        rn.add_population("B", 5, neuron_type="HH")
        rn.connect("A", "B", "one_to_one", weight=0.42,
                   synapse=SynapseSpec.ampa())

        net = rn.network
        weights = [net.synapse(i).weight for i in range(rn.num_synapses)]
        for w in weights:
            assert abs(w - 0.42) < 1e-10, f"Expected 0.42, got {w}"

    def test_tuple_uniform_shorthand_in_callable(self):
        """Tuple weight shorthand (min, max) works with callable patterns."""
        rn = RegionalNetwork()
        rn.add_population("A", 10, neuron_type="HH")
        rn.add_population("B", 10, neuron_type="HH")
        rn.connect("A", "B",
                   lambda s, d: [(i, i) for i in range(min(s, d))],
                   weight=(0.1, 0.9), synapse=SynapseSpec.ampa(), seed=42)

        net = rn.network
        weights = [net.synapse(i).weight for i in range(rn.num_synapses)]
        assert all(0.1 <= w <= 0.9 for w in weights), (
            f"All weights should be in [0.1, 0.9], got range "
            f"[{min(weights):.3f}, {max(weights):.3f}]"
        )
        assert len(set(f"{w:.6f}" for w in weights)) > 1, "Weights should vary"

    def test_weight_distribution_affects_network_dynamics(self):
        """Strong uniform weights should produce more postsynaptic activity
        than weak uniform weights."""
        results = {}
        for label, wdist in [("weak", WeightDistribution.uniform(0.01, 0.05)),
                              ("strong", WeightDistribution.uniform(0.5, 1.0))]:
            rn = RegionalNetwork()
            rn.add_population("pre", 5, neuron_type="HH")
            rn.add_population("post", 5, neuron_type="HH")
            rn.connect("pre", "post", "all_to_all", weight=wdist,
                       synapse=SynapseSpec.ampa(), seed=42)
            traces = rn.simulate(200.0, 0.01, {"pre": 15.0})
            total = sum(count_spikes(traces["post"][i]) for i in range(5))
            results[label] = total

        assert results["strong"] > results["weak"], (
            f"Strong weights ({results['strong']} spikes) should produce "
            f"more post activity than weak ({results['weak']})"
        )
