"""CUDA runtime regression tests for tasks 17.6 through 17.9."""

from __future__ import annotations

import numpy as np
import pytest

import hodgkin_huxley as hh
import sympy
from hodgkin_huxley._core import GateUpdateForm
from hodgkin_huxley import (
    DopamineGating,
    NeuronModel,
    RecordingConfig,
    STPRule,
    STDPRule,
    V,
)


pytestmark = pytest.mark.skipif(not hh.cuda_is_available(), reason="no CUDA GPU")


def _assert_finite_trace(arr: np.ndarray) -> None:
    assert arr.size > 0
    assert np.all(np.isfinite(arr))


def test_cuda_hh_pool_round_trip_to_cpu():
    rn = hh.RegionalNetwork()
    rn.add_population("hh", 12, neuron_type="HH")

    rn.to(hh.Device.cuda(0))
    gpu_v = rn.simulate(120.0, 0.05, {"hh": 5.0})["hh"]
    _assert_finite_trace(gpu_v)

    rn.to(hh.Device.cpu())
    cpu_v = rn.simulate(20.0, 0.05, {"hh": 5.0})["hh"]
    _assert_finite_trace(cpu_v)


def test_cuda_iz_pool_round_trip_to_cpu():
    rn = hh.RegionalNetwork()
    rn.add_population("iz", 12, neuron_type="izhikevich_rs")

    rn.to(hh.Device.cuda(0))
    gpu_v = rn.simulate(120.0, 0.05, {"iz": 10.0})["iz"]
    _assert_finite_trace(gpu_v)

    rn.to(hh.Device.cpu())
    cpu_v = rn.simulate(20.0, 0.05, {"iz": 10.0})["iz"]
    _assert_finite_trace(cpu_v)


def test_cuda_hh_synapse_path_with_delay():
    rn = hh.RegionalNetwork()
    rn.add_population("A", 8, neuron_type="HH")
    rn.add_population("B", 8, neuron_type="HH")
    rn.connect(
        "A",
        "B",
        "all_to_all",
        weight=0.4,
        synapse=hh.SynapseSpec.ampa(),
        delay=5.0,
        seed=1,
    )

    rn.to(hh.Device.cuda(0))
    out = rn.simulate(80.0, 0.05, {"A": 8.0, "B": 0.0})
    _assert_finite_trace(out["A"])
    _assert_finite_trace(out["B"])


def test_cuda_iz_synapse_path_zero_delay():
    rn = hh.RegionalNetwork()
    rn.add_population("A", 8, neuron_type="izhikevich_rs")
    rn.add_population("B", 8, neuron_type="izhikevich_rs")
    rn.connect(
        "A",
        "B",
        "all_to_all",
        weight=0.6,
        synapse=hh.SynapseSpec.gaba_a(),
        delay=0.0,
        seed=1,
    )

    rn.to(hh.Device.cuda(0))
    out = rn.simulate(80.0, 0.05, {"A": 12.0, "B": 8.0})
    _assert_finite_trace(out["A"])
    _assert_finite_trace(out["B"])


@pytest.mark.parametrize(
    ("plasticity", "expect_stdp", "expect_stp"),
    [
        (STDPRule(), True, False),
        (STPRule(), False, True),
    ],
    ids=["stdp", "stp"],
)
def test_cuda_plasticity_network_falls_back_without_crashing(
    plasticity, expect_stdp: bool, expect_stp: bool
):
    rn = hh.RegionalNetwork()
    rn.add_population("pre", 2, model=hh.NeuronModelSpec.hh_default())
    rn.add_population("post", 2, model=hh.NeuronModelSpec.hh_default())
    rn.connect(
        "pre",
        "post",
        "all_to_all",
        weight=0.5,
        synapse=hh.SynapseSpec.ampa(),
        plasticity=plasticity,
    )

    rn.to(hh.Device.cuda(0))
    out = rn.simulate(120.0, 0.05, {"pre": 8.0, "post": 0.0})
    _assert_finite_trace(out["pre"])
    _assert_finite_trace(out["post"])
    assert rn.network.has_stdp is expect_stdp
    assert rn.network.has_stp is expect_stp


def test_cuda_composable_default_round_trip_to_cpu():
    rn = hh.RegionalNetwork()
    rn.add_population("E", 10, model=hh.NeuronModelSpec.hh_default())

    rn.to(hh.Device.cuda(0))
    gpu_v = rn.simulate(120.0, 0.05, {"E": 6.0})["E"]
    _assert_finite_trace(gpu_v)

    rn.to(hh.Device.cpu())
    cpu_v = rn.simulate(20.0, 0.05, {"E": 6.0})["E"]
    _assert_finite_trace(cpu_v)


def test_cuda_custom_expr_gate_records_finite_gates_and_round_trips():
    novel_inf = (
        1 / (1 + sympy.exp(-(V + 40) / 8))
    ) * sympy.Rational(999, 1000) + sympy.Rational(1, 1000)
    novel_tau = sympy.Float(2.0) + sympy.sqrt(V**2 + 1) * sympy.Float(0.0)

    model = NeuronModel("custom_gate", C_m=1.0, V_init=-65.0)
    model.add_gate("m", inf=novel_inf, tau=novel_tau)
    model.add_channel("Leak", g=0.1, E_rev=-65.0)
    spec = model.to_spec()
    assert spec.gates[0].update_form == GateUpdateForm.CUSTOM_EXPR

    rn = hh.RegionalNetwork()
    rn.add_population("pop", 4, model=spec)
    rn.to(hh.Device.cuda(0))

    cfg = RecordingConfig(["V", "gates"])
    out = rn.simulate(50.0, 0.05, {"pop": 5.0}, record=cfg)
    _assert_finite_trace(out["pop"]["V"])
    _assert_finite_trace(out["pop"]["gates"])

    rn.to(hh.Device.cpu())
    cpu_out = rn.simulate(10.0, 0.05, {"pop": 5.0}, record=cfg)
    _assert_finite_trace(cpu_out["pop"]["V"])
    _assert_finite_trace(cpu_out["pop"]["gates"])


def test_cuda_synapse_g_modulation_changes_post_current_and_round_trips():
    cfg = RecordingConfig(["I_syn", "calcium"])

    def run_case(da_level: float):
        rn = hh.RegionalNetwork()
        rn.add_population("pre", 1, model=hh.NeuronModelSpec.hh_default())
        rn.add_population("post", 1, model=hh.NeuronModelSpec.hh_default())
        rn.connect("pre", "post", "all_to_all", weight=1.0)
        rn.add_intracellular(
            DopamineGating(da_level=da_level, k_decay=0.0001),
            populations=["post"],
        )
        rn.to(hh.Device.cuda(0))
        gpu_out = rn.simulate(200.0, 0.05, {"pre": 8.0, "post": 0.0}, record=cfg)
        _assert_finite_trace(gpu_out["post"]["I_syn"])
        _assert_finite_trace(gpu_out["post"]["calcium"])

        rn.to(hh.Device.cpu())
        cpu_out = rn.simulate(20.0, 0.05, {"pre": 8.0, "post": 0.0}, record=cfg)
        _assert_finite_trace(cpu_out["post"]["I_syn"])
        _assert_finite_trace(cpu_out["post"]["calcium"])

        return float(np.max(gpu_out["post"]["I_syn"]))

    low = run_case(0.1)
    high = run_case(0.9)
    assert high > low
