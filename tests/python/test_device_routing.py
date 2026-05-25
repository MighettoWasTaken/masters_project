"""Tests for CUDA device routing (tasks 17.3 and 17.4)."""
import pytest
import numpy as np
import hodgkin_huxley as hh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_rn():
    rn = hh.RegionalNetwork()
    rn.add_population("a", 2, neuron_type="HH")
    return rn


def _run(rn, duration=5.0, dt=0.1):
    return rn.simulate(duration, dt)  # returns {pop_name: V_array}


# ---------------------------------------------------------------------------
# hh.device() string helper (task 17.4)
# ---------------------------------------------------------------------------

class TestDeviceHelper:

    def test_cpu(self):
        assert hh.device("cpu") == hh.Device.cpu()

    def test_cuda_no_index(self):
        d = hh.device("cuda")
        assert d.type == hh.Device.Type.CUDA
        assert d.index == 0

    def test_cuda_with_index(self):
        assert hh.device("cuda:1").index == 1

    def test_cuda_index_zero(self):
        assert hh.device("cuda:0") == hh.Device.cuda(0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown device spec"):
            hh.device("gpu")

    def test_invalid_empty_raises(self):
        with pytest.raises(ValueError):
            hh.device("")


# ---------------------------------------------------------------------------
# RegionalNetwork.to() / current_device() (tasks 17.3 + 17.4)
# ---------------------------------------------------------------------------

class TestRegionalNetworkDevice:

    def test_default_device_is_cpu(self):
        rn = _minimal_rn()
        assert rn.current_device() == hh.Device.cpu()

    def test_to_cpu_returns_self(self):
        rn = _minimal_rn()
        result = rn.to(hh.Device.cpu())
        assert result is rn

    def test_to_cpu_device_is_cpu(self):
        rn = _minimal_rn()
        rn.to(hh.Device.cpu())
        assert rn.current_device() == hh.Device.cpu()

    def test_to_cpu_then_simulate(self):
        rn = _minimal_rn()
        rn.to(hh.Device.cpu())
        results = _run(rn)
        assert np.all(np.isfinite(results["a"]))

    def test_chaining(self):
        rn = _minimal_rn()
        results = rn.to(hh.Device.cpu()).simulate(5.0, 0.1)
        assert results is not None

    @pytest.mark.skipif(not hh.cuda_is_available(), reason="no CUDA GPU")
    def test_to_cuda_device_is_cuda(self):
        rn = _minimal_rn()
        rn.to(hh.Device.cuda(0))
        assert rn.current_device() == hh.Device.cuda(0)

    @pytest.mark.skipif(not hh.cuda_is_available(), reason="no CUDA GPU")
    def test_to_cuda_then_simulate(self):
        # This exercises the real CUDA routing path: pool construction,
        # synchronization, and result handoff back to Python.
        rn = _minimal_rn()
        rn.to(hh.Device.cuda(0))
        results = _run(rn)
        assert np.all(np.isfinite(results["a"]))

    @pytest.mark.skipif(not hh.cuda_is_available(), reason="no CUDA GPU")
    def test_cuda_result_matches_cpu(self):
        rn_cpu = _minimal_rn()
        rn_cpu.to(hh.Device.cpu())
        cpu_V = _run(rn_cpu)["a"]

        rn_gpu = _minimal_rn()
        rn_gpu.to(hh.Device.cuda(0))
        gpu_V = _run(rn_gpu)["a"]

        np.testing.assert_array_equal(cpu_V, gpu_V)

    @pytest.mark.skipif(not hh.cuda_is_available(), reason="no CUDA GPU")
    def test_device_string_helper_with_to(self):
        rn = _minimal_rn()
        rn.to(hh.device("cuda:0"))
        assert rn.current_device().type == hh.Device.Type.CUDA

    def test_to_cuda_raises_when_unavailable(self):
        if hh.cuda_is_available():
            pytest.skip("CUDA is available — cannot test unavailability guard")
        rn = _minimal_rn()
        with pytest.raises(RuntimeError, match="CUDA"):
            rn.to(hh.Device.cuda(0))
