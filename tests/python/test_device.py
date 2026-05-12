"""Tests for Device struct and cuda query functions (task 17.1)."""

import hodgkin_huxley as hh


class TestDeviceDefaults:

    def test_cpu_type(self):
        assert hh.Device.cpu().type == hh.Device.Type.CPU

    def test_cpu_index(self):
        assert hh.Device.cpu().index == 0

    def test_cuda_type(self):
        assert hh.Device.cuda(0).type == hh.Device.Type.CUDA

    def test_cuda_index(self):
        assert hh.Device.cuda(1).index == 1

    def test_cuda_default_index(self):
        assert hh.Device.cuda().index == 0


class TestDeviceStr:

    def test_cpu_str(self):
        assert repr(hh.Device.cpu()) == "cpu"

    def test_cuda_str(self):
        assert repr(hh.Device.cuda(0)) == "cuda:0"

    def test_cuda_str_nonzero(self):
        assert repr(hh.Device.cuda(2)) == "cuda:2"


class TestDeviceEquality:

    def test_cpu_equals_cpu(self):
        assert hh.Device.cpu() == hh.Device.cpu()

    def test_cuda_equals_cuda(self):
        assert hh.Device.cuda(0) == hh.Device.cuda(0)

    def test_cpu_not_equal_cuda(self):
        assert hh.Device.cpu() != hh.Device.cuda(0)

    def test_different_cuda_indices(self):
        assert hh.Device.cuda(0) != hh.Device.cuda(1)


class TestCudaQueries:

    def test_device_count_is_int(self):
        assert isinstance(hh.cuda_device_count(), int)

    def test_device_count_non_negative(self):
        assert hh.cuda_device_count() >= 0

    def test_is_available_is_bool(self):
        assert isinstance(hh.cuda_is_available(), bool)

    def test_is_available_consistent_with_count(self):
        assert hh.cuda_is_available() == (hh.cuda_device_count() > 0)

    def test_no_cuda_build_no_crash(self):
        # On non-CUDA builds both must return without raising
        count = hh.cuda_device_count()
        avail = hh.cuda_is_available()
        assert count >= 0 and isinstance(avail, bool)
