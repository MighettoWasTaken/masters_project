"""
Tests for NoiseInjector — the noise adapter utility.

Tests verify shape handling, broadcasting, dt mismatch warnings,
duration trimming, from_generator, apply_to, and properties.
"""

import warnings
import numpy as np
import pytest

from hodgkin_huxley import NoiseInjector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_1d(n_steps=1000, std=1.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std, n_steps)


def make_2d(n_neurons=10, n_steps=1000, std=1.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std, (n_neurons, n_steps))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_1d_accepted(self):
        arr = make_1d()
        inj = NoiseInjector(arr, dt=0.1)
        assert inj.is_broadcast

    def test_2d_accepted(self):
        arr = make_2d()
        inj = NoiseInjector(arr, dt=0.1)
        assert not inj.is_broadcast

    def test_3d_raises(self):
        arr = np.zeros((5, 10, 100))
        with pytest.raises(ValueError, match="1-D.*2-D"):
            NoiseInjector(arr, dt=0.1)

    def test_invalid_dt_raises(self):
        arr = make_1d()
        with pytest.raises(ValueError, match="dt must be"):
            NoiseInjector(arr, dt=0.0)
        with pytest.raises(ValueError, match="dt must be"):
            NoiseInjector(arr, dt=-0.1)

    def test_from_array_alias(self):
        arr = make_2d()
        inj = NoiseInjector.from_array(arr, dt=0.1)
        assert inj.n_neurons == arr.shape[0]
        assert inj.n_steps == arr.shape[1]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:

    def test_broadcast_n_neurons(self):
        arr = make_1d(n_steps=500)
        inj = NoiseInjector(arr, dt=0.05)
        # 1-D is stored as (1, n_steps) internally
        assert inj.n_neurons == 1

    def test_2d_n_neurons(self):
        arr = make_2d(n_neurons=25, n_steps=500)
        inj = NoiseInjector(arr, dt=0.1)
        assert inj.n_neurons == 25

    def test_n_steps(self):
        arr = make_1d(n_steps=800)
        inj = NoiseInjector(arr, dt=0.1)
        assert inj.n_steps == 800

    def test_dt_stored(self):
        inj = NoiseInjector(make_1d(), dt=0.025)
        assert inj.dt == pytest.approx(0.025)

    def test_array_property_shape(self):
        arr = make_2d(n_neurons=5, n_steps=200)
        inj = NoiseInjector(arr, dt=0.1)
        # array shape should be (n_neurons, n_steps)
        assert inj.array.shape == (5, 200)

    def test_repr_contains_info(self):
        inj = NoiseInjector(make_2d(n_neurons=3, n_steps=100), dt=0.1)
        r = repr(inj)
        assert "3" in r
        assert "100" in r


# ---------------------------------------------------------------------------
# get() — shape and broadcast
# ---------------------------------------------------------------------------

class TestGet:

    def test_broadcast_1d_output_shape(self):
        arr = make_1d(n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        out = inj.get(n_neurons=8, total_duration=50.0, dt=0.1)
        assert out.shape == (8, 500)

    def test_broadcast_all_rows_equal(self):
        arr = make_1d(n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        out = inj.get(n_neurons=5, total_duration=50.0, dt=0.1)
        for row in out[1:]:
            np.testing.assert_array_equal(row, out[0])

    def test_2d_output_shape(self):
        arr = make_2d(n_neurons=10, n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        out = inj.get(n_neurons=10, total_duration=50.0, dt=0.1)
        assert out.shape == (10, 500)

    def test_2d_subset_neurons(self):
        arr = make_2d(n_neurons=20, n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = inj.get(n_neurons=10, total_duration=50.0, dt=0.1)
        assert out.shape == (10, 500)

    def test_too_few_neurons_raises(self):
        arr = make_2d(n_neurons=5, n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        with pytest.raises(ValueError, match="5 neuron traces but 10"):
            inj.get(n_neurons=10, total_duration=50.0, dt=0.1)

    def test_too_many_rows_warns(self):
        arr = make_2d(n_neurons=20, n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = inj.get(n_neurons=5, total_duration=50.0, dt=0.1)
        assert any("20 rows" in str(x.message) for x in w)
        assert out.shape == (5, 500)

    def test_duration_trimming(self):
        arr = make_1d(n_steps=2000)
        inj = NoiseInjector(arr, dt=0.1)
        out = inj.get(n_neurons=3, total_duration=50.0, dt=0.1)  # needs 500 steps
        assert out.shape[1] == 500

    def test_noise_too_short_raises(self):
        arr = make_1d(n_steps=100)
        inj = NoiseInjector(arr, dt=0.1)
        with pytest.raises(ValueError, match="100 steps"):
            inj.get(n_neurons=3, total_duration=50.0, dt=0.1)  # needs 500

    def test_dt_mismatch_warns(self):
        arr = make_1d(n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inj.get(n_neurons=3, total_duration=50.0, dt=0.05)
        assert any("dt=0.1" in str(x.message) for x in w)

    def test_dt_exact_no_warning(self):
        arr = make_1d(n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inj.get(n_neurons=3, total_duration=50.0, dt=0.1)
        dt_warns = [x for x in w if "dt" in str(x.message).lower() and "align" in str(x.message).lower()]
        assert len(dt_warns) == 0

    def test_returns_copy(self):
        arr = make_2d(n_neurons=5, n_steps=1000)
        inj = NoiseInjector(arr, dt=0.1)
        out = inj.get(n_neurons=5, total_duration=50.0, dt=0.1)
        out[:] = 0.0
        # Modifying output should not affect stored noise
        assert not np.all(inj.array == 0.0)


# ---------------------------------------------------------------------------
# apply_to() — base current shapes
# ---------------------------------------------------------------------------

class TestApplyTo:

    def setup_method(self):
        self.arr = make_2d(n_neurons=10, n_steps=1000)
        self.inj = NoiseInjector(self.arr, dt=0.1)
        self.n = 10
        self.dur = 50.0
        self.dt = 0.1
        self.n_steps = 500

    def test_scalar_base(self):
        out = self.inj.apply_to(10.0, self.n, self.dur, self.dt)
        assert out.shape == (self.n, self.n_steps)
        # Mean should be near 10 (noise has zero mean)
        assert abs(out.mean() - 10.0) < 1.0

    def test_1d_base(self):
        base = np.full(self.n_steps, 5.0)
        out = self.inj.apply_to(base, self.n, self.dur, self.dt)
        assert out.shape == (self.n, self.n_steps)

    def test_2d_base(self):
        base = np.full((self.n, self.n_steps), 7.0)
        out = self.inj.apply_to(base, self.n, self.dur, self.dt)
        assert out.shape == (self.n, self.n_steps)

    def test_3d_base_raises(self):
        base = np.zeros((2, self.n, self.n_steps))
        with pytest.raises(ValueError, match="0-D, 1-D, or 2-D"):
            self.inj.apply_to(base, self.n, self.dur, self.dt)

    def test_noise_added_not_replaced(self):
        out = self.inj.apply_to(0.0, self.n, self.dur, self.dt)
        noise = self.inj.get(self.n, self.dur, self.dt)
        np.testing.assert_array_almost_equal(out, noise)

    def test_scalar_base_broadcast_inj(self):
        arr1d = make_1d(n_steps=1000)
        inj = NoiseInjector(arr1d, dt=0.1)
        out = inj.apply_to(5.0, n_neurons=4, total_duration=50.0, dt=0.1)
        assert out.shape == (4, 500)
        # All rows should be equal (broadcast)
        for row in out[1:]:
            np.testing.assert_array_equal(row, out[0])


# ---------------------------------------------------------------------------
# from_generator()
# ---------------------------------------------------------------------------

class TestFromGenerator:

    def test_numpy_lambda(self):
        inj = NoiseInjector.from_generator(
            lambda n, rng: rng.normal(0, 1.5, n),
            n_neurons=10, n_steps=500, dt=0.1, seed=42
        )
        assert inj.n_neurons == 10
        assert inj.n_steps == 500
        assert not inj.is_broadcast

    def test_seed_reproducibility(self):
        fn = lambda n, rng: rng.standard_normal(n)
        inj1 = NoiseInjector.from_generator(fn, n_neurons=5, n_steps=200, dt=0.1, seed=7)
        inj2 = NoiseInjector.from_generator(fn, n_neurons=5, n_steps=200, dt=0.1, seed=7)
        np.testing.assert_array_equal(inj1.array, inj2.array)

    def test_different_seeds_differ(self):
        fn = lambda n, rng: rng.standard_normal(n)
        inj1 = NoiseInjector.from_generator(fn, n_neurons=5, n_steps=200, dt=0.1, seed=1)
        inj2 = NoiseInjector.from_generator(fn, n_neurons=5, n_steps=200, dt=0.1, seed=2)
        assert not np.array_equal(inj1.array, inj2.array)

    def test_no_seed_still_works(self):
        fn = lambda n, rng: rng.standard_normal(n)
        inj = NoiseInjector.from_generator(fn, n_neurons=3, n_steps=100, dt=0.1)
        assert inj.n_neurons == 3

    def test_rows_independent(self):
        # Each row should be drawn independently — not all identical
        fn = lambda n, rng: rng.standard_normal(n)
        inj = NoiseInjector.from_generator(fn, n_neurons=5, n_steps=200, dt=0.1, seed=0)
        rows_identical = all(
            np.array_equal(inj.array[0], inj.array[i]) for i in range(1, 5)
        )
        assert not rows_identical

    def test_zero_mean_approximately(self):
        fn = lambda n, rng: rng.normal(0, 1.0, n)
        inj = NoiseInjector.from_generator(fn, n_neurons=20, n_steps=5000, dt=0.1, seed=0)
        assert abs(inj.array.mean()) < 0.1

    def test_scipy_filtfilt_compatible(self):
        """from_generator works with scipy filtfilt pattern."""
        pytest.importorskip("scipy")
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 0.05)
        inj = NoiseInjector.from_generator(
            lambda n, rng: filtfilt(b, a, rng.standard_normal(n)),
            n_neurons=5, n_steps=500, dt=0.1, seed=0
        )
        assert inj.n_neurons == 5
        assert inj.n_steps == 500
        assert np.all(np.isfinite(inj.array))


# ---------------------------------------------------------------------------
# Integration: use with Network simulation
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_apply_to_iext_hh_network(self):
        """NoiseInjector.apply_to() output can be passed directly as I_ext."""
        from hodgkin_huxley import RegionalNetwork, NeuronModelSpec

        n_neurons = 3
        duration = 50.0
        dt = 0.1
        n_steps = int(duration / dt)

        arr = make_2d(n_neurons=n_neurons, n_steps=n_steps, std=0.5, seed=1)
        inj = NoiseInjector(arr, dt=dt)
        I_noisy = inj.apply_to(10.0, n_neurons=n_neurons,
                                total_duration=duration, dt=dt)

        assert I_noisy.shape == (n_neurons, n_steps)

        rn = RegionalNetwork()
        rn.add_population("E", n_neurons, model=NeuronModelSpec.hh_default())
        traces = rn.simulate(duration, dt, {"E": I_noisy})["E"]
        assert traces.shape == (n_neurons, n_steps)
        # With I_ext≈10 + small noise, neurons should spike
        assert np.any(traces > 0.0)

    def test_broadcast_1d_noise_hh(self):
        """1-D broadcast noise is applied identically to all neurons."""
        from hodgkin_huxley import RegionalNetwork, NeuronModelSpec

        n_neurons = 4
        duration = 30.0
        dt = 0.1
        n_steps = int(duration / dt)

        arr1d = make_1d(n_steps=n_steps, std=0.3)
        inj = NoiseInjector(arr1d, dt=dt)
        I_noisy = inj.apply_to(10.0, n_neurons=n_neurons,
                                total_duration=duration, dt=dt)

        # All rows of I_noisy should be identical
        for row in I_noisy[1:]:
            np.testing.assert_array_equal(row, I_noisy[0])

        rn = RegionalNetwork()
        rn.add_population("E", n_neurons, model=NeuronModelSpec.hh_default())
        traces = rn.simulate(duration, dt, {"E": I_noisy})["E"]
        assert traces.shape == (n_neurons, n_steps)
