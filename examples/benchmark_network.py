#!/usr/bin/env python3
"""
Network Performance Benchmarks: C++ Backend vs Pure NumPy

Compares wall-clock time of the compiled C++ network simulation against a
pure-NumPy reimplementation of the same HH neuron + exponential synapse model.

Topologies tested:
  1. Chain  (0->1->2->...->N-1)
  2. Ring   (chain + N-1->0)
  3. All-to-all (every neuron to every other)
  4. Star   (hub neuron 0 <-> all others)

Output:
  examples/figs/benchmark_time_vs_neurons.png
  examples/figs/benchmark_speedup.png
"""

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

from hodgkin_huxley import Network
import os
import time
import threading
import gc
import psutil
import numpy as np


# =============================================================================
# Pure NumPy HH Network Implementation
# =============================================================================

def _safe_exp(x: np.ndarray) -> np.ndarray:
    """Numerically safe exponential (clamp argument to [-700, 700])."""
    return np.exp(np.clip(x, -700.0, 700.0))


def _alpha_m(V: np.ndarray) -> np.ndarray:
    x = V + 40.0
    # Avoid division by zero at x==0 using limit: lim x->0 of x/(1-e^{-x/10}) = 10
    out = np.where(
        np.abs(x) < 1e-7,
        1.0,
        0.1 * x / (1.0 - _safe_exp(-x / 10.0))
    )
    return out


def _beta_m(V: np.ndarray) -> np.ndarray:
    return 4.0 * _safe_exp(-(V + 65.0) / 18.0)


def _alpha_h(V: np.ndarray) -> np.ndarray:
    return 0.07 * _safe_exp(-(V + 65.0) / 20.0)


def _beta_h(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0))


def _alpha_n(V: np.ndarray) -> np.ndarray:
    x = V + 55.0
    out = np.where(
        np.abs(x) < 1e-7,
        0.1,
        0.01 * x / (1.0 - _safe_exp(-x / 10.0))
    )
    return out


def _beta_n(V: np.ndarray) -> np.ndarray:
    return 0.125 * _safe_exp(-(V + 65.0) / 80.0)


# HH default parameters
_C_m = 1.0
_g_Na = 120.0
_g_K = 36.0
_g_L = 0.3
_E_Na = 50.0
_E_K = -77.0
_E_L = -54.387


def _hh_derivs(V, m, h, n, I):
    """Compute derivatives for all neurons (vectorised)."""
    I_Na = _g_Na * m**3 * h * (V - _E_Na)
    I_K = _g_K * n**4 * (V - _E_K)
    I_L = _g_L * (V - _E_L)

    dV = (I - I_Na - I_K - I_L) / _C_m
    dm = _alpha_m(V) * (1.0 - m) - _beta_m(V) * m
    dh = _alpha_h(V) * (1.0 - h) - _beta_h(V) * h
    dn = _alpha_n(V) * (1.0 - n) - _beta_n(V) * n
    return dV, dm, dh, dn


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _clamp_V(V: np.ndarray) -> np.ndarray:
    """Clamp membrane voltage to biophysically reasonable range."""
    return np.clip(V, -200.0, 200.0)


def _hh_rk4_step(V, m, h, n, I, dt):
    """Single RK4 step for the entire population."""
    k1 = _hh_derivs(V, m, h, n, I)

    V2 = _clamp_V(V + 0.5 * dt * k1[0])
    m2 = _clamp01(m + 0.5 * dt * k1[1])
    h2 = _clamp01(h + 0.5 * dt * k1[2])
    n2 = _clamp01(n + 0.5 * dt * k1[3])
    k2 = _hh_derivs(V2, m2, h2, n2, I)

    V3 = _clamp_V(V + 0.5 * dt * k2[0])
    m3 = _clamp01(m + 0.5 * dt * k2[1])
    h3 = _clamp01(h + 0.5 * dt * k2[2])
    n3 = _clamp01(n + 0.5 * dt * k2[3])
    k3 = _hh_derivs(V3, m3, h3, n3, I)

    V4 = _clamp_V(V + dt * k3[0])
    m4 = _clamp01(m + dt * k3[1])
    h4 = _clamp01(h + dt * k3[2])
    n4 = _clamp01(n + dt * k3[3])
    k4 = _hh_derivs(V4, m4, h4, n4, I)

    V_new = _clamp_V(V + (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]))
    m_new = _clamp01(m + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]))
    h_new = _clamp01(h + (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]))
    n_new = _clamp01(n + (dt / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]))

    return V_new, m_new, h_new, n_new


class NumpyHHNetwork:
    """
    Pure-NumPy HH network with conductance-based synapses.

    Supports three synapse types matching the C++ backend:
      - Exponential decay  (add_synapse)
      - Alpha function     (add_alpha_synapse)
      - Double exponential (add_double_exp_synapse)

    All synapse types support axonal conduction delays.
    """

    def __init__(self, num_neurons: int):
        self.N = num_neurons
        # Neuron state
        self.V = np.full(num_neurons, -65.0)
        am = _alpha_m(self.V)
        bm = _beta_m(self.V)
        self.m = am / (am + bm)
        ah = _alpha_h(self.V)
        bh = _beta_h(self.V)
        self.h = ah / (ah + bh)
        an = _alpha_n(self.V)
        bn = _beta_n(self.V)
        self.n = an / (an + bn)

        # Build-phase lists (converted to arrays in _finalise)
        self._exp_list = []    # (pre, post, weight, E_syn, tau, delay)
        self._alpha_list = []  # (pre, post, weight, E_syn, tau, delay)
        self._dexp_list = []   # (pre, post, weight, E_syn, tau_r, tau_d, delay)
        self._finalised = False
        self._delays_ready = False

    # ------------------------------------------------------------------
    # Synapse creation
    # ------------------------------------------------------------------

    def add_synapse(self, pre: int, post: int, weight: float,
                    E_syn: float = 0.0, tau: float = 2.0,
                    delay: float = 0.0):
        """Add an exponential-decay synapse."""
        self._exp_list.append((pre, post, weight, E_syn, tau, delay))

    def add_alpha_synapse(self, pre: int, post: int, weight: float,
                          E_syn: float = 0.0, tau: float = 2.0,
                          delay: float = 0.0):
        """Add an alpha-function synapse (peaks at t = tau)."""
        self._alpha_list.append((pre, post, weight, E_syn, tau, delay))

    def add_double_exp_synapse(self, pre: int, post: int, weight: float,
                               E_syn: float = 0.0,
                               tau_rise: float = 0.4, tau_decay: float = 2.5,
                               delay: float = 0.0):
        """Add a double-exponential synapse (separate rise/decay)."""
        self._dexp_list.append((pre, post, weight, E_syn,
                                tau_rise, tau_decay, delay))

    # ------------------------------------------------------------------
    # Finalise: convert lists -> numpy arrays
    # ------------------------------------------------------------------

    def _finalise(self):
        from collections import deque  # noqa: used later for delays

        # --- Exponential synapses ---
        S = len(self._exp_list)
        self.S_exp = S
        if S > 0:
            pre, post, w, e, tau, dly = zip(*self._exp_list)
            self.exp_pre = np.array(pre, dtype=np.intp)
            self.exp_post = np.array(post, dtype=np.intp)
            self.exp_weight = np.array(w)
            self.exp_E_syn = np.array(e)
            self.exp_tau = np.array(tau)
            self.exp_delay = np.array(dly)
            self.exp_g = np.zeros(S)
            self.exp_V_prev = np.full(S, -65.0)
        else:
            self.exp_delay = np.empty(0)

        # --- Alpha synapses ---
        S = len(self._alpha_list)
        self.S_alpha = S
        if S > 0:
            pre, post, w, e, tau, dly = zip(*self._alpha_list)
            self.alpha_pre = np.array(pre, dtype=np.intp)
            self.alpha_post = np.array(post, dtype=np.intp)
            self.alpha_weight = np.array(w)
            self.alpha_E_syn = np.array(e)
            self.alpha_tau = np.array(tau)
            self.alpha_delay = np.array(dly)
            self.alpha_g = np.zeros(S)
            self.alpha_x = np.zeros(S)   # auxiliary variable for ODE
            self.alpha_V_prev = np.full(S, -65.0)
        else:
            self.alpha_delay = np.empty(0)

        # --- Double-exponential synapses ---
        S = len(self._dexp_list)
        self.S_dexp = S
        if S > 0:
            pre, post, w, e, tr, td, dly = zip(*self._dexp_list)
            self.dexp_pre = np.array(pre, dtype=np.intp)
            self.dexp_post = np.array(post, dtype=np.intp)
            self.dexp_weight = np.array(w)
            self.dexp_E_syn = np.array(e)
            self.dexp_tau_rise = np.array(tr)
            self.dexp_tau_decay = np.array(td)
            self.dexp_delay = np.array(dly)
            self.dexp_g = np.zeros(S)
            self.dexp_g_rise = np.zeros(S)
            self.dexp_g_decay = np.zeros(S)
            self.dexp_V_prev = np.full(S, -65.0)

            # normalisation so peak conductance = weight
            tp = (self.dexp_tau_decay * self.dexp_tau_rise) / \
                 (self.dexp_tau_decay - self.dexp_tau_rise) * \
                 np.log(self.dexp_tau_decay / self.dexp_tau_rise)
            pk = np.exp(-tp / self.dexp_tau_decay) - \
                 np.exp(-tp / self.dexp_tau_rise)
            self.dexp_norm = 1.0 / pk
        else:
            self.dexp_delay = np.empty(0)

        self._finalised = True

    # ------------------------------------------------------------------
    # Delay buffers — simple per-synapse deques (initialised lazily
    # because we need dt which is only known at simulate-time)
    # ------------------------------------------------------------------

    def _setup_delay_buffers(self, dt):
        from collections import deque

        def _make_buffers(delay_arr, count):
            bufs = [None] * count
            any_delay = False
            for i in range(count):
                d = delay_arr[i]
                if d > 0:
                    n_steps = int(round(d / dt))
                    if n_steps > 0:
                        bufs[i] = deque([False] * n_steps,
                                        maxlen=n_steps)
                        any_delay = True
            return bufs, any_delay

        self._exp_dbufs, self._has_exp_dly = \
            _make_buffers(self.exp_delay, self.S_exp)
        self._alpha_dbufs, self._has_alpha_dly = \
            _make_buffers(self.alpha_delay, self.S_alpha)
        self._dexp_dbufs, self._has_dexp_dly = \
            _make_buffers(self.dexp_delay, self.S_dexp)
        self._delays_ready = True

    def _apply_delays(self, spiked, buffers):
        """Route spikes through per-synapse delay buffers."""
        delayed = spiked.copy()
        for i, buf in enumerate(buffers):
            if buf is not None:
                delayed[i] = buf[0]       # oldest entry
                buf.append(spiked[i])      # push current (auto-pops oldest)
        return delayed

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------

    def simulate(self, duration: float, dt: float,
                 I_ext: np.ndarray) -> np.ndarray:
        """
        Run simulation, return (N, T) voltage traces.

        I_ext: shape (N, T)  where T = int(duration/dt)
        """
        if not self._finalised:
            self._finalise()
        if not self._delays_ready:
            self._setup_delay_buffers(dt)

        num_steps = int(duration / dt)
        traces = np.empty((self.N, num_steps))

        # Pre-compute constants that don't change per step
        if self.S_exp > 0:
            exp_decay = np.exp(-dt / self.exp_tau)
        if self.S_alpha > 0:
            alpha_inv_tau = 1.0 / self.alpha_tau
        if self.S_dexp > 0:
            dexp_rise_decay = np.exp(-dt / self.dexp_tau_rise)
            dexp_fall_decay = np.exp(-dt / self.dexp_tau_decay)

        for t in range(num_steps):
            # Record voltages
            traces[:, t] = self.V

            # ---- Synaptic currents (all types contribute) ----
            I_syn = np.zeros(self.N)

            if self.S_exp > 0:
                V_post = self.V[self.exp_post]
                np.add.at(I_syn, self.exp_post,
                          self.exp_g * (self.exp_E_syn - V_post))

            if self.S_alpha > 0:
                V_post = self.V[self.alpha_post]
                np.add.at(I_syn, self.alpha_post,
                          self.alpha_g * (self.alpha_E_syn - V_post))

            if self.S_dexp > 0:
                V_post = self.V[self.dexp_post]
                np.add.at(I_syn, self.dexp_post,
                          self.dexp_g * (self.dexp_E_syn - V_post))

            I_total = I_ext[:, t] + I_syn

            # ---- RK4 neuron step ----
            self.V, self.m, self.h, self.n = _hh_rk4_step(
                self.V, self.m, self.h, self.n, I_total, dt
            )

            # ---- Update exponential synapses ----
            if self.S_exp > 0:
                V_pre = self.V[self.exp_pre]
                spiked = (V_pre > 0.0) & (self.exp_V_prev <= 0.0)
                self.exp_V_prev = V_pre.copy()

                if self._has_exp_dly:
                    spiked = self._apply_delays(spiked, self._exp_dbufs)

                self.exp_g[spiked] += self.exp_weight[spiked]
                self.exp_g *= exp_decay
                np.clip(self.exp_g, 0.0, 1000.0, out=self.exp_g)

            # ---- Update alpha synapses ----
            if self.S_alpha > 0:
                V_pre = self.V[self.alpha_pre]
                spiked = (V_pre > 0.0) & (self.alpha_V_prev <= 0.0)
                self.alpha_V_prev = V_pre.copy()

                if self._has_alpha_dly:
                    spiked = self._apply_delays(spiked, self._alpha_dbufs)

                # on spike: x += weight * e  (Euler's number)
                self.alpha_x[spiked] += self.alpha_weight[spiked] * np.e

                # ODE step:  dx/dt = -x/tau,  dg/dt = (x - g)/tau
                dx = -self.alpha_x * alpha_inv_tau
                dg = (self.alpha_x - self.alpha_g) * alpha_inv_tau
                self.alpha_x += dt * dx
                self.alpha_g += dt * dg
                np.clip(self.alpha_g, 0.0, None, out=self.alpha_g)

            # ---- Update double-exponential synapses ----
            if self.S_dexp > 0:
                V_pre = self.V[self.dexp_pre]
                spiked = (V_pre > 0.0) & (self.dexp_V_prev <= 0.0)
                self.dexp_V_prev = V_pre.copy()

                if self._has_dexp_dly:
                    spiked = self._apply_delays(spiked, self._dexp_dbufs)

                # on spike: bump both rise and decay components
                self.dexp_g_rise[spiked] += 1.0
                self.dexp_g_decay[spiked] += 1.0

                # exponential decay of both components
                self.dexp_g_rise *= dexp_rise_decay
                self.dexp_g_decay *= dexp_fall_decay

                # conductance = weight * norm * (decay - rise)
                self.dexp_g = self.dexp_weight * self.dexp_norm * \
                              (self.dexp_g_decay - self.dexp_g_rise)
                np.clip(self.dexp_g, 0.0, None, out=self.dexp_g)

        return traces


# =============================================================================
# Topology builders
# =============================================================================

def _build_chain(N: int, weight: float = 0.5, E_syn: float = 0.0,
                 tau: float = 2.0):
    """Return list of (pre, post, weight, E_syn, tau) for a chain."""
    return [(i, i + 1, weight, E_syn, tau) for i in range(N - 1)]


def _build_ring(N: int, weight: float = 0.5, E_syn: float = 0.0,
                tau: float = 2.0):
    syns = _build_chain(N, weight, E_syn, tau)
    syns.append((N - 1, 0, weight, E_syn, tau))
    return syns


def _build_all_to_all(N: int, weight: float = 0.5, E_syn: float = 0.0,
                      tau: float = 2.0):
    return [(i, j, weight, E_syn, tau)
            for i in range(N) for j in range(N) if i != j]


def _build_star(N: int, weight: float = 0.5, E_syn: float = 0.0,
                tau: float = 2.0):
    syns = []
    for i in range(1, N):
        syns.append((0, i, weight, E_syn, tau))
        syns.append((i, 0, weight, E_syn, tau))
    return syns


TOPOLOGIES = {
    "Chain": _build_chain,
    "Ring": _build_ring,
    "All-to-All": _build_all_to_all,
    "Star": _build_star,
}


# =============================================================================
# Benchmark runner
# =============================================================================

def _bench_cpp(N: int, synapses, duration: float, dt: float,
               I_val: float) -> float:
    """Return wall time (s) for C++ Network.simulate()."""
    net = Network(N)
    for pre, post, w, e, tau in synapses:
        net.add_synapse(pre, post, w, e, tau)
    num_steps = int(duration / dt)
    I_ext = np.full((N, num_steps), I_val)

    start = time.perf_counter()
    net.simulate(duration, dt, I_ext)
    return time.perf_counter() - start


def _bench_numpy(N: int, synapses, duration: float, dt: float,
                 I_val: float) -> float:
    """Return wall time (s) for NumpyHHNetwork.simulate()."""
    net = NumpyHHNetwork(N)
    for pre, post, w, e, tau in synapses:
        net.add_synapse(pre, post, w, e, tau)
    num_steps = int(duration / dt)
    I_ext = np.full((N, num_steps), I_val)

    start = time.perf_counter()
    net.simulate(duration, dt, I_ext)
    return time.perf_counter() - start


def validate_implementations(duration: float = 50.0, dt: float = 0.05,
                             I_val: float = 10.0):
    """
    Sanity check: run both implementations on a small chain and compare
    voltage traces.  Prints max absolute error per neuron.
    Returns True if traces are reasonably close.
    """
    N = 3
    synapses = _build_chain(N, weight=0.5, E_syn=0.0, tau=2.0)
    num_steps = int(duration / dt)
    I_ext = np.full((N, num_steps), I_val)

    # C++
    cpp_net = Network(N)
    for pre, post, w, e, tau in synapses:
        cpp_net.add_synapse(pre, post, w, e, tau)
    cpp_traces = np.array(cpp_net.simulate(duration, dt, I_ext))

    # NumPy
    np_net = NumpyHHNetwork(N)
    for pre, post, w, e, tau in synapses:
        np_net.add_synapse(pre, post, w, e, tau)
    np_traces = np_net.simulate(duration, dt, I_ext)

    print("Validation: C++ vs NumPy on 3-neuron chain")
    print(f"  Trace shapes: C++={cpp_traces.shape}  NumPy={np_traces.shape}")

    ok = True
    for i in range(N):
        max_err = np.max(np.abs(cpp_traces[i] - np_traces[i]))
        mean_err = np.mean(np.abs(cpp_traces[i] - np_traces[i]))
        print(f"  Neuron {i}: max_err={max_err:.4f} mV  mean_err={mean_err:.4f} mV")
        if max_err > 5.0:
            print(f"    WARNING: large discrepancy!")
            ok = False

    if ok:
        print("  PASS: implementations agree\n")
    else:
        print("  FAIL: significant disagreement — check numpy model\n")

    return ok


# Maximum N per topology to avoid excessively long runs
# All-to-all has N*(N-1) synapses — N=2000 would be ~4M synapses
TOPO_MAX_N = {
    "Chain": None,
    "Ring": None,
    "All-to-All": 500,
    "Star": None,
}


def run_benchmarks(
    sizes: List[int],
    duration: float = 100.0,
    dt: float = 0.05,
    I_val: float = 10.0,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run benchmarks for all topologies and sizes.

    Returns:
        {topology_name: {"cpp": [times], "numpy": [times], "sizes": [N]}}
    """
    results: Dict[str, Dict[str, list]] = {}

    for topo_name, builder in TOPOLOGIES.items():
        cpp_times: List[float] = []
        numpy_times: List[float] = []
        max_n = TOPO_MAX_N.get(topo_name)
        topo_sizes = [N for N in sizes if max_n is None or N <= max_n]

        if len(topo_sizes) < len(sizes):
            print(f"  ({topo_name}: capping at N={max_n} "
                  f"— {max_n*(max_n-1)} synapses already)")

        for N in topo_sizes:
            synapses = builder(N)
            n_syn = len(synapses)
            print(f"  {topo_name:>12s}  N={N:>4d}  synapses={n_syn:>7d} ... ",
                  end="", flush=True)

            t_cpp = _bench_cpp(N, synapses, duration, dt, I_val)
            t_np = _bench_numpy(N, synapses, duration, dt, I_val)

            cpp_times.append(t_cpp)
            numpy_times.append(t_np)
            speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
            print(f"C++={t_cpp:.3f}s  NumPy={t_np:.3f}s  "
                  f"speedup={speedup:.1f}x")

        results[topo_name] = {
            "cpp": cpp_times,
            "numpy": numpy_times,
            "sizes": topo_sizes,
        }

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_timing(results: Dict, sizes: List[int], figs_dir: Path):
    """Time vs neuron count (log-log), one subplot per topology."""
    n_topo = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (topo_name, data) in enumerate(results.items()):
        ax = axes[idx]
        topo_sizes = data["sizes"]
        ax.loglog(topo_sizes, data["cpp"], 'o-', color='tab:blue',
                  linewidth=2, markersize=6, label='C++ backend')
        ax.loglog(topo_sizes, data["numpy"], 's--', color='tab:red',
                  linewidth=2, markersize=6, label='NumPy')
        ax.set_title(topo_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of neurons')
        ax.set_ylabel('Wall time (s)')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Network Simulation: C++ vs NumPy Timing',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_time_vs_neurons.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {figs_dir / 'benchmark_time_vs_neurons.png'}")


def plot_speedup(results: Dict, sizes: List[int], figs_dir: Path):
    """Speedup ratio (numpy/cpp) vs neuron count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for idx, (topo_name, data) in enumerate(results.items()):
        topo_sizes = data["sizes"]
        speedups = [n / c if c > 0 else 0
                    for c, n in zip(data["cpp"], data["numpy"])]
        ax.plot(topo_sizes, speedups, f'{markers[idx]}-', color=colors[idx],
                linewidth=2, markersize=7, label=topo_name)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Parity')
    ax.set_xscale('log')
    ax.set_xlabel('Number of neurons', fontsize=12)
    ax.set_ylabel('Speedup (NumPy time / C++ time)', fontsize=12)
    ax.set_title('C++ Backend Speedup Over Pure NumPy',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_speedup.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_speedup.png'}")


# =============================================================================
# Isolation benchmarks: neurons vs synapses
# =============================================================================

def _build_random_synapses(N: int, num_synapses: int,
                           weight: float = 0.5, E_syn: float = 0.0,
                           tau: float = 2.0, seed: int = 42):
    """Build a fixed number of random (pre, post) synapses for N neurons."""
    rng = np.random.default_rng(seed)
    synapses = []
    max_possible = N * (N - 1)
    num_synapses = min(num_synapses, max_possible)
    # Generate unique (pre, post) pairs
    pairs = set()
    while len(pairs) < num_synapses:
        pre = int(rng.integers(0, N))
        post = int(rng.integers(0, N))
        if pre != post and (pre, post) not in pairs:
            pairs.add((pre, post))
    for pre, post in pairs:
        synapses.append((pre, post, weight, E_syn, tau))
    return synapses


def _build_sparse_ring(N: int, k: int = 2,
                       weight: float = 0.5, E_syn: float = 0.0,
                       tau: float = 2.0):
    """Each neuron connects to its next k neighbours. Synapses = N*k."""
    synapses = []
    for i in range(N):
        for offset in range(1, k + 1):
            post = (i + offset) % N
            synapses.append((i, post, weight, E_syn, tau))
    return synapses


def run_synapse_scaling(duration: float = 100.0, dt: float = 0.05,
                        I_val: float = 10.0):
    """
    Fix neuron count, increase synapse count.
    Returns dict with timing data.
    """
    N = 200  # fixed neuron count
    synapse_counts = [50, 200, 500, 1000, 2000, 5000, 10000, 20000,
                      N * (N - 1)]  # last = fully connected
    cpp_times = []
    numpy_times = []
    actual_syn_counts = []

    print(f"\n--- Synapse scaling (fixed N={N}) ---")
    for target_s in synapse_counts:
        synapses = _build_random_synapses(N, target_s)
        actual_s = len(synapses)
        actual_syn_counts.append(actual_s)
        print(f"  N={N}  synapses={actual_s:>6d} ... ", end="", flush=True)

        t_cpp = _bench_cpp(N, synapses, duration, dt, I_val)
        t_np = _bench_numpy(N, synapses, duration, dt, I_val)
        cpp_times.append(t_cpp)
        numpy_times.append(t_np)
        speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
        print(f"C++={t_cpp:.3f}s  NumPy={t_np:.3f}s  speedup={speedup:.1f}x")

    return {"synapse_counts": actual_syn_counts,
            "cpp": cpp_times, "numpy": numpy_times, "N": N}


def run_neuron_scaling(duration: float = 100.0, dt: float = 0.05,
                       I_val: float = 10.0):
    """
    Increase neuron count, keep synapses per neuron fixed (sparse ring k=2).
    Synapses grow linearly: S = 2*N.
    """
    neuron_counts = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
    k = 2  # each neuron connects to 2 neighbours
    cpp_times = []
    numpy_times = []
    syn_counts = []

    print(f"\n--- Neuron scaling (sparse ring, k={k}, synapses=2*N) ---")
    for N in neuron_counts:
        synapses = _build_sparse_ring(N, k=k)
        actual_s = len(synapses)
        syn_counts.append(actual_s)
        print(f"  N={N:>5d}  synapses={actual_s:>6d} ... ",
              end="", flush=True)

        t_cpp = _bench_cpp(N, synapses, duration, dt, I_val)
        t_np = _bench_numpy(N, synapses, duration, dt, I_val)
        cpp_times.append(t_cpp)
        numpy_times.append(t_np)
        speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
        print(f"C++={t_cpp:.3f}s  NumPy={t_np:.3f}s  speedup={speedup:.1f}x")

    return {"neuron_counts": neuron_counts, "synapse_counts": syn_counts,
            "cpp": cpp_times, "numpy": numpy_times, "k": k}


def plot_isolation(syn_data: Dict, neuron_data: Dict, figs_dir: Path):
    """Plot the neuron-vs-synapse isolation benchmark."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top-left: Synapse scaling, absolute time ---
    ax = axes[0, 0]
    ax.loglog(syn_data["synapse_counts"], syn_data["cpp"], 'o-',
              color='tab:blue', linewidth=2, markersize=6, label='C++')
    ax.loglog(syn_data["synapse_counts"], syn_data["numpy"], 's--',
              color='tab:red', linewidth=2, markersize=6, label='NumPy')
    ax.set_xlabel('Number of synapses')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Synapse Scaling (fixed N={syn_data["N"]})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # --- Top-right: Synapse scaling, speedup ---
    ax = axes[0, 1]
    speedups = [n / c if c > 0 else 0
                for c, n in zip(syn_data["cpp"], syn_data["numpy"])]
    ax.semilogx(syn_data["synapse_counts"], speedups, 'D-',
                color='tab:purple', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Number of synapses')
    ax.set_ylabel('Speedup (NumPy / C++)')
    ax.set_title(f'Speedup vs Synapse Count (fixed N={syn_data["N"]})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    # --- Bottom-left: Neuron scaling, absolute time ---
    ax = axes[1, 0]
    ax.loglog(neuron_data["neuron_counts"], neuron_data["cpp"], 'o-',
              color='tab:blue', linewidth=2, markersize=6, label='C++')
    ax.loglog(neuron_data["neuron_counts"], neuron_data["numpy"], 's--',
              color='tab:red', linewidth=2, markersize=6, label='NumPy')
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Neuron Scaling (sparse, {neuron_data["k"]} synapses/neuron)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # --- Bottom-right: Neuron scaling, speedup ---
    ax = axes[1, 1]
    speedups = [n / c if c > 0 else 0
                for c, n in zip(neuron_data["cpp"], neuron_data["numpy"])]
    ax.semilogx(neuron_data["neuron_counts"], speedups, 'D-',
                color='tab:purple', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Speedup (NumPy / C++)')
    ax.set_title(f'Speedup vs Neuron Count ({neuron_data["k"]} syn/neuron)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Isolating Scaling: Neurons vs Synapses',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_isolation.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_isolation.png'}")


# =============================================================================
# Duration scaling benchmark
# =============================================================================

def run_duration_scaling(dt: float = 0.05, I_val: float = 10.0):
    """
    Fixed moderate network (100 neurons, sparse ring k=5 → 500 synapses),
    sweep simulation duration.
    """
    N = 100
    k = 5
    synapses = _build_sparse_ring(N, k=k)
    durations = [10, 25, 50, 100, 250, 500, 1000, 2000]
    cpp_times = []
    numpy_times = []

    print(f"\n--- Duration scaling (N={N}, k={k}, "
          f"synapses={len(synapses)}) ---")
    for dur in durations:
        num_steps = int(dur / dt)
        I_ext = np.full((N, num_steps), I_val)

        # C++
        cpp_net = Network(N)
        for pre, post, w, e, tau in synapses:
            cpp_net.add_synapse(pre, post, w, e, tau)
        start = time.perf_counter()
        cpp_net.simulate(dur, dt, I_ext)
        t_cpp = time.perf_counter() - start

        # NumPy
        np_net = NumpyHHNetwork(N)
        for pre, post, w, e, tau in synapses:
            np_net.add_synapse(pre, post, w, e, tau)
        start = time.perf_counter()
        np_net.simulate(dur, dt, I_ext)
        t_np = time.perf_counter() - start

        cpp_times.append(t_cpp)
        numpy_times.append(t_np)
        speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
        print(f"  duration={dur:>5.0f} ms  steps={num_steps:>6d} ... "
              f"C++={t_cpp:.3f}s  NumPy={t_np:.3f}s  speedup={speedup:.1f}x")

    return {"durations": durations, "cpp": cpp_times, "numpy": numpy_times,
            "N": N, "k": k, "S": len(synapses)}


def plot_duration_scaling(dur_data: Dict, figs_dir: Path):
    """Plot timing and speedup vs simulation duration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    durations = dur_data["durations"]
    N = dur_data["N"]
    S = dur_data["S"]

    # --- Left: absolute time ---
    ax = axes[0]
    ax.loglog(durations, dur_data["cpp"], 'o-', color='tab:blue',
              linewidth=2, markersize=6, label='C++')
    ax.loglog(durations, dur_data["numpy"], 's--', color='tab:red',
              linewidth=2, markersize=6, label='NumPy')
    ax.set_xlabel('Simulation duration (ms)')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Duration Scaling (N={N}, {S} synapses)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # --- Right: speedup ---
    ax = axes[1]
    speedups = [n / c if c > 0 else 0
                for c, n in zip(dur_data["cpp"], dur_data["numpy"])]
    ax.semilogx(durations, speedups, 'D-', color='tab:purple',
                linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Simulation duration (ms)')
    ax.set_ylabel('Speedup (NumPy / C++)')
    ax.set_title('Speedup vs Duration', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Fixed Network, Scaling Simulation Time',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_duration.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_duration.png'}")


# =============================================================================
# Memory benchmark
# =============================================================================

import os
import time
import threading
import gc
import psutil
import numpy as np


def _peak_rss_mb_during(callable_fn, *args, sample_ms: int = 2, **kwargs) -> float:
    """
    Poll process RSS while callable_fn runs; return peak RSS (MB).
    Note: This measures *total process memory* (Python + NumPy native + C++ native).
    """
    proc = psutil.Process(os.getpid())
    peak = proc.memory_info().rss
    done = False
    exc: Exception | None = None

    def runner():
        nonlocal done, exc
        try:
            callable_fn(*args, **kwargs)
        except Exception as e:
            exc = e
        finally:
            done = True

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    # Poll RSS until the worker finishes
    while not done:
        rss = proc.memory_info().rss
        if rss > peak:
            peak = rss
        time.sleep(sample_ms / 1000.0)

    t.join()

    if exc is not None:
        raise exc

    return peak / (1024 * 1024)


def _measure_memory_cpp(N: int, synapses, duration: float, dt: float, I_val: float) -> float:
    """Return peak process RSS (MB) while running C++ simulate()."""
    num_steps = int(duration / dt)

    # Try to reduce noise from previous runs
    gc.collect()

    I_ext = np.full((N, num_steps), I_val)

    net = Network(N)
    for pre, post, w, e, tau in synapses:
        net.add_synapse(pre, post, w, e, tau)

    # IMPORTANT: do not keep traces around; we want peak during simulate, not after
    peak_mb = _peak_rss_mb_during(net.simulate, duration, dt, I_ext)

    # Cleanup to reduce cross-run carryover (RSS may not immediately drop due to allocators)
    del net, I_ext
    gc.collect()

    return peak_mb


def _measure_memory_numpy(N: int, synapses, duration: float, dt: float, I_val: float) -> float:
    """Return peak process RSS (MB) while running NumPy simulate()."""
    num_steps = int(duration / dt)

    gc.collect()

    I_ext = np.full((N, num_steps), I_val)

    net = NumpyHHNetwork(N)
    for pre, post, w, e, tau in synapses:
        net.add_synapse(pre, post, w, e, tau)

    peak_mb = _peak_rss_mb_during(net.simulate, duration, dt, I_ext)

    del net, I_ext
    gc.collect()

    return peak_mb



def run_memory_benchmark(duration: float = 100.0, dt: float = 0.05,
                         I_val: float = 10.0):
    """Measure peak memory for all-to-all networks up to 500 neurons."""
    sizes = [5, 10, 20, 50, 100, 200, 500]
    cpp_mem = []
    numpy_mem = []

    print(f"\n--- Memory benchmark (all-to-all) ---")
    for N in sizes:
        synapses = _build_all_to_all(N)
        n_syn = len(synapses)
        print(f"  N={N:>4d}  synapses={n_syn:>7d} ... ", end="", flush=True)

        m_cpp = _measure_memory_cpp(N, synapses, duration, dt, I_val)
        m_np = _measure_memory_numpy(N, synapses, duration, dt, I_val)

        cpp_mem.append(m_cpp)
        numpy_mem.append(m_np)
        print(f"C++={m_cpp:.2f} MB  NumPy={m_np:.2f} MB")

    return {"sizes": sizes, "cpp": cpp_mem, "numpy": numpy_mem}


def plot_memory(mem_data: Dict, figs_dir: Path):
    """Plot memory usage comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = mem_data["sizes"]
    ax.plot(sizes, mem_data["cpp"], 'o-', color='tab:blue',
            linewidth=2, markersize=7, label='C++ backend')
    ax.plot(sizes, mem_data["numpy"], 's--', color='tab:red',
            linewidth=2, markersize=7, label='NumPy')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of neurons', fontsize=12)
    ax.set_ylabel('Peak memory (MB)', fontsize=12)
    ax.set_title('Peak Memory Usage (All-to-All Network)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_memory.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_memory.png'}")


# =============================================================================
# Main
# =============================================================================

def setup_output_dir():
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


def main():
    print("=" * 60)
    print("Network Benchmark: C++ Backend vs Pure NumPy")
    print("=" * 60)

    figs_dir = setup_output_dir()

    sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000]
    duration = 100.0  # ms
    dt = 0.05         # ms (RK4-safe for HH)
    I_val = 10.0      # uA/cm^2

    print(f"\nSimulation: {duration} ms, dt={dt} ms, I_ext={I_val}")
    print(f"Network sizes: {sizes}")
    print(f"Output: {figs_dir}\n")

    # Validate that both implementations produce matching results
    validate_implementations(duration=50.0, dt=dt, I_val=I_val)

    results = run_benchmarks(sizes, duration, dt, I_val)

    plot_timing(results, sizes, figs_dir)
    plot_speedup(results, sizes, figs_dir)

    # Isolation benchmarks: tease apart neuron vs synapse cost
    syn_data = run_synapse_scaling(duration, dt, I_val)
    neuron_data = run_neuron_scaling(duration, dt, I_val)
    plot_isolation(syn_data, neuron_data, figs_dir)

    # Duration scaling: fixed network, increasing simulation time
    dur_data = run_duration_scaling(dt=dt, I_val=I_val)
    plot_duration_scaling(dur_data, figs_dir)

    # Memory benchmark
    mem_data = run_memory_benchmark(duration, dt, I_val)
    plot_memory(mem_data, figs_dir)

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
