"""
str_gate_diagnostic.py — compare benchmark forward-Euler vs library exp-Euler
for a single striatal neuron initialised at various voltages.

Answers the core question: does the benchmark model also produce errant spikes
when a neuron is initialised above threshold (e.g. V=-53.45 mV)?

Run from the project root:
    python benchmarks/str_gate_diagnostic.py
"""

import sys, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT + "/src")

# ---------------------------------------------------------------------------
# Benchmark gate functions (exact copies from simulate_network_model.py)
# ---------------------------------------------------------------------------

def alpham(V): return (0.32 * (54 + V)) / (1 - np.exp((-54 - V) / 4))
def betam(V):  return 0.28 * (V + 27) / (np.exp((27 + V) / 5) - 1)
def alphah(V): return 0.128 * np.exp((-50 - V) / 18)
def betah(V):  return 4 / (1 + np.exp((-27 - V) / 5))
def alphan(V): return (0.032 * (52 + V)) / (1 - np.exp((-52 - V) / 5))
def betan(V):  return 0.5 * np.exp((-57 - V) / 40)
def alphap(V): return (3.209e-4 * (30 + V)) / (1 - np.exp((-30 - V) / 9))
def betap(V):  return (-3.209e-4 * (30 + V)) / (1 - np.exp((30 + V) / 9))


def _safe_ss(alpha_fn, beta_fn, V):
    """alpha/(alpha+beta), handling singularity at denominator ≈ 0."""
    a = alpha_fn(V)
    b = beta_fn(V)
    denom = a + b
    if abs(denom) < 1e-10:
        return 0.5
    return a / denom


def _safe_tau(alpha_fn, beta_fn, V):
    """1/(alpha+beta)."""
    denom = alpha_fn(V) + beta_fn(V)
    if abs(denom) < 1e-10:
        return 1e10
    return 1.0 / denom


# ---------------------------------------------------------------------------
# Channel parameters (identical to benchmark gna[3], gk[3], gl[3], gm)
# ---------------------------------------------------------------------------
GNA, ENA = 100.0, 50.0
GK,  EK  = 80.0, -100.0
GL,  EL  = 0.1,  -67.0
GM,  EM  = 2.6,  -100.0   # healthy (pd=0)


def net_current(V, m, h, n, p):
    """Outward current (positive = repolarising, negative = depolarising)."""
    I_Na = GNA * m**3 * h * (V - ENA)
    I_K  = GK  * n**4     * (V - EK)
    I_L  = GL             * (V - EL)
    I_M  = GM  * p        * (V - EM)
    return I_Na + I_K + I_L + I_M


# ---------------------------------------------------------------------------
# Simulate a single neuron using the benchmark's forward-Euler scheme
# ---------------------------------------------------------------------------

def sim_forward_euler(V_init, t_end=2.0, dt=0.01):
    """Benchmark update order: compute currents → update V → update gates."""
    V  = V_init
    m  = _safe_ss(alpham, betam, V)
    h  = _safe_ss(alphah, betah, V)
    n  = _safe_ss(alphan, betan, V)
    p  = _safe_ss(alphap, betap, V)

    n_steps = int(round(t_end / dt))
    V_trace = np.empty(n_steps + 1)
    V_trace[0] = V

    for i in range(n_steps):
        I_ion = net_current(V, m, h, n, p)        # old gates
        V_new = V + dt * (-I_ion)                  # update V
        m = m + dt * (alpham(V) * (1 - m) - betam(V) * m)   # update gates (old V)
        h = h + dt * (alphah(V) * (1 - h) - betah(V) * h)
        n = n + dt * (alphan(V) * (1 - n) - betan(V) * n)
        p = p + dt * (alphap(V) * (1 - p) - betap(V) * p)
        V = V_new
        V_trace[i + 1] = V

    return V_trace


# ---------------------------------------------------------------------------
# Simulate using library exp-Euler ordering (gates updated first, then V)
# ---------------------------------------------------------------------------

def sim_exp_euler(V_init, t_end=2.0, dt=0.01):
    """Library update order: update gates → compute current (new gates) → update V."""
    V  = V_init
    m  = _safe_ss(alpham, betam, V)
    h  = _safe_ss(alphah, betah, V)
    n  = _safe_ss(alphan, betan, V)
    p  = _safe_ss(alphap, betap, V)

    n_steps = int(round(t_end / dt))
    V_trace = np.empty(n_steps + 1)
    V_trace[0] = V

    for i in range(n_steps):
        # exp-Euler gate update (uses old V)
        def _step(x, af, bf):
            xss  = _safe_ss(af, bf, V)
            taux = _safe_tau(af, bf, V)
            return xss + (x - xss) * np.exp(-dt / taux)

        m = _step(m, alpham, betam)
        h = _step(h, alphah, betah)
        n = _step(n, alphan, betan)
        p = _step(p, alphap, betap)

        I_ion = net_current(V, m, h, n, p)   # new gates, old V
        V = V + dt * (-I_ion)
        V_trace[i + 1] = V

    return V_trace


# ---------------------------------------------------------------------------
# Print a trajectory comparison table
# ---------------------------------------------------------------------------

def compare_at_voltage(V_init, t_end=2.0, dt=0.01, print_step=25,
                        spike_threshold=-10.0):
    V_bm  = sim_forward_euler(V_init, t_end, dt)
    V_lib = sim_exp_euler(V_init, t_end, dt)
    t_arr = np.arange(len(V_bm)) * dt

    # Gate SS values at init
    m0 = _safe_ss(alpham, betam, V_init)
    h0 = _safe_ss(alphah, betah, V_init)
    n0 = _safe_ss(alphan, betan, V_init)
    p0 = _safe_ss(alphap, betap, V_init)
    I0 = net_current(V_init, m0, h0, n0, p0)

    print(f"\n{'='*65}")
    print(f"  V_init = {V_init:.2f} mV   (std = {(V_init - (-63.8)) / 5:.2f} σ from mean -63.8)")
    print(f"  Gate SS: m={m0:.4f}  h={h0:.4f}  n={n0:.4f}  p={p0:.4f}")
    print(f"  Net I(t=0) = {I0:+.3f} µA/cm²  (neg = depolarising → {'FIRES' if I0 < 0 else 'repolarises'})")
    print(f"  dV/dt(t=0) = {-I0:+.3f} mV/ms")
    print(f"{'='*65}")

    print(f"\n{'t(ms)':>6s} | {'V_bench(mV)':>11s} | {'V_lib(mV)':>10s} | {'diff(mV)':>10s}")
    print("-" * 46)
    for i in range(0, len(t_arr), print_step):
        print(f"{t_arr[i]:6.2f} | {V_bm[i]:11.3f} | {V_lib[i]:10.3f} | {V_lib[i]-V_bm[i]:10.5f}")
    print("-" * 46)

    bm_spike_t  = t_arr[np.argmax(V_bm  > spike_threshold)] if np.any(V_bm  > spike_threshold) else None
    lib_spike_t = t_arr[np.argmax(V_lib > spike_threshold)] if np.any(V_lib > spike_threshold) else None
    print(f"\n  Benchmark fires: {'YES at t=' + f'{bm_spike_t:.3f} ms' if bm_spike_t is not None else 'NO'}")
    print(f"  Library fires:   {'YES at t=' + f'{lib_spike_t:.3f} ms' if lib_spike_t is not None else 'NO'}")


# ---------------------------------------------------------------------------
# Find the effective spike threshold (where net I crosses zero)
# ---------------------------------------------------------------------------

def find_threshold():
    print("\n--- Net ionic current at SS as a function of V ---")
    print(f"  (negative = net inward = depolarising → above threshold)")
    print(f"  {'V(mV)':>7s}  {'I_net(µA/cm²)':>14s}  {'dV/dt(mV/ms)':>13s}")
    print("  " + "-" * 40)
    for V in np.arange(-75, -45, 1.0):
        m = _safe_ss(alpham, betam, V)
        h = _safe_ss(alphah, betah, V)
        n = _safe_ss(alphan, betan, V)
        p = _safe_ss(alphap, betap, V)
        I = net_current(V, m, h, n, p)
        mark = " ← threshold" if abs(I) < 1.0 else ""
        print(f"  {V:7.1f}  {I:14.4f}  {-I:13.4f}{mark}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  STRIATAL NEURON ERRANT SPIKE DIAGNOSTIC")
    print("  Compares benchmark forward-Euler vs library exp-Euler")
    print("=" * 65)

    # 1. Show threshold
    find_threshold()

    # 2. Compare at specific voltages of interest
    compare_at_voltage(-53.45)    # N08 from seed 6536 (IC-driven errant spike)
    compare_at_voltage(-58.0)     # intermediate
    compare_at_voltage(-63.8)     # benchmark mean init voltage
    compare_at_voltage(-69.0)     # near library resting potential

    # 3. Sweep: find the lowest V_init that causes an errant spike in EACH scheme
    print("\n\n--- Spike threshold scan (2 ms window, dt=0.01 ms) ---")
    print(f"  Identifies lowest V_init that fires in each integration scheme.")
    print(f"  {'V_init':>8s} | {'bench fires':>11s} | {'lib fires':>11s}")
    print("  " + "-" * 40)
    for V_init in np.arange(-75, -48, 1.0):
        V_bm  = sim_forward_euler(V_init)
        V_lib = sim_exp_euler(V_init)
        bm  = np.any(V_bm  > -10)
        lib = np.any(V_lib > -10)
        if bm or lib:
            print(f"  {V_init:8.1f} | {'FIRES':>11s} | {'FIRES':>11s}" if (bm and lib)
                  else (f"  {V_init:8.1f} | {'FIRES':>11s} | {'no':>11s}" if bm
                  else f"  {V_init:8.1f} | {'no':>11s} | {'FIRES':>11s}"))
        else:
            print(f"  {V_init:8.1f} | {'no':>11s} | {'no':>11s}")
