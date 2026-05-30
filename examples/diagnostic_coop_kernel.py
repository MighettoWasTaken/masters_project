"""
Cooperative kernel bottleneck diagnostics.

Measures µs/step for systematically varied configurations to distinguish:
  - Fixed overhead (stream sync, kernel launch)
  - grid.sync() barrier overhead (7 per step × barrier cost)
  - Per-neuron compute (composable_step_single)
  - Per-synapse compute (accumulate_isyn + update_synapse_state)

Run:
    python examples/diagnostic_coop_kernel.py
"""

from timeit import default_timer as timer
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hodgkin_huxley as hh
from hodgkin_huxley import RegionalNetwork, SynapseSpec, Device, RecordingConfig
from hodgkin_huxley._core import NeuronModelSpec, IzhikevichType
from benchmarks.ctxbgth_model import build_network

DT       = 0.01   # ms
N_WARM   = 3      # warmup repetitions
N_REP    = 3      # timed repetitions

CTXBG_I_EXT = {"TH": 1.2, "GPe": 3.0, "GPi": 3.0}
SPIKE_CFG   = RecordingConfig(["spikes"], spike_threshold=-10.0)


def timed(net, tmax, I_ext, cfg, n_rep=N_REP):
    times = []
    for _ in range(n_rep):
        t0 = timer()
        net.simulate(tmax, DT, I_ext, record=cfg)
        times.append(timer() - t0)
    return float(np.mean(times))


def us_per_step(t_sec, tmax):
    return t_sec * 1e6 / (tmax / DT)


def hline():
    print("-" * 55)


# ============================================================
# 1. Duration scaling — isolates per-step vs fixed overhead
# ============================================================
print("=" * 55)
print("1. CTX-BG-TH (n=10, GPU): µs/step vs simulation duration")
print("=" * 55)
print(f"{'tmax (ms)':>10}  {'n_steps':>8}  {'total (s)':>10}  {'µs/step':>9}")
hline()

ctxbg_results = {}
net10 = build_network(n=10)
net10.to(Device.cuda(0))
# warmup
for _ in range(N_WARM):
    net10.simulate(1.0, DT, CTXBG_I_EXT, record=SPIKE_CFG)

for tmax in [1.0, 10.0, 100.0, 1000.0]:
    t = timed(net10, tmax, CTXBG_I_EXT, SPIKE_CFG)
    ctxbg_results[tmax] = t
    print(f"{tmax:>10.1f}  {int(tmax/DT):>8}  {t:>10.4f}  {us_per_step(t, tmax):>9.2f}")

print()
print("If µs/step is FLAT  → per-step overhead dominates (grid.sync or compute).")
print("If µs/step DROPS    → fixed overhead (stream sync) is significant.")

# ============================================================
# 2. Neuron count scaling — per-neuron compute contribution
# ============================================================
print()
print("=" * 55)
print("2. CTX-BG-TH (GPU, 100ms): µs/step vs n per population")
print("=" * 55)
print(f"{'n/pop':>6}  {'neurons':>8}  {'synapses':>10}  {'total (s)':>10}  {'µs/step':>9}")
hline()

TMAX = 100.0
for n in [2, 5, 10, 20]:
    net = build_network(n=n)
    net.to(Device.cuda(0))
    for _ in range(N_WARM):
        net.simulate(1.0, DT, CTXBG_I_EXT, record=SPIKE_CFG)
    t = timed(net, TMAX, CTXBG_I_EXT, SPIKE_CFG)
    n_syn = net._rnet.num_synapses
    print(f"{n:>6}  {n*8:>8}  {n_syn:>10}  {t:>10.4f}  {us_per_step(t, TMAX):>9.2f}")

print()
print("FLAT w.r.t. n → grid.sync() L2-flush bottleneck (independent of work size).")
print("LINEAR in n   → neuron compute is the bottleneck.")

# ============================================================
# 3. Pure Izhikevich network — overhead floor
# ============================================================
print()
print("=" * 55)
print("3. Simple IZ network (GPU, 100ms): overhead floor")
print("=" * 55)
print(f"{'neurons':>8}  {'synapses':>10}  {'total (s)':>10}  {'µs/step':>9}")
hline()

iz_spec = NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING)

def build_iz(n_neurons, n_syns_per_neuron=0, on_cuda=True):
    net = RegionalNetwork()
    net.add_population("E", n_neurons, model=iz_spec)
    if n_syns_per_neuron > 0:
        k = min(n_syns_per_neuron, n_neurons - 1)
        for _ in range(k):
            net.connect("E", "E", "random_permutation",
                        weight=0.1, synapse=SynapseSpec.ampa(),
                        seed=np.random.randint(1, 2**31))
    if on_cuda:
        net.to(Device.cuda(0))
    return net

for (nn, ns) in [(1, 0), (10, 0), (80, 0), (80, 1), (80, 6)]:
    net = build_iz(nn, ns)
    for _ in range(N_WARM):
        net.simulate(1.0, DT, {}, record=SPIKE_CFG)
    t = timed(net, TMAX, {}, SPIKE_CFG)
    n_syns_actual = net._rnet.num_synapses
    print(f"{nn:>8}  {n_syns_actual:>10}  {t:>10.4f}  {us_per_step(t, TMAX):>9.2f}")

print()
print("80 neurons, 0 syns → pure neuron-step + 7×grid.sync overhead.")
print("Difference 0-syn vs N-syn → marginal synapse processing cost.")

# ============================================================
# 4. CPU vs GPU for CTX-BG-TH
# ============================================================
print()
print("=" * 55)
print("4. CTX-BG-TH: CPU vs GPU (100ms, n=10)")
print("=" * 55)
print(f"{'device':>8}  {'total (s)':>10}  {'µs/step':>9}  {'speedup':>8}")
hline()

cpu_t = None
for on_cuda, label in [(False, "CPU"), (True, "GPU")]:
    net = build_network(n=10)
    if on_cuda:
        net.to(Device.cuda(0))
    for _ in range(N_WARM):
        net.simulate(1.0, DT, CTXBG_I_EXT, record=SPIKE_CFG)
    t = timed(net, TMAX, CTXBG_I_EXT, SPIKE_CFG)
    if label == "CPU":
        cpu_t = t
        speedup_str = "  —"
    else:
        speedup_str = f"{cpu_t/t:>6.2f}×" if t > 0 else "  ?"
    print(f"{label:>8}  {t:>10.4f}  {us_per_step(t, TMAX):>9.2f}  {speedup_str:>8}")

# ============================================================
# 5. Fit: fixed + per-step cost
# ============================================================
print()
print("=" * 55)
print("5. Linear fit: T = fixed_overhead + per_step_cost × n_steps")
print("=" * 55)

steps_arr = np.array([int(x / DT) for x in [1.0, 10.0, 100.0, 1000.0]], dtype=float)
times_arr = np.array([ctxbg_results[x] for x in [1.0, 10.0, 100.0, 1000.0]])

# Least-squares fit
A = np.column_stack([np.ones_like(steps_arr), steps_arr])
result = np.linalg.lstsq(A, times_arr, rcond=None)
coeffs = result[0]
fixed_s  = coeffs[0]
slope_s  = coeffs[1]

print(f"  Fixed overhead (kernel launch + stream sync): {fixed_s*1000:.1f} ms")
print(f"  Per-step cost:                                {slope_s*1e6:.2f} µs/step")
print(f"  7 grid.sync() barriers → ~{slope_s*1e6/7:.2f} µs each if barrier-bound")
print()
t_1000ms = ctxbg_results.get(1000.0, times_arr[-1])
print(f"  Actual 1000ms GPU:  {t_1000ms:.3f}s  ({us_per_step(t_1000ms, 1000.0):.1f} µs/step)")
print(f"  Target:             ≤2.000s  ({2e6/(1000.0/DT):.1f} µs/step)")
print(f"  Gap factor:         {t_1000ms/2.0:.1f}×")
