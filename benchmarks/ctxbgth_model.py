"""
CTX-BG-TH Network Model — hodgkin_huxley library recreation.

Recreates the rat Parkinson's disease model (Hahn et al. 2019,
nihms803352) using the hodgkin_huxley library's built-in neuron
presets, composable model builder, and C++-accelerated simulation.

Network populations:
  TH       — Thalamus            (custom spec matching benchmark K-channel mechanism)
  STN      — Subthalamic Nucleus (NeuronModelSpec.stn preset)
  GPe      — Globus Pallidus ext (NeuronModelSpec.gpe preset)
  GPi      — Globus Pallidus int (NeuronModelSpec.gpi preset)
  Str_D2   — Striatum indirect   (HH-style composable, D2 receptors)
  Str_D1   — Striatum direct     (HH-style composable, D1 receptors)
  CTX_e    — Cortex excitatory   (Izhikevich Regular Spiking)
  CTX_i    — Cortex inhibitory   (Izhikevich Fast Spiking)
"""

import numpy as np
from timeit import default_timer as timer

from hodgkin_huxley import (
    RegionalNetwork,
    NeuronModelSpec, NeuronModel, Boltzmann, RateFunc, Tau,
    SynapseSpec,
    KineticSynapseSpec, KineticUpdateForm, KineticCurrentForm,
    IzhikevichParameters, IzhikevichType,
    DBSStimulator, DBSParameters,
    RecordingConfig,
    mtspectrumpt, beta_band_power,
)


# ---------------------------------------------------------------------------
# Thalamic relay cell matching benchmark (Wang 1994 / Rubin-Terman style)
# Key: K-channel depends on Na inactivation (1-h_Na), NOT T-current inactivation.
# With standard Rubin-Terman preset, h_T≈0.076 at rest → n_K=0.69 → huge K
# current requiring ~30 µA/cm² drive.  This custom spec fires with Iapp=1.2.
# ---------------------------------------------------------------------------

def _make_th_spec(V_init: float = -62.0) -> NeuronModelSpec:
    """
    Thalamic relay cell with benchmark-matching parameters.
    Gates: h_Na (alpha-beta), r (INF_TAU, T-current inactivation),
           m_Na (instant), p (instant), n_K (derived = 0.75*(1-h_Na)).
    Channels: Na (m^3·h), K (n_K^4), T (p^2·r), Leak.
    """
    model = NeuronModel("TH", C_m=1.0, V_init=V_init)

    # h_Na: alpha-beta (ah/bh from Wang 1994)
    #   ah(V) = 0.128*exp(-(V+46)/18)  → EXP_DECAY(0.128, 46, -18)
    #   bh(V) = 4/(1+exp(-(V+23)/5))   → SIGMOID(4, 23, -5)
    h_na_idx = model.add_gate("h_Na", update_form="alpha_beta",
        alpha=RateFunc.exp_decay(0.128, 46.0, -18.0),
        beta =RateFunc.sigmoid(4.0, 23.0, -5.0))

    # r (T-current inactivation): INF_TAU
    #   r_inf = 1/(1+exp((V+84)/4))         → Boltzmann(-84, -4)
    #   tau_r ≈ 0.15*(28+exp(-(V+25)/10.5)) → ~11 ms at V=-65 (constant approx)
    r_idx = model.add_gate("r", update_form="inf_tau", initial_value=0.25,
        inf=Boltzmann(-84.0, -4.0),
        tau=Tau.constant(11.0))

    # m_Na (instantaneous Na activation)
    m_na_idx = model.add_gate("m_Na", update_form="instant",
        inf=Boltzmann(-37.0, 7.0))

    # p (instantaneous T-current activation)
    #   p_inf = 1/(1+exp(-(V+60)/6.2)) → Boltzmann(-60, 6.2)
    p_idx = model.add_gate("p", update_form="instant",
        inf=Boltzmann(-60.0, 6.2))

    # n_K = 0.75*(1-h_Na): DERIVED from h_Na
    #   derived: gate = derived_a * (derived_b + derived_c * source)
    #          = 0.75 * (1 + (-1) * h_Na) = 0.75*(1-h_Na)
    nk_idx = model.add_gate("n_K", update_form="derived",
        derived_source_gate=h_na_idx,
        derived_a=0.75, derived_b=1.0, derived_c=-1.0)

    model.add_channel("Na",   g=3.0,   E_rev=50.0,   gates=[(m_na_idx, 3), (h_na_idx, 1)])
    model.add_channel("K",    g=5.0,   E_rev=-75.0,  gates=[(nk_idx, 4)])
    model.add_channel("T",    g=5.0,   E_rev=0.0,    gates=[(p_idx, 2), (r_idx, 1)])
    model.add_channel("Leak", g=0.05,  E_rev=-70.0)

    return model.to_spec()


# ---------------------------------------------------------------------------
# Striatum composable model (HH-style with Na, K-DR, Leak, M-type K)
# ---------------------------------------------------------------------------

def _make_striatum_spec(pd: int = 0, V_init: float = -63.8) -> NeuronModelSpec:
    """
    HH-style striatal neuron with Na, K-DR, Leak, and M-type K channels.
    Alpha-beta gates match the benchmark's alpham/betam/etc. functions.
    gm factor: (2.6 - 1.1*pd) encodes the PD-dependent M-channel reduction.
    """
    model = NeuronModel("Striatum", C_m=1.0, V_init=V_init)

    # Na channel gating: m^3 * h
    #   alpham = 0.32*(V+54)/(1-exp(-(V+54)/4))  → LINEAR_OVER_EXPM1
    #   betam  = 0.28*(V+27)/(exp((V+27)/5)-1)    → LINEAR_OVER_EXP
    m_idx = model.add_gate("m", update_form="alpha_beta",
        alpha=RateFunc.linear_over_expm1(0.32, 54.0, 4.0),
        beta =RateFunc.linear_over_exp  (0.28, 27.0, 5.0))

    #   alphah = 0.128*exp(-(V+50)/18)             → EXP_DECAY  (C<0 flips sign)
    #   betah  = 4/(1+exp(-(V+27)/5))              → SIGMOID     (C<0 flips sign)
    h_idx = model.add_gate("h", update_form="alpha_beta",
        alpha=RateFunc.exp_decay(0.128,  50.0, -18.0),
        beta =RateFunc.sigmoid  (4.0,    27.0,  -5.0))

    # K-DR channel gating: n^4
    #   alphan = 0.032*(V+52)/(1-exp(-(V+52)/5))  → LINEAR_OVER_EXPM1
    #   betan  = 0.5*exp(-(V+57)/40)              → EXP_DECAY
    n_idx = model.add_gate("n", update_form="alpha_beta",
        alpha=RateFunc.linear_over_expm1(0.032, 52.0,  5.0),
        beta =RateFunc.exp_decay        (0.5,   57.0, -40.0))

    # M-type K gating (slow, Kir-like): p^1
    #   alphap = 3.209e-4*(V+30)/(1-exp(-(V+30)/9)) → LINEAR_OVER_EXPM1
    #   betap  = 3.209e-4*(V+30)/(exp((V+30)/9)-1)  → LINEAR_OVER_EXP
    p_idx = model.add_gate("p", update_form="alpha_beta",
        alpha=RateFunc.linear_over_expm1(3.209e-4, 30.0, 9.0),
        beta =RateFunc.linear_over_exp  (3.209e-4, 30.0, 9.0))

    gm_eff = (2.6 - 1.1 * pd)   # PD reduces M-channel conductance

    model.add_channel("Leak", g=0.1,     E_rev=-67.0)
    model.add_channel("Na",   g=100.0,   E_rev=50.0,   gates=[(m_idx, 3), (h_idx, 1)])
    model.add_channel("K_DR", g=80.0,    E_rev=-100.0, gates=[(n_idx, 4)])
    model.add_channel("K_M",  g=gm_eff,  E_rev=-100.0, gates=[(p_idx, 1)])

    return model.to_spec()


# ---------------------------------------------------------------------------
# Kinetic GABA-A synapse (Ggaba(V) = 2*(1+tanh(V/4)))
# Used for intra-striatal inhibition (Str→Str)
# dS/dt = Ggaba(V)*(1-S) - S/tau_i  (exact-exponential integration in C++)
# ---------------------------------------------------------------------------

def _gaba_kinetic(tau_i: float = 13.0, E_syn: float = -80.0) -> KineticSynapseSpec:
    ks = KineticSynapseSpec()
    ks.update_form = KineticUpdateForm.TANH_GATE
    ks.tanh_amp    = 2.0   # Ggaba amplitude: 2*(1+tanh(...))
    ks.tanh_vh     = 0.0   # V half
    ks.tanh_k      = 4.0   # slope (V/4 inside tanh)
    ks.tau_decay   = tau_i
    ks.E_syn       = E_syn
    return ks


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------

def build_network(n: int = 10, pd: int = 0, seed: int = 42) -> RegionalNetwork:
    """
    Build the CTX-BG-TH network.

    Parameters
    ----------
    n   : neurons per population
    pd  : 0=healthy, 1=Parkinson's disease
    seed: RNG seed
    """
    rng = np.random.default_rng(seed)
    net = RegionalNetwork()

    # ---- Populations -------------------------------------------------------
    net.add_population("TH",     n, model=_make_th_spec())
    net.add_population("STN",    n, model=NeuronModelSpec.stn())
    net.add_population("GPe",    n, model=NeuronModelSpec.gpe())
    net.add_population("GPi",    n, model=NeuronModelSpec.gpi())
    net.add_population("Str_D2", n, model=_make_striatum_spec(pd=pd, V_init=-63.8))
    net.add_population("Str_D1", n, model=_make_striatum_spec(pd=pd, V_init=-63.8))

    iz_rs = IzhikevichParameters(); iz_rs.a, iz_rs.b, iz_rs.c, iz_rs.d = 0.02, 0.2, -65.0, 8.0
    iz_fs = IzhikevichParameters(); iz_fs.a, iz_fs.b, iz_fs.c, iz_fs.d = 0.10, 0.2, -65.0, 2.0
    net.add_population("CTX_e", n, parameters=iz_rs)
    net.add_population("CTX_i", n, parameters=iz_fs)

    # Randomise initial membrane potentials (matches benchmark scatter)
    net.randomize_membrane_potentials("TH",     -62.0, 5.0, seed=seed)
    net.randomize_membrane_potentials("STN",    -62.0, 5.0, seed=seed + 1)
    net.randomize_membrane_potentials("GPe",    -62.0, 5.0, seed=seed + 2)
    net.randomize_membrane_potentials("GPi",    -62.0, 5.0, seed=seed + 3)
    net.randomize_membrane_potentials("Str_D2", -63.8, 5.0, seed=seed + 4)
    net.randomize_membrane_potentials("Str_D1", -63.8, 5.0, seed=seed + 5)

    # ---- Synapse parameter shorthands (from benchmark) --------------------
    tau    = 5.0    # ms — alpha-function time constant
    gpeak  = 0.43
    gpeak1 = 0.3

    # ---- GPi → TH  (inhibitory alpha-function, delay=5ms) -----------------
    net.connect("GPi", "TH", "all_to_all",
                weight=0.112 / n,
                synapse=SynapseSpec.alpha(-85.0, tau), delay=5.0)

    # ---- GPe → STN  (inhibitory double-exponential, delay=4ms) ------------
    # Two cyclic-shifted copies: STN[j] receives from GPe[j+1] and GPe[j+2]
    for shift in (1, 2):
        net.connect("GPe", "STN", "shifted",
                    weight=0.5 / 2,
                    synapse=SynapseSpec.double_exponential(-85.0, tau_rise=0.4, tau_decay=7.7),
                    delay=4.0, shift=shift)

    # ---- STN → GPe  (excitatory AMPA + NMDA, sparse, delay=2ms) -----------
    # 2 of n STN neurons active per synapse type; each hits GPe[j] & GPe[j+1]
    ampa_idx = rng.permutation(n)[:2]
    nmda_idx = rng.permutation(n)[:2]
    gsngea   = 0.3  * rng.random(2)
    gsngen   = 0.002 * rng.random(2)
    for k, i in enumerate(ampa_idx):
        for j_off in (0, 1):
            net.add_connection("STN", int(i), "GPe", (int(i) + j_off) % n,
                               float(gsngea[k]),
                               SynapseSpec.double_exponential(0.0, 0.4, 2.5), delay=2.0)
    for k, i in enumerate(nmda_idx):
        for j_off in (0, 1):
            net.add_connection("STN", int(i), "GPe", (int(i) + j_off) % n,
                               float(gsngen[k]),
                               SynapseSpec.double_exponential(0.0, 2.0, 67.0), delay=2.0)

    # ---- STN → GPi  (excitatory alpha, 5 of n neurons, delay=1.5ms) -------
    stn_gpi_idx = rng.permutation(n)[:5]
    for i in stn_gpi_idx:
        for j_off in (0, 1):
            net.add_connection("STN", int(i), "GPi", (int(i) + j_off) % n,
                               0.15, SynapseSpec.alpha(0.0, tau), delay=1.5)

    # ---- GPe → GPi  (inhibitory alpha, shifted, delay=3ms) ----------------
    for shift in (1, -2 % n):
        net.connect("GPe", "GPi", "shifted",
                    weight=gpeak1 / 2,
                    synapse=SynapseSpec.alpha(-85.0, tau), delay=3.0, shift=shift)

    # ---- GPe → GPe  (inhibitory alpha, PD-scaled random weights) ----------
    ggege_scale   = 0.25 * (pd * 3 + 1)
    ggege_weights = ggege_scale * rng.random(n)
    for i in range(n):
        for j_off, shift_sign in ((1, 1), (-2, -1)):
            net.add_connection("GPe", i, "GPe", (i + j_off) % n,
                               float(ggege_weights[i]) * gpeak1 / 2,
                               SynapseSpec.alpha(-85.0, tau), delay=1.0)

    # ---- Str_D2 → GPe  (inhibitory alpha, all-to-all, delay=5ms) ----------
    net.connect("Str_D2", "GPe", "all_to_all",
                weight=0.5 / n,
                synapse=SynapseSpec.alpha(-85.0, tau), delay=5.0)

    # ---- Str_D1 → GPi  (inhibitory alpha, all-to-all, delay=4ms) ----------
    net.connect("Str_D1", "GPi", "all_to_all",
                weight=0.5 / n,
                synapse=SynapseSpec.alpha(-85.0, tau), delay=4.0)

    # ---- CTX_e → Str_D2  (excitatory alpha, one-to-one, delay=5.1ms) -----
    net.connect("CTX_e", "Str_D2", "one_to_one",
                weight=0.07,
                synapse=SynapseSpec.alpha(0.0, tau), delay=5.1)

    # ---- CTX_e → Str_D1  (excitatory alpha, per-neuron weight) ------------
    gcordrstr = (0.07 - 0.044 * pd) + 0.001 * rng.random(n)
    for i in range(n):
        net.add_connection("CTX_e", i, "Str_D1", i,
                           float(gcordrstr[i]),
                           SynapseSpec.alpha(0.0, tau), delay=5.1)

    # ---- CTX_e → STN  (AMPA + NMDA double-exp, per-neuron, delay=5.9ms) --
    gcorsna = 0.3   * rng.random(n)   # AMPA
    gcorsnn = 0.003 * rng.random(n)   # NMDA
    for i in range(n):
        for j_off in (0, 1):
            j = (i + j_off) % n
            net.add_connection("CTX_e", i, "STN", j, float(gcorsna[i]),
                               SynapseSpec.double_exponential(0.0, 0.5, 2.49), delay=5.9)
            net.add_connection("CTX_e", i, "STN", j, float(gcorsnn[i]),
                               SynapseSpec.double_exponential(0.0, 2.0, 90.0), delay=5.9)

    # ---- TH → CTX_e  (excitatory alpha, random permutation, delay=5ms) ---
    perm = rng.permutation(n)
    for i in range(n):
        net.add_connection("TH", i, "CTX_e", int(perm[i]),
                           0.15, SynapseSpec.alpha(0.0, tau), delay=5.0)

    # ---- CTX_i → CTX_e  (inhibitory, 4 random permutations) ---------------
    for _ in range(4):
        perm = rng.permutation(n)
        for i in range(n):
            net.add_connection("CTX_i", i, "CTX_e", int(perm[i]),
                               0.2 / 4, SynapseSpec.alpha(-85.0, tau))

    # ---- CTX_e → CTX_i  (excitatory, 4 random permutations) ---------------
    for _ in range(4):
        perm = rng.permutation(n)
        for i in range(n):
            net.add_connection("CTX_e", i, "CTX_i", int(perm[i]),
                               0.1 / 4, SynapseSpec.alpha(0.0, tau))

    # ---- Intra-striatal GABA-A  (kinetic, Ggaba(V) gate) ------------------
    gaba_kin = _gaba_kinetic(tau_i=13.0, E_syn=-80.0)
    ggaba = 0.1
    # Str_D2 → Str_D2 (4 random permutations, GABA-A)
    for _ in range(4):
        perm = rng.permutation(n)
        for i in range(n):
            net.add_kinetic_connection("Str_D2", i, "Str_D2", int(perm[i]),
                                       ggaba / 4, gaba_kin)
    # Str_D1 → Str_D1 (3 random permutations, GABA-A)
    for _ in range(3):
        perm = rng.permutation(n)
        for i in range(n):
            net.add_kinetic_connection("Str_D1", i, "Str_D1", int(perm[i]),
                                       ggaba / 3, gaba_kin)

    return net


# ---------------------------------------------------------------------------
# Top-level simulate function  (mirrors benchmark signature)
# ---------------------------------------------------------------------------

def simulate_ctxbgth(
    n: int          = 10,
    pd: int         = 0,
    tmax: float     = 1000.0,
    dt: float       = 0.01,
    dbs_freq: float = 0.0,
    PW: float       = 0.1,
    amplitude: float= 0.0,
    corstim: int    = 0,
    seed: int       = 42,
):
    """
    Simulate the CTX-BG-TH network and return GPi spectral metrics.

    Parameters
    ----------
    n         : neurons per population (default 10)
    pd        : 0=healthy, 1=Parkinson's disease
    tmax      : duration in ms
    dt        : time step in ms (default 0.01)
    dbs_freq  : DBS frequency in Hz (0=off)
    PW        : DBS pulse width in ms
    amplitude : DBS amplitude (µA/cm²)
    corstim   : cortical stimulation pulse (0=off, 1=on at t=1000ms)
    seed      : RNG seed

    Returns
    -------
    gpi_alpha_beta_area : float  — GPi beta-band spectral power (7–35 Hz)
    gpi_S               : ndarray — full GPi power spectrum
    gpi_f               : ndarray — frequency axis
    spike_times         : dict   — {pop: list of spike-time arrays (s)}
    """
    net = build_network(n=n, pd=pd, seed=seed)

    # ---- External currents -------------------------------------------------
    # Note: the benchmark uses Iapp values calibrated for its specific parameter
    # sets (TH: 1.2, GPe/GPi: 3.0).  Our presets encode the same neuron types
    # from the same literature but with slightly different absolute conductances,
    # so the drives below are calibrated to match the published firing-rate
    # targets for each population in the healthy state:
    #   TH ~8-12 Hz, STN ~5-10 Hz, GPe/GPi ~25 Hz, CTX ~20-30 Hz
    n_steps = int(tmax / dt)
    I_ext: dict = {
        "TH":    0.5,                                # thalamic relay: ~8-12 Hz (lower than benchmark Iappth=1.2 — h_Na recovers faster in alpha_beta form)
        "STN":   5.0,                                # STN: needs extra drive to overcome GPe inhibition
        "GPe":   float(1.0 - 0.3 * pd),             # GPe: healthy 1.0, PD 0.7
        "GPi":   1.0,                                # GPi: ~25 Hz
        "CTX_e": 5.0,                                # RS cortical drive
        "CTX_i": 2.0,                                # FS interneuron drive
    }

    # Cortical stimulus pulse at t=1000ms (if corstim=1)
    if corstim:
        I_cortex = np.zeros(n_steps)
        s0 = int(1000.0 / dt)
        s1 = int((1000.0 + 0.3) / dt)
        I_cortex[s0:s1] = 350.0
        I_ext["CTX_e"] = I_cortex
        I_ext["CTX_i"] = I_cortex

    # DBS on STN via built-in stimulator
    if dbs_freq > 0:
        params = DBSParameters()
        params.frequency   = dbs_freq
        params.pulse_width = PW
        params.amplitude   = amplitude
        net.attach_stimulator("STN", DBSStimulator(params))

    # ---- Simulate ----------------------------------------------------------
    start = timer()
    result = net.simulate(tmax, dt, I_ext,
                          record=RecordingConfig.spike_stats())
    elapsed = timer() - start

    # ---- Spectral analysis of GPi (multitaper, matches benchmark) ----------
    dt_s      = dt * 1e-3          # ms → s
    duration_s = tmax * 1e-3       # ms → s
    GPi_spike_times = [sp / 1000.0 for sp in result["GPi"]["spikes"]]  # ms → s
    gpi_S, gpi_f = mtspectrumpt(
        GPi_spike_times,
        duration=duration_s,
        Fs=1.0 / dt_s,
        fpass=(1, 100),
        tapers=(3, 5),
    )
    beta_mask = (gpi_f > 7) & (gpi_f < 35)
    gpi_alpha_beta_area = float(np.trapezoid(gpi_S[beta_mask], gpi_f[beta_mask]))

    spike_times = {
        pop: result[pop]["spikes"]
        for pop in ("TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i")
    }

    return gpi_alpha_beta_area, gpi_S, gpi_f, spike_times, elapsed


# ---------------------------------------------------------------------------
# Quick-run entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    time = 1 # seconds  
    print(f"Building CTX-BG-TH network (healthy, no DBS, n=10, {time}s) ...")
    area, S, f, spikes, t_sim = simulate_ctxbgth(n=10, pd=0, tmax=time*1000, dt=0.01)
    print(f"  Simulation time : {t_sim:.2f} s")
    print(f"  GPi β-band power: {area:.4f}")
    for pop, sp_list in spikes.items():
        rates = [len(s) / time for s in sp_list]
        print(f"  {pop:8s}: mean rate = {np.mean(rates):.1f} spk/s")
