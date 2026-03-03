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
    mtspectrumpt,
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
    # Benchmark: TH/STN/GPe/GPi/Str are randomised; CTX_e/CTX_i are NOT (all at -65).
    net.randomize_membrane_potentials("TH",     -62.0, 5.0, seed=seed, reset_gates=True)
    net.randomize_membrane_potentials("STN",    -62.0, 5.0, seed=seed + 1, reset_gates=True)
    net.randomize_membrane_potentials("GPe",    -62.0, 5.0, seed=seed + 2, reset_gates=True)
    net.randomize_membrane_potentials("GPi",    -62.0, 5.0, seed=seed + 3, reset_gates=True)
    # Str: randomise V AND reset gate steady states to avoid V/gate mismatch transient.
    net.randomize_membrane_potentials("Str_D2", -63.8, 5.0, seed=seed + 4, reset_gates=True)
    net.randomize_membrane_potentials("Str_D1", -63.8, 5.0, seed=seed + 5, reset_gates=True)


    # ---- Synapse parameter shorthands (from benchmark) --------------------
    tau    = 5.0    # ms — alpha-function time constant
    gpeak  = 0.43   # peak of excitatory alpha kernel (const = gpeak/(tau*exp(-1)))
    gpeak1 = 0.3    # peak of inhibitory alpha kernel (const1 = gpeak1/(tau*exp(-1)))
    # Note: benchmark stores synapse kernels normalized to peak at gpeak/gpeak1.
    # Benchmark conductance: g_ij * S(t), where S peaks at gpeak or gpeak1.
    # Library conductance: weight * kernel(t), where kernel peaks at 1.
    # Therefore: weight = g_ij * gpeak (or gpeak1) to match benchmark effective conductance.

    # ---- GPi → TH  (one-to-one, delay=5ms) --------------------------------
    # Benchmark: Igith = ggith*(V1-Esyn[5])*S4, element-wise → TH[k] ← GPi[k]
    # ggith=0.112, S4 peaks at gpeak1=0.3 → weight = 0.112*0.3 = 0.0336
    net.connect("GPi", "TH", "one_to_one",
                weight=0.112 * gpeak1,
                synapse=SynapseSpec.alpha(-85.0, tau), delay=5.0)

    # ---- GPe → STN  (shift=0 and shift=1, delay=4ms) ----------------------
    # Benchmark: Igesn = ggesn*(V2-Esyn[0])*(S3a+S31a)
    #   S3a[k] from GPe[k] → shift=0; S31a[k]=S3a[k+1] from GPe[k+1] → shift=1
    # ggesn=0.5, kernel peaks at gpeak1=0.3 → weight = 0.5*0.3 = 0.15 per connection
    net.connect("GPe", "STN", "one_to_one",
                weight=0.5 * gpeak1,
                synapse=SynapseSpec.double_exponential(-85.0, tau_rise=0.4, tau_decay=7.7),
                delay=4.0)
    net.connect("GPe", "STN", "shifted",
                weight=0.5 * gpeak1,
                synapse=SynapseSpec.double_exponential(-85.0, tau_rise=0.4, tau_decay=7.7),
                delay=4.0, shift=1)

    # ---- STN → GPe  (sparse receiver-selected AMPA + NMDA, delay=2ms) ----
    # Benchmark: gsngea[k]*(S2a[k]+S21a[k]) where S21a[k]=S2a[k-1]
    #   gsngea = zeros(n), 2 GPe RECEIVERS nonzero (0 to 0.3 each)
    #   GPe[k] ← STN[k] and STN[k-1] with weight gsngea[k]*gpeak
    gsngea = np.zeros(n)
    gsngea[rng.permutation(n)[:2]] = 0.3 * rng.random(2)
    gsngen = np.zeros(n)
    gsngen[rng.permutation(n)[:2]] = 0.002 * rng.random(2)
    for k in range(n):
        if gsngea[k] > 0:
            net.add_connection("STN", k,          "GPe", k, float(gsngea[k]) * gpeak,
                               SynapseSpec.double_exponential(0.0, 0.4, 2.5), delay=2.0)
            net.add_connection("STN", (k-1)%n,    "GPe", k, float(gsngea[k]) * gpeak,
                               SynapseSpec.double_exponential(0.0, 0.4, 2.5), delay=2.0)
        if gsngen[k] > 0:
            net.add_connection("STN", k,          "GPe", k, float(gsngen[k]) * gpeak,
                               SynapseSpec.double_exponential(0.0, 2.0, 67.0), delay=2.0)
            net.add_connection("STN", (k-1)%n,    "GPe", k, float(gsngen[k]) * gpeak,
                               SynapseSpec.double_exponential(0.0, 2.0, 67.0), delay=2.0)

    # ---- STN → GPi  (sparse receiver-selected alpha, delay=1.5ms) ---------
    # Benchmark: gsngi[k]*(S2b[k]+S21b[k]) where S21b[k]=S2b[k-1]
    #   gsngi = zeros(n), 5 GPi RECEIVERS = 0.15 each
    #   GPi[k] ← STN[k] and STN[k-1] with weight gsngi[k]*gpeak = 0.15*0.43 = 0.0645
    gsngi = np.zeros(n)
    gsngi[rng.permutation(n)[:5]] = 0.15
    for k in range(n):
        if gsngi[k] > 0:
            net.add_connection("STN", k,       "GPi", k, float(gsngi[k]) * gpeak,
                               SynapseSpec.alpha(0.0, tau), delay=1.5)
            net.add_connection("STN", (k-1)%n, "GPi", k, float(gsngi[k]) * gpeak,
                               SynapseSpec.alpha(0.0, tau), delay=1.5)

    # ---- GPe → GPi  (shift=1 and shift=n-2, delay=3ms) --------------------
    # Benchmark: Igigi = ggigi*(V4-Esyn[4])*(S31b+S32b)
    #   S31b[k]=S3b[k+1] → GPi[k]←GPe[k+1] (shift=1)
    #   S32b[k]=S3b[k-2] → GPi[k]←GPe[k-2] (shift=n-2)
    # ggigi=0.5, kernel peaks at gpeak1=0.3 → weight = 0.5*0.3 = 0.15 each
    for shift in (1, -2 % n):
        net.connect("GPe", "GPi", "shifted",
                    weight=0.5 * gpeak1,
                    synapse=SynapseSpec.alpha(-85.0, tau), delay=3.0, shift=shift)

    # ---- GPe → GPe  (receiver-indexed random weights, delay=1ms) ----------
    # Benchmark: Igege = ggege_scale * ggege[k] * (S31c[k]+S32c[k])
    #   S31c[k]=S3c[k+1] → GPe[k]←GPe[k+1]; S32c[k]=S3c[k-2] → GPe[k]←GPe[k-2]
    #   Weight is per RECEIVER: ggege_scale * ggege[k] * gpeak1
    ggege_scale   = 0.25 * (pd * 3 + 1)
    ggege_weights = rng.random(n)   # ggege[k] per receiver
    for k in range(n):
        net.add_connection("GPe", (k+1) % n,    "GPe", k,
                           float(ggege_weights[k]) * ggege_scale * gpeak1,
                           SynapseSpec.alpha(-85.0, tau), delay=1.0)
        net.add_connection("GPe", (k-2+n) % n,  "GPe", k,
                           float(ggege_weights[k]) * ggege_scale * gpeak1,
                           SynapseSpec.alpha(-85.0, tau), delay=1.0)

    # ---- Str_D2 → GPe  (inhibitory alpha, all-to-all, delay=5ms) ----------
    # Benchmark: gstrgpe=0.5, S peaks at gpeak1=0.3 → weight=0.5*gpeak1/n per connection
    net.connect("Str_D2", "GPe", "all_to_all",
                weight=0.5 * gpeak1,
                synapse=SynapseSpec.alpha(-85.0, tau), delay=5.0)

    # ---- Str_D1 → GPi  (inhibitory alpha, all-to-all, delay=4ms) ----------
    # Benchmark: gstrgpi=0.5, kernel peaks at gpeak1=0.3
    net.connect("Str_D1", "GPi", "all_to_all",
                weight=0.5 * gpeak1,
                synapse=SynapseSpec.alpha(-85.0, tau), delay=4.0)

    # ---- CTX_e → Str_D2  (excitatory alpha, one-to-one, delay=5.1ms) -----
    # Benchmark: Icorstr5 = gcorindrstr*(V5-Esyn[1])*S6a, one-to-one
    # gcorindrstr=0.07, S6a peaks at gpeak=0.43 → weight=0.07*0.43=0.0301
    net.connect("CTX_e", "Str_D2", "one_to_one",
                weight=0.07 * gpeak,
                synapse=SynapseSpec.alpha(0.0, tau), delay=5.1)

    # ---- CTX_e → Str_D1  (excitatory alpha, one-to-one, per-neuron weight) -
    # Benchmark: Icorstr6 = gcordrstr[k]*(V6-Esyn[1])*S6a, gcordrstr peaks at gpeak
    
    gcordrstr = (0.07 - 0.044 * pd) + 0.001 * rng.random(n)
    
    for i in range(n):
        net.add_connection("CTX_e", i, "Str_D1", i,
                           float(gcordrstr[i]) * gpeak,
                           SynapseSpec.alpha(0.0, tau), delay=5.1)
    

    # ---- CTX_e → STN  (receiver-indexed AMPA + NMDA, delay=5.9ms) --------
    # Benchmark: gcorsna[k]*(S6b[k]+S61b[k]) where S61b[k]=S6b[k+1]
    #   gcorsna is (n,1), all neurons nonzero (0 to 0.3)
    #   STN[k] ← CTX[k] and CTX[k+1], weight = gcorsna[k]*gpeak (receiver-indexed)
    net.connect("CTX_e", "STN", "all_to_all", weight=(0.0, 0.003 * gpeak),
                synapse=SynapseSpec.double_exponential(0.0, 2.0, 90.0), delay=5.9)
    
    net.connect("CTX_e", "STN", "one_to_one", weight=(0.0, 0.3 * gpeak),
                synapse=SynapseSpec.double_exponential(0.0, 0.5, 2.49), delay=5.9)
    
    net.connect("CTX_e", "STN", "shifted", weight=(0.0, 0.3 * gpeak),
                synapse=SynapseSpec.double_exponential(0.0, 0.5, 2.49), delay=5.9, shift=1)

    # ---- TH → CTX_e  (one-to-one, no permutation, delay=5ms) --------------
    # Benchmark: t_d_th_cor = 5 ms, gthcor=0.15, kernel peaks at gpeak=0.43.
    # weight = gthcor * gpeak = 0.15 * 0.43
    net.connect("TH", "CTX_e", "one_to_one", weight=0.15 * gpeak,
                synapse=SynapseSpec.alpha(0.0, tau), delay=5.0)

    # ---- CTX_i → CTX_e  (4 random permutations) ---------------------------
    # Benchmark: gie=0.2, S1b via 2nd-order ODE (no explicit delay, starts at spike).
    # weight = gie * gpeak = 0.2 * gpeak; delay=0.01 matches benchmark's ~0 effective delay.
    for _ in range(4):
        perm = rng.permutation(n)
        for i in range(n):
            net.add_connection("CTX_i", i, "CTX_e", int(perm[i]),
                               0.2 * gpeak, SynapseSpec.alpha(-85.0, tau), delay=0.01)

    # ---- CTX_e → CTX_i  (4 random permutations) ---------------------------
    # Benchmark: gei=0.1, S1a via 2nd-order ODE (no explicit delay, starts at spike).
    # weight = gei * gpeak = 0.1 * gpeak; delay=0.01 matches benchmark's ~0 effective delay.
    for _ in range(4):
        perm = rng.permutation(n)
        for i in range(n):
            net.add_connection("CTX_e", i, "CTX_i", int(perm[i]),
                               0.1 * gpeak, SynapseSpec.alpha(0.0, tau), delay=0.01)

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
    # I_ext values match benchmark exactly:
    #   Iappth=1.2, (no Iappstn — STN fires spontaneously), Iappgpe=3, Iappgpi=3
    #   CTX and Str receive NO constant drive — cortex is driven only by TH synaptic input
    #   Benchmark: Iappgpe = 3 - 2*corstim*(not pd)  [only reduced during corstim in healthy]
    I_ext: dict = {
        "TH":    1.2,
        "GPe":   float(3.0 - 2.0 * corstim * (1 - pd)),
        "GPi":   3.0,
    }

    # Cortical stimulus pulse at t=1000ms (if corstim=1)
    # Benchmark: CTX has zero constant Iapp; corstim adds a 350 µA/cm² pulse
    if corstim:
        I_cortex_e = np.zeros(n_steps)
        I_cortex_i = np.zeros(n_steps)
        s0 = int(1000.0 / dt)
        s1 = int((1000.0 + 0.3) / dt)
        I_cortex_e[s0:s1] = 350.0
        I_cortex_i[s0:s1] = 350.0
        I_ext["CTX_e"] = I_cortex_e
        I_ext["CTX_i"] = I_cortex_i

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
                          record=RecordingConfig(
                              ["V", "spikes", "spike_count", "firing_rate",
                               "ISI_mean", "ISI_cv", "spike_events"],
                              interval=1, spike_threshold=-10.0))
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

    pops_ordered = ("TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i")
    spike_times = {pop: result[pop]["spikes"] for pop in pops_ordered}
    spike_events = {pop: result[pop]["spike_events"] for pop in pops_ordered}

    return gpi_alpha_beta_area, gpi_S, gpi_f, spike_times, spike_events, elapsed, result


# ---------------------------------------------------------------------------
# Quick-run entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    time = 2 # seconds
    window_s = 0.25  # time window for rate-over-time reporting (seconds)

    print(f"Building CTX-BG-TH network (healthy, no DBS, n=10, {time}s) ...")
    area, S, f, spikes, syn_events, t_sim, sim_result = simulate_ctxbgth(n=10, pd=0, tmax=time*1000, dt=0.01, corstim=0, seed=6536)
    print(f"  Simulation time : {t_sim:.2f} s")
    print(f"  GPi β-band power: {area:.4f}")

    # Overall mean rates
    print("\n  Overall mean firing rates:")
    for pop, sp_list in spikes.items():
        rates = [len(s) / time for s in sp_list]
        print(f"    {pop:8s}: {np.mean(rates):.1f} spk/s")

    # Rates in successive time windows to check for drift/transients
    tmax_ms   = time * 1000.0
    win_ms    = window_s * 1000.0
    n_windows = int(tmax_ms / win_ms)
    edges_ms  = [i * win_ms for i in range(n_windows + 1)]

    pops = list(spikes.keys())
    hdr  = f"  {'Window':>16s} | " + " | ".join(f"{p:>8s}" for p in pops)
    print(f"\n  Firing rates by {window_s*1000:.0f} ms window (spk/s):")
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for w in range(n_windows):
        t0, t1 = edges_ms[w], edges_ms[w + 1]
        label  = f"{t0/1000:.2f}–{t1/1000:.2f} s"
        row_rates = []
        for pop in pops:
            sp_list = spikes[pop]
            count = sum(np.sum((s >= t0) & (s < t1)) for s in sp_list)
            mean_rate = count / len(sp_list) / window_s if sp_list else 0.0
            row_rates.append(mean_rate)
        print(f"  {label:>16s} | " + " | ".join(f"{r:>8.1f}" for r in row_rates))
    print("  " + "-" * (len(hdr) - 2))

    # Synapse-detected vs voltage-based spike count comparison
    # syn_events[pop] shape: (n_neurons, n_rec); each value = #steps neuron fired (synapse view)
    # spikes[pop] is list of spike-time arrays (voltage view)
    """
    dt_ms = 0.01
    t_axis = np.arange(0, time * 1000, dt_ms * 1)  # interval=1, so n_rec = n_steps

    print(f"\n  Synapse-detected vs voltage-based spike counts by {window_s*1000:.0f} ms window:")
    print(f"  (syn = sum of synapse-detected firing steps; V = voltage-threshold crossings)")
    for pop in pops:
        ev = syn_events[pop]          # (n_neurons, n_rec)
        sp_list = spikes[pop]
        n_pop = ev.shape[0]
        tmax_ms_val = time * 1000.0
        # time axis for syn_events: each column j covers t = j * dt_ms (interval=1)
        t_rec = np.arange(ev.shape[1]) * dt_ms

        syn_row = []
        v_row = []
        for w in range(n_windows):
            t0, t1 = edges_ms[w], edges_ms[w + 1]
            # synapse count: sum of event steps in window, averaged over neurons
            mask = (t_rec >= t0) & (t_rec < t1)
            syn_count = float(ev[:, mask].sum()) / n_pop
            # voltage count: spike events in window
            v_count = sum(np.sum((s >= t0) & (s < t1)) for s in sp_list) / n_pop
            syn_row.append(syn_count)
            v_row.append(v_count)

        print(f"\n    {pop}:")
        win_labels = [f"{edges_ms[w]/1000:.2f}-{edges_ms[w+1]/1000:.2f}s" for w in range(n_windows)]
        label_w = max(len(l) for l in win_labels)
        for w, lbl in enumerate(win_labels):
            delta = syn_row[w] - v_row[w]
            print(f"      {lbl:{label_w}s}  syn={syn_row[w]:6.1f}  V={v_row[w]:6.1f}  Δ={delta:+.1f}")
    print()
    """

    # ---- Voltage diagnostic: Str_D1 from 0–2 ms at 0.25 ms resolution ------
    print("\n--- Str_D1 voltage diagnostic (0–2 ms, every 0.25 ms) ---")
    _V_all = sim_result["Str_D1"]["V"]   # (n, n_steps) at interval=1, dt=0.01ms
    _t_all = sim_result["Str_D1"].time   # ms
    _sp    = sim_result["Str_D1"]["spikes"]

    # 0.25 ms resolution = every 25 steps at dt=0.01ms; window 0–2 ms
    _step = 25
    _idx = np.arange(0, _V_all.shape[1], _step)
    _mask = _t_all[_idx] <= 2.0
    _idx = _idx[_mask]
    _V = _V_all[:, _idx]
    _t = _t_all[_idx]
    _sp_100 = [sp[sp <= 2.0] for sp in _sp]

    n_d1 = _V.shape[0]
    _hdr = f"{'t(ms)':>7s} | " + " | ".join(f"  N{i:02d} " for i in range(n_d1))
    print(_hdr)
    print("-" * len(_hdr))
    for _ti in range(len(_t)):
        _row = " | ".join(f"{_V[i, _ti]:+6.2f}" for i in range(n_d1))
        print(f"{_t[_ti]:7.1f} | {_row}")
    print("-" * len(_hdr))
    print("Spikes in first 100 ms:")
    for i, sp in enumerate(_sp_100):
        print(f"  N{i:02d}: {np.round(sp, 2).tolist()}")
