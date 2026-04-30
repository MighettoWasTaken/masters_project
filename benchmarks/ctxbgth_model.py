"""
CTX-BG-TH Network Model — hodgkin_huxley library recreation.

Recreates the rat Parkinson's disease model (Hahn et al. 2019,
nihms803352) using the hodgkin_huxley library's built-in neuron
presets, composable model builder, and C++-accelerated simulation.

Network populations:
  TH       — Thalamus            (custom spec matching benchmark K-channel mechanism)
  STN      — Subthalamic Nucleus (custom NeuronModel builder)
  GPe      — Globus Pallidus ext (custom NeuronModel builder)
  GPi      — Globus Pallidus int (custom NeuronModel builder, same params as GPe)
  Str_D2   — Striatum indirect   (HH-style composable, D2 receptors)
  Str_D1   — Striatum direct     (HH-style composable, D1 receptors)
  CTX_e    — Cortex excitatory   (Izhikevich Regular Spiking)
  CTX_i    — Cortex inhibitory   (Izhikevich Fast Spiking)

Intracellular dynamics (task14 API):
  STN : Ca²⁺ DRIVEN_DECAY_NERNST  — dCa/dt = ε*(-I_CaL - I_T - K_Ca*Ca)
  GPe : Ca²⁺ DRIVEN_DECAY         — dCa/dt = 1e-4*(-I_T - I_CaL - 15*Ca)
  GPi : Ca²⁺ DRIVEN_DECAY         — same parameters as GPe
"""

from timeit import default_timer as timer

import numpy as np
import sympy as sp

from hodgkin_huxley import (
    DBSParameters,
    DBSStimulator,
    IntracellularDynamics,
    IzhikevichType,
    NeuronModel,
    NeuronModelSpec,
    PulseStimulator,
    RecordingConfig,
    RegionalNetwork,
    SynapseModel,
    V,
    analyze_beta_power,
)
from hodgkin_huxley._codegen import Ca, I_source

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
    No intracellular dynamics (TH has no calcium in this model).
    """
    model = NeuronModel("TH", C_m=1.0, V_init=V_init)

    # h_Na: alpha-beta (ah/bh from Wang 1994)
    #   ah(V) = 0.128 * exp(-(V+46)/18)
    #   bh(V) = 4 / (1 + exp(-(V+23)/5))
    h_na_idx = model.add_gate(
        "h_Na",
        update_form="alpha_beta",
        alpha=sp.Float(0.128) * sp.exp(-(V + 46) / 18),
        beta=4 / (1 + sp.exp(-(V + 23) / 5)),
    )

    # r (T-current inactivation): INF_TAU
    #   r_inf = 1 / (1 + exp((V+84)/4))
    #   tau_r = 4.2 + 0.15*exp(-(V+25)/10.5)
    r_idx = model.add_gate(
        "r",
        update_form="inf_tau",
        initial_value=0.25,
        inf=1 / (1 + sp.exp((V + 84) / 4)),
        tau=sp.Float(4.2) + sp.Float(0.15) / (sp.exp((V + 25) / sp.Float(10.5)) + sp.exp(-(V + 1000) / 1)),
    )

    # m_Na (instantaneous Na activation)
    #   m_inf = 1 / (1 + exp(-(V+37)/7))
    m_na_idx = model.add_gate(
        "m_Na", update_form="instant", inf=1 / (1 + sp.exp(-(V + 37) / 7))
    )

    # p (instantaneous T-current activation)
    #   p_inf = 1 / (1 + exp(-(V+60)/6.2))
    p_idx = model.add_gate(
        "p", update_form="instant", inf=1 / (1 + sp.exp(-(V + 60) / sp.Float(6.2)))
    )

    # n_K = 0.75*(1-h_Na): DERIVED from h_Na
    nk_idx = model.add_gate(
        "n_K",
        update_form="derived",
        derived_source_gate=h_na_idx,
        derived_a=0.75,
        derived_b=1.0,
        derived_c=-1.0,
    )

    model.add_channel("Na", g=3.0, E_rev=50.0, gating=m_na_idx**3 * h_na_idx)
    model.add_channel("K", g=5.0, E_rev=-75.0, gating=nk_idx**4)
    model.add_channel("T", g=5.0, E_rev=0.0, gating=p_idx**2 * r_idx)
    model.add_channel("Leak", g=0.05, E_rev=-70.0)

    return model.to_spec()


# ---------------------------------------------------------------------------
# Striatum composable model (HH-style with Na, K-DR, Leak, M-type K)
# No calcium dynamics.
# ---------------------------------------------------------------------------


def _make_striatum_spec(pd: int = 0, V_init: float = -63.8) -> NeuronModelSpec:
    """
    HH-style striatal neuron with Na, K-DR, Leak, and M-type K channels.
    Alpha-beta gates match the benchmark's alpham/betam/etc. functions.
    gm factor: (2.6 - 1.1*pd) encodes the PD-dependent M-channel reduction.
    """
    model = NeuronModel("Striatum", C_m=1.0, V_init=V_init)

    # Na channel gating: m^3 * h
    m_idx = model.add_gate(
        "m",
        update_form="alpha_beta",
        alpha=sp.Float(0.32) * (V + 54) / (1 - sp.exp(-(V + 54) / 4)),
        beta=sp.Float(0.28) * (V + 27) / (sp.exp((V + 27) / 5) - 1),
    )
    h_idx = model.add_gate(
        "h",
        update_form="alpha_beta",
        alpha=sp.Float(0.128) * sp.exp(-(V + 50) / 18),
        beta=4 / (1 + sp.exp(-(V + 27) / 5)),
    )

    # K-DR channel gating: n^4
    n_idx = model.add_gate(
        "n",
        update_form="alpha_beta",
        alpha=sp.Float(0.032) * (V + 52) / (1 - sp.exp(-(V + 52) / 5)),
        beta=sp.Float(0.5) * sp.exp(-(V + 57) / 40),
    )

    # M-type K gating (slow, Kir-like): p^1
    p_idx = model.add_gate(
        "p",
        update_form="alpha_beta",
        alpha=sp.Float(3.209e-4) * (V + 30) / (1 - sp.exp(-(V + 30) / 9)),
        beta=sp.Float(3.209e-4) * (V + 30) / (sp.exp((V + 30) / 9) - 1),
    )

    gm_eff = 2.6 - 1.1 * pd  # PD reduces M-channel conductance

    model.add_channel("Leak", g=0.1, E_rev=-67.0)
    model.add_channel("Na", g=100.0, E_rev=50.0, gating=m_idx**3 * h_idx)
    model.add_channel("K_DR", g=80.0, E_rev=-100.0, gating=n_idx**4)
    model.add_channel("K_M", g=gm_eff, E_rev=-100.0, gating=p_idx)

    return model.to_spec()


# ---------------------------------------------------------------------------
# STN composable model (Rubin & Terman 2004 / Hahn et al. 2019)
# Channels: Na(m³h), K-DR(n⁴), A(a²b), CaL(c²d1d2), T(p²q), AHP-K(r²), Leak
#
# Ca²⁺ dynamics (DRIVEN_DECAY_NERNST):
#   dCa/dt = epsilon*(-I_CaL - I_T - K_Ca*Ca)
#   E_Ca updates via standard Nernst: (RT/zF)*ln(Ca_o/Ca)
#
# The CaL and T channels use nernst_substance_idx=0 so their E_rev tracks
# the intracellular [Ca²⁺] through the Nernst reversal of substance 0.
# ---------------------------------------------------------------------------


def _make_stn_spec() -> tuple[NeuronModelSpec, IntracellularDynamics]:
    """Return (spec, ca_dynamics).  Attach ca_dynamics via add_intracellular()."""
    model = NeuronModel("STN", C_m=1.0, V_init=-62.0)

    m = model.add_gate(
        "m_Na",
        update_form="inf_tau",
        initial_value=0.060,
        inf=1 / (1 + sp.exp(-(V + 40) / 8)),
        tau=sp.Float(0.2) + 3 / (1 + sp.exp(-(V - sp.Float(-53.0)) / sp.Float(-0.7))),
    )
    h = model.add_gate(
        "h_Na",
        update_form="inf_tau",
        initial_value=0.929,
        inf=1 / (1 + sp.exp(-(V - sp.Float(-45.5)) / sp.Float(-6.4))),
        tau=sp.Float(24.5) / (sp.exp((V + 50) / 15) + sp.exp(-(V + 50) / 16)),
    )
    n = model.add_gate(
        "n_K",
        update_form="inf_tau",
        initial_value=0.182,
        inf=1 / (1 + sp.exp(-(V + 41) / 14)),
        tau=sp.Float(11.0) / (sp.exp((V + 40) / 40) + sp.exp(-(V + 40) / 50)),
    )
    a = model.add_gate(
        "a_A",
        update_form="inf_tau",
        initial_value=0.239,
        inf=1 / (1 + sp.exp(-(V + 45) / sp.Float(14.7))),
        tau=sp.Float(1.0) + 1 / (1 + sp.exp(-(V - sp.Float(-40.0)) / sp.Float(-0.5))),
    )
    b = model.add_gate(
        "b_A",
        update_form="inf_tau",
        initial_value=0.023,
        inf=1 / (1 + sp.exp(-(V - sp.Float(-90.0)) / sp.Float(-7.5))),
        tau=sp.Float(200.0) / (sp.exp((V + 60) / 30) + sp.exp(-(V + 40) / 10)),
    )
    c = model.add_gate(
        "c_CaL",
        update_form="inf_tau",
        initial_value=0.002,
        inf=1 / (1 + sp.exp(-(V + sp.Float(30.6)) / 5)),
        tau=sp.Float(45.0)
        + sp.Float(10.0) / (sp.exp((V + 27) / 20) + sp.exp(-(V + 50) / 15)),
    )
    d1 = model.add_gate(
        "d1_CaL",
        update_form="inf_tau",
        initial_value=0.566,
        inf=1 / (1 + sp.exp(-(V - sp.Float(-60.0)) / sp.Float(-7.5))),
        tau=sp.Float(400.0)
        + sp.Float(500.0) / (sp.exp((V + 40) / 15) + sp.exp(-(V + 20) / 20)),
    )
    d2 = model.add_gate(
        "d2_CaL",
        update_form="inf_tau",
        initial_value=1.0,
        inf=1 / (1 + sp.exp(-(V - sp.Float(0.1)) / sp.Float(-0.02))),
        tau=sp.Float(130.0),
    )
    p = model.add_gate(
        "p_T",
        update_form="inf_tau",
        initial_value=0.290,
        inf=1 / (1 + sp.exp(-(V + 56) / sp.Float(6.7))),
        tau=sp.Float(5.0)
        + sp.Float(0.33) / (sp.exp((V + 27) / 10) + sp.exp(-(V + 102) / 15)),
    )
    q = model.add_gate(
        "q_T",
        update_form="inf_tau",
        initial_value=0.019,
        inf=1 / (1 + sp.exp(-(V - sp.Float(-85.0)) / sp.Float(-5.8))),
        tau=sp.Float(400.0) / (sp.exp((V + 50) / 15) + sp.exp(-(V + 50) / 16)),
    )
    r = model.add_gate(
        "r_AHP",
        update_form="inf_tau",
        initial_value=0.0,
        inf=1 / (1 + sp.exp(-(V - sp.Float(0.17)) / sp.Float(0.08))),
        tau=sp.Float(2.0),
    )

    model.add_channel("Na", g=49.0, E_rev=60.0, gating=m**3 * h)
    model.add_channel("K", g=57.0, E_rev=-90.0, gating=n**4)
    model.add_channel("A", g=5.0, E_rev=-90.0, gating=a**2 * b)
    # CaL and T: nernst_substance_idx=0 → E_rev tracks Ca²⁺ (substance 0)
    cal = model.add_channel(
        "CaL", g=15.0, E_rev=0.0,
        nernst_substance_idx=0,
        gating=c**2 * d1 * d2,
    )
    t_ch = model.add_channel(
        "T", g=5.0, E_rev=0.0,
        nernst_substance_idx=0,
        gating=p**2 * q,
    )
    model.add_channel("AHP_K", g=1.0, E_rev=-90.0, gating=r**2)
    model.add_channel("Leak", g=0.35, E_rev=-60.0)

    spec = model.to_spec()

    # Ca²⁺ dynamics: DRIVEN_DECAY_NERNST
    #   dCa/dt = epsilon * (-I_CaL - I_T - K_Ca*Ca)
    #   Standard Nernst: E_Ca = (RT/zF) * ln(Ca_o/Ca)
    Ca_o = 2000.0
    R, T_K, z, F = 8314.0, 298.0, 2.0, 96485.0
    ca_dynamics = IntracellularDynamics(
        "Ca",
        ode=sp.Float(5.182e-6) * (-I_source - sp.Float(386.0) * Ca),
        source_channels=["CaL", "T"],
        nernst=(R * T_K / (z * F)) * sp.log(sp.Float(Ca_o) / Ca),
        initial=0.005,
    )

    return spec, ca_dynamics


# ---------------------------------------------------------------------------
# GPe/GPi composable model (Rubin & Terman 2004)
# Channels: Na(m³h), K(n⁴), T(a³r), CaL(s²), AHP(Ca-dep K), Leak
#
# Ca²⁺ dynamics (DRIVEN_DECAY):
#   dCa/dt = 1e-4 * (-I_T - I_CaL - 15*Ca)
#
# The AHP channel uses ahp_substance_idx=0, drawing the Ca-dependent
# conductance from substance 0 (Ca²⁺).  T and CaL use E_rev=120 mV directly
# (no Nernst update — GPe/GPi model does not update E_Ca dynamically).
# ---------------------------------------------------------------------------


def _make_gpe_spec() -> tuple[NeuronModelSpec, IntracellularDynamics]:
    """Return (spec, ca_dynamics).  Attach ca_dynamics via add_intracellular()."""
    model = NeuronModel("GPe", C_m=1.0, V_init=-62.0)

    m = model.add_gate(
        "m_Na",
        update_form="instant",
        initial_value=0.076,
        inf=1 / (1 + sp.exp(-(V + 37) / 10)),
    )
    h = model.add_gate(
        "h_Na",
        update_form="inf_tau",
        scale=0.05,
        initial_value=0.583,
        inf=1 / (1 + sp.exp(-(V - sp.Float(-58.0)) / sp.Float(-12.0))),
        tau=sp.Float(0.05)
        + sp.Float(0.27) / (1 + sp.exp(-(V - sp.Float(-40.0)) / sp.Float(-12.0))),
    )
    n = model.add_gate(
        "n_K",
        update_form="inf_tau",
        scale=0.1,
        initial_value=0.298,
        inf=1 / (1 + sp.exp(-(V + 50) / 14)),
        tau=sp.Float(0.05)
        + sp.Float(0.27) / (1 + sp.exp(-(V - sp.Float(-40.0)) / sp.Float(-12.0))),
    )
    a = model.add_gate(
        "a_T",
        update_form="instant",
        initial_value=0.0,
        inf=1 / (1 + sp.exp(-(V + 57) / 2)),
    )
    r = model.add_gate(
        "r_T",
        update_form="inf_tau",
        initial_value=0.018,
        inf=1 / (1 + sp.exp(-(V - sp.Float(-70.0)) / sp.Float(-2.0))),
        tau=sp.Float(30.0),
    )
    s = model.add_gate(
        "s_CaL",
        update_form="instant",
        initial_value=0.0,
        inf=1 / (1 + sp.exp(-(V + 35) / 2)),
    )

    model.add_channel("Na", g=120.0, E_rev=55.0, gating=m**3 * h)
    model.add_channel("K", g=30.0, E_rev=-80.0, gating=n**4)
    t_ch = model.add_channel("T", g=0.5, E_rev=120.0, gating=a**3 * r)
    cal = model.add_channel("CaL", g=0.15, E_rev=120.0, gating=s**2)
    # AHP: g_AHP = g * Ca/(Ca + k1) — ahp_substance_idx=0 selects substance 0 (Ca²⁺)
    model.add_channel("AHP", g=10.0, E_rev=-80.0,
                      is_ahp=True, ahp_k1=10.0, ahp_substance_idx=0)
    model.add_channel("Leak", g=0.1, E_rev=-65.0)

    spec = model.to_spec()

    # Ca²⁺ dynamics: DRIVEN_DECAY (no Nernst update for GPe/GPi)
    #   dCa/dt = 1e-4 * (-I_T - I_CaL - 15*Ca)
    ca_dynamics = IntracellularDynamics(
        "Ca",
        ode=sp.Float(1e-4) * (-I_source - sp.Float(15.0) * Ca),
        source_channels=["T", "CaL"],
        initial=0.1,
    )

    return spec, ca_dynamics


def _make_gpi_spec() -> tuple[NeuronModelSpec, IntracellularDynamics]:
    """GPi: identical neuron parameters to GPe."""
    spec, ca_dynamics = _make_gpe_spec()
    spec.name = "GPi"
    return spec, ca_dynamics


# ---------------------------------------------------------------------------
# Kinetic GABA-A synapse (Ggaba(V) = 2*(1+tanh(V/4)))
# Used for intra-striatal inhibition (Str→Str)
# dS/dt = Ggaba(V)*(1-S) - S/tau_i  (exact-exponential integration in C++)
# ---------------------------------------------------------------------------


def _gaba_kinetic(tau_i: float = 13.0, E_syn: float = -80.0) -> SynapseModel:
    return SynapseModel.tanh_gate(
        name="GABA_kin",
        amp=2.0,
        v_half=0.0,
        k=4.0,
        tau_decay=tau_i,
        g=0.1,
        E_syn=E_syn,
    )


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

    # ---- Build specs (spec-builders now return (spec, ca_dynamics)) --------
    stn_spec, stn_ca = _make_stn_spec()
    gpe_spec, gpe_ca = _make_gpe_spec()
    gpi_spec, gpi_ca = _make_gpi_spec()

    # ---- Populations -------------------------------------------------------
    net.add_population("TH",     n, model=_make_th_spec())
    net.add_population("STN",    n, model=stn_spec)
    net.add_population("GPe",    n, model=gpe_spec)
    net.add_population("GPi",    n, model=gpi_spec)
    net.add_population("Str_D2", n, model=_make_striatum_spec(pd=pd, V_init=-63.8))
    net.add_population("Str_D1", n, model=_make_striatum_spec(pd=pd, V_init=-63.8))

    net.add_population("CTX_e", n, model=NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING))
    net.add_population("CTX_i", n, model=NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING))

    # ---- Attach Ca²⁺ intracellular dynamics --------------------------------
    net.add_intracellular(stn_ca, "STN")
    net.add_intracellular(gpe_ca, "GPe")
    net.add_intracellular(gpi_ca, "GPi")

    # Randomise initial membrane potentials (matches benchmark scatter)
    net.randomize_membrane_potentials("TH",  -62.0, 5.0, seed=seed,     reset_gates=True)
    net.randomize_membrane_potentials("STN", -62.0, 5.0, seed=seed + 1, reset_gates=True)
    net.randomize_membrane_potentials("GPe", -62.0, 5.0, seed=seed + 2, reset_gates=True)
    net.randomize_membrane_potentials("GPi", -62.0, 5.0, seed=seed + 3, reset_gates=True)
    net.randomize_membrane_potentials("Str_D2", -63.8, 5.0, seed=seed + 4, reset_gates=True)
    net.randomize_membrane_potentials("Str_D1", -63.8, 5.0, seed=seed + 5, reset_gates=True)

    # ---- Synapse parameter shorthands (from benchmark) --------------------
    tau    = 5.0   # ms — alpha-function time constant
    gpeak  = 0.43  # peak of excitatory alpha kernel
    gpeak1 = 0.3   # peak of inhibitory alpha kernel

    # ---- GPi → TH  (one-to-one, delay=5ms) --------------------------------
    net.connect(
        "GPi", "TH", "one_to_one",
        weight=0.112 * gpeak1,
        synapse=SynapseModel.alpha_function(tau=tau, E_syn=-85.0),
        delay=5.0,
    )

    # ---- GPe → STN  (shift=0 and shift=1, delay=4ms) ----------------------
    net.connect(
        "GPe", "STN", "one_to_one",
        weight=0.5 * gpeak1,
        synapse=SynapseModel.double_exponential(tau_rise=0.4, tau_decay=7.7, E_syn=-85.0),
        delay=4.0,
    )
    net.connect(
        "GPe", "STN", "shifted",
        weight=0.5 * gpeak1,
        synapse=SynapseModel.double_exponential(tau_rise=0.4, tau_decay=7.7, E_syn=-85.0),
        delay=4.0,
        shift=1,
    )

    # ---- STN → GPe  (sparse receiver-selected AMPA + NMDA, delay=2ms) ----
    gsngea = np.zeros(n)
    gsngea[rng.permutation(n)[:2]] = 0.3 * rng.random(2)
    gsngen = np.zeros(n)
    gsngen[rng.permutation(n)[:2]] = 0.002 * rng.random(2)
    for k in range(n):
        if gsngea[k] > 0:
            net.add_connection("STN", k,          "GPe", k, float(gsngea[k]) * gpeak,
                               SynapseModel.double_exponential(tau_rise=0.4, tau_decay=2.5), delay=2.0)
            net.add_connection("STN", (k-1) % n,  "GPe", k, float(gsngea[k]) * gpeak,
                               SynapseModel.double_exponential(tau_rise=0.4, tau_decay=2.5), delay=2.0)
        if gsngen[k] > 0:
            net.add_connection("STN", k,          "GPe", k, float(gsngen[k]) * gpeak,
                               SynapseModel.double_exponential(tau_rise=2.0, tau_decay=67.0), delay=2.0)
            net.add_connection("STN", (k-1) % n,  "GPe", k, float(gsngen[k]) * gpeak,
                               SynapseModel.double_exponential(tau_rise=2.0, tau_decay=67.0), delay=2.0)

    # ---- STN → GPi  (sparse receiver-selected alpha, delay=1.5ms) ---------
    gsngi = np.zeros(n)
    gsngi[rng.permutation(n)[:5]] = 0.15
    for k in range(n):
        if gsngi[k] > 0:
            net.add_connection("STN", k,         "GPi", k, float(gsngi[k]) * gpeak,
                               SynapseModel.alpha_function(tau=tau), delay=1.5)
            net.add_connection("STN", (k-1) % n, "GPi", k, float(gsngi[k]) * gpeak,
                               SynapseModel.alpha_function(tau=tau), delay=1.5)

    # ---- GPe → GPi  (shift=1 and shift=n-2, delay=3ms) --------------------
    for shift in (1, -2 % n):
        net.connect(
            "GPe", "GPi", "shifted",
            weight=0.5 * gpeak1,
            synapse=SynapseModel.alpha_function(tau=tau, E_syn=-85.0),
            delay=3.0,
            shift=shift,
        )

    # ---- GPe → GPe  (receiver-indexed random weights, delay=1ms) ----------
    ggege_scale   = 0.25 * (pd * 3 + 1)
    ggege_weights = rng.random(n)
    for k in range(n):
        net.add_connection("GPe", (k+1) % n,       "GPe", k,
                           float(ggege_weights[k]) * ggege_scale * gpeak1,
                           SynapseModel.alpha_function(tau=tau, E_syn=-85.0), delay=1.0)
        net.add_connection("GPe", (k-2+n) % n,     "GPe", k,
                           float(ggege_weights[k]) * ggege_scale * gpeak1,
                           SynapseModel.alpha_function(tau=tau, E_syn=-85.0), delay=1.0)

    # ---- Str_D2 → GPe  (inhibitory alpha, all-to-all, delay=5ms) ----------
    net.connect(
        "Str_D2", "GPe", "all_to_all",
        weight=0.5 * gpeak1,
        synapse=SynapseModel.alpha_function(tau=tau, E_syn=-85.0),
        delay=5.0,
    )

    # ---- Str_D1 → GPi  (inhibitory alpha, all-to-all, delay=4ms) ----------
    net.connect(
        "Str_D1", "GPi", "all_to_all",
        weight=0.5 * gpeak1,
        synapse=SynapseModel.alpha_function(tau=tau, E_syn=-85.0),
        delay=4.0,
    )

    # ---- CTX_e → Str_D2  (excitatory alpha, one-to-one, delay=5.1ms) -----
    net.connect(
        "CTX_e", "Str_D2", "one_to_one",
        weight=0.07 * gpeak,
        synapse=SynapseModel.alpha_function(tau=tau),
        delay=5.1,
    )

    # ---- CTX_e → Str_D1  (excitatory alpha, one-to-one, uniform weight) ----
    w_d1_base = (0.07 - 0.044 * pd) * gpeak
    net.connect(
        "CTX_e", "Str_D1", "one_to_one",
        weight=(w_d1_base, w_d1_base + 0.001 * gpeak),
        synapse=SynapseModel.alpha_function(tau=tau),
        delay=5.1,
        seed=int(rng.integers(1, 2**31)),
    )

    # ---- CTX_e → STN  (2→1 AMPA + NMDA, delay=5.9ms) ----------------------
    net.connect(
        "CTX_e", "STN", "one_to_one",
        weight=(0.0, 0.3 * gpeak),
        synapse=SynapseModel.double_exponential(tau_rise=0.5, tau_decay=2.49),
        delay=5.9,
        seed=int(rng.integers(1, 2**31)),
    )
    net.connect(
        "CTX_e", "STN", "shifted",
        weight=(0.0, 0.3 * gpeak),
        synapse=SynapseModel.double_exponential(tau_rise=0.5, tau_decay=2.49),
        delay=5.9,
        shift=1,
        seed=int(rng.integers(1, 2**31)),
    )
    net.connect(
        "CTX_e", "STN", "one_to_one",
        weight=(0.0, 0.003 * gpeak),
        synapse=SynapseModel.double_exponential(tau_rise=2.0, tau_decay=90.0),
        delay=5.9,
        seed=int(rng.integers(1, 2**31)),
    )
    net.connect(
        "CTX_e", "STN", "shifted",
        weight=(0.0, 0.003 * gpeak),
        synapse=SynapseModel.double_exponential(tau_rise=2.0, tau_decay=90.0),
        delay=5.9,
        shift=1,
        seed=int(rng.integers(1, 2**31)),
    )

    # ---- TH → CTX_e  (one-to-one, delay=5ms) ------------------------------
    net.connect(
        "TH", "CTX_e", "one_to_one",
        weight=0.15 * gpeak,
        synapse=SynapseModel.alpha_function(tau=tau),
        delay=5.0,
    )

    # ---- CTX_i → CTX_e  (4 random permutations) ---------------------------
    for _ in range(4):
        net.connect(
            "CTX_i", "CTX_e", "random_permutation",
            weight=0.2 * gpeak,
            synapse=SynapseModel.alpha_function(tau=tau, E_syn=-85.0),
            delay=0.01,
            seed=int(rng.integers(1, 2**31)),
        )

    # ---- CTX_e → CTX_i  (4 random permutations) ---------------------------
    for _ in range(4):
        net.connect(
            "CTX_e", "CTX_i", "random_permutation",
            weight=0.1 * gpeak,
            synapse=SynapseModel.alpha_function(tau=tau),
            delay=0.01,
            seed=int(rng.integers(1, 2**31)),
        )

    # ---- Intra-striatal GABA-A  (kinetic, Ggaba(V) gate) ------------------
    gaba_kin = _gaba_kinetic(tau_i=13.0, E_syn=-80.0)
    ggaba = 0.1
    for _ in range(4):
        net.connect(
            "Str_D2", "Str_D2", "random_permutation",
            weight=ggaba / 4, synapse=gaba_kin,
            allow_self=True, seed=int(rng.integers(1, 2**31)),
        )
    for _ in range(3):
        net.connect(
            "Str_D1", "Str_D1", "random_permutation",
            weight=ggaba / 3, synapse=gaba_kin,
            allow_self=True, seed=int(rng.integers(1, 2**31)),
        )

    return net


# ---------------------------------------------------------------------------
# Top-level simulate function  (mirrors benchmark signature)
# ---------------------------------------------------------------------------


def simulate_ctxbgth(
    n: int = 10,
    pd: int = 0,
    tmax: float = 1000.0,
    dt: float = 0.01,
    dbs_freq: float = 0.0,
    PW: float = 0.1,
    amplitude: float = 0.0,
    corstim: int = 0,
    seed: int = 42,
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

    POPULATIONS = ["TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i"]
    groups = {f"g{i}": [name] for i, name in enumerate(POPULATIONS)}
    net.set_thread_groups(groups)   # Phase 2: one thread per population

    # ---- External currents -------------------------------------------------
    I_ext: dict = {
        "TH":  1.2,
        "GPe": float(3.0 - 2.0 * corstim * (1 - pd)),
        "GPi": 3.0,
    }

    if corstim:
        ctx_pulse = PulseStimulator.single(onset=1000.0, duration=0.3, amplitude=350.0)
        I_ext["CTX_e"] = ctx_pulse.generate(tmax, dt)
        I_ext["CTX_i"] = ctx_pulse.generate(tmax, dt)

    if dbs_freq > 0:
        params = DBSParameters()
        params.frequency = dbs_freq
        params.pulse_width = PW
        params.amplitude = amplitude
        net.attach_stimulator("STN", DBSStimulator(params))

    # ---- Simulate ----------------------------------------------------------
    start = timer()
    result = net.simulate(
        tmax, dt, I_ext, record=RecordingConfig(["spikes"], spike_threshold=-10.0)
    )
    elapsed = timer() - start

    # ---- Spectral analysis of GPi (multitaper, matches benchmark) ----------
    gpi_beta              = analyze_beta_power(result["GPi"], duration_ms=tmax)
    gpi_alpha_beta_area   = gpi_beta["power"]
    gpi_S                 = gpi_beta["spectrum"]
    gpi_f                 = gpi_beta["frequencies"]

    spike_times  = {pop: result[pop]["spikes"] for pop in POPULATIONS}

    return gpi_alpha_beta_area, gpi_S, gpi_f, spike_times, elapsed


# ---------------------------------------------------------------------------
# Quick-run entry point
# ---------------------------------------------------------------------------

def main():
    time = 1  # seconds
    window_s = 0.25  # time window for rate-over-time reporting (seconds)

    print(f"Building CTX-BG-TH network (healthy, no DBS, n=10, {time}s) ...")
    area, S, f, spikes, t_sim = simulate_ctxbgth(
        n=10, pd=0, tmax=time * 1000, dt=0.01, corstim=0, seed=6536, amplitude=1.0
    )
    print(f"  Simulation time : {t_sim:.2f} s")
    print(f"  GPi beta-band power: {area:.4f}")

    print("\n  Overall mean firing rates:")
    for pop, sp_list in spikes.items():
        rates = [len(s) / time for s in sp_list]
        print(f"    {pop:8s}: {np.mean(rates):.1f} spk/s")

    tmax_ms  = time * 1000.0
    win_ms   = window_s * 1000.0
    n_windows = int(tmax_ms / win_ms)
    edges_ms = [i * win_ms for i in range(n_windows + 1)]

    pops = list(spikes.keys())
    hdr  = f"  {'Window':>16s} | " + " | ".join(f"{p:>8s}" for p in pops)
    print(f"\n  Firing rates by {window_s * 1000:.0f} ms window (spk/s):")
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for w in range(n_windows):
        t0, t1 = edges_ms[w], edges_ms[w + 1]
        label  = f"{t0 / 1000:.2f}–{t1 / 1000:.2f} s"
        row_rates = []
        for pop in pops:
            sp_list = spikes[pop]
            count   = sum(np.sum((s >= t0) & (s < t1)) for s in sp_list)
            mean_rate = count / len(sp_list) / window_s if sp_list else 0.0
            row_rates.append(mean_rate)
        print(f"  {label:>16s} | " + " | ".join(f"{r:>8.1f}" for r in row_rates))
    print("  " + "-" * (len(hdr) - 2))

if __name__ == "__main__":
    main()
