"""
Neuron spec builder functions for use in tests.

These Python implementations replicate the parameter values that were
previously defined as static preset factories on NeuronModelSpec
(thalamic, stn, gpe, gpi, striatum).  They are kept here rather than
in the library because the models are circuit-specific and unlikely to be
needed outside this test suite.

References
----------
Rubin & Terman (2004) — thalamic, STN, GPe/GPi
McCarthy et al. (2011) / Humphries et al. (2009) — striatum
"""

from hodgkin_huxley import (
    NeuronModel,
    NeuronModelSpec,
    Boltzmann,
    RateFunc,
    Tau,
)


def make_thalamic(V_init: float = -65.0) -> NeuronModelSpec:
    """
    Thalamic relay cell (Rubin & Terman 2004).
    Channels: Na(m_Na^3·h_Na), K(n_K^4), T(m_T^2·h_T), Leak.
    n_K is derived: 0.75*(1 - h_T).
    """
    model = NeuronModel("TH", C_m=1.0, V_init=V_init)

    h_na = model.add_gate("h_Na", update_form="inf_tau", initial_value=0.6,
                           inf=Boltzmann(-41.0, -4.0),
                           tau=Tau.scaled_exp(1.0, -41.0, -4.0))

    h_t = model.add_gate("h_T", update_form="inf_tau", initial_value=0.35,
                          inf=Boltzmann(-80.0, -6.0),
                          tau=Tau.double_exp_sum(20.0, 100.0, 39.7, 9.32, 0.6, 7.4))

    m_na = model.add_gate("m_Na", update_form="instant", initial_value=0.05,
                           inf=Boltzmann(-37.0, 7.0))

    m_t = model.add_gate("m_T", update_form="instant", initial_value=0.0,
                          inf=Boltzmann(-57.0, 6.2))

    # n_K = 0.75 * (1 - h_T)
    n_k = model.add_gate("n_K", update_form="derived",
                          derived_source_gate=h_t,
                          derived_a=0.75, derived_b=1.0, derived_c=-1.0)

    model.add_channel("Na",   g=3.0,  E_rev=50.0,  gates=[(m_na, 3), (h_na, 1)])
    model.add_channel("K",    g=5.0,  E_rev=-90.0, gates=[(n_k, 4)])
    model.add_channel("T",    g=5.0,  E_rev=0.0,   gates=[(m_t, 2), (h_t, 1)])
    model.add_channel("Leak", g=0.05, E_rev=-70.0)

    return model.to_spec()


def make_stn() -> NeuronModelSpec:
    """
    Subthalamic nucleus (Rubin & Terman 2004 / Hahn et al. 2019).
    Channels: Na(m^3·h), K-DR(n^4), A(a^2·b), CaL(c^2·d1·d2),
              T(p^2·q), AHP-K(r^2), Leak.
    """
    model = NeuronModel("STN", C_m=1.0, V_init=-62.0)

    m  = model.add_gate("m_Na",   update_form="inf_tau", initial_value=0.060,
                         inf=Boltzmann(-40.0,  8.0),
                         tau=Tau.boltzmann(0.2, 3.0, -53.0, -0.7))
    h  = model.add_gate("h_Na",   update_form="inf_tau", initial_value=0.929,
                         inf=Boltzmann(-45.5, -6.4),
                         tau=Tau.double_exp_sum(0.0, 24.5, 50.0, 15.0, 50.0, 16.0))
    n  = model.add_gate("n_K",    update_form="inf_tau", initial_value=0.182,
                         inf=Boltzmann(-41.0, 14.0),
                         tau=Tau.double_exp_sum(0.0, 11.0, 40.0, 40.0, 40.0, 50.0))
    a  = model.add_gate("a_A",    update_form="inf_tau", initial_value=0.239,
                         inf=Boltzmann(-45.0, 14.7),
                         tau=Tau.boltzmann(1.0, 1.0, -40.0, -0.5))
    b  = model.add_gate("b_A",    update_form="inf_tau", initial_value=0.023,
                         inf=Boltzmann(-90.0, -7.5),
                         tau=Tau.double_exp_sum(0.0, 200.0, 60.0, 30.0, 40.0, 10.0))
    c  = model.add_gate("c_CaL",  update_form="inf_tau", initial_value=0.002,
                         inf=Boltzmann(-30.6,  5.0),
                         tau=Tau.double_exp_sum(45.0, 10.0, 27.0, 20.0, 50.0, 15.0))
    d1 = model.add_gate("d1_CaL", update_form="inf_tau", initial_value=0.566,
                         inf=Boltzmann(-60.0, -7.5),
                         tau=Tau.double_exp_sum(400.0, 500.0, 40.0, 15.0, 20.0, 20.0))
    d2 = model.add_gate("d2_CaL", update_form="inf_tau", initial_value=1.0,
                         inf=Boltzmann(0.1, -0.02),
                         tau=Tau.constant(130.0))
    p  = model.add_gate("p_T",    update_form="inf_tau", initial_value=0.290,
                         inf=Boltzmann(-56.0,  6.7),
                         tau=Tau.double_exp_sum(5.0, 0.33, 27.0, 10.0, 102.0, 15.0))
    q  = model.add_gate("q_T",    update_form="inf_tau", initial_value=0.019,
                         inf=Boltzmann(-85.0, -5.8),
                         tau=Tau.double_exp_sum(0.0, 400.0, 50.0, 15.0, 50.0, 16.0))
    r  = model.add_gate("r_AHP",  update_form="inf_tau", initial_value=0.0,
                         inf=Boltzmann(0.17, 0.08),
                         tau=Tau.constant(2.0))

    model.add_channel("Na",    g=49.0,  E_rev= 60.0, gates=[(m, 3), (h, 1)])
    model.add_channel("K",     g=57.0,  E_rev=-90.0, gates=[(n, 4)])
    model.add_channel("A",     g= 5.0,  E_rev=-90.0, gates=[(a, 2), (b, 1)])
    cal = model.add_channel("CaL",  g= 0.5,  E_rev=  0.0,
                             use_calcium_nernst=True, gates=[(c, 2), (d1, 1), (d2, 1)])
    t   = model.add_channel("T",    g= 5.0,  E_rev=  0.0,
                             use_calcium_nernst=True, gates=[(p, 2), (q, 1)])
    model.add_channel("AHP_K", g= 1.0,  E_rev=-90.0, gates=[(r, 2)])
    model.add_channel("Leak",  g=0.35,  E_rev=-60.0)

    model.set_calcium(epsilon=5.182e-6, K_Ca=386.0, Ca_init=0.005,
                      use_nernst=True, Ca_o=2000.0, source_channels=[cal, t])
    return model.to_spec()


def make_gpe() -> NeuronModelSpec:
    """
    Globus pallidus externus (Rubin & Terman 2004).
    Channels: Na(m^3·h), K(n^4), T(a^3·r), CaL(s^2), AHP(Ca-dep K), Leak.
    """
    model = NeuronModel("GPe", C_m=1.0, V_init=-62.0)

    m = model.add_gate("m_Na",  update_form="instant", initial_value=0.076,
                        inf=Boltzmann(-37.0, 10.0))
    h = model.add_gate("h_Na",  update_form="inf_tau", scale=0.05, initial_value=0.583,
                        inf=Boltzmann(-58.0, -12.0),
                        tau=Tau.boltzmann(0.05, 0.27, -40.0, -12.0))
    n = model.add_gate("n_K",   update_form="inf_tau", scale=0.1,  initial_value=0.298,
                        inf=Boltzmann(-50.0,  14.0),
                        tau=Tau.boltzmann(0.05, 0.27, -40.0, -12.0))
    a = model.add_gate("a_T",   update_form="instant", initial_value=0.0,
                        inf=Boltzmann(-57.0,   2.0))
    r = model.add_gate("r_T",   update_form="inf_tau", initial_value=0.018,
                        inf=Boltzmann(-70.0,  -2.0),
                        tau=Tau.constant(30.0))
    s = model.add_gate("s_CaL", update_form="instant", initial_value=0.0,
                        inf=Boltzmann(-35.0,   2.0))

    model.add_channel("Na",   g=120.0, E_rev= 55.0, gates=[(m, 3), (h, 1)])
    model.add_channel("K",    g= 30.0, E_rev=-80.0, gates=[(n, 4)])
    t_ch = model.add_channel("T",   g=  0.5, E_rev=120.0, gates=[(a, 3), (r, 1)])
    cal  = model.add_channel("CaL", g= 0.15, E_rev=120.0, gates=[(s, 2)])
    model.add_channel("AHP",  g= 10.0, E_rev=-80.0, is_ahp=True, ahp_k1=10.0)
    model.add_channel("Leak", g=  0.1, E_rev=-65.0)

    model.set_calcium(epsilon=1e-4, K_Ca=15.0, Ca_init=0.1,
                      source_channels=[t_ch, cal])
    return model.to_spec()


def make_gpi() -> NeuronModelSpec:
    """GPi: identical neuron parameters to GPe."""
    spec = make_gpe()
    spec.name = "GPi"
    return spec


def make_striatum(pd: float = 0.0) -> NeuronModelSpec:
    """
    Striatal medium spiny neuron (McCarthy et al. 2011).
    Channels: Na(m^3·h), K-DR(n^4), M-type K(m_M^1, dopamine-modulated), Leak.
    pd=0.0 (healthy): g_M=1.2;  pd=1.0 (fully depleted): g_M=0.4.
    """
    model = NeuronModel("Striatum", C_m=1.0, V_init=-87.0)

    m = model.add_gate("m_Na", update_form="alpha_beta", initial_value=0.03,
                        alpha=RateFunc.linear_over_expm1(0.32,  54.0,  4.0),
                        beta =RateFunc.linear_over_exp  (0.28,  27.0,  5.0))
    h = model.add_gate("h_Na", update_form="alpha_beta", initial_value=0.99,
                        alpha=RateFunc.exp_decay(0.128, 50.0, -18.0),
                        beta =RateFunc.sigmoid  (4.0,   27.0,  -5.0))
    n = model.add_gate("n_K",  update_form="alpha_beta", initial_value=0.01,
                        alpha=RateFunc.linear_over_expm1(0.032, 52.0,  5.0),
                        beta =RateFunc.exp_decay        (0.5,   57.0, -40.0))
    p = model.add_gate("m_M",  update_form="inf_tau", initial_value=0.01,
                        inf=Boltzmann(-30.0, 9.0),
                        tau=Tau.scaled_exp(1000.0, -35.0, 20.0))

    model.add_channel("Na",   g=100.0,          E_rev= 50.0, gates=[(m, 3), (h, 1)])
    model.add_channel("K",    g= 80.0,          E_rev=-100.0, gates=[(n, 4)])
    model.add_channel("M",    g=1.2 - 0.8 * pd, E_rev=-100.0, gates=[(p, 1)])
    model.add_channel("Leak", g=  0.1,          E_rev= -67.0)

    return model.to_spec()
