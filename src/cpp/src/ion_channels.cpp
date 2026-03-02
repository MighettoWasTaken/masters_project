#include "hodgkin_huxley/ion_channels.hpp"
#include <iostream>
#include <stdexcept>

namespace hodgkin_huxley {

// =============================================================================
// Thalamic relay neuron (TH)
// Based on Rubin & Terman (2004) thalamocortical relay model
// Channels: Na, K, T-type Ca (low-threshold), Leak
// Gates: m_Na (instant), h_Na, m_T (instant), h_T, n_K (derived from h_T)
// =============================================================================

NeuronModelSpec NeuronModelSpec::thalamic() {
    NeuronModelSpec spec;
    spec.name = "TH";
    spec.C_m = 1.0;
    spec.V_init = -65.0;

    // Gate 0: h_Na (Na inactivation, dynamic)
    {
        GateSpec g;
        g.name = "h_Na";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.6;
        g.inf = {-41.0, -4.0};  // h_inf = 1/(1+exp((V+41)/4))
        g.tau.form = TauParams::Form::SCALED_EXP;
        g.tau.params[0] = 1.0;    // scale
        g.tau.params[1] = -41.0;  // v_half
        g.tau.params[2] = -4.0;   // k (used as: tau = scale / cosh((V-v_half)/(2*k)))
        spec.gates.push_back(g);
    }

    // Gate 1: h_T (T-current inactivation, dynamic)
    {
        GateSpec g;
        g.name = "h_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.35;
        g.inf = {-80.0, -6.0};  // h_T_inf = 1/(1+exp((V+80)/6))
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        // tau_h_T = 20 + 100 / (exp((V+39.7)/9.32) + exp(-(V+0.6)/7.4))
        g.tau.params[0] = 20.0;   // base
        g.tau.params[1] = 100.0;  // amplitude (numerator)
        g.tau.params[2] = 39.7;   // v1 offset
        g.tau.params[3] = 9.32;   // s1
        g.tau.params[4] = 0.0;    // (unused for sum form denominator)
        g.tau.params[5] = 0.6;    // v2 offset  (was -0.6: sign error caused tau~1535ms)
        g.tau.params[6] = 7.4;    // s2         (was -7.4: double negation -> wrong exponent)
        spec.gates.push_back(g);
    }

    // Gate 2: m_Na (Na activation, instant)
    {
        GateSpec g;
        g.name = "m_Na";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.05;
        g.inf = {-37.0, 7.0};  // m_inf = 1/(1+exp(-(V+37)/7))  (k>0 means positive slope)
        spec.gates.push_back(g);
    }

    // Gate 3: m_T (T-current activation, instant)
    {
        GateSpec g;
        g.name = "m_T";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.0;
        g.inf = {-57.0, 6.2};  // m_T_inf = 1/(1+exp(-(V+57)/6.2))
        spec.gates.push_back(g);
    }

    // Gate 4: n_K (K activation, derived from h_T)
    // n_K = 0.75 * (1 - h_T)
    {
        GateSpec g;
        g.name = "n_K";
        g.update_form = GateSpec::UpdateForm::DERIVED;
        g.derived_source_gate = 1;  // h_T
        g.derived_a = 0.75;
        g.derived_b = 1.0;
        g.derived_c = -1.0;  // n_K = 0.75*(1 - h_T)
        g.initial_value = 0.0;
        spec.gates.push_back(g);
    }

    // Channel 0: Na
    {
        ChannelSpec c;
        c.name = "Na";
        c.g = 3.0;
        c.E_rev = 50.0;
        c.gates = {{2, 3}, {0, 1}};  // m_Na^3 * h_Na
        spec.channels.push_back(c);
    }

    // Channel 1: K
    {
        ChannelSpec c;
        c.name = "K";
        c.g = 5.0;
        c.E_rev = -90.0;
        c.gates = {{4, 4}};  // n_K^4
        spec.channels.push_back(c);
    }

    // Channel 2: T-type Ca (low threshold)
    {
        ChannelSpec c;
        c.name = "T";
        c.g = 5.0;
        c.E_rev = 0.0;
        c.gates = {{3, 2}, {1, 1}};  // m_T^2 * h_T
        spec.channels.push_back(c);
    }

    // Channel 3: Leak
    {
        ChannelSpec c;
        c.name = "Leak";
        c.g = 0.05;
        c.E_rev = -70.0;
        spec.channels.push_back(c);
    }

    return spec;
}

// =============================================================================
// Subthalamic Nucleus (STN)
// Rubin & Terman (2004) / Hahn et al. (2019) parameterisation.
// All gate inf/tau functions match stn_xinf/stn_taux from benchmark exactly.
// Channels: Na(m³h), K-DR(n⁴), A(a²b), CaL(c²d1d2), T(p²q), AHP-K(r²), Leak
// Ca dynamics: dCa/dt = epsilon*(-I_Ca - K_Ca*Ca)
//   epsilon=alp=5e-5, K_Ca=Kca/alp=0.002/5e-5=40
// =============================================================================

NeuronModelSpec NeuronModelSpec::stn() {
    NeuronModelSpec spec;
    spec.name = "STN";
    spec.C_m = 1.0;
    spec.V_init = -62.0;

    // Gate 0: m_Na — Na activation (dynamic)
    // stn_minf = 1/(1+exp(-(V+40)/8))
    // stn_taum = 0.2 + 3/(1+exp((V+53)/0.7))  → BOLTZMANN(0.2, 3, -53, -0.7)
    {
        GateSpec g;
        g.name = "m_Na";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.060;
        g.inf = {-40.0, 8.0};
        g.tau.form = TauParams::Form::BOLTZMANN;
        g.tau.params[0] = 0.2;    // base
        g.tau.params[1] = 3.0;    // amp
        g.tau.params[2] = -53.0;  // v_half
        g.tau.params[3] = -0.7;   // k
        spec.gates.push_back(g);
    }

    // Gate 1: h_Na — Na inactivation (dynamic)
    // stn_hinf = 1/(1+exp((V+45.5)/6.4))
    // stn_tauh = 24.5/(exp((V+50)/15) + exp(-(V+50)/16))
    {
        GateSpec g;
        g.name = "h_Na";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.929;
        g.inf = {-45.5, -6.4};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 0.0;    // base
        g.tau.params[1] = 24.5;   // amp
        g.tau.params[2] = 50.0;   // v1
        g.tau.params[3] = 15.0;   // s1
        g.tau.params[4] = 0.0;
        g.tau.params[5] = 50.0;   // v2
        g.tau.params[6] = 16.0;   // s2
        spec.gates.push_back(g);
    }

    // Gate 2: n_K — K-DR activation (dynamic)
    // stn_ninf = 1/(1+exp(-(V+41)/14))
    // stn_taun = 11/(exp((V+40)/40) + exp(-(V+40)/50))
    {
        GateSpec g;
        g.name = "n_K";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.182;
        g.inf = {-41.0, 14.0};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 0.0;    // base
        g.tau.params[1] = 11.0;   // amp
        g.tau.params[2] = 40.0;   // v1
        g.tau.params[3] = 40.0;   // s1
        g.tau.params[4] = 0.0;
        g.tau.params[5] = 40.0;   // v2
        g.tau.params[6] = 50.0;   // s2
        spec.gates.push_back(g);
    }

    // Gate 3: a_A — A-type K activation (dynamic)
    // stn_ainf = 1/(1+exp(-(V+45)/14.7))
    // stn_taua = 1 + 1/(1+exp((V+40)/0.5))  → BOLTZMANN(1, 1, -40, -0.5)
    {
        GateSpec g;
        g.name = "a_A";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.239;
        g.inf = {-45.0, 14.7};
        g.tau.form = TauParams::Form::BOLTZMANN;
        g.tau.params[0] = 1.0;    // base
        g.tau.params[1] = 1.0;    // amp
        g.tau.params[2] = -40.0;  // v_half
        g.tau.params[3] = -0.5;   // k
        spec.gates.push_back(g);
    }

    // Gate 4: b_A — A-type K inactivation (dynamic)
    // stn_binf = 1/(1+exp((V+90)/7.5))
    // stn_taub = 200/(exp((V+60)/30) + exp(-(V+40)/10))
    {
        GateSpec g;
        g.name = "b_A";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.023;
        g.inf = {-90.0, -7.5};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 0.0;    // base
        g.tau.params[1] = 200.0;  // amp
        g.tau.params[2] = 60.0;   // v1
        g.tau.params[3] = 30.0;   // s1
        g.tau.params[4] = 0.0;
        g.tau.params[5] = 40.0;   // v2
        g.tau.params[6] = 10.0;   // s2
        spec.gates.push_back(g);
    }

    // Gate 5: c_CaL — L-type Ca activation (dynamic)
    // stn_cinf = 1/(1+exp(-(V+30.6)/5))
    // stn_tauc = 45 + 10/(exp((V+27)/20) + exp(-(V+50)/15))
    {
        GateSpec g;
        g.name = "c_CaL";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.002;
        g.inf = {-30.6, 5.0};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 45.0;   // base
        g.tau.params[1] = 10.0;   // amp
        g.tau.params[2] = 27.0;   // v1
        g.tau.params[3] = 20.0;   // s1
        g.tau.params[4] = 0.0;
        g.tau.params[5] = 50.0;   // v2
        g.tau.params[6] = 15.0;   // s2
        spec.gates.push_back(g);
    }

    // Gate 6: d1_CaL — L-type Ca slow voltage inactivation (dynamic)
    // stn_d1inf = 1/(1+exp((V+60)/7.5))
    // stn_taud1 = 400 + 500/(exp((V+40)/15) + exp(-(V+20)/20))
    {
        GateSpec g;
        g.name = "d1_CaL";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.566;
        g.inf = {-60.0, -7.5};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 400.0;  // base
        g.tau.params[1] = 500.0;  // amp
        g.tau.params[2] = 40.0;   // v1
        g.tau.params[3] = 15.0;   // s1
        g.tau.params[4] = 0.0;
        g.tau.params[5] = 20.0;   // v2
        g.tau.params[6] = 20.0;   // s2
        spec.gates.push_back(g);
    }

    // Gate 7: d2_CaL — L-type Ca fast inactivation (voltage-dep, NOT Ca-dep)
    // stn_d2inf = 1/(1+exp((V-0.1)/0.02))  → Boltzmann(0.1, -0.02)
    // td2 = 130 ms (constant)
    {
        GateSpec g;
        g.name = "d2_CaL";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 1.0;
        g.inf = {0.1, -0.02};
        g.dependency = GateSpec::Dependency::VOLTAGE;  // explicitly voltage-dep
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 130.0;
        spec.gates.push_back(g);
    }

    // Gate 8: p_T — T-type Ca activation (dynamic)
    // stn_pinf = 1/(1+exp(-(V+56)/6.7))
    // stn_taup = 5 + 0.33/(exp((V+27)/10) + exp(-(V+102)/15))
    {
        GateSpec g;
        g.name = "p_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.290;
        g.inf = {-56.0, 6.7};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 5.0;    // base
        g.tau.params[1] = 0.33;   // amp
        g.tau.params[2] = 27.0;   // v1
        g.tau.params[3] = 10.0;   // s1
        g.tau.params[4] = 0.0;
        g.tau.params[5] = 102.0;  // v2
        g.tau.params[6] = 15.0;   // s2
        spec.gates.push_back(g);
    }

    // Gate 9: q_T — T-type Ca inactivation (dynamic)
    // stn_qinf = 1/(1+exp((V+85)/5.8))
    // stn_tauq = 400/(exp((V+50)/15) + exp(-(V+50)/16))
    {
        GateSpec g;
        g.name = "q_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.019;
        g.inf = {-85.0, -5.8};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 0.0;    // base
        g.tau.params[1] = 400.0;  // amp
        g.tau.params[2] = 50.0;   // v1
        g.tau.params[3] = 15.0;   // s1
        g.tau.params[4] = 0.0;
        g.tau.params[5] = 50.0;   // v2
        g.tau.params[6] = 16.0;   // s2
        spec.gates.push_back(g);
    }

    // Gate 10: r_AHP — AHP K activation (voltage-dep near AP peak, NOT Ca-dep)
    // stn_rinf = 1/(1+exp(-(V-0.17)/0.08))  → Boltzmann(0.17, 0.08)
    // tr2 = 2 ms (constant)
    {
        GateSpec g;
        g.name = "r_AHP";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.0;
        g.inf = {0.17, 0.08};
        g.dependency = GateSpec::Dependency::VOLTAGE;  // explicitly voltage-dep
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 2.0;
        spec.gates.push_back(g);
    }

    // Channel 0: Na — g=49, Ena=60
    {
        ChannelSpec c;
        c.name = "Na";
        c.g = 49.0;
        c.E_rev = 60.0;
        c.gates = {{0, 3}, {1, 1}};  // m_Na^3 * h_Na
        spec.channels.push_back(c);
    }

    // Channel 1: K (delayed rectifier) — g=57, Ek=-90
    {
        ChannelSpec c;
        c.name = "K";
        c.g = 57.0;
        c.E_rev = -90.0;
        c.gates = {{2, 4}};  // n_K^4
        spec.channels.push_back(c);
    }

    // Channel 2: A-type K — g=5, Ek=-90
    {
        ChannelSpec c;
        c.name = "A";
        c.g = 5.0;
        c.E_rev = -90.0;
        c.gates = {{3, 2}, {4, 1}};  // a_A^2 * b_A
        spec.channels.push_back(c);
    }

    // Channel 3: L-type Ca — g=0.5, Nernst
    {
        ChannelSpec c;
        c.name = "CaL";
        c.g = 0.5;
        c.E_rev = 0.0;
        c.use_calcium_nernst = true;
        c.gates = {{5, 2}, {6, 1}, {7, 1}};  // c_CaL^2 * d1_CaL * d2_CaL
        spec.channels.push_back(c);
    }

    // Channel 4: T-type Ca — g=5, Nernst
    {
        ChannelSpec c;
        c.name = "T";
        c.g = 5.0;
        c.E_rev = 0.0;
        c.use_calcium_nernst = true;
        c.gates = {{8, 2}, {9, 1}};  // p_T^2 * q_T
        spec.channels.push_back(c);
    }

    // Channel 5: AHP K — g=1, Ek=-90, gate-based r^2 (NOT is_ahp)
    {
        ChannelSpec c;
        c.name = "AHP_K";
        c.g = 1.0;
        c.E_rev = -90.0;
        c.gates = {{10, 2}};  // r_AHP^2
        spec.channels.push_back(c);
    }

    // Channel 6: Leak — g=0.35, El=-60
    {
        ChannelSpec c;
        c.name = "Leak";
        c.g = 0.35;
        c.E_rev = -60.0;
        spec.channels.push_back(c);
    }

    // Calcium dynamics: dCa/dt = epsilon*(-I_Ca - K_Ca*Ca)
    // matches benchmark: dCa/dt = -alp*(IL + IT) - Kca*Ca
    // alp = 1/(Z*F) = 1/(2*96485) = 5.182e-6   (NOT 5e-5 from fast version)
    // Kca = 2e-3 = 0.002
    // K_Ca = Kca / alp = 0.002 / 5.182e-6 ≈ 386
    spec.calcium.enabled = true;
    spec.calcium.use_nernst = true;
    spec.calcium.epsilon = 5.182e-6;  // = 1/(2*96485)
    spec.calcium.K_Ca = 386.0;        // = Kca/alp = 0.002/5.182e-6
    spec.calcium.Ca_init = 0.005;
    spec.calcium.Ca_o = 2000.0;
    spec.calcium.source_channels = {3, 4};  // CaL (ch3) and T (ch4)

    return spec;
}

// =============================================================================
// Globus Pallidus external (GPe)
// Based on Rubin & Terman (2004)
// Channels: Na, K, T-type Ca, AHP (Ca-dep K), Leak
// Calcium with simple exponential decay
// =============================================================================

NeuronModelSpec NeuronModelSpec::gpe() {
    NeuronModelSpec spec;
    spec.name = "GPe";
    spec.C_m = 1.0;
    spec.V_init = -62.0;

    // Gate 0: m_Na (instant Na activation)
    //   gpe_minf(V) = 1/(1+exp(-(V+37)/10)) = Boltzmann(-37, 10)
    {
        GateSpec g;
        g.name = "m_Na";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.076;
        g.inf = {-37.0, 10.0};
        spec.gates.push_back(g);
    }

    // Gate 1: h_Na (Na inactivation)
    //   gpe_hinf(V) = 1/(1+exp((V+58)/12)) = Boltzmann(-58, -12)
    //   tau_h(V) = 0.05 + 0.27/(1+exp((V+40)/12)), phi_h=0.05
    {
        GateSpec g;
        g.name = "h_Na";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.scale = 0.05;  // phi_h = 0.05
        g.initial_value = 0.583;
        g.inf = {-58.0, -12.0};
        g.tau.form = TauParams::Form::BOLTZMANN;
        g.tau.params[0] = 0.05;   // base
        g.tau.params[1] = 0.27;   // amp
        g.tau.params[2] = -40.0;  // vh
        g.tau.params[3] = -12.0;  // k
        spec.gates.push_back(g);
    }

    // Gate 2: n_K (K delayed-rectifier activation)
    //   gpe_ninf(V) = 1/(1+exp(-(V+50)/14)) = Boltzmann(-50, 14)
    //   tau_n(V) = 0.05 + 0.27/(1+exp((V+40)/12)), phi_n=0.1
    {
        GateSpec g;
        g.name = "n_K";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.scale = 0.1;  // phi_n = 0.1
        g.initial_value = 0.298;
        g.inf = {-50.0, 14.0};
        g.tau.form = TauParams::Form::BOLTZMANN;
        g.tau.params[0] = 0.05;   // base
        g.tau.params[1] = 0.27;   // amp
        g.tau.params[2] = -40.0;  // vh
        g.tau.params[3] = -12.0;  // k
        spec.gates.push_back(g);
    }

    // Gate 3: a_T (T-type Ca activation, instant)
    //   gpe_ainf(V) = 1/(1+exp(-(V+57)/2)) = Boltzmann(-57, 2)
    {
        GateSpec g;
        g.name = "a_T";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.0;
        g.inf = {-57.0, 2.0};
        spec.gates.push_back(g);
    }

    // Gate 4: r_T (T-type Ca inactivation)
    //   gpe_rinf(V) = 1/(1+exp((V+70)/2)) = Boltzmann(-70, -2)
    //   tau_r = 30 ms (constant), phi=1
    {
        GateSpec g;
        g.name = "r_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.scale = 1.0;
        g.initial_value = 0.018;
        g.inf = {-70.0, -2.0};
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 30.0;
        spec.gates.push_back(g);
    }

    // Gate 5: s_CaL (L-type Ca activation, instant)
    //   gpe_sinf(V) = 1/(1+exp(-(V+35)/2)) = Boltzmann(-35, 2)
    {
        GateSpec g;
        g.name = "s_CaL";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.0;
        g.inf = {-35.0, 2.0};
        spec.gates.push_back(g);
    }

    // Channel 0: Na — m^3 * h
    {
        ChannelSpec c;
        c.name = "Na";
        c.g = 120.0;
        c.E_rev = 55.0;
        c.gates = {{0, 3}, {1, 1}};
        spec.channels.push_back(c);
    }

    // Channel 1: K — n^4
    {
        ChannelSpec c;
        c.name = "K";
        c.g = 30.0;
        c.E_rev = -80.0;
        c.gates = {{2, 4}};
        spec.channels.push_back(c);
    }

    // Channel 2: T-type Ca — a^3 * r
    //   It = gt[2] * a^3 * r * (V - Eca[2]),  gt[2]=0.5, Eca[2]=120
    {
        ChannelSpec c;
        c.name = "T";
        c.g = 0.5;
        c.E_rev = 120.0;
        c.gates = {{3, 3}, {4, 1}};
        spec.channels.push_back(c);
    }

    // Channel 3: L-type Ca — s^2
    //   Ica = gca[2] * s^2 * (V - Eca[2]),  gca[2]=0.15, Eca[2]=120
    {
        ChannelSpec c;
        c.name = "CaL";
        c.g = 0.15;
        c.E_rev = 120.0;
        c.gates = {{5, 2}};
        spec.channels.push_back(c);
    }

    // Channel 4: AHP (Ca-dependent K)
    //   Iahp = gahp[2] * (Ca/(Ca+k1[2])) * (V-Ek),  gahp[2]=10, k1[2]=10
    {
        ChannelSpec c;
        c.name = "AHP";
        c.g = 10.0;
        c.E_rev = -80.0;
        c.is_ahp = true;
        c.ahp_k1 = 10.0;
        spec.channels.push_back(c);
    }

    // Channel 5: Leak
    //   Il = gl[2] * (V - El[2]),  gl[2]=0.1, El[2]=-65
    {
        ChannelSpec c;
        c.name = "Leak";
        c.g = 0.1;
        c.E_rev = -65.0;
        spec.channels.push_back(c);
    }

    // Calcium dynamics
    //   dCa/dt = 1e-4 * (-Ica - It - kca[2]*Ca),  kca[2]=15
    spec.calcium.enabled = true;
    spec.calcium.use_nernst = false;
    spec.calcium.epsilon = 1e-4;
    spec.calcium.K_Ca = 15.0;
    spec.calcium.Ca_init = 0.1;
    spec.calcium.source_channels = {2, 3};  // T-type and L-type Ca

    return spec;
}

// =============================================================================
// GPi - identical neuron parameters to GPe (benchmark uses same gna/gk/etc.)
// =============================================================================

NeuronModelSpec NeuronModelSpec::gpi() {
    NeuronModelSpec spec = gpe();
    spec.name = "GPi";
    return spec;
}

// =============================================================================
// Striatal medium spiny neuron (MSN)
// Based on McCarthy et al. (2011) / Humphries et al. (2009)
// Channels: Na, K, Leak, M-type K (modulated by dopamine)
// Alpha-beta gate kinetics (HH-style)
// =============================================================================

NeuronModelSpec NeuronModelSpec::striatum(double pd) {
    NeuronModelSpec spec;
    spec.name = "Striatum";
    spec.C_m = 1.0;
    spec.V_init = -87.0;

    // Gate 0: m_Na (alpha-beta)
    {
        GateSpec g;
        g.name = "m_Na";
        g.update_form = GateSpec::UpdateForm::ALPHA_BETA;
        g.initial_value = 0.03;
        g.alpha.form = RateFuncParams::Form::LINEAR_OVER_EXPM1;
        g.alpha.A = 0.32;
        g.alpha.B = 54.0;   // alpha_m = 0.32*(V+54) / (1 - exp(-(V+54)/4))
        g.alpha.C = 4.0;
        g.beta.form = RateFuncParams::Form::LINEAR_OVER_EXPM1;
        g.beta.A = -0.28;
        g.beta.B = 27.0;    // beta_m = -0.28*(V+27) / (1 - exp((V+27)/5))
        g.beta.C = -5.0;
        spec.gates.push_back(g);
    }

    // Gate 1: h_Na (alpha-beta)
    {
        GateSpec g;
        g.name = "h_Na";
        g.update_form = GateSpec::UpdateForm::ALPHA_BETA;
        g.initial_value = 0.99;
        g.alpha.form = RateFuncParams::Form::EXP_DECAY;
        g.alpha.A = 0.128;
        g.alpha.B = 50.0;   // alpha_h = 0.128 * exp(-(V+50)/18)
        g.alpha.C = -18.0;
        g.beta.form = RateFuncParams::Form::SIGMOID;
        g.beta.A = 4.0;
        g.beta.B = 27.0;    // beta_h = 4 / (1 + exp(-(V+27)/5))
        g.beta.C = -5.0;
        spec.gates.push_back(g);
    }

    // Gate 2: n_K (alpha-beta)
    {
        GateSpec g;
        g.name = "n_K";
        g.update_form = GateSpec::UpdateForm::ALPHA_BETA;
        g.initial_value = 0.01;
        g.alpha.form = RateFuncParams::Form::LINEAR_OVER_EXPM1;
        g.alpha.A = 0.032;
        g.alpha.B = 52.0;   // alpha_n = 0.032*(V+52) / (1 - exp(-(V+52)/5))
        g.alpha.C = 5.0;
        g.beta.form = RateFuncParams::Form::EXP_DECAY;
        g.beta.A = 0.5;
        g.beta.B = 57.0;    // beta_n = 0.5 * exp(-(V+57)/40)
        g.beta.C = -40.0;
        spec.gates.push_back(g);
    }

    // Gate 3: m_M (M-current, inf-tau with slow kinetics)
    {
        GateSpec g;
        g.name = "m_M";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.01;
        g.inf = {-30.0, 9.0};
        g.tau.form = TauParams::Form::SCALED_EXP;
        g.tau.params[0] = 1000.0;  // tau_M = 1000 / (3.3*(exp((V+35)/20)+exp(-(V+35)/20)))
        g.tau.params[1] = -35.0;
        g.tau.params[2] = 20.0;
        spec.gates.push_back(g);
    }

    // Channel 0: Na
    {
        ChannelSpec c;
        c.name = "Na";
        c.g = 100.0;
        c.E_rev = 50.0;
        c.gates = {{0, 3}, {1, 1}};
        spec.channels.push_back(c);
    }

    // Channel 1: K (delayed rectifier)
    {
        ChannelSpec c;
        c.name = "K";
        c.g = 80.0;
        c.E_rev = -100.0;
        c.gates = {{2, 4}};
        spec.channels.push_back(c);
    }

    // Channel 2: M-current (slow K, dopamine-modulated)
    // pd=0 (healthy): g_M=1.2, pd=1 (fully depleted): g_M=0.4
    {
        ChannelSpec c;
        c.name = "M";
        c.g = 1.2 - 0.8 * pd;  // Linear interpolation
        c.E_rev = -100.0;
        c.gates = {{3, 1}};
        spec.channels.push_back(c);
    }

    // Channel 3: Leak
    {
        ChannelSpec c;
        c.name = "Leak";
        c.g = 0.1;
        c.E_rev = -67.0;
        spec.channels.push_back(c);
    }

    return spec;
}

// =============================================================================
// KineticSynapseSpec presets
// =============================================================================

KineticSynapseSpec KineticSynapseSpec::gaba_kinetic() {
    KineticSynapseSpec s;
    s.name = "GABA_kinetic";
    s.update_form = UpdateForm::TANH_GATE;
    s.tanh_amp = 2.0;
    s.tanh_vh = 0.0;
    s.tanh_k = 4.0;
    s.tau_decay = 13.0;
    s.current_form = CurrentForm::LINEAR;
    s.g = 0.1;
    s.E_syn = -80.0;
    s.power = 1;
    s.S_init = 0.0;
    return s;
}

KineticSynapseSpec KineticSynapseSpec::nmda_kinetic() {
    KineticSynapseSpec s;
    s.name = "NMDA_kinetic";
    s.update_form = UpdateForm::BOLTZMANN_GATE;
    s.s_inf.v_half = -20.0;
    s.s_inf.k = 16.0;
    s.tau.form = TauParams::Form::CONSTANT;
    s.tau.params[0] = 80.0;  // ms — NMDA slow gating
    s.current_form = CurrentForm::MG_BLOCK;
    s.g = 0.1;
    s.E_syn = 0.0;
    s.power = 1;
    s.mg_conc = 1.0;
    s.mg_scale = 0.062;
    s.mg_denom = 3.57;
    s.S_init = 0.0;
    return s;
}

KineticSynapseSpec KineticSynapseSpec::gaba_b() {
    KineticSynapseSpec s;
    s.name = "GABA_B";
    s.update_form = UpdateForm::BOLTZMANN_GATE;
    s.s_inf.v_half = -60.0;
    s.s_inf.k = -8.0;   // negative k → inverted sigmoid (decreases with V)
    s.tau.form = TauParams::Form::CONSTANT;
    s.tau.params[0] = 200.0;  // ms — very slow GABA-B
    s.current_form = CurrentForm::LINEAR;
    s.g = 0.1;
    s.E_syn = -95.0;
    s.power = 4;         // GABA-B has cooperative gating
    s.S_init = 0.0;
    return s;
}

void NeuronModelSpec::validate() const {
    const int n_gates    = static_cast<int>(gates.size());
    const int n_channels = static_cast<int>(channels.size());

    // C_m must be positive
    if (C_m <= 0.0)
        throw std::invalid_argument(
            "NeuronModelSpec '" + name + "': C_m must be > 0, got " + std::to_string(C_m));
    if (C_m < 0.01 || C_m > 100.0)
        std::cerr << "[WARNING] NeuronModelSpec '" << name
                  << "': C_m=" << C_m << " is outside typical range [0.01, 100]\n";

    // Channel checks
    for (int ci = 0; ci < n_channels; ++ci) {
        const auto& ch = channels[ci];
        if (ch.g < 0.0)
            throw std::invalid_argument(
                "NeuronModelSpec '" + name + "': channel '" + ch.name
                + "' has negative conductance g=" + std::to_string(ch.g));
        for (const auto& gp : ch.gates) {
            if (gp.first < 0 || gp.first >= n_gates)
                throw std::invalid_argument(
                    "NeuronModelSpec '" + name + "': channel '" + ch.name
                    + "' references gate index " + std::to_string(gp.first)
                    + " but spec has only " + std::to_string(n_gates) + " gate(s)");
        }
    }

    // Gate checks
    for (int gi = 0; gi < n_gates; ++gi) {
        const auto& g = gates[gi];
        if (g.update_form == GateSpec::UpdateForm::DERIVED) {
            if (g.derived_source_gate < 0 || g.derived_source_gate >= n_gates)
                throw std::invalid_argument(
                    "NeuronModelSpec '" + name + "': gate '" + g.name
                    + "' derived_source_gate=" + std::to_string(g.derived_source_gate)
                    + " is out of range [0, " + std::to_string(n_gates) + ")");
        }
        if (g.tau.form == TauParams::Form::CONSTANT && g.tau.params[0] <= 0.0)
            std::cerr << "[WARNING] NeuronModelSpec '" << name << "': gate '" << g.name
                      << "' CONSTANT tau=" << g.tau.params[0] << " is <= 0\n";
    }

    // Calcium source channel checks
    for (int ch_idx : calcium.source_channels) {
        if (ch_idx < 0 || ch_idx >= n_channels)
            throw std::invalid_argument(
                "NeuronModelSpec '" + name + "': calcium.source_channels contains index "
                + std::to_string(ch_idx) + " but spec has only "
                + std::to_string(n_channels) + " channel(s)");
    }
}

} // namespace hodgkin_huxley
