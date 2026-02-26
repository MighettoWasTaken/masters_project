#include "hodgkin_huxley/ion_channels.hpp"

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
        g.tau.params[5] = -0.6;   // v2 offset
        g.tau.params[6] = -7.4;   // s2
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
// Based on Terman et al. (2002) / Rubin & Terman (2004)
// Channels: Na, K, T-type Ca, Ca-dep K (AHP), L-type Ca, Leak, A-type K(?)
// Calcium dynamics with Nernst equation
// =============================================================================

NeuronModelSpec NeuronModelSpec::stn() {
    NeuronModelSpec spec;
    spec.name = "STN";
    spec.C_m = 1.0;
    spec.V_init = -62.0;

    // Gate 0: h_Na
    {
        GateSpec g;
        g.name = "h_Na";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.6;
        g.inf = {-39.0, -3.1};
        g.tau.form = TauParams::Form::SCALED_EXP;
        g.tau.params[0] = 1.0;
        g.tau.params[1] = -39.0;
        g.tau.params[2] = -3.1;
        spec.gates.push_back(g);
    }

    // Gate 1: n_K
    {
        GateSpec g;
        g.name = "n_K";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.1;
        g.inf = {-32.0, 8.0};
        g.tau.form = TauParams::Form::SCALED_EXP;
        g.tau.params[0] = 1.0;
        g.tau.params[1] = -32.0;
        g.tau.params[2] = 8.0;
        spec.gates.push_back(g);
    }

    // Gate 2: m_Na (instant)
    {
        GateSpec g;
        g.name = "m_Na";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.05;
        g.inf = {-30.0, 15.0};
        spec.gates.push_back(g);
    }

    // Gate 3: a_T (T-current activation, dynamic)
    {
        GateSpec g;
        g.name = "a_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.0;
        g.inf = {-63.0, 7.8};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 1.0;    // base
        g.tau.params[1] = 5.0;    // amp
        g.tau.params[2] = 0.0;    // v1 (numerator over exp sum)
        g.tau.params[3] = 1e6;    // s1 (very large -> exp term ~ 1)
        g.tau.params[4] = 0.0;
        g.tau.params[5] = -68.0;
        g.tau.params[6] = -2.2;
        spec.gates.push_back(g);
    }

    // Gate 4: b_T (T-current inactivation, dynamic)
    {
        GateSpec g;
        g.name = "b_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.4;
        g.inf = {-0.014, -0.059};  // r_inf uses special calcium-based form
        g.dependency = GateSpec::Dependency::CALCIUM;
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 0.24;
        spec.gates.push_back(g);
    }

    // Gate 5: r (AHP gate, calcium-dependent)
    {
        GateSpec g;
        g.name = "r";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.17;
        g.inf = {-0.017, -0.08};  // Boltzmann with Ca as input
        g.dependency = GateSpec::Dependency::CALCIUM;
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 2.0;
        spec.gates.push_back(g);
    }

    // Gate 6: s_CaL (L-type Ca activation, instant)
    {
        GateSpec g;
        g.name = "s_CaL";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.0;
        g.inf = {-39.0, 8.0};
        spec.gates.push_back(g);
    }

    // Gate 7: h_T_stn (T inact, dynamic)
    {
        GateSpec g;
        g.name = "h_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.35;
        g.inf = {-80.0, -6.0};
        g.tau.form = TauParams::Form::DOUBLE_EXP_SUM;
        g.tau.params[0] = 20.0;
        g.tau.params[1] = 100.0;
        g.tau.params[2] = 39.7;
        g.tau.params[3] = 9.32;
        g.tau.params[4] = 0.0;
        g.tau.params[5] = -0.6;
        g.tau.params[6] = -7.4;
        spec.gates.push_back(g);
    }

    // Gate 8: m_T (T activation, instant)
    {
        GateSpec g;
        g.name = "m_T_stn";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.0;
        g.inf = {-57.0, 6.2};
        spec.gates.push_back(g);
    }

    // Gate 9: a_stn (A-type K activation, instant)
    {
        GateSpec g;
        g.name = "a_stn";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.0;
        g.inf = {-45.0, 14.7};
        spec.gates.push_back(g);
    }

    // Gate 10: b_stn (A-type K inactivation, dynamic)
    {
        GateSpec g;
        g.name = "b_stn";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.6;
        g.inf = {-90.0, -7.5};
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 20.0;
        spec.gates.push_back(g);
    }

    // Channel 0: Na
    {
        ChannelSpec c;
        c.name = "Na";
        c.g = 37.5;
        c.E_rev = 55.0;
        c.gates = {{2, 3}, {0, 1}};  // m^3 * h
        spec.channels.push_back(c);
    }

    // Channel 1: K
    {
        ChannelSpec c;
        c.name = "K";
        c.g = 45.0;
        c.E_rev = -80.0;
        c.gates = {{1, 4}};  // n^4
        spec.channels.push_back(c);
    }

    // Channel 2: T-type Ca
    {
        ChannelSpec c;
        c.name = "T";
        c.g = 0.5;
        c.E_rev = 140.0;
        c.gates = {{8, 2}, {7, 1}};  // m_T^2 * h_T
        spec.channels.push_back(c);
    }

    // Channel 3: L-type Ca
    {
        ChannelSpec c;
        c.name = "CaL";
        c.g = 15.0;
        c.E_rev = 140.0;
        c.gates = {{6, 1}};  // s_CaL
        spec.channels.push_back(c);
    }

    // Channel 4: AHP (Ca-dependent K)
    {
        ChannelSpec c;
        c.name = "AHP";
        c.g = 9.0;
        c.E_rev = -80.0;
        c.is_ahp = true;
        c.ahp_k1 = 15.0;  // I_AHP = g * Ca/(Ca+k1) * (V-E)
        spec.channels.push_back(c);
    }

    // Channel 5: A-type K
    {
        ChannelSpec c;
        c.name = "A";
        c.g = 5.0;
        c.E_rev = -80.0;
        c.gates = {{9, 2}, {10, 1}};  // a^2 * b
        spec.channels.push_back(c);
    }

    // Channel 6: Leak
    {
        ChannelSpec c;
        c.name = "Leak";
        c.g = 2.25;
        c.E_rev = -60.0;
        spec.channels.push_back(c);
    }

    // Calcium dynamics with Nernst
    spec.calcium.enabled = true;
    spec.calcium.use_nernst = true;
    spec.calcium.epsilon = 5e-6;  // Ca influx scaling
    spec.calcium.K_Ca = 15.0;     // Ca decay rate
    spec.calcium.Ca_init = 0.1;
    spec.calcium.Ca_o = 2000.0;
    spec.calcium.source_channels = {2, 3};  // T and CaL channels

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

    // Gate 0: h_Na
    {
        GateSpec g;
        g.name = "h_Na";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.scale = 0.05;  // phi_h = 0.05
        g.initial_value = 0.6;
        g.inf = {-58.3, -6.7};
        g.tau.form = TauParams::Form::SCALED_EXP;
        g.tau.params[0] = 0.27;
        g.tau.params[1] = -40.0;
        g.tau.params[2] = -12.0;
        spec.gates.push_back(g);
    }

    // Gate 1: n_K
    {
        GateSpec g;
        g.name = "n_K";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.scale = 0.1;  // phi_n = 0.1
        g.initial_value = 0.1;
        g.inf = {-32.0, 8.0};
        g.tau.form = TauParams::Form::SCALED_EXP;
        g.tau.params[0] = 0.27;
        g.tau.params[1] = -40.0;
        g.tau.params[2] = 8.0;
        spec.gates.push_back(g);
    }

    // Gate 2: r (Ca-dependent AHP gate)
    {
        GateSpec g;
        g.name = "r";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.initial_value = 0.1;
        g.inf = {-0.017, -0.08};
        g.dependency = GateSpec::Dependency::CALCIUM;
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 2.0;
        spec.gates.push_back(g);
    }

    // Gate 3: m_Na (instant)
    {
        GateSpec g;
        g.name = "m_Na";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.05;
        g.inf = {-37.0, 10.0};
        spec.gates.push_back(g);
    }

    // Gate 4: a_T (T-current activation, instant)
    {
        GateSpec g;
        g.name = "a_T";
        g.update_form = GateSpec::UpdateForm::INSTANT;
        g.initial_value = 0.0;
        g.inf = {-57.0, 2.0};
        spec.gates.push_back(g);
    }

    // Gate 5: s_T (T-current slow inactivation, dynamic)
    {
        GateSpec g;
        g.name = "s_T";
        g.update_form = GateSpec::UpdateForm::INF_TAU;
        g.scale = 1.0;
        g.initial_value = 0.3;
        g.inf = {-35.0, -2.0};
        g.tau.form = TauParams::Form::CONSTANT;
        g.tau.params[0] = 400.0;
        spec.gates.push_back(g);
    }

    // Channel 0: Na
    {
        ChannelSpec c;
        c.name = "Na";
        c.g = 120.0;
        c.E_rev = 55.0;
        c.gates = {{3, 3}, {0, 1}};  // m^3 * h
        spec.channels.push_back(c);
    }

    // Channel 1: K
    {
        ChannelSpec c;
        c.name = "K";
        c.g = 30.0;
        c.E_rev = -80.0;
        c.gates = {{1, 4}};  // n^4
        spec.channels.push_back(c);
    }

    // Channel 2: T-type Ca
    {
        ChannelSpec c;
        c.name = "T";
        c.g = 0.5;
        c.E_rev = 120.0;
        c.gates = {{4, 3}, {5, 1}};  // a_T^3 * s_T
        spec.channels.push_back(c);
    }

    // Channel 3: AHP (Ca-dep K)
    {
        ChannelSpec c;
        c.name = "AHP";
        c.g = 30.0;
        c.E_rev = -80.0;
        c.is_ahp = true;
        c.ahp_k1 = 15.0;
        spec.channels.push_back(c);
    }

    // Channel 4: Leak
    {
        ChannelSpec c;
        c.name = "Leak";
        c.g = 0.1;
        c.E_rev = -55.0;
        spec.channels.push_back(c);
    }

    // Calcium
    spec.calcium.enabled = true;
    spec.calcium.use_nernst = false;
    spec.calcium.epsilon = 1e-4;
    spec.calcium.K_Ca = 15.0;
    spec.calcium.Ca_init = 0.1;
    spec.calcium.source_channels = {2};  // T-type

    return spec;
}

// =============================================================================
// GPi - same structure as GPe, different conductance params
// =============================================================================

NeuronModelSpec NeuronModelSpec::gpi() {
    NeuronModelSpec spec = gpe();
    spec.name = "GPi";

    // GPi has different maximal conductances
    spec.channels[0].g = 120.0;  // Na
    spec.channels[1].g = 30.0;   // K
    spec.channels[2].g = 0.5;    // T
    spec.channels[3].g = 20.0;   // AHP (lower than GPe)
    spec.channels[4].g = 0.1;    // Leak

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

} // namespace hodgkin_huxley
