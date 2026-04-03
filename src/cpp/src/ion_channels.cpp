#include "hodgkin_huxley/ion_channels.hpp"
#include <iostream>
#include <stdexcept>

namespace hodgkin_huxley {

// Circuit-specific preset factories (thalamic, stn, gpe, gpi, striatum) were
// removed. Python equivalents live in tests/python/neuron_specs.py and
// benchmarks/ctxbgth_model.py.

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

// =============================================================================
// NeuronModelSpec::hh_default()
// Classic Hodgkin-Huxley squid giant axon (HH 1952).
// Gates: m (Na activation), h (Na inactivation), n (K activation) — all ALPHA_BETA.
// Channels: Na (g=120, E=50, m^3*h), K (g=36, E=-77, n^4), Leak (g=0.3, E=-54.3).
// =============================================================================

NeuronModelSpec NeuronModelSpec::hh_default() {
    NeuronModelSpec spec;
    spec.name = "HH";
    spec.C_m = 1.0;
    spec.V_init = -65.0;

    // Gate 0: m (Na activation, ALPHA_BETA)
    // alpha_m(V) = 0.1*(V+40) / (1 - exp(-(V+40)/10))
    // beta_m(V)  = 4 * exp(-(V+65)/18)
    {
        GateSpec g;
        g.name = "m";
        g.update_form = GateSpec::UpdateForm::ALPHA_BETA;
        g.initial_value = 0.05;
        g.alpha = {RateFuncParams::Form::LINEAR_OVER_EXPM1, 0.1, 40.0, 10.0};
        g.beta  = {RateFuncParams::Form::EXP_DECAY,         4.0, 65.0, -18.0};
        spec.gates.push_back(g);
    }

    // Gate 1: h (Na inactivation, ALPHA_BETA)
    // alpha_h(V) = 0.07 * exp(-(V+65)/20)
    // beta_h(V)  = 1 / (1 + exp(-(V+35)/10))
    {
        GateSpec g;
        g.name = "h";
        g.update_form = GateSpec::UpdateForm::ALPHA_BETA;
        g.initial_value = 0.6;
        g.alpha = {RateFuncParams::Form::EXP_DECAY, 0.07, 65.0, -20.0};
        g.beta  = {RateFuncParams::Form::SIGMOID,   1.0,  35.0, -10.0};
        spec.gates.push_back(g);
    }

    // Gate 2: n (K activation, ALPHA_BETA)
    // alpha_n(V) = 0.01*(V+55) / (1 - exp(-(V+55)/10))
    // beta_n(V)  = 0.125 * exp(-(V+65)/80)
    {
        GateSpec g;
        g.name = "n";
        g.update_form = GateSpec::UpdateForm::ALPHA_BETA;
        g.initial_value = 0.32;
        g.alpha = {RateFuncParams::Form::LINEAR_OVER_EXPM1, 0.01,  55.0,  10.0};
        g.beta  = {RateFuncParams::Form::EXP_DECAY,         0.125, 65.0, -80.0};
        spec.gates.push_back(g);
    }

    // Channel 0: Na — g=120, E=50, gates m^3*h
    {
        ChannelSpec c;
        c.name = "Na";
        c.g = 120.0;
        c.E_rev = 50.0;
        c.gates = {{0, 3}, {1, 1}};  // m^3 * h
        spec.channels.push_back(c);
    }

    // Channel 1: K — g=36, E=-77, gates n^4
    {
        ChannelSpec c;
        c.name = "K";
        c.g = 36.0;
        c.E_rev = -77.0;
        c.gates = {{2, 4}};  // n^4
        spec.channels.push_back(c);
    }

    // Channel 2: Leak — g=0.3, E=-54.3, no gates (always on)
    {
        ChannelSpec c;
        c.name = "Leak";
        c.g = 0.3;
        c.E_rev = -54.3;
        // empty gates vector → always-on conductance
        spec.channels.push_back(c);
    }

    return spec;
}

// =============================================================================
// NeuronModelSpec::izhikevich()
// Returns a spec with is_izhikevich=true; RegionalNetwork::add_population
// dispatches to the IzPool path when it detects this flag.
// =============================================================================

NeuronModelSpec NeuronModelSpec::izhikevich(IzhikevichNeuron::Type type) {
    NeuronModelSpec spec;
    spec.name = "Izhikevich";
    spec.is_izhikevich = true;
    spec.iz_params = IzhikevichNeuron::get_preset(type);
    spec.V_init = spec.iz_params.c;
    return spec;
}

NeuronModelSpec NeuronModelSpec::izhikevich(const IzhikevichNeuron::Parameters& params) {
    NeuronModelSpec spec;
    spec.name = "Izhikevich";
    spec.is_izhikevich = true;
    spec.iz_params = params;
    spec.V_init = params.c;
    return spec;
}

} // namespace hodgkin_huxley
