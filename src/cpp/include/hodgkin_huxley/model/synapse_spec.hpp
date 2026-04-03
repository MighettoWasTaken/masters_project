#pragma once

#include "hodgkin_huxley/model/gate_spec.hpp"
#include <string>

namespace hodgkin_huxley {

// =============================================================================
// KineticSynapseSpec — composable kinetic synapse
// =============================================================================

struct KineticSynapseSpec {
    std::string name;

    enum class UpdateForm { ALPHA_BETA, TANH_GATE, BOLTZMANN_GATE };
    UpdateForm update_form = UpdateForm::TANH_GATE;

    // ALPHA_BETA: dS/dt = alpha(V)*(1-S) - beta(V)*S
    RateFuncParams alpha, beta;

    // TANH_GATE: dS/dt = tanh_amp*(1+tanh((V-tanh_vh)/tanh_k))*(1-S) - S/tau_decay
    double tanh_amp = 2.0, tanh_vh = 0.0, tanh_k = 4.0, tau_decay = 13.0;

    // BOLTZMANN_GATE: dS/dt = (S_inf(V) - S) / tau(V)
    BoltzmannParams s_inf;
    TauParams tau;

    enum class CurrentForm { LINEAR, MG_BLOCK };
    CurrentForm current_form = CurrentForm::LINEAR;

    double g = 0.1, E_syn = -80.0;
    int power = 1;

    // MG_BLOCK (NMDA)
    double mg_conc = 1.0, mg_scale = 0.062, mg_denom = 3.57;

    double S_init = 0.0;

    static KineticSynapseSpec gaba_kinetic();  // Kumaravelu 2016
    static KineticSynapseSpec nmda_kinetic();  // NMDA + Mg block
    static KineticSynapseSpec gaba_b();        // slow GABA-B
};

} // namespace hodgkin_huxley
