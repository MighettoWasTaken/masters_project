#include "hodgkin_huxley/synapse_base.hpp"
#include "hodgkin_huxley/network.hpp"

namespace hodgkin_huxley {

double SynapseBase::conductance() const {
    return net_->syn_arrays().g[idx_];
}

double SynapseBase::reversal_potential() const {
    return net_->syn_arrays().E_syn[idx_];
}

double SynapseBase::weight() const {
    return net_->syn_arrays().weight[idx_];
}

size_t SynapseBase::pre_idx() const {
    return net_->syn_arrays().pre[idx_];
}

size_t SynapseBase::post_idx() const {
    return net_->syn_arrays().post[idx_];
}

double SynapseBase::delay() const {
    return net_->syn_arrays().delay[idx_];
}

std::string SynapseBase::type_name() const {
    using UF = SynapseSpec::UpdateForm;
    switch (spec().update_form) {
        case UF::EXP_DECAY:    return "exponential";
        case UF::ALPHA_FUNC:   return "alpha_function";
        case UF::DOUBLE_EXP:   return "double_exponential";
        case UF::TANH_GATE:    return "tanh_gate";
        case UF::BOLTZMANN_GATE: return "boltzmann_gate";
        case UF::ALPHA_BETA:   return "alpha_beta";
        case UF::CUSTOM_EXPR:  return "custom_expr";
        default:               return "unknown";
    }
}

const SynapseSpec& SynapseBase::spec() const {
    return net_->synapse_spec(net_->syn_arrays().spec_idx[idx_]);
}

} // namespace hodgkin_huxley
