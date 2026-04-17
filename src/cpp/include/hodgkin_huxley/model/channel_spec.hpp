#pragma once

#include "hodgkin_huxley/model/gate_spec.hpp"
#include <string>
#include <vector>
#include <utility>

namespace hodgkin_huxley {

struct ChannelSpec {
    std::string name;
    double g = 0.0;
    double E_rev = 0.0;

    /// Index into NeuronModelSpec::intracellular for the Nernst reversal potential.
    /// -1 = none (use E_rev directly); >= 0 = use E_nernst_[nernst_substance_idx].
    /// Replaces the old bool use_calcium_nernst (which defaulted to substance 0 = calcium).
    int nernst_substance_idx = -1;

    std::vector<std::pair<int, int>> gates;  // (gate_index, power)
    bool   is_ahp = false;
    double ahp_k1 = 0.0;

    /// Index into NeuronModelSpec::intracellular for the AHP substance.
    /// Default 0 = calcium by convention.
    int ahp_substance_idx = 0;

    VmExpr gate_product_vm;  // if non-empty, replaces gates[] integer-power product
};

} // namespace hodgkin_huxley
