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
    bool use_calcium_nernst = false;
    std::vector<std::pair<int, int>> gates;  // (gate_index, power)
    bool is_ahp = false;
    double ahp_k1 = 0.0;
};

} // namespace hodgkin_huxley
