#pragma once

#include "hodgkin_huxley/model/gate_spec.hpp"
#include "hodgkin_huxley/model/channel_spec.hpp"
#include "hodgkin_huxley/izhikevich.hpp"
#include <string>
#include <vector>

namespace hodgkin_huxley {

struct NeuronModelSpec {
    std::string name;
    double C_m = 1.0;
    double V_init = -65.0;
    std::vector<GateSpec>             gates;
    std::vector<ChannelSpec>          channels;
    std::vector<IntracellularSpec>    intracellular;  // replaces CalciumSpec calcium

    // Izhikevich override — mutually exclusive with gates/channels.
    // Set by izhikevich() factory; RegionalNetwork::add_population dispatches
    // to the IzPool path when this is true.
    bool is_izhikevich = false;
    IzhikevichNeuron::Parameters iz_params;

    // Generic presets — replaces NetworkNeuronType enum
    static NeuronModelSpec hh_default();
    static NeuronModelSpec izhikevich(IzhikevichNeuron::Type type =
                                          IzhikevichNeuron::Type::REGULAR_SPIKING);
    static NeuronModelSpec izhikevich(const IzhikevichNeuron::Parameters& params);

    // Validation — throws std::invalid_argument on structural errors
    void validate() const;

    // --- Convenience helpers ---

    /// Returns true if any IntracellularSpec has nernst_enabled or is a driven/decay form.
    bool has_intracellular() const { return !intracellular.empty(); }

    /// Returns index of first IntracellularSpec with the given name, or -1 if not found.
    int intracellular_index(const std::string& nm) const {
        for (int i = 0; i < static_cast<int>(intracellular.size()); ++i) {
            if (intracellular[i].name == nm) return i;
        }
        return -1;
    }
};

} // namespace hodgkin_huxley
