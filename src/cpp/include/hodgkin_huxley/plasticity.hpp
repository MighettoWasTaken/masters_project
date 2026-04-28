#pragma once
#include <cstddef>
#include <cstdint>

namespace hodgkin_huxley {

enum class PlasticityType : uint8_t { NONE = 0, STDP = 1, STP = 2 };

struct STDPParams {
    double A_plus    = 0.005;
    double A_minus   = 0.006;
    double tau_plus  = 20.0;
    double tau_minus = 20.0;
    double w_min     = 0.0;
    double w_max     = 1.0;
    // Neuromodulator gating (-1 = disabled)
    int    modulator_pop_start     = -1;  // global neuron index of pop start
    int    modulator_substance_idx = -1;
    double modulator_scale         = 1.0;
};

struct STPParams {
    double U     = 0.5;
    double tau_u = 1000.0;
    double tau_x = 100.0;
};

struct PlasticitySpec {
    PlasticityType type = PlasticityType::NONE;
    STDPParams stdp;
    STPParams  stp;
};

} // namespace hodgkin_huxley
