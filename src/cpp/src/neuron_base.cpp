#include "hodgkin_huxley/neuron_base.hpp"
#include <stdexcept>

namespace hodgkin_huxley {

std::vector<double> NeuronBase::simulate(double duration, double dt, double I_ext) {
    size_t num_steps = static_cast<size_t>(duration / dt);
    std::vector<double> trace;
    trace.reserve(num_steps);

    for (size_t i = 0; i < num_steps; ++i) {
        trace.push_back(membrane_potential());
        step(dt, I_ext);
    }

    return trace;
}

std::vector<double> NeuronBase::simulate(double duration, double dt,
                                         const std::vector<double>& I_ext) {
    size_t num_steps = static_cast<size_t>(duration / dt);
    if (I_ext.size() < num_steps) {
        throw std::invalid_argument("I_ext vector too short for simulation duration");
    }

    std::vector<double> trace;
    trace.reserve(num_steps);

    for (size_t i = 0; i < num_steps; ++i) {
        trace.push_back(membrane_potential());
        step(dt, I_ext[i]);
    }

    return trace;
}

} // namespace hodgkin_huxley
