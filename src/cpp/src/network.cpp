#include "hodgkin_huxley/network.hpp"
#include <cmath>
#include <stdexcept>

namespace hodgkin_huxley {

Network::Network(size_t num_neurons) {
    neurons_.reserve(num_neurons);
    for (size_t i = 0; i < num_neurons; ++i) {
        neurons_.emplace_back();
    }
}

size_t Network::add_neuron() {
    neurons_.emplace_back();
    return neurons_.size() - 1;
}

size_t Network::add_neuron(const HHNeuron::Parameters& params) {
    neurons_.emplace_back(params);
    return neurons_.size() - 1;
}

void Network::add_synapse(size_t pre_idx, size_t post_idx, double weight,
                          double E_syn, double tau) {
    if (pre_idx >= neurons_.size() || post_idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }

    Synapse syn;
    syn.pre_idx = pre_idx;
    syn.post_idx = post_idx;
    syn.weight = weight;
    syn.E_syn = E_syn;
    syn.tau = tau;
    syn.g = 0.0;

    synapses_.push_back(syn);
}

std::vector<double> Network::get_potentials() const {
    std::vector<double> potentials;
    potentials.reserve(neurons_.size());
    for (const auto& neuron : neurons_) {
        potentials.push_back(neuron.membrane_potential());
    }
    return potentials;
}

void Network::reset() {
    for (auto& neuron : neurons_) {
        neuron.reset();
    }
    for (auto& synapse : synapses_) {
        synapse.g = 0.0;
    }
}

std::vector<double> Network::compute_synaptic_currents() const {
    std::vector<double> currents(neurons_.size(), 0.0);

    for (const auto& syn : synapses_) {
        double V_post = neurons_[syn.post_idx].membrane_potential();
        double I_syn = syn.g * (V_post - syn.E_syn);
        currents[syn.post_idx] -= I_syn;  // Negative because it's an input current
    }

    return currents;
}

void Network::update_synapses(double dt) {
    // Simple exponential synapse model
    // When pre-synaptic neuron spikes, g increases by weight
    // Then decays exponentially with time constant tau

    const double spike_threshold = 0.0;  // mV

    for (auto& syn : synapses_) {
        double V_pre = neurons_[syn.pre_idx].membrane_potential();

        // Check for spike (simple threshold crossing)
        // A more sophisticated implementation would track the derivative
        if (V_pre > spike_threshold) {
            syn.g += syn.weight;
        }

        // Exponential decay
        syn.g *= std::exp(-dt / syn.tau);
    }
}

void Network::step(double dt, const std::vector<double>& I_ext) {
    if (I_ext.size() != neurons_.size()) {
        throw std::invalid_argument("I_ext size must match number of neurons");
    }

    // Compute synaptic currents
    auto I_syn = compute_synaptic_currents();

    // Step each neuron
    for (size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i].step(dt, I_ext[i] + I_syn[i]);
    }

    // Update synapse states
    update_synapses(dt);
}

std::vector<std::vector<double>> Network::simulate(
    double duration,
    double dt,
    const std::vector<std::vector<double>>& I_ext
) {
    size_t num_steps = static_cast<size_t>(duration / dt);
    size_t n_neurons = neurons_.size();

    // Validate input
    if (I_ext.size() != n_neurons) {
        throw std::invalid_argument("I_ext outer size must match number of neurons");
    }
    for (const auto& curr : I_ext) {
        if (curr.size() < num_steps) {
            throw std::invalid_argument("I_ext vectors too short for simulation duration");
        }
    }

    // Initialize output
    std::vector<std::vector<double>> traces(n_neurons);
    for (auto& trace : traces) {
        trace.reserve(num_steps);
    }

    // Run simulation
    std::vector<double> I_step(n_neurons);
    for (size_t t = 0; t < num_steps; ++t) {
        // Record voltages
        for (size_t i = 0; i < n_neurons; ++i) {
            traces[i].push_back(neurons_[i].membrane_potential());
            I_step[i] = I_ext[i][t];
        }

        // Step network
        step(dt, I_step);
    }

    return traces;
}

} // namespace hodgkin_huxley
