#include "hodgkin_huxley/regional_network.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace hodgkin_huxley {

// =============================================================================
// SynapseSpec factories
// =============================================================================

SynapseSpec SynapseSpec::ampa() {
    return {Type::DOUBLE_EXPONENTIAL, 0.0, 0.0, 0.5, 2.5};
}

SynapseSpec SynapseSpec::nmda() {
    return {Type::DOUBLE_EXPONENTIAL, 0.0, 0.0, 2.0, 67.0};
}

SynapseSpec SynapseSpec::gaba_a() {
    return {Type::DOUBLE_EXPONENTIAL, -80.0, 0.0, 0.4, 7.7};
}

SynapseSpec SynapseSpec::exponential(double E_syn, double tau) {
    return {Type::EXPONENTIAL, E_syn, tau, 0.0, 0.0};
}

SynapseSpec SynapseSpec::alpha(double E_syn, double tau) {
    return {Type::ALPHA, E_syn, tau, 0.0, 0.0};
}

SynapseSpec SynapseSpec::double_exponential(double E_syn, double tau_rise, double tau_decay) {
    return {Type::DOUBLE_EXPONENTIAL, E_syn, 0.0, tau_rise, tau_decay};
}

// =============================================================================
// WeightDistribution
// =============================================================================

WeightDistribution WeightDistribution::constant(double value) {
    return {WeightDistType::CONSTANT, value, 0.0};
}

WeightDistribution WeightDistribution::uniform(double min, double max) {
    return {WeightDistType::UNIFORM, min, max};
}

WeightDistribution WeightDistribution::normal(double mean, double std) {
    return {WeightDistType::NORMAL, mean, std};
}

double WeightDistribution::sample(std::mt19937& rng) const {
    switch (type) {
        case WeightDistType::CONSTANT:
            return param1;
        case WeightDistType::UNIFORM: {
            std::uniform_real_distribution<double> dist(param1, param2);
            return dist(rng);
        }
        case WeightDistType::NORMAL: {
            std::normal_distribution<double> dist(param1, param2);
            return dist(rng);
        }
    }
    return param1;  // unreachable
}

// =============================================================================
// RegionalNetwork — population management
// =============================================================================

void RegionalNetwork::add_population(const std::string& name, size_t count,
                                     Network::NeuronType type) {
    if (pop_index_.count(name)) {
        throw std::runtime_error("Population '" + name + "' already exists");
    }
    size_t start = net_.num_neurons();
    for (size_t i = 0; i < count; ++i) {
        net_.add_neuron(type);
    }
    pop_index_[name] = populations_.size();
    populations_.push_back({name, start, count});
}

void RegionalNetwork::add_population(const std::string& name, size_t count,
                                     const HHNeuron::Parameters& params) {
    if (pop_index_.count(name)) {
        throw std::runtime_error("Population '" + name + "' already exists");
    }
    size_t start = net_.num_neurons();
    for (size_t i = 0; i < count; ++i) {
        net_.add_neuron(params);
    }
    pop_index_[name] = populations_.size();
    populations_.push_back({name, start, count});
}

void RegionalNetwork::add_population(const std::string& name, size_t count,
                                     const IzhikevichNeuron::Parameters& params) {
    if (pop_index_.count(name)) {
        throw std::runtime_error("Population '" + name + "' already exists");
    }
    size_t start = net_.num_neurons();
    for (size_t i = 0; i < count; ++i) {
        net_.add_neuron(params);
    }
    pop_index_[name] = populations_.size();
    populations_.push_back({name, start, count});
}

void RegionalNetwork::add_population(const std::string& name, size_t count,
                                     const NeuronModelSpec& spec) {
    if (pop_index_.count(name)) {
        throw std::runtime_error("Population '" + name + "' already exists");
    }
    size_t start = net_.num_neurons();
    for (size_t i = 0; i < count; ++i) {
        net_.add_neuron(spec);
    }
    pop_index_[name] = populations_.size();
    populations_.push_back({name, start, count});
}

// =============================================================================
// Population queries
// =============================================================================

void RegionalNetwork::validate_population(const std::string& name) const {
    if (!pop_index_.count(name)) {
        throw std::runtime_error("Unknown population: '" + name + "'");
    }
}

const Population& RegionalNetwork::population(const std::string& name) const {
    validate_population(name);
    return populations_[pop_index_.at(name)];
}

std::vector<std::string> RegionalNetwork::population_names() const {
    std::vector<std::string> names;
    names.reserve(populations_.size());
    for (const auto& pop : populations_) {
        names.push_back(pop.name);
    }
    return names;
}

size_t RegionalNetwork::population_size(const std::string& name) const {
    return population(name).count;
}

size_t RegionalNetwork::population_start(const std::string& name) const {
    return population(name).start_idx;
}

size_t RegionalNetwork::num_populations() const {
    return populations_.size();
}

// =============================================================================
// Delegation
// =============================================================================

size_t RegionalNetwork::num_neurons() const { return net_.num_neurons(); }
size_t RegionalNetwork::num_synapses() const { return net_.num_synapses(); }
void RegionalNetwork::set_fast_math(bool enabled) { net_.set_fast_math(enabled); }
bool RegionalNetwork::fast_math() const { return net_.fast_math(); }
void RegionalNetwork::reset() { net_.reset(); }
Network& RegionalNetwork::network() { return net_; }
const Network& RegionalNetwork::network() const { return net_; }

// =============================================================================
// Synapse creation from spec
// =============================================================================

void RegionalNetwork::add_synapse_from_spec(size_t pre, size_t post,
                                            double weight, const SynapseSpec& spec,
                                            double delay) {
    switch (spec.type) {
        case SynapseSpec::Type::EXPONENTIAL:
            net_.add_synapse(pre, post, weight, spec.E_syn, spec.tau, delay);
            break;
        case SynapseSpec::Type::ALPHA:
            net_.add_alpha_synapse(pre, post, weight, spec.E_syn, spec.tau, delay);
            break;
        case SynapseSpec::Type::DOUBLE_EXPONENTIAL:
            net_.add_double_exp_synapse(pre, post, weight, spec.E_syn,
                                        spec.tau_rise, spec.tau_decay, delay);
            break;
    }
}

// =============================================================================
// Single connection (for Python custom patterns)
// =============================================================================

void RegionalNetwork::add_connection(const std::string& src, size_t src_local,
                                     const std::string& dst, size_t dst_local,
                                     double weight, const SynapseSpec& synapse,
                                     double delay) {
    const auto& src_pop = population(src);
    const auto& dst_pop = population(dst);
    if (src_local >= src_pop.count) {
        throw std::out_of_range("Source local index " + std::to_string(src_local) +
                                " out of range for population '" + src + "' (size " +
                                std::to_string(src_pop.count) + ")");
    }
    if (dst_local >= dst_pop.count) {
        throw std::out_of_range("Destination local index " + std::to_string(dst_local) +
                                " out of range for population '" + dst + "' (size " +
                                std::to_string(dst_pop.count) + ")");
    }
    size_t pre = src_pop.start_idx + src_local;
    size_t post = dst_pop.start_idx + dst_local;
    add_synapse_from_spec(pre, post, weight, synapse, delay);
}

// =============================================================================
// Kinetic connection (single, between two populations)
// =============================================================================

void RegionalNetwork::add_kinetic_connection(const std::string& src, size_t i,
                                              const std::string& dst, size_t j,
                                              double weight, const KineticSynapseSpec& spec,
                                              double delay) {
    const auto& src_pop = population(src);
    const auto& dst_pop = population(dst);
    if (i >= src_pop.count) {
        throw std::out_of_range("Source local index " + std::to_string(i) +
                                " out of range for population '" + src + "' (size " +
                                std::to_string(src_pop.count) + ")");
    }
    if (j >= dst_pop.count) {
        throw std::out_of_range("Destination local index " + std::to_string(j) +
                                " out of range for population '" + dst + "' (size " +
                                std::to_string(dst_pop.count) + ")");
    }
    size_t pre  = src_pop.start_idx + i;
    size_t post = dst_pop.start_idx + j;
    net_.add_kinetic_synapse(pre, post, weight, spec, delay);
}

// =============================================================================
// Bulk connectivity
// =============================================================================

void RegionalNetwork::connect(const std::string& src, const std::string& dst,
                              ConnectivityPattern pattern,
                              const SynapseSpec& synapse,
                              const WeightDistribution& weight,
                              double delay, int shift, double probability,
                              bool allow_self, unsigned int seed) {
    const auto& src_pop = population(src);
    const auto& dst_pop = population(dst);

    std::mt19937 rng;
    if (seed != 0) {
        rng.seed(seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    generate_connections(src_pop, dst_pop, pattern, synapse, weight,
                         delay, shift, probability, allow_self, rng);
}

void RegionalNetwork::generate_connections(
    const Population& src, const Population& dst,
    ConnectivityPattern pattern,
    const SynapseSpec& synapse,
    const WeightDistribution& weight,
    double delay, int shift, double probability,
    bool allow_self, std::mt19937& rng)
{
    bool same_pop = (src.start_idx == dst.start_idx && src.count == dst.count);

    switch (pattern) {
        case ConnectivityPattern::ALL_TO_ALL: {
            for (size_t i = 0; i < src.count; ++i) {
                for (size_t j = 0; j < dst.count; ++j) {
                    size_t pre = src.start_idx + i;
                    size_t post = dst.start_idx + j;
                    if (!allow_self && same_pop && i == j) continue;
                    add_synapse_from_spec(pre, post, weight.sample(rng), synapse, delay);
                }
            }
            break;
        }
        case ConnectivityPattern::ONE_TO_ONE: {
            if (src.count != dst.count) {
                throw std::runtime_error(
                    "ONE_TO_ONE requires equal population sizes ('" +
                    src.name + "': " + std::to_string(src.count) + ", '" +
                    dst.name + "': " + std::to_string(dst.count) + ")");
            }
            for (size_t k = 0; k < src.count; ++k) {
                add_synapse_from_spec(src.start_idx + k, dst.start_idx + k,
                                      weight.sample(rng), synapse, delay);
            }
            break;
        }
        case ConnectivityPattern::SHIFTED: {
            if (src.count != dst.count) {
                throw std::runtime_error(
                    "SHIFTED requires equal population sizes ('" +
                    src.name + "': " + std::to_string(src.count) + ", '" +
                    dst.name + "': " + std::to_string(dst.count) + ")");
            }
            int n = static_cast<int>(dst.count);
            for (size_t k = 0; k < src.count; ++k) {
                int dst_k = (static_cast<int>(k) + ((shift % n) + n)) % n;
                add_synapse_from_spec(src.start_idx + k,
                                      dst.start_idx + static_cast<size_t>(dst_k),
                                      weight.sample(rng), synapse, delay);
            }
            break;
        }
        case ConnectivityPattern::RANDOM_SPARSE: {
            std::uniform_real_distribution<double> coin(0.0, 1.0);
            for (size_t i = 0; i < src.count; ++i) {
                for (size_t j = 0; j < dst.count; ++j) {
                    if (!allow_self && same_pop && i == j) continue;
                    if (coin(rng) < probability) {
                        add_synapse_from_spec(src.start_idx + i, dst.start_idx + j,
                                              weight.sample(rng), synapse, delay);
                    }
                }
            }
            break;
        }
        case ConnectivityPattern::RANDOM_PERMUTATION: {
            if (src.count != dst.count) {
                throw std::runtime_error(
                    "RANDOM_PERMUTATION requires equal population sizes ('" +
                    src.name + "': " + std::to_string(src.count) + ", '" +
                    dst.name + "': " + std::to_string(dst.count) + ")");
            }
            std::vector<size_t> perm(src.count);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            for (size_t k = 0; k < src.count; ++k) {
                add_synapse_from_spec(src.start_idx + k,
                                      dst.start_idx + perm[k],
                                      weight.sample(rng), synapse, delay);
            }
            break;
        }
    }
}

// =============================================================================
// Randomize membrane potentials
// =============================================================================

void RegionalNetwork::randomize_membrane_potentials(const std::string& name,
                                                     double mean, double std_dev,
                                                     unsigned int seed) {
    const auto& pop = population(name);
    std::mt19937 rng;
    if (seed != 0) {
        rng.seed(seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    std::normal_distribution<double> dist(mean, std_dev);
    for (size_t i = pop.start_idx; i < pop.end_idx(); ++i) {
        net_.neuron(i).set_membrane_potential(dist(rng));
    }
}

// =============================================================================
// Simulate
// =============================================================================

std::vector<std::vector<double>> RegionalNetwork::simulate(
    double duration, double dt,
    const std::vector<std::vector<double>>& I_ext)
{
    return net_.simulate(duration, dt, I_ext);
}

} // namespace hodgkin_huxley
