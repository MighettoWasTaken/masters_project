#pragma once

#include "network.hpp"
#include <vector>
#include <string>
#include <map>
#include <random>
#include <stdexcept>

namespace hodgkin_huxley {

// --- Population bookkeeping ---
struct Population {
    std::string name;
    size_t start_idx;
    size_t count;
    size_t end_idx() const { return start_idx + count; }
};

// --- Connectivity presets (C++ fast path) ---
enum class ConnectivityPattern {
    ALL_TO_ALL,
    ONE_TO_ONE,
    SHIFTED,
    RANDOM_SPARSE,
    RANDOM_PERMUTATION
};

// --- Synapse configuration ---
struct SynapseSpec {
    enum class Type { EXPONENTIAL, ALPHA, DOUBLE_EXPONENTIAL };
    Type type;
    double E_syn;
    double tau;        // exponential, alpha
    double tau_rise;   // double_exp only
    double tau_decay;  // double_exp only

    // Preset receptor factories
    static SynapseSpec ampa();
    static SynapseSpec nmda();
    static SynapseSpec gaba_a();

    // Custom factories
    static SynapseSpec exponential(double E_syn, double tau);
    static SynapseSpec alpha(double E_syn, double tau);
    static SynapseSpec double_exponential(double E_syn, double tau_rise, double tau_decay);
};

// --- Weight distribution ---
enum class WeightDistType { CONSTANT, UNIFORM, NORMAL };

struct WeightDistribution {
    WeightDistType type = WeightDistType::CONSTANT;
    double param1 = 0.0;  // constant: value; uniform: min; normal: mean
    double param2 = 0.0;  // constant: unused; uniform: max; normal: std

    static WeightDistribution constant(double value);
    static WeightDistribution uniform(double min, double max);
    static WeightDistribution normal(double mean, double std);
    double sample(std::mt19937& rng) const;
};

// --- Main class ---
class RegionalNetwork {
public:
    RegionalNetwork() = default;
    RegionalNetwork(const RegionalNetwork&) = delete;
    RegionalNetwork& operator=(const RegionalNetwork&) = delete;
    RegionalNetwork(RegionalNetwork&&) = default;
    RegionalNetwork& operator=(RegionalNetwork&&) = default;

    // Add populations
    void add_population(const std::string& name, size_t count,
                        Network::NeuronType type);
    void add_population(const std::string& name, size_t count,
                        const HHNeuron::Parameters& params);
    void add_population(const std::string& name, size_t count,
                        const IzhikevichNeuron::Parameters& params);

    // Bulk connectivity (preset pattern, dispatched to C++)
    void connect(const std::string& src, const std::string& dst,
                 ConnectivityPattern pattern,
                 const SynapseSpec& synapse,
                 const WeightDistribution& weight,
                 double delay = 0.0,
                 int shift = 1,
                 double probability = 0.1,
                 bool allow_self = false,
                 unsigned int seed = 0);

    // Add a single synapse between two populations using local indices
    void add_connection(const std::string& src, size_t src_local,
                        const std::string& dst, size_t dst_local,
                        double weight, const SynapseSpec& synapse,
                        double delay = 0.0);

    // Heterogeneous initial conditions
    void randomize_membrane_potentials(const std::string& name,
                                       double mean, double std_dev,
                                       unsigned int seed = 0);

    // Population queries
    const Population& population(const std::string& name) const;
    std::vector<std::string> population_names() const;
    size_t population_size(const std::string& name) const;
    size_t population_start(const std::string& name) const;
    size_t num_populations() const;

    // Delegation
    size_t num_neurons() const;
    size_t num_synapses() const;
    void set_fast_math(bool enabled);
    bool fast_math() const;
    void reset();
    Network& network();
    const Network& network() const;

    // Simulate (flat I_ext — Python wrapper handles dict conversion)
    std::vector<std::vector<double>> simulate(
        double duration, double dt,
        const std::vector<std::vector<double>>& I_ext);

private:
    Network net_;
    std::vector<Population> populations_;
    std::map<std::string, size_t> pop_index_;

    void validate_population(const std::string& name) const;
    void add_synapse_from_spec(size_t pre, size_t post, double weight,
                               const SynapseSpec& spec, double delay);
    void generate_connections(const Population& src, const Population& dst,
                              ConnectivityPattern pattern,
                              const SynapseSpec& synapse,
                              const WeightDistribution& weight,
                              double delay, int shift, double probability,
                              bool allow_self, std::mt19937& rng);
};

} // namespace hodgkin_huxley
