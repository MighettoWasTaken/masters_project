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

// SynapseSpec is the unified synapse model specification from model/synapse_spec.hpp
// (included transitively via network.hpp → ion_channels.hpp).

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
    void add_population(const std::string& name, size_t count,
                        const NeuronModelSpec& spec);
    void add_population(const std::string& name,
                        const std::vector<NeuronModelSpec>& specs);

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

    // Overload with plasticity rule
    void add_connection(const std::string& src, size_t src_local,
                        const std::string& dst, size_t dst_local,
                        double weight, const SynapseSpec& synapse,
                        double delay, const PlasticitySpec& plast);

    // Add a kinetic synapse between two populations using local indices
    void add_kinetic_connection(const std::string& src, size_t i,
                                const std::string& dst, size_t j,
                                double weight, const SynapseSpec& spec,
                                double delay = 0.0);

    // Heterogeneous initial conditions
    void randomize_membrane_potentials(const std::string& name,
                                       double mean, double std_dev,
                                       unsigned int seed = 0,
                                       bool reset_gates = false);

    // Population queries
    const Population& population(const std::string& name) const;
    std::vector<std::string> population_names() const;
    size_t population_size(const std::string& name) const;
    size_t population_start(const std::string& name) const;
    size_t num_populations() const;

    // Update the NeuronModelSpec for a named population and rebuild pools.
    // Called by Python add_intracellular() after appending an IntracellularSpec.
    void update_population_spec(const std::string& name, const NeuronModelSpec& spec);

    // --- Phase 2: thread-group API ---
    // groups: group_id → list of population names; each inter-group synapse must have delay >= dt.
    void set_thread_groups(const std::map<int, std::vector<std::string>>& groups);
    void clear_thread_groups();
    bool has_thread_groups() const { return !group_populations_.empty(); }

    // Run a parallel simulation using stored thread groups with full buffer recording.
    // All buffer pointers may be nullptr to skip recording that metric.
    // spike_event_buf only counts intra-group spike arrivals (inter-group omitted).
    void simulate_parallel(const StimPlan& stim, double duration, double dt,
                           double* V_buf,
                           double* gate_buf,        size_t max_gates,
                           double* calcium_buf,
                           double* u_buf,
                           double* g_syn_buf,
                           double* I_syn_buf,
                           double* spike_event_buf,
                           size_t n_rec, size_t interval,
                           double spike_threshold = 0.0);

    // --- CUDA device routing (task17) ---
    void   to(const Device& device);
    Device current_device() const { return current_device_; }

    // Delegation
    size_t num_neurons() const;
    size_t num_synapses() const;
    void set_fast_math(bool enabled);
    bool fast_math() const;
    void reset();
    Network& network();
    const Network& network() const;

private:
    Network net_;
    Device  current_device_ = Device::cpu();
    std::vector<Population> populations_;
    std::map<std::string, size_t> pop_index_;
    std::map<std::string, NeuronModelSpec> pop_specs_;  // per-population stored spec (task14)

    // Phase 2 thread groups: group_id → list of population names
    std::map<int, std::vector<std::string>> group_populations_;

    void validate_population(const std::string& name) const;
    void generate_connections(const Population& src, const Population& dst,
                              ConnectivityPattern pattern,
                              const SynapseSpec& synapse,
                              const WeightDistribution& weight,
                              double delay, int shift, double probability,
                              bool allow_self, std::mt19937& rng);
};

} // namespace hodgkin_huxley
