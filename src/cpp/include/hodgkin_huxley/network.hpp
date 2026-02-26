#pragma once

#include "neuron_base.hpp"
#include "neuron.hpp"
#include "izhikevich.hpp"
#include "synapse.hpp"
#include "hh_pool.hpp"
#include "iz_pool.hpp"
#include "composable_neuron.hpp"
#include "composable_pool.hpp"
#include "ion_channels.hpp"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace hodgkin_huxley {

/**
 * @brief Network of neurons with polymorphic neuron support
 *
 * Allows simulation of interconnected neurons (HH, Izhikevich, or mixed)
 * with synaptic connections.
 *
 * Synapse data is stored in Structure-of-Arrays (SoA) layout for
 * cache-friendly inner loops. The polymorphic synapse objects are kept
 * only for API access (synapse() getter).
 */
class Network {
public:
    /**
     * @brief Receptor types with biologically accurate default kinetics
     */
    enum class ReceptorType {
        AMPA,     // Fast excitatory: E_syn=0, tau_rise=0.5, tau_decay=2.5
        NMDA,     // Slow excitatory: E_syn=0, tau_rise=2.0, tau_decay=67.0
        GABA_A    // Inhibitory:      E_syn=-80, tau_rise=0.4, tau_decay=7.7
    };

    /**
     * @brief Enum for neuron types when adding neurons
     */
    enum class NeuronType {
        HH,
        IZHIKEVICH_RS,
        IZHIKEVICH_FS,
        IZHIKEVICH_IB,
        IZHIKEVICH_CH,
        IZHIKEVICH_LTS,
        IZHIKEVICH_CUSTOM,
        COMPOSABLE
    };

    Network() = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;

    /**
     * @brief Create network with N HH neurons (backward compatible)
     */
    explicit Network(size_t num_neurons);

    /**
     * @brief Create network with N neurons of specified type
     */
    Network(size_t num_neurons, NeuronType type);

    // Add neurons
    size_t add_neuron();  // Add default HH neuron
    size_t add_neuron(const HHNeuron::Parameters& params);
    size_t add_neuron(NeuronType type);
    size_t add_neuron(const IzhikevichNeuron::Parameters& params);
    size_t add_neuron(const NeuronModelSpec& spec);

    /**
     * @brief Add a neuron with explicit type specification
     */
    size_t add_hh_neuron();
    size_t add_hh_neuron(const HHNeuron::Parameters& params);
    size_t add_izhikevich_neuron(IzhikevichNeuron::Type type = IzhikevichNeuron::Type::REGULAR_SPIKING);
    size_t add_izhikevich_neuron(const IzhikevichNeuron::Parameters& params);

    // Add kinetic synapse
    size_t add_kinetic_synapse(size_t pre, size_t post, double weight,
                               const KineticSynapseSpec& spec, double delay = 0.0);

    // Add synaptic connections
    void add_synapse(size_t pre_idx, size_t post_idx, double weight,
                     double E_syn = 0.0, double tau = 2.0,
                     double delay = 0.0);
    void add_alpha_synapse(size_t pre_idx, size_t post_idx, double weight,
                           double E_syn = 0.0, double tau = 2.0,
                           double delay = 0.0);
    void add_double_exp_synapse(size_t pre_idx, size_t post_idx, double weight,
                                double E_syn = 0.0,
                                double tau_rise = 0.4, double tau_decay = 2.5,
                                double delay = 0.0);

    // Receptor-type convenience methods (double-exponential with preset kinetics)
    void add_ampa_synapse(size_t pre_idx, size_t post_idx, double weight,
                          double delay = 0.0);
    void add_nmda_synapse(size_t pre_idx, size_t post_idx, double weight,
                          double delay = 0.0);
    void add_gaba_a_synapse(size_t pre_idx, size_t post_idx, double weight,
                            double delay = 0.0);
    void add_receptor_synapse(size_t pre_idx, size_t post_idx, double weight,
                              ReceptorType receptor, double delay = 0.0);

    // Getters
    [[nodiscard]] size_t num_neurons() const { return neurons_.size(); }
    [[nodiscard]] size_t num_synapses() const { return synapses_.size(); }

    // Kinetic synapse state accessor (for testing / inspection)
    [[nodiscard]] double get_kin_S(size_t synapse_idx) const;
    [[nodiscard]] double get_kin_g(size_t synapse_idx) const;

    /**
     * @brief Get neuron by index (polymorphic access)
     */
    [[nodiscard]] const NeuronBase& neuron(size_t idx) const { return *neurons_[idx]; }
    [[nodiscard]] NeuronBase& neuron(size_t idx) { return *neurons_[idx]; }

    /**
     * @brief Get neuron as HH (throws if wrong type)
     */
    [[nodiscard]] const HHNeuron& hh_neuron(size_t idx) const;
    [[nodiscard]] HHNeuron& hh_neuron(size_t idx);

    /**
     * @brief Get neuron as Izhikevich (throws if wrong type)
     */
    [[nodiscard]] const IzhikevichNeuron& iz_neuron(size_t idx) const;
    [[nodiscard]] IzhikevichNeuron& iz_neuron(size_t idx);

    /**
     * @brief Get neuron type name
     */
    [[nodiscard]] std::string neuron_type(size_t idx) const { return neurons_[idx]->type_name(); }

    /**
     * @brief Get synapse by index (lazy-syncs conductance from SoA)
     */
    [[nodiscard]] const SynapseBase& synapse(size_t idx) const;

    // Fast math toggle (affects exp() in HH pool — ~8 digits vs full precision)
    void set_fast_math(bool enabled) { fast_math_ = enabled; }
    [[nodiscard]] bool fast_math() const { return fast_math_; }

    // Get all membrane potentials
    [[nodiscard]] std::vector<double> get_potentials() const;

    // Reset all neurons
    void reset();

    // Step the entire network
    void step(double dt, const std::vector<double>& I_ext);

    // Simulate network, returns matrix of voltage traces (neurons x time)
    std::vector<std::vector<double>> simulate(
        double duration,
        double dt,
        const std::vector<std::vector<double>>& I_ext
    );

private:
    std::vector<std::unique_ptr<NeuronBase>> neurons_;
    std::vector<std::unique_ptr<SynapseBase>> synapses_;  // API access only

    // =========================================================================
    // Structure-of-Arrays (SoA) synapse data for cache-friendly inner loops.
    // Eliminates pointer chasing and enables SIMD auto-vectorization.
    // =========================================================================
    enum class SynType : uint8_t { SYN_EXP = 0, SYN_ALPHA = 1, SYN_DEXP = 2, SYN_KINETIC = 3 };

    struct SynArrays {
        // Common fields (all synapse types)
        std::vector<size_t> pre;
        std::vector<size_t> post;
        std::vector<double> weight;
        std::vector<double> E_syn;
        std::vector<double> g;          // mutable conductance

        std::vector<SynType> type;

        // Spike detection
        std::vector<double> V_pre_prev;

        // Delay
        std::vector<double> delay;
        std::vector<std::vector<bool>> spike_buf;
        std::vector<size_t> buf_head;
        std::vector<bool> delay_init;

        // Exponential-specific
        std::vector<double> exp_tau;
        std::vector<double> exp_decay;  // cached exp(-dt/tau)

        // Alpha-specific
        std::vector<double> alpha_x;
        std::vector<double> alpha_inv_tau;

        // Double-exponential-specific
        std::vector<double> dexp_g_rise;
        std::vector<double> dexp_g_decay;
        std::vector<double> dexp_tau_rise;
        std::vector<double> dexp_tau_decay;
        std::vector<double> dexp_rise_decay;  // cached exp(-dt/tau_rise)
        std::vector<double> dexp_fall_decay;  // cached exp(-dt/tau_decay)
        std::vector<double> dexp_norm;

        // Kinetic-specific
        std::vector<double> kin_S;           // gating variable (0 for non-kinetic)
        std::vector<size_t> kin_spec_idx;    // index into Network::kinetic_specs_

        double cached_dt = -1.0;
        size_t size() const { return pre.size(); }

        // Push default values for all type-specific fields
        void push_type_defaults() {
            exp_tau.push_back(0.0);
            exp_decay.push_back(0.0);
            alpha_x.push_back(0.0);
            alpha_inv_tau.push_back(0.0);
            dexp_g_rise.push_back(0.0);
            dexp_g_decay.push_back(0.0);
            dexp_tau_rise.push_back(0.0);
            dexp_tau_decay.push_back(0.0);
            dexp_rise_decay.push_back(0.0);
            dexp_fall_decay.push_back(0.0);
            dexp_norm.push_back(0.0);
            kin_S.push_back(0.0);
            kin_spec_idx.push_back(0);
        }
    } sa_;

    // Pre-allocated working buffers
    std::vector<double> I_syn_buffer_;
    std::vector<double> V_cache_;

    // Type-separated synapse index lists for branch-free inner loops
    struct SynapseGroups {
        std::vector<size_t> exp;
        std::vector<size_t> alpha;
        std::vector<size_t> dexp;
        std::vector<size_t> kinetic;
    } syn_groups_;
    std::vector<uint8_t> spike_detected_;  // per-synapse spike flag buffer

    // Kinetic synapse specs (deduped by name)
    std::vector<KineticSynapseSpec> kinetic_specs_;

    // Use fast polynomial exp approximation (~8 digits) vs Eigen's built-in
    bool fast_math_ = true;

    // Lazy sync: SoA is source of truth during simulation
    mutable bool soa_dirty_ = false;
    bool soa_sorted_ = false;

    void cache_voltages();
    void ensure_buffers();
    void compute_synaptic_currents();
    void update_synapses(double dt);
    void update_synapses_grouped(double dt);
    void update_decay_factors(double dt);
    void build_synapse_groups();
    void sync_soa_to_objects() const;
    void sort_synapses_by_pre();
};

} // namespace hodgkin_huxley
