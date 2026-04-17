#pragma once

#include "hodgkin_huxley/neuron_base.hpp"
#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/izhikevich.hpp"
#include "hodgkin_huxley/synapse_base.hpp"
#include "hodgkin_huxley/hh_pool.hpp"
#include "hodgkin_huxley/iz_pool.hpp"
#include "hodgkin_huxley/composable_neuron.hpp"
#include "hodgkin_huxley/composable_pool.hpp"
#include "hodgkin_huxley/ion_channels.hpp"
#include "hodgkin_huxley/network/pool_manager.hpp"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace hodgkin_huxley {

// =============================================================================
// Compact stimulation plan — avoids dense (n_neurons × n_steps) I_ext matrix
// =============================================================================

/// A rectangular current pulse applied to a contiguous neuron range.
struct PulseEvent {
    size_t neuron_start;  ///< First neuron index (inclusive)
    size_t neuron_end;    ///< Last neuron index (exclusive)
    size_t onset_step;    ///< First time step of pulse (inclusive)
    size_t end_step;      ///< First time step after pulse (exclusive)
    double amplitude;     ///< Current amplitude (µA/cm²)
};

/// A periodic DBS pulse train applied to a contiguous neuron range.
struct DBSEvent {
    size_t neuron_start;  ///< First neuron index (inclusive)
    size_t neuron_end;    ///< Last neuron index (exclusive)
    size_t isi_steps;     ///< Inter-stimulus interval in time steps
    size_t pw_steps;      ///< Pulse width in time steps
    double amplitude;     ///< Current amplitude (µA/cm²)
};

/// Compact representation of all external stimulation for one simulation run.
struct StimPlan {
    std::vector<double>     I_const;   ///< Per-neuron constant baseline current
    std::vector<PulseEvent> pulses;    ///< Sparse rectangular pulse events
    std::vector<DBSEvent>   dbs;       ///< Periodic DBS pulse trains
};

/**
 * @brief Network of neurons connected by synapses.
 *
 * All synapse update logic is centralised in update_synapses_grouped().
 * Synapse state is stored in SynArrays (Structure-of-Arrays) for cache-
 * friendly inner loops.  SynapseBase objects provide a read-only view for
 * API access; they are always in sync (no lazy copy needed).
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

    explicit Network(size_t num_neurons);
    Network(size_t num_neurons, NeuronType type);

    // -------------------------------------------------------------------------
    // Add neurons
    // -------------------------------------------------------------------------
    size_t add_neuron();
    size_t add_neuron(const HHNeuron::Parameters& params);
    size_t add_neuron(NeuronType type);
    size_t add_neuron(const IzhikevichNeuron::Parameters& params);
    size_t add_neuron(const NeuronModelSpec& spec);

    size_t add_hh_neuron();
    size_t add_hh_neuron(const HHNeuron::Parameters& params);
    size_t add_izhikevich_neuron(IzhikevichNeuron::Type type = IzhikevichNeuron::Type::REGULAR_SPIKING);
    size_t add_izhikevich_neuron(const IzhikevichNeuron::Parameters& params);

    // -------------------------------------------------------------------------
    // Add synapses — unified interface
    // -------------------------------------------------------------------------

    /// Primary method: add a synapse described by a SynapseSpec.
    size_t add_synapse(size_t pre, size_t post, double weight,
                       const SynapseSpec& spec, double delay = 0.0);

    // Backward-compatible convenience wrappers — delegate to add_synapse()
    void add_synapse(size_t pre_idx, size_t post_idx, double weight,
                     double E_syn = 0.0, double tau = 2.0, double delay = 0.0);
    void add_alpha_synapse(size_t pre_idx, size_t post_idx, double weight,
                           double E_syn = 0.0, double tau = 2.0, double delay = 0.0);
    void add_double_exp_synapse(size_t pre_idx, size_t post_idx, double weight,
                                double E_syn = 0.0,
                                double tau_rise = 0.4, double tau_decay = 2.5,
                                double delay = 0.0);
    size_t add_kinetic_synapse(size_t pre, size_t post, double weight,
                               const SynapseSpec& spec, double delay = 0.0);

    // Receptor-type convenience methods
    void add_ampa_synapse(size_t pre_idx, size_t post_idx, double weight,
                          double delay = 0.0);
    void add_nmda_synapse(size_t pre_idx, size_t post_idx, double weight,
                          double delay = 0.0);
    void add_gaba_a_synapse(size_t pre_idx, size_t post_idx, double weight,
                            double delay = 0.0);
    void add_receptor_synapse(size_t pre_idx, size_t post_idx, double weight,
                              ReceptorType receptor, double delay = 0.0);

    // -------------------------------------------------------------------------
    // Getters
    // -------------------------------------------------------------------------
    [[nodiscard]] size_t num_neurons()  const { return neurons_.size(); }
    [[nodiscard]] size_t num_synapses() const { return synapses_.size(); }

    [[nodiscard]] double get_kin_S(size_t synapse_idx) const;
    [[nodiscard]] double get_kin_g(size_t synapse_idx) const;

    [[nodiscard]] const NeuronBase& neuron(size_t idx) const { return *neurons_[idx]; }
    [[nodiscard]] NeuronBase&       neuron(size_t idx)       { return *neurons_[idx]; }
    [[nodiscard]] const HHNeuron&   hh_neuron(size_t idx) const;
    [[nodiscard]] HHNeuron&         hh_neuron(size_t idx);
    [[nodiscard]] const IzhikevichNeuron& iz_neuron(size_t idx) const;
    [[nodiscard]] IzhikevichNeuron&       iz_neuron(size_t idx);
    [[nodiscard]] std::string neuron_type(size_t idx) const { return neurons_[idx]->type_name(); }

    [[nodiscard]] const SynapseBase& synapse(size_t idx) const;

    void set_fast_math(bool enabled) { fast_math_ = enabled; }
    [[nodiscard]] bool fast_math() const { return fast_math_; }

    // Called by RegionalNetwork::update_population_spec() after modifying a pop's spec
    void mark_pools_dirty() { pools_dirty_ = true; }

    [[nodiscard]] std::vector<double> get_potentials() const;

    void reset();
    void step(double dt, const std::vector<double>& I_ext);

    std::vector<std::vector<double>> simulate(
        double duration, double dt,
        const std::vector<std::vector<double>>& I_ext);

    size_t max_gate_count() const;
    std::vector<size_t> get_synapse_pre_indices()  const;
    std::vector<size_t> get_synapse_post_indices() const;

    void simulate_into_buffers(
        double duration, double dt,
        const std::vector<std::vector<double>>& I_ext,
        double* V_buf,
        double* gate_buf,   size_t max_gates,
        double* calcium_buf,
        double* u_buf,
        double* g_syn_buf,
        double* I_syn_buf,
        double* spike_event_buf,
        size_t  interval,
        size_t  n_rec,
        double  spike_threshold = 0.0
    );

    void simulate_with_descriptors(
        double duration, double dt,
        const StimPlan& stim,
        double* V_buf,
        double* gate_buf,       size_t max_gates,
        double* calcium_buf,
        double* u_buf,
        double* g_syn_buf,
        double* I_syn_buf,
        double* spike_event_buf,
        size_t  interval,
        size_t  n_rec,
        double  spike_threshold = 0.0
    );

private:
    std::vector<std::unique_ptr<NeuronBase>> neurons_;
    std::vector<SynapseBase> synapses_;   // lightweight views — always in sync

    // =========================================================================
    // Structure-of-Arrays synapse data (all types unified)
    // =========================================================================
    struct SynArrays {
        // Common
        std::vector<size_t> pre, post;
        std::vector<double> weight, E_syn, g;
        std::vector<double> V_pre_prev;
        std::vector<double> delay;
        std::vector<std::vector<bool>> spike_buf;
        std::vector<size_t> buf_head;
        std::vector<bool>   delay_init;

        // Unified state (all types)
        std::vector<double> S;           // primary gating / conductance variable
        std::vector<double> A;           // auxiliary variable (0 if unused)
        std::vector<double> delta_S;     // on-spike additive increment for S
        std::vector<double> delta_A;     // on-spike additive increment for A
        std::vector<double> tau_S;       // S time constant (EXP_DECAY, DOUBLE_EXP)
        std::vector<double> tau_A;       // A time constant (ALPHA_FUNC, DOUBLE_EXP)
        std::vector<double> inv_tau_A;   // 1/tau_A cached (ALPHA_FUNC Euler)
        std::vector<double> norm;        // DOUBLE_EXP peak normalization
        std::vector<double> decay_S;     // cached exp(-dt/tau_S)
        std::vector<double> decay_A;     // cached exp(-dt/tau_A)

        std::vector<size_t> spec_idx;    // index into Network::synapse_specs_

        double cached_dt = -1.0;
        size_t size() const { return pre.size(); }

        void push_defaults() {
            S.push_back(0.0);
            A.push_back(0.0);
            delta_S.push_back(0.0);
            delta_A.push_back(0.0);
            tau_S.push_back(0.0);
            tau_A.push_back(0.0);
            inv_tau_A.push_back(0.0);
            norm.push_back(1.0);
            decay_S.push_back(0.0);
            decay_A.push_back(0.0);
            spec_idx.push_back(0);
        }
    } sa_;

public:
    // -------------------------------------------------------------------------
    // SoA accessor for SynapseBase view (called from SynapseBase member fns)
    // -------------------------------------------------------------------------
    [[nodiscard]] const SynArrays& syn_arrays() const { return sa_; }
    [[nodiscard]] const SynapseSpec& synapse_spec(size_t spec_idx) const {
        return synapse_specs_[spec_idx];
    }

private:
    // Pre-allocated working buffers
    std::vector<double> I_syn_buffer_;
    std::vector<double> V_cache_;
    std::vector<double> synapse_g_scale_;  // per-neuron synapse conductance multiplier (task14)

    // UpdateForm-based synapse groups for branch-free inner loops
    struct SynapseGroups {
        std::vector<size_t> exp_decay;      // EXP_DECAY
        std::vector<size_t> alpha_func;     // ALPHA_FUNC
        std::vector<size_t> double_exp;     // DOUBLE_EXP
        std::vector<size_t> voltage_gated;  // TANH_GATE, BOLTZMANN_GATE, ALPHA_BETA, CUSTOM_EXPR
    } syn_groups_;
    std::vector<uint8_t> spike_detected_;

    // Synapse specs (deduped by name)
    std::vector<SynapseSpec> synapse_specs_;

    bool fast_math_ = true;
    double spike_threshold_ = 0.0;

    mutable bool soa_dirty_ = false;
    bool soa_sorted_ = false;
    bool groups_built_ = false;

    PoolManager pool_mgr_;
    bool pools_dirty_ = true;

    void cache_voltages();
    void ensure_buffers();
    void compute_synaptic_currents();
    void update_synapses_grouped(double dt);
    void update_decay_factors(double dt);
    void build_synapse_groups();
    void sort_synapses_by_pre();
};

} // namespace hodgkin_huxley
