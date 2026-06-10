#pragma once

#include "hodgkin_huxley/parallel_sim.hpp"
#include "hodgkin_huxley/plasticity.hpp"
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
#include "hodgkin_huxley/device.hpp"
#include "hodgkin_huxley/stim_plan.hpp"
#ifdef HH_USE_CUDA
#  include "hodgkin_huxley/cuda_synapse.hpp"
#endif
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace hodgkin_huxley {

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
    ~Network();
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
    Network(Network&&) = delete;
    Network& operator=(Network&&) = delete;

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

    /// Overload with optional plasticity rule.
    size_t add_synapse(size_t pre, size_t post, double weight,
                       const SynapseSpec& spec, double delay,
                       const PlasticitySpec& plast);

    // -------------------------------------------------------------------------
    // Bulk-construction fast path (used by pattern generators).
    //   intern_synapse_spec : dedup + cache a spec once, return its index
    //   reserve_synapses    : pre-reserve SoA capacity for N more synapses
    //   add_synapse_interned: append one synapse with a pre-resolved spec index
    //                         (skips the per-synapse dedup scan)
    // Resolve the spec once, reserve once, then loop add_synapse_interned().
    // -------------------------------------------------------------------------
    size_t intern_synapse_spec(const SynapseSpec& spec);
    void   reserve_synapses(size_t additional) { sa_.reserve_additional(additional); }
    size_t add_synapse_interned(size_t pre, size_t post, double weight,
                                size_t sidx, const SynapseSpec& spec, double delay);

    /// Append many synapses sharing one spec from precomputed global index/weight
    /// arrays in a single call (interns the spec + reserves once). Lets the Python
    /// connect path build connectivity in numpy and cross into C++ exactly once.
    void add_synapses_bulk(const std::vector<uint32_t>& pre,
                           const std::vector<uint32_t>& post,
                           const std::vector<double>& weight,
                           const SynapseSpec& spec, double delay);

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
    [[nodiscard]] size_t num_synapses() const { return sa_.size(); }  // sa_ is source of truth

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

    void set_num_threads(int n);
    [[nodiscard]] int num_threads() const { return num_threads_; }

    // Called by RegionalNetwork::update_population_spec() after modifying a pop's spec
    void mark_pools_dirty() { pools_dirty_ = true; }

    [[nodiscard]] std::vector<double> get_potentials() const;
    [[nodiscard]] std::vector<double> get_synapse_weights() const;
    [[nodiscard]] bool has_stdp() const { return has_stdp_; }
    [[nodiscard]] bool has_stp()  const { return has_stp_; }

    void reset();
    void step(double dt, const std::vector<double>& I_ext);

    // --- CUDA device routing (task17) ---
    void   set_device(const Device& device);
    Device get_device() const;

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
        uint8_t* spike_buf_uint8,
        size_t  interval,
        size_t  n_rec,
        double  spike_threshold = 0.0,
        bool    use_float32     = true
    );

    /// Phase 2 parallel simulate: each GroupDef runs in its own std::thread.
    /// Pre-condition: groups partition all neurons; each inter-group synapse has delay >= dt.
    /// STDP/STP are not applied per-step in this path (require network-wide barrier).
    /// spike_event_buf records only intra-group spike arrivals (inter-group spikes omitted
    /// to avoid data races on spike_detected_ across threads).
    void simulate_with_descriptors_parallel(
        double duration, double dt,
        const StimPlan& stim,
        const std::map<int, std::vector<size_t>>& group_neurons,
        const std::map<int, std::vector<std::string>>& group_pools,
        double* V_buf,
        double* gate_buf,        size_t max_gates,
        double* calcium_buf,
        double* u_buf,
        double* g_syn_buf,
        double* I_syn_buf,
        double* spike_event_buf,
        size_t  interval,
        size_t  n_rec,
        double  spike_threshold = 0.0
    );

    // Forward-injection spike delivery (task26); syn_idx narrow to uint32_t (task27.3)
    struct SynapseRef {
        uint32_t syn_idx;       // index into SynArrays — uint32_t saves 8 bytes/syn via padding removal
        uint32_t delay_steps;   // precomputed: round(sa_.delay[i] / dt)
    };

    // Compressed per-neuron connectivity (task27.2)
    struct NeuronConnectivity {
        std::vector<uint8_t>  post_deltas;    // delta-encoded sorted post indices
        std::vector<uint32_t> post_overflow;  // pairs (local_j, full_post) where delta == 255
        bool   uniform_weight = false;
        double weight_value   = 0.0;
        size_t syn_start      = 0;
        size_t syn_count      = 0;
    };

    // Spec-derived constants shared across all synapses of the same spec (task27.1)
    struct SynapseSpecCache {
        double delta_S   = 0.0;
        double delta_A   = 0.0;
        double tau_S     = 0.0;
        double tau_A     = 0.0;
        double inv_tau_A = 0.0;
        double norm      = 1.0;
        double decay_S   = 0.0;  // dt-dependent; rebuilt when dt changes
        double decay_A   = 0.0;  // dt-dependent
    };

private:
    std::vector<std::unique_ptr<NeuronBase>> neurons_;
    mutable std::vector<SynapseBase> synapses_;   // lazy: built on first synapse(idx) call (task27.5)

    // =========================================================================
    // Structure-of-Arrays synapse data (all types unified)
    // =========================================================================
    struct SynArrays {
        // Common — pre/post uint32_t (task27.2); delay float (task27.2); E_syn float (task27.4)
        std::vector<uint32_t> pre, post;
        std::vector<double>   weight, g;
        std::vector<float>    E_syn;     // float: biological values ≤3 sig figs (task27.4)
        std::vector<float>    delay;

        // Unified state (all types)
        std::vector<double> S;           // primary gating / conductance variable
        std::vector<double> A;           // auxiliary variable (0 if unused)

        std::vector<uint32_t> spec_idx;  // index into synapse_specs_ / spec_caches_ (task27.4)

        // Plasticity state (task15) — all empty when no plasticity used
        std::vector<PlasticityType> plast_type;     // full-length, NONE by default
        std::vector<int32_t>  plast_state_idx;      // full-length, -1 = no state
        std::vector<uint32_t> plast_spec_idx_arr;   // full-length, → plasticity_specs_ (task27.4)
        std::vector<double>   plast_x_pre;          // sparse (n_STDP_synapses)
        std::vector<double>   plast_x_post;         // sparse (n_STDP_synapses)
        std::vector<double>   stp_u;                // sparse (n_STP_synapses)
        std::vector<double>   stp_x;                // sparse (n_STP_synapses)

        // Active synapse set (task16.5)
        std::vector<bool> is_active;         // [S] O(1) membership check

        double cached_dt = -1.0;
        size_t size() const { return S.size(); }  // pre/post may be cleared after build

        // Reserve capacity for `n` additional synapses across all per-synapse
        // arrays — avoids incremental reallocation during bulk connect.
        void reserve_additional(size_t n) {
            const size_t cap = S.size() + n;
            pre.reserve(cap);   post.reserve(cap);   weight.reserve(cap);
            g.reserve(cap);     E_syn.reserve(cap);  delay.reserve(cap);
            S.reserve(cap);     A.reserve(cap);      spec_idx.reserve(cap);
            plast_type.reserve(cap);        plast_state_idx.reserve(cap);
            plast_spec_idx_arr.reserve(cap); is_active.reserve(cap);
        }

        void push_defaults() {
            S.push_back(0.0);
            A.push_back(0.0);
            spec_idx.push_back(0);
            // Plasticity defaults: NONE, no state
            plast_type.push_back(PlasticityType::NONE);
            plast_state_idx.push_back(-1);
            plast_spec_idx_arr.push_back(0);
            is_active.push_back(false);
        }
    } sa_;

public:
    // -------------------------------------------------------------------------
    // SoA accessors for SynapseBase (called from synapse_base.cpp)
    // -------------------------------------------------------------------------
    [[nodiscard]] const SynArrays& syn_arrays() const { return sa_; }
    [[nodiscard]] const SynapseSpec& synapse_spec(size_t spec_idx) const {
        return synapse_specs_[spec_idx];
    }
    // pre/post indices: prefer decoded arrays (available after first simulate);
    // fall back to sa_.pre/post when accessed before the first build.
    [[nodiscard]] size_t pre_at(size_t syn_idx) const {
        return pre_decoded_.empty() ? sa_.pre[syn_idx] : pre_decoded_[syn_idx];
    }
    [[nodiscard]] size_t post_at(size_t syn_idx) const {
        return post_decoded_.empty() ? sa_.post[syn_idx] : post_decoded_[syn_idx];
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
        // Plasticity index lists (task15)
        std::vector<size_t> stdp;           // synapses with STDP
        std::vector<size_t> stp;            // synapses with STP
        // Active subsets (task16.5) — spike-driven types with S/A > epsilon
        std::vector<size_t> active_exp_decay;
        std::vector<size_t> active_alpha_func;
        std::vector<size_t> active_double_exp;
        std::vector<size_t> active_g;       // union of all types with g > epsilon
    } syn_groups_;
    std::vector<uint8_t> spike_detected_;

    // Plasticity (task15)
    std::vector<PlasticitySpec> plasticity_specs_;
    bool has_stdp_ = false;
    bool has_stp_  = false;
    std::vector<uint8_t> post_spiked_;   // per-neuron; allocated when has_stdp_
    std::vector<double>  V_all_prev_;    // per-neuron; allocated when has_stdp_

    // Synapse specs (deduped by name) + per-spec constant cache (task27.1)
    std::vector<SynapseSpec>      synapse_specs_;
    std::vector<SynapseSpecCache> synapse_spec_caches_;

    bool fast_math_ = true;
    double spike_threshold_ = 0.0;
    int  num_threads_ = 0;

    mutable bool soa_dirty_ = false;
    bool soa_sorted_ = false;
    bool groups_built_ = false;

    PoolManager pool_mgr_;
    bool pools_dirty_ = true;

    // Pinned host memory for async GPU DMA (task17). Null when CUDA not active.
    bool    use_pinned_memory_ = false;
    double* V_cache_pinned_    = nullptr;
    double* I_syn_pinned_      = nullptr;
    size_t  pinned_size_       = 0;

#ifdef HH_USE_CUDA
    double* d_V_cache_         = nullptr;
    double* d_I_syn_           = nullptr;
    double* d_synapse_g_scale_ = nullptr;
    size_t  cuda_buffer_size_  = 0;
    DeviceSynapseArrays dev_syn_;

    cudaStream_t sim_stream_      = nullptr;
    // Chunked async recording streaming: double-buffered (ping/pong) per-chunk
    // device recording buffers + a dedicated copy stream that overlaps the D2H
    // of chunk k with the compute of chunk k+1. Bounds recording VRAM to
    // 2 * (n_neurons * chunk_rec) instead of the whole-sim n_neurons * n_rec.
    cudaStream_t copy_stream_       = nullptr;
    double*      d_V_out_[2]        = {nullptr, nullptr};
    uint8_t*     d_spike_buf_[2]    = {nullptr, nullptr};
    double*      V_stage_pinned_[2] = {nullptr, nullptr};   // pinned host staging
    uint8_t*     spk_stage_pinned_[2] = {nullptr, nullptr};
    cudaEvent_t  compute_done_[2]   = {nullptr, nullptr};
    cudaEvent_t  copy_done_[2]      = {nullptr, nullptr};
    size_t       rec_buf_cap_       = 0;  // per-buffer capacity in recording columns (n_neurons*chunk_rec)
#endif

    // Forward-injection spike delivery state (task26)
    std::vector<std::vector<SynapseRef>> post_from_;    // [pre_neuron] → {syn_idx, delay_steps}
    std::vector<std::vector<size_t>>     event_slots_;  // [step % cap] → syn_indices due this step
    std::vector<uint32_t>                delay_steps_;  // precomputed per-synapse delay/dt (task27.2: uint32)
    std::vector<double>                  V_prev_;       // per-neuron previous voltage
    size_t                               current_step_ = 0;

    // Connectivity compression (task27.2)
    std::vector<NeuronConnectivity> neuron_connectivity_;  // [pre_neuron]
    std::vector<uint32_t>           post_decoded_;         // flat post indices, 4 bytes/syn
    std::vector<uint32_t>           pre_decoded_;          // flat pre indices, 4 bytes/syn
    bool                            pre_post_cleared_       = false;
    bool                            injection_tables_built_ = false;

    mutable bool synapses_dirty_ = true;  // cleared on first synapse(idx) call (task27.5)

    void reallocate_pinned_buffers(size_t n);
    void ensure_synapses_built() const;  // lazy SynapseBase construction (task27.5)
    void cache_voltages();
    void ensure_buffers();
    void compute_synaptic_currents();
    void update_synapses_grouped(double dt);
    void rebuild_spec_caches(double dt);
    void build_synapse_groups();
    void build_injection_tables(double dt);
    void build_neuron_connectivity();
    void reconstruct_pre_post();
    void sort_synapses_by_pre();
    void apply_stdp(double dt);
    void apply_stp(double dt);
    size_t add_or_find_plasticity_spec(const PlasticitySpec& ps);
#ifdef HH_USE_CUDA
    bool can_use_cuda_synapse_path() const;
    void free_cuda_runtime();
    void ensure_cuda_runtime_buffers(size_t n_neurons);
    void prepare_cuda_synapses(double dt);
    void sync_cuda_synapses_to_host(bool sync_pending_events);
#endif
};

} // namespace hodgkin_huxley
