#pragma once

#include <atomic>
#include <unordered_map>
#include <vector>
#include <string>

namespace hodgkin_huxley {

/// Spike-delivery reference used in forward-injection tables.
/// Defined at namespace scope (moved from Network nested struct) so parallel_sim.hpp can use it.
struct SynapseRef {
    uint32_t syn_idx;       // index into SynArrays
    uint32_t delay_steps;   // precomputed: round(sa_.delay[i] / dt)
};

/// SPSC event queue for one ordered group pair using §26.3 event-queue routing.
/// Pre-allocated per-step slots avoid the overwrite race when the producer runs ahead.
/// src thread writes fired_per_step[t] then increments spike_done;
/// dst thread waits spike_done >= t+1 then reads fired_per_step[t].
struct CrossGroupQueue {
    std::vector<std::vector<uint32_t>> fired_per_step;  // indexed by step t; sized = num_steps
    std::atomic<size_t>                spike_done{0};
};

/// Per-thread-group bookkeeping for delay-decomposition parallel simulation (Phase 2).
/// Built once by Network::simulate_with_descriptors_parallel before launching threads.
struct GroupDef {
    int id = 0;

    /// Global neuron indices owned by this group.
    std::vector<size_t> neuron_indices;

    /// Spec names of ComposablePools whose neurons are entirely in this group.
    std::vector<std::string> pool_names;

    /// Synapse indices where pre[k] is in this group — for spike detection + g update.
    /// Only contains intra-group and shared-g cross-group synapses (not queue-routed cross-group).
    std::vector<size_t> pre_all;           // all owned pre-synapses (spike detection Phase 1)
    std::vector<size_t> pre_exp_decay;     // Phase 2a
    std::vector<size_t> pre_alpha_func;    // Phase 2b
    std::vector<size_t> pre_double_exp;    // Phase 2c
    std::vector<size_t> pre_voltage_gated; // Phase 2d
    // Active subsets — spike-driven synapses with S/A > kSynapseEpsilon (task16.5)
    std::vector<size_t> active_pre_exp_decay;
    std::vector<size_t> active_pre_alpha_func;
    std::vector<size_t> active_pre_double_exp;

    /// Cross-group spike-driven synapses owned by THIS (post) group via §26.3 queue routing.
    /// These were previously owned by the pre-group under shared-g; now the post-group
    /// receives fired neuron indices and runs the conductance update locally.
    std::vector<size_t> cross_pre_exp_decay;
    std::vector<size_t> cross_pre_alpha_func;
    std::vector<size_t> cross_pre_double_exp;
    std::vector<size_t> active_cross_pre_exp_decay;
    std::vector<size_t> active_cross_pre_alpha_func;
    std::vector<size_t> active_cross_pre_double_exp;

    /// Synapse indices where post[k] is in this group — for I_syn accumulation.
    std::vector<size_t> post_syn;

    // -------------------------------------------------------------------------
    // Cross-group routing (replaces monolithic src_group_ids / consumer_group_ids)
    // -------------------------------------------------------------------------

    /// Shared-g source groups: wait for step_done[src] >= t before Phase A.
    std::vector<int> sharedg_src_ids;

    /// Event-queue source groups: wait for spike_done on the incoming queue before Phase QI.
    std::vector<int> queue_src_ids;

    /// Shared-g consumer groups: W2 barrier (wait read_done >= t+1 before writing g[k]).
    std::vector<int> sharedg_consumer_ids;

    /// Event-queue consumer groups: no W2 barrier needed (post-group owns the g[k] update).
    std::vector<int> queue_consumer_ids;

    /// Incoming queues, parallel to queue_src_ids.
    std::vector<CrossGroupQueue*> incoming_queues;

    /// Outgoing queues, parallel to queue_consumer_ids.
    std::vector<CrossGroupQueue*> outgoing_queues;

    // -------------------------------------------------------------------------
    // Per-group forward-injection tables
    // -------------------------------------------------------------------------

    /// local_post_from[n]: SynapseRefs for synapses where pre=n AND
    /// (post ∈ this group OR pair is shared-g cross-group with this group as pre).
    /// Used in Phase U1 to inject spikes into event_slots.
    /// Size = N_neurons; most entries empty.
    std::vector<std::vector<SynapseRef>> local_post_from;

    /// cross_post_from[src_gid][n]: SynapseRefs where pre=n ∈ src_gid AND post ∈ this group,
    /// for queue-routed pairs only. Used in Phase QI to inject cross-group spike arrivals.
    std::unordered_map<int, std::vector<std::vector<SynapseRef>>> cross_post_from;

    /// Pool-local indices for HH and Izhikevich neurons in this group.
    std::vector<size_t> hh_local_indices;
    std::vector<size_t> iz_local_indices;

    /// Synapse indices where BOTH pre[k] and post[k] are in this group.
    /// Used for spike_event accumulation without cross-group data races.
    std::vector<size_t> intra_syn;

    /// Unique pre-neuron indices for synapses in pre_all + cross-group queue synapses.
    /// Used for spike detection in Phase U1 and fired-neuron collection for Phase Q.
    std::vector<size_t> pre_neurons;

    /// Per-group circular event buffer for forward-injection spike delivery.
    /// Covers max delay across all synapses injected by this group (local + cross-group queue).
    std::vector<std::vector<size_t>> event_slots;
};

} // namespace hodgkin_huxley
