#pragma once

#include <vector>
#include <string>

namespace hodgkin_huxley {

/// Per-thread-group bookkeeping for delay-decomposition parallel simulation (Phase 2).
/// Built once by Network::simulate_with_descriptors_parallel before launching threads.
struct GroupDef {
    int id = 0;

    /// Global neuron indices owned by this group.
    std::vector<size_t> neuron_indices;

    /// Spec names of ComposablePools whose neurons are entirely in this group.
    std::vector<std::string> pool_names;

    /// Synapse indices where pre[k] is in this group — for spike detection + g update.
    std::vector<size_t> pre_all;           // all pre-synapses (spike detection Phase 1)
    std::vector<size_t> pre_exp_decay;     // Phase 2a
    std::vector<size_t> pre_alpha_func;    // Phase 2b
    std::vector<size_t> pre_double_exp;    // Phase 2c
    std::vector<size_t> pre_voltage_gated; // Phase 2d

    /// Synapse indices where post[k] is in this group — for I_syn accumulation.
    std::vector<size_t> post_syn;

    /// Groups this group waits on (Phase W) before accumulating I_syn.
    /// Specifically: wait for step_done_U1[src] >= t so src has completed
    /// Phase U1 of step t-1 (which means src has NOT yet written new g[k]
    /// values for step t — safe to read).
    std::vector<int> src_group_ids;

    /// Groups that read this group's g[k] values (consumers of this group's
    /// pre-synapses). This group waits for consumer read_done >= t+1 before
    /// writing new g[k] in Phase U2 of step t, preventing a write-after-read race.
    std::vector<int> consumer_group_ids;

    /// Pool-local indices for HH and Izhikevich neurons in this group.
    /// Used by PoolManager subset step/gather/scatter — enables Phase 2
    /// parallelism for all neuron types, not just ComposablePool.
    std::vector<size_t> hh_local_indices;
    std::vector<size_t> iz_local_indices;

    /// Synapse indices where BOTH pre[k] and post[k] are in this group.
    /// Used for spike_event accumulation without cross-group data races.
    std::vector<size_t> intra_syn;
};

} // namespace hodgkin_huxley
