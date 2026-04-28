#pragma once

// =============================================================================
// network/pool_manager.hpp — owns and orchestrates batched neuron pools
//
// Encapsulates pool construction (the former dynamic_cast scan inside
// simulate_into_buffers) and provides a uniform interface for the hot loop.
//
// Held as a Network member with a pools_dirty_ flag:
//   - pools_dirty_ = true  → rebuild on next simulate (after add_neuron / reset)
//   - pools_dirty_ = false → reuse existing pools (task20: continuable simulation)
//
// Insertion point for task16 (OpenMP): replace step_all with a parallel version.
// Insertion point for task17 (CUDA):   swap concrete pool types for CUDA variants.
// =============================================================================

#include "hodgkin_huxley/hh_pool.hpp"
#include "hodgkin_huxley/iz_pool.hpp"
#include "hodgkin_huxley/composable_pool.hpp"
#include "hodgkin_huxley/neuron_base.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace hodgkin_huxley {

class PoolManager {
public:
    // Build (or rebuild) pools from the current API neuron state.
    // Called once before the first simulate(); skipped on subsequent calls
    // when pools_dirty_ is false (pools retain state from prior run).
    void build_from_neurons(const std::vector<std::unique_ptr<NeuronBase>>& neurons,
                            bool fast_math);

    // --- Per-step hot-loop delegates ---
    void scatter_all_voltages(double* V_cache) const;
    void gather_all_currents(const double* I_buf);
    void step_all(double dt);
    void sync_all_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const;

    // --- Recording delegates ---
    void scatter_gates(double* gate_buf, size_t max_gates, size_t n_rec, size_t tr) const;
    void scatter_calcium(double* ca_buf, size_t n_rec, size_t tr) const;
    void scatter_recoveries(double* u_buf, size_t n_rec, size_t tr) const;

    // --- Intracellular substance delegates (task14) ---
    /// Scatter X_[subst_idx] for all ComposablePools into buf[net_idx * n_rec + tr]
    void scatter_substances(size_t subst_idx, double* buf, size_t n_rec, size_t tr) const;
    /// Reset buf[net_idx] = 1.0 then let each ComposablePool write its synapse_g_scale
    void scatter_synapse_g_scale(double* buf) const;

    bool empty() const {
        return hh_pool_.empty() && iz_pool_.empty() && comp_pools_.empty();
    }

    // True if any ComposablePool has a SYNAPSE_G modulation (set after build)
    bool has_synapse_g_mods() const { return has_synapse_g_mods_; }

    /// Return intracellular substance concentration for a global neuron index.
    /// Scans all ComposablePools; returns 0 if neuron has no intracellular dynamics.
    double get_substance(size_t global_neuron_idx, size_t subst_idx) const {
        for (const auto& kv : comp_pools_) {
            if (kv.second.contains_neuron(global_neuron_idx))
                return kv.second.get_substance_at(global_neuron_idx, subst_idx);
        }
        return 0.0;
    }

private:
    bool has_synapse_g_mods_ = false;
    HHPool hh_pool_;
    IzPool iz_pool_;
    std::map<std::string, ComposablePool> comp_pools_;
};

} // namespace hodgkin_huxley
