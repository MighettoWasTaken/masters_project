#pragma once

// =============================================================================
// pool/pool_base.hpp — abstract pool interface
//
// Defines the behavioral contract shared by HHPool, IzPool, and ComposablePool.
// All pools manage a batch of neurons of one type, using Eigen SoA arrays.
//
// Insertion points:
//   task16 (OpenMP): replace step() in each pool with a parallel version
//   task17 (CUDA):   add a CudaHHPool / CudaComposablePool that overrides step()
//   task20 (continuable): PoolManager holds PoolBase* across simulate() calls
// =============================================================================

#include "hodgkin_huxley/neuron_base.hpp"
#include <memory>
#include <vector>

namespace hodgkin_huxley {

class PoolBase {
public:
    virtual ~PoolBase() = default;

    // --- Lifecycle ---
    virtual void reset() {}  // optional: reset internal state to initial values

    // --- Per-step interface (called by Network hot loop) ---
    virtual void scatter_voltages(double* V_buf) const = 0;
    virtual void gather_currents(const double* I_buf) = 0;
    virtual void step(double dt) = 0;
    virtual void sync_to_neurons(
        std::vector<std::unique_ptr<NeuronBase>>& neurons) const = 0;

    // --- Recording (called when t % interval == 0; default: no-op) ---
    virtual void scatter_gate_states_into(double* /*gate_buf*/, size_t /*max_gates*/,
                                          size_t /*n_rec*/, size_t /*t_rec*/) const {}
    virtual void scatter_calcium_into(double* /*ca_buf*/,
                                      size_t /*n_rec*/, size_t /*t_rec*/) const {}
    virtual void scatter_recoveries(double* /*u_buf*/,
                                    size_t /*n_rec*/, size_t /*tr*/) const {}

    // --- Metadata ---
    virtual size_t size() const = 0;
    virtual int    n_gates() const { return 0; }
    virtual bool   has_calcium() const { return false; }
};

} // namespace hodgkin_huxley
