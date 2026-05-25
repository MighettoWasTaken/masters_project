#pragma once

#include "hodgkin_huxley/izhikevich.hpp"
#include "hodgkin_huxley/pool/pool_base.hpp"
#include <Eigen/Core>
#include <vector>
#include <memory>

namespace hodgkin_huxley {

/**
 * @brief Batched pool for Izhikevich neurons using Eigen SIMD
 *
 * Processes all Izhikevich neurons simultaneously via vectorized array
 * operations.  Spike detection and reset are handled branchlessly using
 * Eigen mask operations (select).  All working buffers are pre-allocated.
 *
 * Used internally by Network::simulate() for the hot loop.
 */
class IzPool : public PoolBase {
public:
    virtual ~IzPool();
    IzPool() = default;
    explicit IzPool(size_t capacity);

    void add(size_t network_idx, const IzhikevichNeuron::Parameters& params,
             const IzhikevichNeuron::State& state);

    size_t size() const override { return N_; }
    bool empty() const { return N_ == 0; }
    size_t network_idx(size_t pool_idx) const { return net_idx_[pool_idx]; }

    void scatter_voltages(double* V_buf) const override;
    void scatter_recoveries(double* u_buf, size_t n_rec, size_t tr) const override;
    void gather_currents(const double* I_buf) override;
    // PoolBase::step() delegates to step_euler()
    void step(double dt) override { step_euler(dt); }
    void step_euler(double dt);
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const override;

    // Per-group subset ops for Phase 2: operate only on specified pool-local indices.
    void gather_currents_subset(const std::vector<size_t>& local_indices, const double* I_buf);
    void step_subset(const std::vector<size_t>& local_indices, double dt);
    void scatter_voltages_subset(const std::vector<size_t>& local_indices, double* V_buf) const;
    void scatter_recoveries_subset(const std::vector<size_t>& local_indices,
                                   double* u_buf, size_t n_rec, size_t tr) const;

protected:
    size_t N_ = 0;
    std::vector<size_t> net_idx_;

    // State (SoA)
    Eigen::ArrayXd v_, u_;

    // Parameters (SoA)
    Eigen::ArrayXd a_, b_, c_, d_;

    // Per-step current input
    Eigen::ArrayXd I_ext_;

    // Pre-allocated working buffers
    Eigen::ArrayXd dv_, du_;

    static constexpr double SPIKE_THRESHOLD = 30.0;

    // True when pool indices are [start, start+1, ..., start+N-1]
    bool contiguous_ = false;
};

} // namespace hodgkin_huxley
