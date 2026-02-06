#pragma once

#include "neuron.hpp"
#include <Eigen/Core>
#include <vector>
#include <memory>

namespace hodgkin_huxley {

/**
 * @brief Batched pool for Hodgkin-Huxley neurons using Eigen SIMD
 *
 * Processes all HH neurons simultaneously via vectorized array operations.
 * All working buffers are pre-allocated in the constructor so the hot loop
 * does zero heap allocations.
 *
 * Used internally by Network::simulate() for the hot loop.
 */
class HHPool {
public:
    HHPool() = default;
    explicit HHPool(size_t capacity);

    void add(size_t network_idx, const HHNeuron::Parameters& params,
             const HHNeuron::State& state);

    size_t size() const { return N_; }
    bool empty() const { return N_ == 0; }
    size_t network_idx(size_t pool_idx) const { return net_idx_[pool_idx]; }

    void scatter_voltages(double* V_buf) const;
    void gather_currents(const double* I_buf);
    void step_rk4(double dt);
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const;

private:
    size_t N_ = 0;
    std::vector<size_t> net_idx_;

    // State (SoA)
    Eigen::ArrayXd V_, m_, h_, n_;

    // Parameters (SoA)
    Eigen::ArrayXd C_m_, g_Na_, g_K_, g_L_, E_Na_, E_K_, E_L_;

    // Per-step current input
    Eigen::ArrayXd I_ext_;

    // =========================================================================
    // Pre-allocated working buffers (zero heap allocation in hot loop)
    // =========================================================================

    // RK4: current k-vector (overwritten each stage)
    Eigen::ArrayXd kV_, km_, kh_, kn_;

    // RK4: running accumulator  k1 + 2*k2 + 2*k3 + k4
    Eigen::ArrayXd accV_, accm_, acch_, accn_;

    // RK4: saved original state
    Eigen::ArrayXd V0_, m0_, h0_, n0_;

    // Derivative intermediates (reused across all 4 RK4 stages)
    Eigen::ArrayXd tmp_n2_, tmp_Vp65_, tmp_dVm_, tmp_dVn_;

    // Rate function buffers — materialized for guaranteed SIMD packet eval
    Eigen::ArrayXd tmp_am_, tmp_bm_, tmp_ah_, tmp_bh_, tmp_an_, tmp_bn_;

    // True when pool indices are [start, start+1, ..., start+N-1]
    bool contiguous_ = false;

    // Writes derivatives into kV_/km_/kh_/kn_ (passed as dV/dm/dh/dn).
    // Non-const because it writes to tmp_ members.
    void compute_derivatives(
        const Eigen::ArrayXd& V, const Eigen::ArrayXd& m,
        const Eigen::ArrayXd& h, const Eigen::ArrayXd& n,
        const Eigen::ArrayXd& I_ext,
        Eigen::ArrayXd& dV, Eigen::ArrayXd& dm,
        Eigen::ArrayXd& dh, Eigen::ArrayXd& dn);
};

} // namespace hodgkin_huxley
