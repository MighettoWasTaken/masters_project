#pragma once

#include "ion_channels.hpp"
#include "neuron_base.hpp"
#include <Eigen/Core>
#include <vector>
#include <memory>

namespace hodgkin_huxley {

class ComposablePool {
public:
    ComposablePool() = default;
    explicit ComposablePool(const NeuronModelSpec& model, size_t capacity, bool fast_math = true);

    void add(size_t network_idx, double V_init,
             const std::vector<double>& gate_inits, double Ca_init);

    size_t size() const { return N_; }
    bool empty() const { return N_ == 0; }
    size_t network_idx(size_t pool_idx) const { return net_idx_[pool_idx]; }

    void scatter_voltages(double* V_buf) const;
    void gather_currents(const double* I_buf);
    void step(double dt);
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const;

    // Write gate_states_[g][i] into buf[net_idx_[i] * max_gates * n_rec + g * n_rec + t_rec]
    void scatter_gate_states_into(double* buf, size_t max_gates,
                                   size_t n_rec, size_t t_rec) const;

    // Write Ca_[i] into buf[net_idx_[i] * n_rec + t_rec]
    void scatter_calcium_into(double* buf, size_t n_rec, size_t t_rec) const;

    size_t n_gates() const { return model_.gates.size(); }

    const NeuronModelSpec& model() const { return model_; }

private:
    NeuronModelSpec model_;
    size_t N_ = 0;
    bool fast_math_ = true;
    std::vector<size_t> net_idx_;
    bool contiguous_ = false;

    // All arrays are resized to exactly N_ (conservative_resize on add)
    Eigen::ArrayXd V_;
    std::vector<Eigen::ArrayXd> gate_states_;
    Eigen::ArrayXd Ca_;
    Eigen::ArrayXd E_Ca_;
    Eigen::ArrayXd I_ext_;

    // Pre-allocated temporaries (resized to N_)
    Eigen::ArrayXd I_total_, tmp_, tmp2_;

    // Helpers (work on full ArrayXd, no head())
    static Eigen::ArrayXd boltzmann_vec(const Eigen::ArrayXd& x, const BoltzmannParams& p);
    static Eigen::ArrayXd compute_tau_vec(const Eigen::ArrayXd& V, const TauParams& tau,
                                           Eigen::ArrayXd& tmp);
    static Eigen::ArrayXd compute_rate_vec(const Eigen::ArrayXd& V, const RateFuncParams& rate,
                                            Eigen::ArrayXd& tmp);

    Eigen::ArrayXd tmp_exp_r_;
    void fast_exp(const Eigen::ArrayXd& src, Eigen::ArrayXd& dst);

    // Called after last add() to trim arrays to N_
    bool finalized_ = false;
    void finalize();
};

} // namespace hodgkin_huxley
