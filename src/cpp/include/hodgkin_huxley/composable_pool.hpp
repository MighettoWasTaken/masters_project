#pragma once

#include "hodgkin_huxley/ion_channels.hpp"
#include "hodgkin_huxley/neuron_base.hpp"
#include "hodgkin_huxley/pool/pool_base.hpp"
#include <Eigen/Core>
#include <vector>
#include <memory>

namespace hodgkin_huxley {

class ComposablePool : public PoolBase {
public:
    ComposablePool() = default;
    explicit ComposablePool(const NeuronModelSpec& model, size_t capacity, bool fast_math = true);

    void add(size_t network_idx, double V_init,
             const std::vector<double>& gate_inits,
             const std::vector<double>& substance_inits);

    size_t size() const override { return N_; }
    bool empty() const { return N_ == 0; }
    size_t network_idx(size_t pool_idx) const { return net_idx_[pool_idx]; }

    void scatter_voltages(double* V_buf) const override;
    void gather_currents(const double* I_buf) override;
    void step(double dt) override;
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const override;

    // Write gate_states_[g][i] into buf[net_idx_[i] * max_gates * n_rec + g * n_rec + t_rec]
    void scatter_gate_states_into(double* buf, size_t max_gates,
                                   size_t n_rec, size_t t_rec) const override;

    // Write X_[subst_idx][i] into buf[net_idx_[i] * n_rec + t_rec]
    void scatter_substance_into(size_t subst_idx, double* buf,
                                size_t n_rec, size_t t_rec) const override;

    // Write synapse_g_scale_[i] into buf[net_idx_[i]]
    void scatter_synapse_g_scale(double* buf) const override;

    int    n_gates() const override { return static_cast<int>(model_.gates.size()); }
    int    n_substances() const override { return static_cast<int>(model_.intracellular.size()); }
    bool   has_calcium() const override { return !model_.intracellular.empty(); }
    bool   has_synapse_g_mods() const { return has_synapse_g_mods_; }

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

    // Intracellular substance state (one ArrayXd per IntracellularSpec)
    std::vector<Eigen::ArrayXd> X_;         // concentrations
    std::vector<Eigen::ArrayXd> E_nernst_;  // Nernst reversals (for specs with nernst_enabled)

    // Per-neuron synaptic g multiplier (written by SYNAPSE_G modulations; read by Network)
    Eigen::ArrayXd synapse_g_scale_;

    Eigen::ArrayXd I_ext_;

    // Pre-allocated temporaries (resized to N_)
    Eigen::ArrayXd I_total_, tmp_, tmp2_;

    Eigen::ArrayXd tmp_exp_r_;
    // fast_exp wraps hodgkin_huxley::fast_exp using tmp_exp_r_ as scratch
    void fast_exp(const Eigen::ArrayXd& src, Eigen::ArrayXd& dst);

    // Called after last add() to trim arrays to N_
    bool finalized_ = false;
    void finalize();

    // Pre-allocated modulation scratch arrays — reset each step, no per-step malloc
    std::vector<Eigen::ArrayXd> mod_ch_g_;       // nc × N_, ones each step
    std::vector<Eigen::ArrayXd> mod_ch_E_ovr_;   // nc, empty = not overridden
    std::vector<Eigen::ArrayXd> mod_gate_shift_; // ng × N_, zeros each step
    std::vector<Eigen::ArrayXd> mod_gate_scale_; // ng × N_, ones each step
    std::vector<Eigen::ArrayXd> mod_gate_tau_;   // ng × N_, ones each step
    std::vector<Eigen::ArrayXd> mod_gate_expr_;  // ng, empty = not overridden

    // Pre-filled constant E_rev arrays per channel — eliminates Constant() malloc per step
    std::vector<Eigen::ArrayXd> ch_E_rev_;       // nc × N_, filled at finalize()

    // Per-channel currents cached during channel loop, reused by update_substances
    std::vector<Eigen::ArrayXd> I_channel_;      // nc × N_

    bool has_any_mods_      = false; // any substance has modulations
    bool has_synapse_g_mods_ = false; // any substance has SYNAPSE_G modulation

    // Helpers for substance dynamics
    void update_substances(double dt, Eigen::ArrayXd* fexp);
    void apply_modulations(Eigen::ArrayXd* fexp);
};

} // namespace hodgkin_huxley
