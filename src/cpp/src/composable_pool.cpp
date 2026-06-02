#include "hodgkin_huxley/composable_pool.hpp"
#include "hodgkin_huxley/composable_neuron.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
#include <cstring>
#include <cmath>
#include <limits>

namespace hodgkin_huxley {

ComposablePool::ComposablePool(const NeuronModelSpec& model, size_t capacity, bool fast_math)
    : model_(model), N_(0), fast_math_(fast_math)
{
    net_idx_.reserve(capacity);
    gate_states_.resize(model_.gates.size());

    const size_t ns = model_.intracellular.size();
    X_.resize(ns);
    E_nernst_.resize(ns);

    // Compute flags once — model is fixed for the pool's lifetime.
    // Done here rather than finalize() so add() can gate synapse_g_scale_ allocation.
    has_intracellular_ = ns > 0;
    has_any_mods_ = false;
    has_synapse_g_mods_ = false;
    for (const auto& ic : model_.intracellular)
        for (const auto& mod : ic.modulations) {
            has_any_mods_ = true;
            if (mod.target == IntracellularModulation::Target::SYNAPSE_G)
                has_synapse_g_mods_ = true;
        }
}

ComposablePool::~ComposablePool() = default;

void ComposablePool::add(size_t network_idx, double V_init,
                          const std::vector<double>& gate_inits,
                          const std::vector<double>& substance_inits) {
    size_t i = N_++;
    net_idx_.push_back(network_idx);

    if (i == 0) contiguous_ = true;
    else if (contiguous_ && network_idx != net_idx_[i - 1] + 1) contiguous_ = false;

    // Grow arrays
    V_.conservativeResize(N_);
    I_ext_.conservativeResize(N_);

    V_(i) = V_init;

    if (has_synapse_g_mods_) {
        synapse_g_scale_.conservativeResize(N_);
        synapse_g_scale_(i) = 1.0;
    }

    // Substance state
    const size_t ns = model_.intracellular.size();
    for (size_t si = 0; si < ns; ++si) {
        const auto& ic = model_.intracellular[si];
        X_[si].conservativeResize(N_);
        E_nernst_[si].conservativeResize(N_);

        double x_init = (si < substance_inits.size()) ? substance_inits[si] : ic.initial;
        X_[si](i) = x_init;

        if (ic.nernst_enabled && x_init > 0.0) {
            E_nernst_[si](i) = (ic.nernst_R * ic.nernst_T)
                               / (ic.nernst_z * ic.nernst_F)
                               * std::log(ic.nernst_Ca_o / x_init);
        } else {
            E_nernst_[si](i) = 120.0;  // default safe value
        }
    }

    for (size_t g = 0; g < gate_states_.size(); ++g) {
        gate_states_[g].conservativeResize(N_);
        if (g < gate_inits.size()) {
            gate_states_[g](i) = gate_inits[g];
        } else {
            gate_states_[g](i) = 0.0;
        }
    }

    finalized_ = false;
}

void ComposablePool::finalize() {
    if (finalized_) return;
    I_total_.resize(N_);
    tmp_.resize(N_);
    tmp2_.resize(N_);
    tmp_exp_r_.resize(N_);

    const size_t nc = model_.channels.size();
    const size_t ng = model_.gates.size();

    // Mod scratch arrays: only allocated when modulations are present.
    // Populations with no add_intracellular() modulations pay zero memory cost.
    if (has_any_mods_) {
        mod_ch_g_.assign(nc, Eigen::ArrayXd::Ones(N_));
        mod_ch_E_ovr_.assign(nc, Eigen::ArrayXd());
        mod_gate_shift_.assign(ng, Eigen::ArrayXd::Zero(N_));
        mod_gate_scale_.assign(ng, Eigen::ArrayXd::Ones(N_));
        mod_gate_tau_.assign(ng, Eigen::ArrayXd::Ones(N_));
        mod_gate_expr_.assign(ng, Eigen::ArrayXd());
    }

    // Per-channel current cache: only needed when substances reference source channels.
    if (has_intracellular_) {
        I_channel_.assign(nc, Eigen::ArrayXd::Zero(N_));
        // Unique set of channels that feed any substance's I_source. Their
        // currents are recomputed at the post-V-update voltage each step.
        std::vector<char> seen(nc, 0);
        ca_source_channels_.clear();
        for (const auto& ic : model_.intracellular)
            for (int ch_idx : ic.source_channels)
                if (ch_idx >= 0 && ch_idx < static_cast<int>(nc) && !seen[ch_idx]) {
                    seen[ch_idx] = 1;
                    ca_source_channels_.push_back(ch_idx);
                }
    }

    ch_E_rev_.resize(nc);
    for (size_t ci = 0; ci < nc; ++ci)
        ch_E_rev_[ci] = Eigen::ArrayXd::Constant(N_, model_.channels[ci].E_rev);

    // Flags were computed in the constructor — no recomputation needed.

    finalized_ = true;
}

void ComposablePool::scatter_voltages(double* V_buf) const {
    if (contiguous_ && N_ > 0) {
        std::memcpy(V_buf + net_idx_[0], V_.data(), N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            V_buf[net_idx_[i]] = V_(i);
    }
}

void ComposablePool::gather_currents(const double* I_buf) {
    if (contiguous_ && N_ > 0) {
        std::memcpy(I_ext_.data(), I_buf + net_idx_[0], N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            I_ext_(i) = I_buf[net_idx_[i]];
    }
}

void ComposablePool::fast_exp(const Eigen::ArrayXd& src, Eigen::ArrayXd& dst) {
    hodgkin_huxley::fast_exp(src, dst, tmp_exp_r_);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void ComposablePool::update_substances(double dt, Eigen::ArrayXd* fexp) {
    const size_t ns = model_.intracellular.size();
    if (ns == 0) return;

    // tmp_  = dX scratch (reuse pre-allocated member, avoids per-step malloc)
    // tmp2_ = I_src scratch
    for (size_t si = 0; si < ns; ++si) {
        const auto& ic = model_.intracellular[si];

        using UF = IntracellularSpec::UpdateForm;

        switch (ic.update_form) {
            case UF::DECAY:
                tmp_ = -ic.k_decay * X_[si];  // dX stored in tmp_
                break;

            case UF::DRIVEN_DECAY:
            case UF::DRIVEN_DECAY_NERNST: {
                tmp2_.setZero();  // I_src in tmp2_
                for (int ch_idx : ic.source_channels)
                    if (ch_idx >= 0 && ch_idx < static_cast<int>(I_channel_.size()))
                        tmp2_ += I_channel_[ch_idx];
                tmp_ = ic.epsilon * (-tmp2_ - ic.k_decay * X_[si]);  // dX in tmp_
                break;
            }

            case UF::CUSTOM_EXPR: {
                if (!ic.ode_vm.empty()) {
                    tmp2_.setZero();  // I_src in tmp2_
                    for (int ch_idx : ic.source_channels)
                        if (ch_idx >= 0 && ch_idx < static_cast<int>(I_channel_.size()))
                            tmp2_ += I_channel_[ch_idx];
                    tmp_ = hodgkin_huxley::vm_eval_substance(
                        ic.ode_vm, tmp2_, X_[si], X_, fexp);  // dX in tmp_
                } else {
                    tmp_.setZero();
                }
                break;
            }
        }

        X_[si] += dt * tmp_;
        X_[si] = X_[si].max(0.0);

        // Update Nernst reversal potential
        if (ic.nernst_enabled) {
            if (!ic.nernst_vm.empty()) {
                E_nernst_[si] = hodgkin_huxley::vm_eval_substance(
                    ic.nernst_vm, X_[si], X_[si], X_, fexp);
            } else {
                // Standard Nernst formula: E = (RT/zF) * ln(Ca_o / X)
                // tmp2_ = x_safe (reuse scratch, avoids malloc)
                tmp2_ = X_[si].max(1e-10);
                E_nernst_[si] = (ic.nernst_R * ic.nernst_T)
                                / (ic.nernst_z * ic.nernst_F)
                                * (ic.nernst_Ca_o / tmp2_).log();
            }
        }
    }
}

void ComposablePool::recompute_source_channel_currents(Eigen::ArrayXd* fexp) {
    // Recompute I_channel_ for calcium-source channels at the CURRENT V_ (called
    // after the voltage update). Mirrors the per-channel computation in step()'s
    // main channel loop exactly, but evaluates the driving force (V_ - E_rev) at
    // the post-update voltage. Only source channels are touched — they are the
    // only ones update_substances() reads.
    const size_t ng = model_.gates.size();
    for (int ci : ca_source_channels_) {
        const auto& ch = model_.channels[ci];

        Eigen::ArrayXd gate_prod;
        if (!ch.gate_product_vm.empty()) {
            gate_prod = hodgkin_huxley::vm_eval_gate_product_vec(
                ch.gate_product_vm, V_, gate_states_, fexp);
        } else {
            gate_prod = Eigen::ArrayXd::Ones(N_);
            for (const auto& gp : ch.gates) {
                int idx = gp.first, power = gp.second;
                if (idx >= 0 && idx < static_cast<int>(ng)) {
                    for (int p = 0; p < power; ++p)
                        gate_prod *= gate_states_[idx];
                }
            }
        }

        const Eigen::ArrayXd& E_rev =
            (has_any_mods_ && mod_ch_E_ovr_[ci].size()) ? mod_ch_E_ovr_[ci]
            : (ch.nernst_substance_idx >= 0
               && ch.nernst_substance_idx < static_cast<int>(E_nernst_.size()))
                ? E_nernst_[ch.nernst_substance_idx]
                : ch_E_rev_[ci];

        Eigen::ArrayXd& I_ci = I_channel_[ci];
        if (ch.is_ahp) {
            int aidx = ch.ahp_substance_idx;
            const Eigen::ArrayXd& X_ahp = (aidx >= 0 && aidx < static_cast<int>(X_.size()))
                ? X_[aidx] : (tmp2_.setZero(), tmp2_);
            Eigen::ArrayXd ca_factor = X_ahp / (X_ahp + ch.ahp_k1).max(1e-10);
            if (has_any_mods_)
                I_ci = ch.g * mod_ch_g_[ci] * ca_factor * (V_ - E_rev);
            else
                I_ci = ch.g * ca_factor * (V_ - E_rev);
        } else {
            if (has_any_mods_)
                I_ci = ch.g * mod_ch_g_[ci] * gate_prod * (V_ - E_rev);
            else
                I_ci = ch.g * gate_prod * (V_ - E_rev);
        }
    }
}

void ComposablePool::apply_modulations(Eigen::ArrayXd* fexp)
{
    const size_t ns = model_.intracellular.size();
    for (size_t si = 0; si < ns; ++si) {
        for (const auto& mod : model_.intracellular[si].modulations) {
            int sidx = mod.substance_idx;
            if (sidx < 0 || sidx >= static_cast<int>(X_.size())) continue;
            const Eigen::ArrayXd& Xmod = X_[sidx];

            using T = IntracellularModulation::Target;
            switch (mod.target) {
                case T::CHANNEL_G:
                    if (mod.target_idx >= 0 && mod.target_idx < static_cast<int>(mod_ch_g_.size())) {
                        if (!mod.mod_vm.empty())
                            mod_ch_g_[mod.target_idx] *= vm_eval_substance(mod.mod_vm, Xmod, Xmod, X_, fexp);
                        else if (mod.shift_scale != 0.0)
                            mod_ch_g_[mod.target_idx] *= Eigen::ArrayXd::Constant(N_, mod.shift_scale);
                    }
                    break;

                case T::CHANNEL_EREV:
                    if (mod.target_idx >= 0 && mod.target_idx < static_cast<int>(mod_ch_E_ovr_.size())) {
                        if (!mod.mod_vm.empty())
                            mod_ch_E_ovr_[mod.target_idx] = vm_eval_substance(mod.mod_vm, Xmod, Xmod, X_, fexp);
                    }
                    break;

                case T::GATE_INF_SHIFT:
                    if (mod.target_idx >= 0 && mod.target_idx < static_cast<int>(mod_gate_shift_.size())) {
                        if (!mod.mod_vm.empty())
                            mod_gate_shift_[mod.target_idx] += vm_eval_substance(mod.mod_vm, Xmod, Xmod, X_, fexp);
                        else
                            mod_gate_shift_[mod.target_idx] += mod.shift_scale * Xmod;
                    }
                    break;

                case T::GATE_INF_SCALE:
                    if (mod.target_idx >= 0 && mod.target_idx < static_cast<int>(mod_gate_scale_.size())) {
                        if (!mod.mod_vm.empty())
                            mod_gate_scale_[mod.target_idx] *= vm_eval_substance(mod.mod_vm, Xmod, Xmod, X_, fexp);
                    }
                    break;

                case T::GATE_TAU_SCALE:
                    if (mod.target_idx >= 0 && mod.target_idx < static_cast<int>(mod_gate_tau_.size())) {
                        if (!mod.mod_vm.empty())
                            mod_gate_tau_[mod.target_idx] *= vm_eval_substance(mod.mod_vm, Xmod, Xmod, X_, fexp);
                    }
                    break;

                case T::GATE_INF_EXPR:
                    if (mod.target_idx >= 0 && mod.target_idx < static_cast<int>(mod_gate_expr_.size())) {
                        if (!mod.mod_vm.empty())
                            mod_gate_expr_[mod.target_idx] = vm_eval_substance(mod.mod_vm, V_, Xmod, X_, fexp);
                    }
                    break;

                case T::SYNAPSE_G:
                    if (!mod.mod_vm.empty())
                        synapse_g_scale_ *= vm_eval_substance(mod.mod_vm, Xmod, Xmod, X_, fexp);
                    break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// step()
// ---------------------------------------------------------------------------

void ComposablePool::step(double dt) {
    if (N_ == 0 || dt == 0.0) return;
    finalize();

    const size_t ng = model_.gates.size();
    const size_t nc = model_.channels.size();
    const size_t ns = model_.intracellular.size();

    const bool use_fast = fast_math_ && N_ >= 50;
    Eigen::ArrayXd* fexp = use_fast ? &tmp_exp_r_ : nullptr;

    // =========================================================================
    // 0. Compute modulations from previous step's substance concentrations.
    //    Uses pre-allocated member arrays — no per-step heap allocation.
    // =========================================================================
    if (has_any_mods_) {
        // Reset to identity values
        for (size_t ci = 0; ci < nc; ++ci) {
            mod_ch_g_[ci].setOnes();
            mod_ch_E_ovr_[ci].resize(0);
        }
        for (size_t gi = 0; gi < ng; ++gi) {
            mod_gate_shift_[gi].setZero();
            mod_gate_scale_[gi].setOnes();
            mod_gate_tau_[gi].setOnes();
            mod_gate_expr_[gi].resize(0);
        }
        if (has_synapse_g_mods_)
            synapse_g_scale_.setOnes();
        apply_modulations(fexp);
    }

    // =========================================================================
    // 1. Update gates (using modulation arrays from above)
    // =========================================================================
    for (size_t gi = 0; gi < ng; ++gi) {
        const auto& gs = model_.gates[gi];
        Eigen::ArrayXd& X_gate = gate_states_[gi];

        // Dependency: voltage or intracellular substance (skip shift when no mods)
        auto get_dep = [&]() -> Eigen::ArrayXd {
            if (gs.dependency == GateSpec::Dependency::INTRACELLULAR) {
                int iidx = gs.intracellular_idx;
                if (iidx >= 0 && iidx < static_cast<int>(X_.size()))
                    return has_any_mods_ ? X_[iidx] + mod_gate_shift_[gi]
                                         : Eigen::ArrayXd(X_[iidx]);
                return Eigen::ArrayXd::Zero(N_);
            }
            return has_any_mods_ ? V_ + mod_gate_shift_[gi] : Eigen::ArrayXd(V_);
        };

        switch (gs.update_form) {
            case GateSpec::UpdateForm::INF_TAU: {
                Eigen::ArrayXd dep = get_dep();
                Eigen::ArrayXd x_inf = hodgkin_huxley::boltzmann_vec(dep, gs.inf);
                if (has_any_mods_) x_inf *= mod_gate_scale_[gi];
                Eigen::ArrayXd tau_x = hodgkin_huxley::compute_tau_vec(V_, gs.tau, tmp_);
                if (has_any_mods_) tau_x *= mod_gate_tau_[gi];
                tau_x = tau_x.max(1e-10);
                if (has_any_mods_ && mod_gate_expr_[gi].size())
                    x_inf = mod_gate_expr_[gi];
                if (use_fast) {
                    tmp_ = -dt * gs.scale / tau_x;
                    fast_exp(tmp_, tmp_);
                    X_gate = x_inf + (X_gate - x_inf) * tmp_;
                } else {
                    X_gate = x_inf + (X_gate - x_inf) * (-dt * gs.scale / tau_x).exp();
                }
                break;
            }

            case GateSpec::UpdateForm::ALPHA_BETA: {
                Eigen::ArrayXd alpha = hodgkin_huxley::compute_rate_vec(V_, gs.alpha, tmp_);
                Eigen::ArrayXd beta  = hodgkin_huxley::compute_rate_vec(V_, gs.beta,  tmp2_);
                Eigen::ArrayXd rate  = (alpha + beta).max(1e-10);
                Eigen::ArrayXd x_inf = alpha / rate;
                if (has_any_mods_) x_inf *= mod_gate_scale_[gi];
                Eigen::ArrayXd tau_x = 1.0 / rate;
                if (has_any_mods_) tau_x *= mod_gate_tau_[gi];
                if (has_any_mods_ && mod_gate_expr_[gi].size())
                    x_inf = mod_gate_expr_[gi];
                if (use_fast) {
                    tmp_ = -dt / tau_x;
                    fast_exp(tmp_, tmp_);
                    X_gate = x_inf + (X_gate - x_inf) * tmp_;
                } else {
                    X_gate = x_inf + (X_gate - x_inf) * (-dt / tau_x).exp();
                }
                break;
            }

            case GateSpec::UpdateForm::INSTANT: {
                Eigen::ArrayXd dep = get_dep();
                if (has_any_mods_ && mod_gate_expr_[gi].size()) {
                    X_gate = mod_gate_expr_[gi];
                } else {
                    X_gate = hodgkin_huxley::boltzmann_vec(dep, gs.inf);
                    if (has_any_mods_) X_gate *= mod_gate_scale_[gi];
                }
                break;
            }

            case GateSpec::UpdateForm::DERIVED: {
                int src = gs.derived_source_gate;
                if (src >= 0 && src < static_cast<int>(ng))
                    X_gate = gs.derived_a * (gs.derived_b + gs.derived_c * gate_states_[src]);
                break;
            }

            case GateSpec::UpdateForm::CUSTOM_EXPR: {
                Eigen::ArrayXd dep = get_dep();
                if (!gs.dxdt_vm.empty()) {
                    auto dxdt = hodgkin_huxley::vm_eval_vec_2arg(gs.dxdt_vm, dep, X_gate, fexp);
                    X_gate += dt * gs.scale * dxdt;
                } else if (!gs.inf_vm.empty() && !gs.tau_vm.empty()) {
                    Eigen::ArrayXd x_inf = hodgkin_huxley::vm_eval_vec(gs.inf_vm, dep, fexp);
                    if (has_any_mods_) {
                        x_inf *= mod_gate_scale_[gi];
                        if (mod_gate_expr_[gi].size()) x_inf = mod_gate_expr_[gi];
                    }
                    Eigen::ArrayXd tau_x = hodgkin_huxley::vm_eval_vec(gs.tau_vm, V_, fexp).max(1e-10);
                    if (has_any_mods_) tau_x *= mod_gate_tau_[gi];
                    if (use_fast) {
                        tmp_ = -dt * gs.scale / tau_x;
                        fast_exp(tmp_, tmp_);
                        X_gate = x_inf + (X_gate - x_inf) * tmp_;
                    } else {
                        X_gate = x_inf + (X_gate - x_inf) * (-dt * gs.scale / tau_x).exp();
                    }
                } else if (!gs.alpha_vm.empty() && !gs.beta_vm.empty()) {
                    auto alpha = hodgkin_huxley::vm_eval_vec(gs.alpha_vm, V_, fexp);
                    auto beta  = hodgkin_huxley::vm_eval_vec(gs.beta_vm,  V_, fexp);
                    auto rate  = (alpha + beta).max(1e-10);
                    Eigen::ArrayXd x_inf = alpha / rate;
                    if (has_any_mods_) {
                        x_inf *= mod_gate_scale_[gi];
                        if (mod_gate_expr_[gi].size()) x_inf = mod_gate_expr_[gi];
                    }
                    if (has_any_mods_) {
                        if (use_fast) {
                            tmp_ = -dt * rate * mod_gate_tau_[gi];
                            fast_exp(tmp_, tmp_);
                        } else {
                            tmp_ = (-dt * rate * mod_gate_tau_[gi]).eval();
                        }
                    } else {
                        if (use_fast) tmp_ = -dt * rate;
                        else tmp_ = (-dt * rate).eval();
                    }
                    if (use_fast) {
                        fast_exp(tmp_, tmp_);
                        X_gate = x_inf + (X_gate - x_inf) * tmp_;
                    } else {
                        X_gate = x_inf + (X_gate - x_inf) * tmp_.exp();
                    }
                } else if (!gs.inf_vm.empty()) {
                    Eigen::ArrayXd x_inf = hodgkin_huxley::vm_eval_vec(gs.inf_vm, dep, fexp);
                    if (has_any_mods_) {
                        x_inf *= mod_gate_scale_[gi];
                        if (mod_gate_expr_[gi].size()) { X_gate = mod_gate_expr_[gi]; break; }
                    }
                    X_gate = x_inf;
                }
                break;
            }
        }

        X_gate = X_gate.max(0.0).min(1.0);
    }

    // =========================================================================
    // 2. Compute total ionic current; cache per-channel for substance ODE reuse
    // =========================================================================
    I_total_.setZero();

    for (size_t ci = 0; ci < nc; ++ci) {
        const auto& ch = model_.channels[ci];

        // Gate product
        Eigen::ArrayXd gate_prod;
        if (!ch.gate_product_vm.empty()) {
            gate_prod = hodgkin_huxley::vm_eval_gate_product_vec(
                ch.gate_product_vm, V_, gate_states_, fexp);
        } else {
            gate_prod = Eigen::ArrayXd::Ones(N_);
            for (const auto& gp : ch.gates) {
                int idx = gp.first, power = gp.second;
                if (idx >= 0 && idx < static_cast<int>(ng)) {
                    for (int p = 0; p < power; ++p)
                        gate_prod *= gate_states_[idx];
                }
            }
        }

        // Reversal potential — ref to pre-allocated array, zero mallocs per step
        const Eigen::ArrayXd& E_rev =
            (has_any_mods_ && mod_ch_E_ovr_[ci].size()) ? mod_ch_E_ovr_[ci]
            : (ch.nernst_substance_idx >= 0
               && ch.nernst_substance_idx < static_cast<int>(E_nernst_.size()))
                ? E_nernst_[ch.nernst_substance_idx]
                : ch_E_rev_[ci];

        // Output buffer: I_channel_[ci] when substances need the cache; tmp_ otherwise.
        // tmp_ is free at this point (gate loop is done) and pre-allocated — zero malloc.
        Eigen::ArrayXd& I_ci = has_intracellular_ ? I_channel_[ci] : tmp_;
        if (ch.is_ahp) {
            int aidx = ch.ahp_substance_idx;
            const Eigen::ArrayXd& X_ahp = (aidx >= 0 && aidx < static_cast<int>(X_.size()))
                ? X_[aidx] : (tmp2_.setZero(), tmp2_);
            Eigen::ArrayXd ca_factor = X_ahp / (X_ahp + ch.ahp_k1).max(1e-10);
            if (has_any_mods_)
                I_ci = ch.g * mod_ch_g_[ci] * ca_factor * (V_ - E_rev);
            else
                I_ci = ch.g * ca_factor * (V_ - E_rev);
        } else {
            if (has_any_mods_)
                I_ci = ch.g * mod_ch_g_[ci] * gate_prod * (V_ - E_rev);
            else
                I_ci = ch.g * gate_prod * (V_ - E_rev);
        }
        I_total_ += I_ci;
    }

    // =========================================================================
    // 3. Voltage update: dV/dt = (-I_ion + I_ext) / C_m
    // =========================================================================
    V_ += dt * (-I_total_ + I_ext_) / model_.C_m;

    // =========================================================================
    // 4. Update intracellular substances (uses post-voltage-update gate/channel state)
    // =========================================================================
    // Recompute calcium source-channel currents at the post-update V so the
    // substance ODE's I_source uses the new voltage's driving force. The main
    // loop above cached I_channel_ at the PRE-update V (needed for the V step);
    // reusing that stale cache here shifts Ca2+-dependent dynamics (STN drift).
    if (has_intracellular_)
        recompute_source_channel_currents(fexp);

    update_substances(dt, fexp);

    // synapse_g_scale_ is already populated from apply_modulations (step 0),
    // which used substance values from the previous step. That is the correct
    // one-step-lag behaviour — Network reads it after step_all().
}

// ---------------------------------------------------------------------------
// Recording scatter methods
// ---------------------------------------------------------------------------

void ComposablePool::scatter_gate_states_into(double* buf, size_t max_gates,
                                               size_t n_rec, size_t t_rec) const {
    const size_t ng = model_.gates.size();
    for (size_t i = 0; i < N_; ++i) {
        size_t net_i = net_idx_[i];
        for (size_t g = 0; g < ng; ++g) {
            buf[net_i * max_gates * n_rec + g * n_rec + t_rec] = gate_states_[g](i);
        }
    }
}

void ComposablePool::scatter_substance_into(size_t subst_idx, double* buf,
                                             size_t n_rec, size_t t_rec) const {
    if (subst_idx >= X_.size()) return;
    const auto& X = X_[subst_idx];
    for (size_t i = 0; i < N_; ++i) {
        buf[net_idx_[i] * n_rec + t_rec] = X(i);
    }
}

void ComposablePool::scatter_synapse_g_scale(double* buf) const {
    if (!has_synapse_g_mods_) return;  // array not allocated — nothing to scatter
    for (size_t i = 0; i < N_; ++i) {
        buf[net_idx_[i]] = synapse_g_scale_(i);
    }
}

// ---------------------------------------------------------------------------
// sync_to_neurons — syncs pool state back to ComposableNeuron objects
// ---------------------------------------------------------------------------

void ComposablePool::sync_to_neurons(
    std::vector<std::unique_ptr<NeuronBase>>& neurons) const
{
    const size_t ng = model_.gates.size();
    const size_t ns = model_.intracellular.size();

    for (size_t i = 0; i < N_; ++i) {
        auto* cn = dynamic_cast<ComposableNeuron*>(neurons[net_idx_[i]].get());
        if (!cn) continue;

        cn->set_membrane_potential(V_(i));

        std::vector<double> gs(ng);
        for (size_t g = 0; g < ng; ++g)
            gs[g] = gate_states_[g](i);
        cn->set_gate_states(gs);

        // Sync substance concentrations — keep backward compat accessor alive
        std::vector<double> xs(ns), es(ns, 120.0);
        for (size_t si = 0; si < ns; ++si) {
            xs[si] = X_[si](i);
            es[si] = E_nernst_[si](i);
        }
        cn->set_substances(xs, es);
    }
}

} // namespace hodgkin_huxley
