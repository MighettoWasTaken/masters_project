#include "hodgkin_huxley/network/pool_manager.hpp"
#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/izhikevich.hpp"
#include "hodgkin_huxley/composable_neuron.hpp"
#ifdef HH_USE_CUDA
#  include <cuda_runtime_api.h>
#  include "hodgkin_huxley/cuda_sim_all.hpp"
#endif
#include <map>
#ifdef HH_USE_OPENMP
#  include <omp.h>
#endif

namespace hodgkin_huxley {

PoolManager::~PoolManager() = default;

void PoolManager::build_from_neurons(
    const std::vector<std::unique_ptr<NeuronBase>>& neurons, bool fast_math)
{
    const size_t nn = neurons.size();

    // Count by type
    size_t n_hh = 0, n_iz = 0;
    std::map<std::string, size_t> composable_counts;
    for (const auto& n : neurons) {
        if (dynamic_cast<HHNeuron*>(n.get()))               ++n_hh;
        else if (dynamic_cast<IzhikevichNeuron*>(n.get()))  ++n_iz;
        else if (auto* cn = dynamic_cast<ComposableNeuron*>(n.get()))
            composable_counts[cn->pool_key()]++;
    }

    // Construct pools sized to capacity — CUDA variants when device is set
    comp_pools_.clear();
#ifdef HH_USE_CUDA
    if (use_cuda_) {
        hh_pool_ = std::make_unique<CudaHHPool>(device_id_, n_hh, fast_math);
        iz_pool_ = std::make_unique<CudaIzPool>(device_id_, n_iz);
        for (const auto& kv : composable_counts) {
            for (const auto& n : neurons) {
                auto* cn = dynamic_cast<ComposableNeuron*>(n.get());
                if (cn && cn->pool_key() == kv.first) {
                    comp_pools_.emplace(kv.first,
                        std::make_unique<CudaComposablePool>(
                            device_id_, cn->model_spec(), kv.second, fast_math));
                    break;
                }
            }
        }
    } else {
#endif
        hh_pool_ = std::make_unique<HHPool>(n_hh, fast_math);
        iz_pool_ = std::make_unique<IzPool>(n_iz);
        for (const auto& kv : composable_counts) {
            for (const auto& n : neurons) {
                auto* cn = dynamic_cast<ComposableNeuron*>(n.get());
                if (cn && cn->pool_key() == kv.first) {
                    comp_pools_.emplace(kv.first,
                        std::make_unique<ComposablePool>(
                            cn->model_spec(), kv.second, fast_math));
                    break;
                }
            }
        }
#ifdef HH_USE_CUDA
    }
#endif

    // Set synapse_g_mods flag
    has_synapse_g_mods_ = false;
    for (const auto& kv : comp_pools_)
        if (kv.second->has_synapse_g_mods()) { has_synapse_g_mods_ = true; break; }

    // Populate with current neuron state; build global→local index maps
    hh_global_to_local_.clear();
    iz_global_to_local_.clear();
    for (size_t i = 0; i < nn; ++i) {
        if (auto* hh = dynamic_cast<HHNeuron*>(neurons[i].get())) {
            hh_global_to_local_[i] = hh_pool_->size();
            hh_pool_->add(i, hh->parameters(), hh->state());
        } else if (auto* iz = dynamic_cast<IzhikevichNeuron*>(neurons[i].get())) {
            iz_global_to_local_[i] = iz_pool_->size();
            iz_pool_->add(i, iz->parameters(), iz->state());
        } else if (auto* cn = dynamic_cast<ComposableNeuron*>(neurons[i].get())) {
            auto it = comp_pools_.find(cn->pool_key());
            if (it != comp_pools_.end())
                it->second->add(i, cn->membrane_potential(),
                                cn->gate_states(), cn->substances());
        }
    }
}

void PoolManager::scatter_all_voltages(double* V_cache) const {
    hh_pool_->scatter_voltages(V_cache);
    iz_pool_->scatter_voltages(V_cache);
    for (const auto& kv : comp_pools_) kv.second->scatter_voltages(V_cache);
}

void PoolManager::gather_all_currents(const double* I_buf) {
    hh_pool_->gather_currents(I_buf);
    iz_pool_->gather_currents(I_buf);
    for (auto& kv : comp_pools_) kv.second->gather_currents(I_buf);
}

void PoolManager::scatter_all_voltages_device(double* d_V_cache) const {
#ifdef HH_USE_CUDA
    if (!use_cuda_) return;
    if (auto* hh = dynamic_cast<CudaHHPool*>(hh_pool_.get()))
        hh->scatter_voltages_device(d_V_cache);
    if (auto* iz = dynamic_cast<CudaIzPool*>(iz_pool_.get()))
        iz->scatter_voltages_device(d_V_cache);
    for (const auto& kv : comp_pools_) {
        if (auto* cp = dynamic_cast<CudaComposablePool*>(kv.second.get()))
            cp->scatter_voltages_device(d_V_cache);
    }
#else
    (void)d_V_cache;
#endif
}

void PoolManager::gather_all_currents_device(const double* d_I_buf) {
#ifdef HH_USE_CUDA
    if (!use_cuda_) return;
    if (auto* hh = dynamic_cast<CudaHHPool*>(hh_pool_.get()))
        hh->gather_currents_device(d_I_buf);
    if (auto* iz = dynamic_cast<CudaIzPool*>(iz_pool_.get()))
        iz->gather_currents_device(d_I_buf);
    for (auto& kv : comp_pools_) {
        if (auto* cp = dynamic_cast<CudaComposablePool*>(kv.second.get()))
            cp->gather_currents_device(d_I_buf);
    }
#else
    (void)d_I_buf;
#endif
}

void PoolManager::step_all(double dt) {
#ifdef HH_USE_OPENMP
    if (num_threads_ > 0) {
        // All pools are independent — one parallel for covers HH (slot 0),
        // Iz (slot 1), and all composable pools (slots 2+). One OpenMP region
        // per call avoids a second fork-join barrier.
        std::vector<ComposablePool*> cpools;
        cpools.reserve(comp_pools_.size());
        for (auto& kv : comp_pools_) cpools.push_back(kv.second.get());
        const int n = static_cast<int>(cpools.size()) + 2;
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
        for (int i = 0; i < n; ++i) {
            if (i == 0)      hh_pool_->step(dt);
            else if (i == 1) iz_pool_->step(dt);
            else             cpools[i - 2]->step(dt);
        }
        synchronize_cuda();
        return;
    }
#endif
    hh_pool_->step(dt);
    iz_pool_->step(dt);
    for (auto& kv : comp_pools_) kv.second->step(dt);
    synchronize_cuda();
}

void PoolManager::sync_all_to_neurons(
    std::vector<std::unique_ptr<NeuronBase>>& neurons) const
{
    hh_pool_->sync_to_neurons(neurons);
    iz_pool_->sync_to_neurons(neurons);
    for (const auto& kv : comp_pools_) kv.second->sync_to_neurons(neurons);
}

void PoolManager::scatter_gates(double* gate_buf, size_t max_gates,
                                 size_t n_rec, size_t tr) const {
    for (const auto& kv : comp_pools_)
        kv.second->scatter_gate_states_into(gate_buf, max_gates, n_rec, tr);
}

void PoolManager::scatter_calcium(double* ca_buf, size_t n_rec, size_t tr) const {
    for (const auto& kv : comp_pools_)
        kv.second->scatter_calcium_into(ca_buf, n_rec, tr);
}

void PoolManager::scatter_recoveries(double* u_buf, size_t n_rec, size_t tr) const {
    iz_pool_->scatter_recoveries(u_buf, n_rec, tr);
}

void PoolManager::scatter_substances(size_t subst_idx, double* buf,
                                      size_t n_rec, size_t tr) const {
    for (const auto& kv : comp_pools_)
        kv.second->scatter_substance_into(subst_idx, buf, n_rec, tr);
}

void PoolManager::scatter_synapse_g_scale(double* buf) const {
    for (const auto& kv : comp_pools_)
        kv.second->scatter_synapse_g_scale(buf);
}

void PoolManager::scatter_synapse_g_scale_device(double* d_buf) const {
#ifdef HH_USE_CUDA
    if (!use_cuda_) return;
    for (const auto& kv : comp_pools_) {
        if (auto* cp = dynamic_cast<CudaComposablePool*>(kv.second.get()))
            cp->scatter_synapse_g_scale_device(d_buf);
    }
#else
    (void)d_buf;
#endif
}

// =============================================================================
// CUDA device routing (task17)
// =============================================================================

void PoolManager::assign_to_device(int id) {
    use_cuda_  = true;
    device_id_ = id;
    // Migrate already-built pools (no-op for CPU pools; meaningful for real CUDA pools)
    if (hh_pool_) hh_pool_->migrate_to_device(id);
    if (iz_pool_) iz_pool_->migrate_to_device(id);
    for (auto& kv : comp_pools_) kv.second->migrate_to_device(id);
}

void PoolManager::assign_to_cpu() {
    use_cuda_  = false;
    device_id_ = 0;
}

void PoolManager::synchronize_cuda() const {
#ifdef HH_USE_CUDA
    if (use_cuda_) {
        const auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("PoolManager::synchronize_cuda failed: ") +
                cudaGetErrorString(err));
        }
    }
#endif
}

// =============================================================================
// Per-group ops for Phase 2 delay-decomposition parallelism
// =============================================================================

void PoolManager::step_for_names(const std::vector<std::string>& names, double dt) {
    for (const auto& name : names) {
        auto it = comp_pools_.find(name);
        if (it != comp_pools_.end()) it->second->step(dt);
    }
}

void PoolManager::scatter_voltages_for_names(const std::vector<std::string>& names,
                                              double* V_cache) const {
    for (const auto& name : names) {
        auto it = comp_pools_.find(name);
        if (it != comp_pools_.end()) it->second->scatter_voltages(V_cache);
    }
}

void PoolManager::gather_currents_for_names(const std::vector<std::string>& names,
                                             const double* I_buf) {
    for (const auto& name : names) {
        auto it = comp_pools_.find(name);
        if (it != comp_pools_.end()) it->second->gather_currents(I_buf);
    }
}

void PoolManager::scatter_synapse_g_scale_for_names(const std::vector<std::string>& names,
                                                     double* buf) const {
    for (const auto& name : names) {
        auto it = comp_pools_.find(name);
        if (it != comp_pools_.end()) it->second->scatter_synapse_g_scale(buf);
    }
}

// =============================================================================
// HHPool / IzPool per-group subset ops
// =============================================================================

void PoolManager::fill_group_hh_iz_indices(
    const std::vector<size_t>& global_neuron_indices,
    std::vector<size_t>& hh_local_out,
    std::vector<size_t>& iz_local_out) const
{
    hh_local_out.clear();
    iz_local_out.clear();
    for (size_t gi : global_neuron_indices) {
        auto hh_it = hh_global_to_local_.find(gi);
        if (hh_it != hh_global_to_local_.end()) {
            hh_local_out.push_back(hh_it->second);
            continue;
        }
        auto iz_it = iz_global_to_local_.find(gi);
        if (iz_it != iz_global_to_local_.end())
            iz_local_out.push_back(iz_it->second);
    }
}

void PoolManager::gather_currents_for_hh(const std::vector<size_t>& local_indices,
                                          const double* I_buf) {
    hh_pool_->gather_currents_subset(local_indices, I_buf);
}
void PoolManager::step_for_hh(const std::vector<size_t>& local_indices, double dt) {
    hh_pool_->step_subset(local_indices, dt);
}
void PoolManager::scatter_voltages_for_hh(const std::vector<size_t>& local_indices,
                                           double* V_cache) const {
    hh_pool_->scatter_voltages_subset(local_indices, V_cache);
}

void PoolManager::gather_currents_for_iz(const std::vector<size_t>& local_indices,
                                          const double* I_buf) {
    iz_pool_->gather_currents_subset(local_indices, I_buf);
}
void PoolManager::step_for_iz(const std::vector<size_t>& local_indices, double dt) {
    iz_pool_->step_subset(local_indices, dt);
}
void PoolManager::scatter_voltages_for_iz(const std::vector<size_t>& local_indices,
                                           double* V_cache) const {
    iz_pool_->scatter_voltages_subset(local_indices, V_cache);
}

void PoolManager::scatter_gates_for_names(const std::vector<std::string>& names,
                                           double* gate_buf, size_t max_gates,
                                           size_t n_rec, size_t tr) const {
    for (const auto& name : names) {
        auto it = comp_pools_.find(name);
        if (it != comp_pools_.end())
            it->second->scatter_gate_states_into(gate_buf, max_gates, n_rec, tr);
    }
}

void PoolManager::scatter_calcium_for_names(const std::vector<std::string>& names,
                                             double* ca_buf, size_t n_rec, size_t tr) const {
    for (const auto& name : names) {
        auto it = comp_pools_.find(name);
        if (it != comp_pools_.end())
            it->second->scatter_calcium_into(ca_buf, n_rec, tr);
    }
}

void PoolManager::scatter_recoveries_for_iz(const std::vector<size_t>& local_indices,
                                              double* u_buf, size_t n_rec, size_t tr) const {
    iz_pool_->scatter_recoveries_subset(local_indices, u_buf, n_rec, tr);
}

#ifdef HH_USE_CUDA

bool PoolManager::has_coop_capable_cuda_pools() const {
    if (!use_cuda_) return false;
    for (const auto& kv : comp_pools_) {
        auto* cp = dynamic_cast<CudaComposablePool*>(kv.second.get());
        if (!cp) return false;
        // The cooperative kernel only carries the lean, pre-instantiated
        // composable buckets. Pools with VM programs, modulations, or an
        // unsupported (n_gates,n_channels,n_substances) signature must use the
        // per-step path so they don't bloat the kernel's register footprint.
        if (!cp->coop_fast_eligible()) return false;
    }
    return true;
}

void PoolManager::collect_hh_descs(std::vector<CudaHHDesc>& out) const {
    out.clear();
    if (auto* hh = dynamic_cast<CudaHHPool*>(hh_pool_.get())) {
        if (!hh->empty()) {
            out.emplace_back();
            hh->fill_coop_desc(out.back());
        }
    }
}

void PoolManager::collect_iz_descs(std::vector<CudaIzDesc>& out) const {
    out.clear();
    if (auto* iz = dynamic_cast<CudaIzPool*>(iz_pool_.get())) {
        if (!iz->empty()) {
            out.emplace_back();
            iz->fill_coop_desc(out.back());
        }
    }
}

void PoolManager::collect_comp_descs(std::vector<CudaComposableDesc>& out) const {
    out.clear();
    for (const auto& kv : comp_pools_) {
        if (auto* cp = dynamic_cast<CudaComposablePool*>(kv.second.get())) {
            if (!cp->empty()) {
                out.emplace_back();
                cp->fill_coop_desc(out.back());
            }
        }
    }
}

#endif // HH_USE_CUDA

} // namespace hodgkin_huxley
