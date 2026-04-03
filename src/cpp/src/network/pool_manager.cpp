#include "hodgkin_huxley/network/pool_manager.hpp"
#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/izhikevich.hpp"
#include "hodgkin_huxley/composable_neuron.hpp"
#include <map>

namespace hodgkin_huxley {

void PoolManager::build_from_neurons(
    const std::vector<std::unique_ptr<NeuronBase>>& neurons, bool fast_math)
{
    const size_t nn = neurons.size();

    // Count by type
    size_t n_hh = 0, n_iz = 0;
    std::map<std::string, size_t> composable_counts;
    for (const auto& n : neurons) {
        if (dynamic_cast<HHNeuron*>(n.get()))           ++n_hh;
        else if (dynamic_cast<IzhikevichNeuron*>(n.get())) ++n_iz;
        else if (auto* cn = dynamic_cast<ComposableNeuron*>(n.get()))
            composable_counts[cn->model_spec().name]++;
    }

    // Construct pools sized to capacity
    hh_pool_ = HHPool(n_hh, fast_math);
    iz_pool_ = IzPool(n_iz);

    comp_pools_.clear();
    for (const auto& kv : composable_counts) {
        for (const auto& n : neurons) {
            auto* cn = dynamic_cast<ComposableNeuron*>(n.get());
            if (cn && cn->model_spec().name == kv.first) {
                comp_pools_.emplace(kv.first,
                    ComposablePool(cn->model_spec(), kv.second, fast_math));
                break;
            }
        }
    }

    // Populate with current neuron state
    for (size_t i = 0; i < nn; ++i) {
        if (auto* hh = dynamic_cast<HHNeuron*>(neurons[i].get())) {
            hh_pool_.add(i, hh->parameters(), hh->state());
        } else if (auto* iz = dynamic_cast<IzhikevichNeuron*>(neurons[i].get())) {
            iz_pool_.add(i, iz->parameters(), iz->state());
        } else if (auto* cn = dynamic_cast<ComposableNeuron*>(neurons[i].get())) {
            auto it = comp_pools_.find(cn->model_spec().name);
            if (it != comp_pools_.end())
                it->second.add(i, cn->membrane_potential(),
                               cn->gate_states(), cn->calcium());
        }
    }
}

void PoolManager::scatter_all_voltages(double* V_cache) const {
    hh_pool_.scatter_voltages(V_cache);
    iz_pool_.scatter_voltages(V_cache);
    for (const auto& kv : comp_pools_) kv.second.scatter_voltages(V_cache);
}

void PoolManager::gather_all_currents(const double* I_buf) {
    hh_pool_.gather_currents(I_buf);
    iz_pool_.gather_currents(I_buf);
    for (auto& kv : comp_pools_) kv.second.gather_currents(I_buf);
}

void PoolManager::step_all(double dt) {
    hh_pool_.step(dt);
    iz_pool_.step(dt);
    for (auto& kv : comp_pools_) kv.second.step(dt);
}

void PoolManager::sync_all_to_neurons(
    std::vector<std::unique_ptr<NeuronBase>>& neurons) const
{
    hh_pool_.sync_to_neurons(neurons);
    iz_pool_.sync_to_neurons(neurons);
    for (const auto& kv : comp_pools_) kv.second.sync_to_neurons(neurons);
}

void PoolManager::scatter_gates(double* gate_buf, size_t max_gates,
                                 size_t n_rec, size_t tr) const {
    for (const auto& kv : comp_pools_)
        kv.second.scatter_gate_states_into(gate_buf, max_gates, n_rec, tr);
}

void PoolManager::scatter_calcium(double* ca_buf, size_t n_rec, size_t tr) const {
    for (const auto& kv : comp_pools_)
        kv.second.scatter_calcium_into(ca_buf, n_rec, tr);
}

void PoolManager::scatter_recoveries(double* u_buf, size_t n_rec, size_t tr) const {
    iz_pool_.scatter_recoveries(u_buf, n_rec, tr);
}

} // namespace hodgkin_huxley
