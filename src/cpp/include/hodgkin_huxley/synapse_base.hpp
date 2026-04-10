#pragma once

#include "hodgkin_huxley/model/synapse_spec.hpp"
#include <cstddef>
#include <string>

namespace hodgkin_huxley {

class Network;  // forward declaration — SynapseBase reads from Network::SynArrays

/**
 * @brief Lightweight non-virtual view of a single synapse stored in SynArrays.
 *
 * All update logic lives in Network::update_synapses_grouped().
 * SynapseBase is pure data-access: it holds an index and a pointer to the
 * owning Network and reads state directly from the SoA arrays.
 *
 * No virtual dispatch, no per-object heap allocation, no lazy sync needed.
 */
class SynapseBase {
public:
    SynapseBase() = default;
    SynapseBase(size_t idx, const Network* net) : idx_(idx), net_(net) {}

    [[nodiscard]] double      conductance()        const;
    [[nodiscard]] double      reversal_potential() const;
    [[nodiscard]] double      weight()             const;
    [[nodiscard]] size_t      pre_idx()            const;
    [[nodiscard]] size_t      post_idx()           const;
    [[nodiscard]] double      delay()              const;
    [[nodiscard]] std::string type_name()          const;  // UpdateForm name string
    [[nodiscard]] const SynapseSpec& spec()        const;  // full spec for this synapse

private:
    size_t         idx_ = 0;
    const Network* net_ = nullptr;
};

} // namespace hodgkin_huxley
