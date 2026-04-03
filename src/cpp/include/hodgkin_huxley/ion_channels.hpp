#pragma once

// =============================================================================
// ion_channels.hpp — backwards-compatible umbrella header
//
// All types and math helpers now live in focused sub-headers under model/.
// This file re-exports everything so existing #include "ion_channels.hpp"
// sites continue to compile without modification.
// =============================================================================

#include "hodgkin_huxley/model/gate_spec.hpp"
#include "hodgkin_huxley/model/channel_spec.hpp"
#include "hodgkin_huxley/model/synapse_spec.hpp"
#include "hodgkin_huxley/model/neuron_spec.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
