#pragma once

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/cuda_pool_descs.hpp"
#include "hodgkin_huxley/cuda_synapse.hpp"
#include "hodgkin_huxley/cuda_composable_pool.hpp"
#include "hodgkin_huxley/stim_plan.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace hodgkin_huxley {

// ---------------------------------------------------------------------------
// Pool descriptor POD structs — filled by fill_coop_desc() on each pool type
// ---------------------------------------------------------------------------

// CudaHHDesc and CudaIzDesc are defined in cuda_pool_descs.hpp (included above)
// so that cuda_hh_pool.hpp / cuda_iz_pool.hpp can include them without pulling
// in the composable/Eigen chain.

struct CudaComposableDesc {
    double                    *d_V, *d_I_ext, *d_synapse_g_scale;
    // Flat state (Stage 1): gate g of neuron i at d_gate_state[g*state_stride + i]
    // — single indirection, coalesced. Replaces the jagged double** pointer chase.
    double                    *d_gate_state;
    double                    *d_substance_state;
    double                    *d_nernst_state;
    int                        state_stride;
    CudaGateDesc              *d_gate_descs;
    CudaChannelDesc           *d_channel_descs;
    CudaChannelGateRef        *d_channel_gate_refs;
    CudaIntracellularDesc     *d_intr_descs;
    int                       *d_intr_source_channels;
    CudaIntracellularModDesc  *d_mod_descs;
    DeviceVmProgram           *d_vm_programs;
    const size_t              *d_net_idx;
    int n, n_gates, n_channels, n_intracellulars, n_gate_refs, n_mods;
    double C_m;
    // 1 = pool uses only pattern-matched standard forms with no modulations and
    // no VM programs → eligible for the gate-major fast path (no per-thread
    // local-memory scratch). 0 = fall back to the generic per-neuron interpreter.
    int fast_path;
};

// ---------------------------------------------------------------------------
// Synapse POD (no std::vector) — trivially built from DeviceSynapseArrays
// ---------------------------------------------------------------------------

struct DeviceSynapseRaw {
    double          *d_weight, *d_g, *d_S, *d_A, *d_E_syn;
    const uint32_t  *d_pre, *d_post, *d_delay_steps, *d_spec_idx;
    DeviceSynapseSpecDesc *d_spec_descs;
    uint8_t         *d_spike_ring, *d_spike_arrived, *d_neuron_spiked;
    double          *d_V_prev;
    DeviceVmProgram *d_vm_programs;
    int n_neurons, n_synapses, ring_size, n_specs;
    // Minimum delay across all synapses, in steps. If >= 1, the kernel can
    // skip the P6→P7 within-step barrier (write_slot != read_slot is guaranteed).
    int min_delay_steps;
    double spike_threshold;
};

// ---------------------------------------------------------------------------
// Stim descriptor types (moved from cuda_synapse.hpp)
// ---------------------------------------------------------------------------

struct DevicePulseDesc {
    uint32_t neuron_start, neuron_end;
    uint32_t onset_step,   end_step;
    double   amplitude;
};

struct DeviceDBSDesc {
    uint32_t neuron_start, neuron_end;
    uint32_t isi_steps,    pw_steps;
    double   amplitude;
};

// ---------------------------------------------------------------------------
// Stim descriptor (replaces DeviceStimDescriptors — now internal to this unit)
// ---------------------------------------------------------------------------

struct CudaStimRaw {
    double          *d_I_const;
    DevicePulseDesc *d_pulses;
    DeviceDBSDesc   *d_dbs;
    int n_pulses, n_dbs, n_neurons;
};

// Upload / free helpers (defined in cuda_sim_all.cu)
CudaStimRaw upload_stim_raw(const StimPlan& stim, size_t n_neurons, int device_id);
void        free_stim_raw(CudaStimRaw& raw);

// Build a DeviceSynapseRaw view of an existing DeviceSynapseArrays (no copy)
inline DeviceSynapseRaw make_syn_raw(const DeviceSynapseArrays& d) {
    DeviceSynapseRaw r{};
    r.d_weight         = d.d_weight;
    r.d_g              = d.d_g;
    r.d_S              = d.d_S;
    r.d_A              = d.d_A;
    r.d_E_syn          = d.d_E_syn;
    r.d_pre            = d.d_pre;
    r.d_post           = d.d_post;
    r.d_delay_steps    = d.d_delay_steps;
    r.d_spec_idx       = d.d_spec_idx;
    r.d_spec_descs     = d.d_spec_descs;
    r.d_spike_ring     = d.d_spike_ring;
    r.d_spike_arrived  = d.d_spike_arrived;
    r.d_neuron_spiked  = d.d_neuron_spiked;
    r.d_V_prev         = d.d_V_prev;
    r.d_vm_programs    = d.d_vm_programs;
    r.n_neurons        = static_cast<int>(d.neuron_count);
    r.n_synapses       = static_cast<int>(d.synapse_count);
    r.ring_size        = static_cast<int>(d.ring_size);
    r.n_specs          = static_cast<int>(d.spec_count);
    r.min_delay_steps  = static_cast<int>(d.min_delay_steps);
    r.spike_threshold  = d.spike_threshold;
    return r;
}

// ---------------------------------------------------------------------------
// Main entry point — launches the cooperative simulation kernel
// ---------------------------------------------------------------------------

void simulate_all_steps(
    const CudaHHDesc*          hh_descs,   int n_hh,
    const CudaIzDesc*          iz_descs,   int n_iz,
    const CudaComposableDesc*  comp_descs, int n_comp,
    DeviceSynapseRaw           syn,
    CudaStimRaw                stim,
    double*                    d_V_cache,
    double*                    d_I_syn,
    double*                    d_V_out,     // (N * n_rec) device buffer, nullptr = no V recording
    uint8_t*                   d_spike_buf, // (N * n_rec) uint8 device buffer, nullptr = no spike recording
    size_t                     n_neurons,
    size_t                     num_steps,
    double                     dt,
    size_t                     step_start,
    size_t                     record_interval,
    size_t                     n_rec,
    cudaStream_t               stream);

} // namespace hodgkin_huxley

#endif // HH_USE_CUDA
