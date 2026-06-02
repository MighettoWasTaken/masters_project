#include "hodgkin_huxley/cuda_sim_all.hpp"
#include "hodgkin_huxley/cuda_vm.hpp"
#include "hodgkin_huxley/stim_plan.hpp"

#include <algorithm>
#include <cooperative_groups.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cg = cooperative_groups;

namespace {

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(oss.str());
    }
}

// ---------------------------------------------------------------------------
// Scalar physics helpers (device-only, no global state)
// ---------------------------------------------------------------------------

__device__ inline double hh_clamp01(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

__device__ inline void hh_derivs(
    double V, double m, double h, double n,
    double Cm, double gNa, double gK, double gL,
    double ENa, double EK, double EL, double I,
    double& dV, double& dm, double& dh, double& dn)
{
    const double Vp40 = V + 40.0;
    const double am = (fabs(Vp40) < 1e-7)
        ? 1.0
        : 0.1 * Vp40 / (1.0 - exp(-0.1 * Vp40));
    const double bm = 4.0 * exp(-(V + 65.0) / 18.0);
    const double ah = 0.07 * exp(-0.05 * (V + 65.0));
    const double bh = 1.0 / (1.0 + exp(-0.1 * (V + 35.0)));
    const double Vp55 = V + 55.0;
    const double an = (fabs(Vp55) < 1e-7)
        ? 0.1
        : 0.01 * Vp55 / (1.0 - exp(-0.1 * Vp55));
    const double bn = 0.125 * exp(-0.0125 * (V + 65.0));
    dV = (I - gNa*m*m*m*h*(V-ENa) - gK*n*n*n*n*(V-EK) - gL*(V-EL)) / Cm;
    dm = am*(1.0-m) - bm*m;
    dh = ah*(1.0-h) - bh*h;
    dn = an*(1.0-n) - bn*n;
}

__device__ inline void hh_step_single(
    hodgkin_huxley::CudaHHDesc p, int i, double I, double dt)
{
    const double Cm  = p.d_C_m[i],  gNa = p.d_g_Na[i], gK = p.d_g_K[i];
    const double gL  = p.d_g_L[i],  ENa = p.d_E_Na[i], EK = p.d_E_K[i];
    const double EL  = p.d_E_L[i];
    double V = p.d_V[i], m = p.d_m[i], h = p.d_h[i], n = p.d_n[i];
    double k1V,k1m,k1h,k1n, k2V,k2m,k2h,k2n, k3V,k3m,k3h,k3n, k4V,k4m,k4h,k4n;
    hh_derivs(V,m,h,n,Cm,gNa,gK,gL,ENa,EK,EL,I,k1V,k1m,k1h,k1n);
    hh_derivs(V+0.5*dt*k1V, hh_clamp01(m+0.5*dt*k1m), hh_clamp01(h+0.5*dt*k1h), hh_clamp01(n+0.5*dt*k1n),
              Cm,gNa,gK,gL,ENa,EK,EL,I, k2V,k2m,k2h,k2n);
    hh_derivs(V+0.5*dt*k2V, hh_clamp01(m+0.5*dt*k2m), hh_clamp01(h+0.5*dt*k2h), hh_clamp01(n+0.5*dt*k2n),
              Cm,gNa,gK,gL,ENa,EK,EL,I, k3V,k3m,k3h,k3n);
    hh_derivs(V+dt*k3V,     hh_clamp01(m+dt*k3m),     hh_clamp01(h+dt*k3h),     hh_clamp01(n+dt*k3n),
              Cm,gNa,gK,gL,ENa,EK,EL,I, k4V,k4m,k4h,k4n);
    const double d6 = dt/6.0;
    p.d_V[i] = V + d6*(k1V+2.0*k2V+2.0*k3V+k4V);
    p.d_m[i] = hh_clamp01(m + d6*(k1m+2.0*k2m+2.0*k3m+k4m));
    p.d_h[i] = hh_clamp01(h + d6*(k1h+2.0*k2h+2.0*k3h+k4h));
    p.d_n[i] = hh_clamp01(n + d6*(k1n+2.0*k2n+2.0*k3n+k4n));
}

__device__ inline void iz_step_single(
    hodgkin_huxley::CudaIzDesc p, int i, double I, double dt)
{
    double vi = p.d_v[i], ui = p.d_u[i];
    const bool fired = vi >= p.threshold;
    vi = fired ? p.d_c[i] : vi;
    ui = fired ? ui + p.d_d[i] : ui;
    const double dv = 0.04*vi*vi + 5.0*vi + 140.0 - ui + I;
    const double du = p.d_a[i] * (p.d_b[i]*vi - ui);
    vi += dt * dv;
    ui += dt * du;
    p.d_v[i] = vi > 100.0 ? 100.0 : vi;
    p.d_u[i] = ui;
}

// Boltzmann / tau / rate helpers (mirrors cuda_composable_pool.cu)
__device__ inline double sa_boltz(double x, const hodgkin_huxley::BoltzmannParams& p) {
    double arg = -(x - p.v_half) / p.k;
    if (arg > 500.0) return 0.0;
    if (arg < -500.0) return 1.0;
    return 1.0 / (1.0 + exp(arg));
}
__device__ inline double sa_tau(double V, const hodgkin_huxley::TauParams& p) {
    using F = hodgkin_huxley::TauParams::Form;
    switch (p.form) {
        case F::CONSTANT:           return p.params[0];
        case F::BOLTZMANN: {
            double arg = fmax(-500.0, fmin(500.0, -(V-p.params[2])/p.params[3]));
            return p.params[0] + p.params[1]/(1.0+exp(arg));
        }
        case F::DOUBLE_EXP_SUM: {
            double d = exp((V+p.params[2])/p.params[3]) + exp(-(V+p.params[5])/p.params[6]);
            return p.params[0] + p.params[1]/fmax(d,1e-10);
        }
        case F::OFFSET_DOUBLE_EXP: {
            double x1=(V+p.params[2])/p.params[3], x2=(V+p.params[5])/p.params[6];
            return p.params[0] + p.params[1]*exp(-x1*x1) + p.params[4]*exp(-x2*x2);
        }
        case F::SCALED_EXP: {
            double ch = cosh(fmax(-500.0,fmin(500.0,(V-p.params[1])/(2.0*p.params[2]))));
            return p.params[0]/fmax(ch,1e-10);
        }
        case F::COMPOUND_AB: {
            double s = p.params[0]*exp((V+p.params[1])/p.params[2])
                     + p.params[3]*exp((V+p.params[4])/p.params[5]);
            return 1.0/fmax(s,1e-10);
        }
    }
    return 1.0;
}
__device__ inline double sa_rate(double V, const hodgkin_huxley::RateFuncParams& p) {
    using F = hodgkin_huxley::RateFuncParams::Form;
    switch (p.form) {
        case F::LINEAR_OVER_EXP: {
            double x=V+p.B, xc=x/p.C;
            return fabs(xc)<1e-6 ? p.A*p.C*(1.0+xc*0.5) : p.A*x/(exp(xc)-1.0);
        }
        case F::EXP_DECAY:
            return p.A*exp(fmax(-500.0,fmin(500.0,(V+p.B)/p.C)));
        case F::LINEAR_OVER_EXPM1: {
            double x=V+p.B, xc=x/p.C;
            return fabs(xc)<1e-6 ? p.A*p.C*(1.0+xc*0.5) : p.A*x/(1.0-exp(-xc));
        }
        case F::SIGMOID:
            return p.A/(1.0+exp(fmax(-500.0,fmin(500.0,(V+p.B)/p.C))));
    }
    return 0.0;
}
__device__ inline double sa_clamp01(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

// Lean, fully-specialized composable step (Stage 2). Templated on the EXACT
// gate/channel/substance counts so loops are compile-time-bounded and unrolled,
// and the per-thread scratch arrays are exactly sized. Handles only the
// pattern-matched standard forms with NO modulations and NO VM programs — the
// mod_* scratch arrays and the entire eval_vm_program path are gone, which is
// what raises occupancy enough to hide the fp64 transcendental latency.
// Dispatched per pool by composable_step_dispatch(); pools with modulations or
// VM programs are routed away from the cooperative kernel entirely.
template <int NG, int NC, int NS>
__device__ void composable_step_unrolled(
    hodgkin_huxley::CudaComposableDesc p, int i, double I_in, double dt)
{
    using namespace hodgkin_huxley;
    const int stride = p.state_stride;

    double v = p.d_V[i];
    const double current = p.d_I_ext[i] + I_in;

    double gate_vals[NG];
    double x_vals[NS > 0 ? NS : 1];
    double e_nernst[NS > 0 ? NS : 1];
    double channel_currents[NC];

    #pragma unroll
    for (int g = 0; g < NG; ++g)
        gate_vals[g] = p.d_gate_state[g * stride + i];
    #pragma unroll
    for (int s = 0; s < NS; ++s) {
        x_vals[s]   = p.d_substance_state[s * stride + i];
        e_nernst[s] = p.d_nernst_state[s * stride + i];
    }

    // Gate update — standard forms only.
    #pragma unroll
    for (int g = 0; g < NG; ++g) {
        const auto& desc = p.d_gate_descs[g];
        const double dep = (desc.dependency == 1 && desc.intracellular_idx >= 0
                            && desc.intracellular_idx < NS)
            ? x_vals[desc.intracellular_idx] : v;
        switch (desc.update_form) {
            case 0: { // INF_TAU
                const double xi  = sa_boltz(dep, desc.inf);
                const double tau = fmax(sa_tau(v, desc.tau), 1e-10);
                gate_vals[g] = xi + (gate_vals[g] - xi) * exp(-dt * desc.scale / tau);
                break;
            }
            case 1: { // ALPHA_BETA  (tau = 1/(al+be) → exp(-dt*(al+be)))
                const double al = sa_rate(v, desc.alpha), be = sa_rate(v, desc.beta);
                const double rt = fmax(al + be, 1e-10);
                const double xi = al / rt;
                gate_vals[g] = xi + (gate_vals[g] - xi) * exp(-dt * rt);
                break;
            }
            case 2: // INSTANT
                gate_vals[g] = sa_boltz(dep, desc.inf);
                break;
            case 3: // DERIVED
                if (desc.derived_source_gate >= 0 && desc.derived_source_gate < NG)
                    gate_vals[g] = desc.derived_a * (desc.derived_b
                        + desc.derived_c * gate_vals[desc.derived_source_gate]);
                break;
        }
        gate_vals[g] = sa_clamp01(gate_vals[g]);
    }

    // Channel currents.
    double I_total = 0.0;
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        const auto& ch = p.d_channel_descs[c];
        double gate_prod = 1.0;
        for (int gi = 0; gi < ch.gate_ref_count; ++gi) {
            const auto& ref = p.d_channel_gate_refs[ch.gate_ref_start + gi];
            if (ref.gate_idx >= 0 && ref.gate_idx < NG)
                for (int pp = 0; pp < ref.power; ++pp) gate_prod *= gate_vals[ref.gate_idx];
        }
        const double E_rev = (ch.nernst_substance_idx >= 0 && ch.nernst_substance_idx < NS)
            ? e_nernst[ch.nernst_substance_idx] : ch.E_rev;
        const double I_ci = ch.is_ahp
            ? ch.g * (x_vals[ch.ahp_substance_idx] / fmax(x_vals[ch.ahp_substance_idx] + ch.ahp_k1, 1e-10)) * (v - E_rev)
            : ch.g * gate_prod * (v - E_rev);
        channel_currents[c] = I_ci;
        I_total += I_ci;
    }

    v += dt * (-I_total + current) / p.C_m;

    // Intracellular substances — DECAY / DRIVEN_DECAY(_NERNST) standard forms.
    #pragma unroll
    for (int s = 0; s < NS; ++s) {
        const auto& intr = p.d_intr_descs[s];
        double dX;
        if (intr.update_form == 0) {            // DECAY
            dX = -intr.k_decay * x_vals[s];
        } else {                                // DRIVEN_DECAY (_NERNST)
            double I_src = 0.0;
            for (int sc = 0; sc < intr.source_count; ++sc) {
                const int ci = p.d_intr_source_channels[intr.source_start + sc];
                if (ci >= 0 && ci < NC) I_src += channel_currents[ci];
            }
            dX = intr.epsilon * (-I_src - intr.k_decay * x_vals[s]);
        }
        x_vals[s] = fmax(0.0, x_vals[s] + dt * dX);
        if (intr.nernst_enabled)
            e_nernst[s] = (intr.nernst_R * intr.nernst_T) / (intr.nernst_z * intr.nernst_F)
                * log(intr.nernst_Ca_o / fmax(x_vals[s], 1e-10));
    }

    // Write back.
    p.d_V[i] = v;
    p.d_synapse_g_scale[i] = 1.0;
    #pragma unroll
    for (int g = 0; g < NG; ++g) p.d_gate_state[g * stride + i] = gate_vals[g];
    #pragma unroll
    for (int s = 0; s < NS; ++s) {
        p.d_substance_state[s * stride + i] = x_vals[s];
        p.d_nernst_state[s * stride + i]    = e_nernst[s];
    }
}

// Per-pool dispatch on the exact structural signature. The bucket list MUST
// stay in sync with is_known_composable_bucket() on the host — the cooperative
// kernel is only launched when every composable pool matches a bucket here, so
// the default is unreachable in practice.
__device__ inline void composable_step_dispatch(
    const hodgkin_huxley::CudaComposableDesc& cd, int i, double I_in, double dt)
{
    const int g = cd.n_gates, c = cd.n_channels, s = cd.n_intracellulars;
    if (g == 3  && c == 3 && s == 0) { composable_step_unrolled<3, 3, 0>(cd, i, I_in, dt);  return; }
    if (g == 4  && c == 4 && s == 0) { composable_step_unrolled<4, 4, 0>(cd, i, I_in, dt);  return; }
    if (g == 5  && c == 4 && s == 0) { composable_step_unrolled<5, 4, 0>(cd, i, I_in, dt);  return; }
    if (g == 6  && c == 6 && s == 1) { composable_step_unrolled<6, 6, 1>(cd, i, I_in, dt);  return; }
    if (g == 11 && c == 7 && s == 1) { composable_step_unrolled<11, 7, 1>(cd, i, I_in, dt); return; }
    // Unreachable: routing guarantees only known buckets reach this kernel.
}

// ---------------------------------------------------------------------------
// Per-synapse helpers
// ---------------------------------------------------------------------------

__device__ inline void accumulate_isyn_single(
    const hodgkin_huxley::DeviceSynapseRaw& syn,
    int k, const double* d_V, double* d_I_syn)
{
    const uint32_t post = syn.d_post[k];
    const double cur = syn.d_g[k] * (syn.d_E_syn[k] - d_V[post]);
    atomicAdd(&d_I_syn[post], cur);
}

__device__ inline void update_synapse_state_single(
    hodgkin_huxley::DeviceSynapseRaw& syn,
    int k, const double* d_V, size_t step, double dt)
{
    using namespace hodgkin_huxley;

    const int read_slot = static_cast<int>(
        (step + syn.ring_size - (syn.d_delay_steps[k] % static_cast<uint32_t>(syn.ring_size))) % syn.ring_size);
    const bool arrived = syn.d_spike_ring[k * syn.ring_size + read_slot] != 0;
    syn.d_spike_arrived[k] = arrived ? 1u : 0u;

    const auto& desc = syn.d_spec_descs[syn.d_spec_idx[k]];
    const double v_pre  = d_V[syn.d_pre[k]];
    const double v_post = d_V[syn.d_post[k]];

    double s = syn.d_S[k], a = syn.d_A[k];

    switch (desc.update_form) {
        case 0: { if (arrived) s += desc.delta_S; s *= exp(-dt/desc.tau_S); if (s<0) s=0; a=0; break; }
        case 1: { if (arrived) a += desc.delta_A; double dS=(a-s)/desc.tau_A, dA=-a/desc.tau_A; s+=dt*dS; a+=dt*dA; if(s<0)s=0; break; }
        case 2: { if (arrived){s+=desc.delta_S;a+=desc.delta_A;} s*=exp(-dt/desc.tau_S); a*=exp(-dt/desc.tau_A); break; }
        case 3: { double ro=desc.tanh_amp*(1.0+tanh((v_pre-desc.tanh_vh)/desc.tanh_k)); double rt=ro+1.0/desc.tau_decay; double si=ro/rt; s=si+(s-si)*exp(-dt*rt); break; }
        case 4: { double si=sa_boltz(v_pre,desc.s_inf); double tau=fmax(sa_tau(v_pre,desc.tau),1e-10); s=si+(s-si)*exp(-dt/tau); break; }
        case 5: { double al=sa_rate(v_pre,desc.alpha),be=sa_rate(v_pre,desc.beta); double rt=al+be; double si=(rt>1e-10)?al/rt:s; s=si+(s-si)*exp(-dt*rt); break; }
        case 6: {
            if (desc.dS_vm_idx >= 0) {
                if (desc.dA_vm_idx >= 0) {
                    double dS=eval_vm_program(syn.d_vm_programs[desc.dS_vm_idx],v_pre,s,a,nullptr,0,nullptr,0);
                    double dA=eval_vm_program(syn.d_vm_programs[desc.dA_vm_idx],v_pre,s,a,nullptr,0,nullptr,0);
                    s=fmax(0.0,fmin(1.0,s+dt*dS)); a+=dt*dA;
                } else {
                    double dS=eval_vm_program(syn.d_vm_programs[desc.dS_vm_idx],v_pre,s,a,nullptr,0,nullptr,0);
                    s=fmax(0.0,fmin(1.0,s+dt*dS));
                }
            }
            break;
        }
    }

    double g_eff = 0.0;
    if (desc.update_form <= 1) {
        g_eff = desc.spec_g * syn.d_weight[k] * s;
    } else if (desc.update_form == 2) {
        double shape = desc.norm_factor * (s - a);
        if (shape < 0.0) shape = 0.0;
        g_eff = desc.spec_g * syn.d_weight[k] * shape;
    } else if (desc.current_form == 2 && desc.current_vm_idx >= 0) {
        g_eff = eval_vm_program(syn.d_vm_programs[desc.current_vm_idx],v_post,s,a,nullptr,0,nullptr,0) * syn.d_weight[k];
    } else {
        double gate = desc.spec_g * syn.d_weight[k];
        for (int p = 0; p < desc.power; ++p) gate *= s;
        if (desc.current_form == 1) {
            double mg = 1.0 + desc.mg_conc*exp(-desc.mg_scale*v_post)/desc.mg_denom;
            gate /= mg;
        }
        g_eff = gate;
    }
    syn.d_S[k] = s; syn.d_A[k] = a; syn.d_g[k] = g_eff;
}

// ---------------------------------------------------------------------------
// Barrier helper — switches between grid.sync() and __syncthreads()
// depending on whether the launch is cooperative (multi-block) or single-block.
// ---------------------------------------------------------------------------

template <bool MultiBlock>
__device__ inline void sync_step(cg::grid_group& grid) {
    if (MultiBlock) grid.sync();
    else            __syncthreads();
}

// ---------------------------------------------------------------------------
// Main simulation kernel — templated on MultiBlock.
// Phase fusion (Phase 1 optimization):
//   P1 (stim + scatter V)                           [barrier]
//   P2 (accumulate I_syn)                           [barrier]
//   P3+P4+P5 fused (step + scatter V + detect)     [barrier]
//   P6 (write spike ring)                           [barrier ONLY if min_delay=0]
//   P7 (update synapse state)                       [barrier, end-of-step]
//
// P3→P4 and P4→P5 barriers are removed: each thread handles its own neuron
// through all three operations with no cross-thread dependency.
//
// P6→P7 barrier is elided when min_delay_steps >= 1, since write_slot
// (step%R) differs from read_slot ((step-delay)%R). For typical networks,
// barrier count is 4 per step (down from 7). With delay=0 it's 5.
// ---------------------------------------------------------------------------

template <bool MultiBlock>
__device__ void simulate_all_kernel_impl(
    hodgkin_huxley::CudaHHDesc*         hh_descs,   int n_hh,
    hodgkin_huxley::CudaIzDesc*         iz_descs,   int n_iz,
    hodgkin_huxley::CudaComposableDesc* comp_descs, int n_comp,
    hodgkin_huxley::DeviceSynapseRaw    syn,
    hodgkin_huxley::CudaStimRaw         stim,
    double*   d_V_cache,
    double*   d_I_syn,
    double*   d_V_out,
    uint8_t*  d_spike_buf,
    int       n_neurons,
    size_t    num_steps,
    double    dt,
    size_t    step_start,
    int       record_interval,
    int       n_rec)
{
    auto grid  = cg::this_grid();
    const int tid   = MultiBlock
        ? (blockIdx.x * blockDim.x + threadIdx.x)
        : static_cast<int>(threadIdx.x);
    const int total = MultiBlock
        ? (gridDim.x * blockDim.x)
        : static_cast<int>(blockDim.x);

    for (size_t t = 0; t < num_steps; ++t) {
        const size_t step = step_start + t;

        // ---- P1: compute stim (→ d_I_syn) and scatter V (→ d_V_cache) ----
        for (int i = tid; i < stim.n_neurons; i += total) {
            double I = stim.d_I_const[i];
            for (int p = 0; p < stim.n_pulses; ++p) {
                const auto& pd = stim.d_pulses[p];
                if (step >= pd.onset_step && step < pd.end_step
                    && static_cast<uint32_t>(i) >= pd.neuron_start
                    && static_cast<uint32_t>(i) < pd.neuron_end)
                    I += pd.amplitude;
            }
            for (int d = 0; d < stim.n_dbs; ++d) {
                const auto& dd = stim.d_dbs[d];
                if (dd.isi_steps == 0) continue;
                if (static_cast<uint32_t>(i) >= dd.neuron_start
                    && static_cast<uint32_t>(i) < dd.neuron_end) {
                    if (static_cast<uint32_t>(step % dd.isi_steps) < dd.pw_steps)
                        I += dd.amplitude;
                }
            }
            d_I_syn[i] = I;
        }
        for (int p = 0; p < n_hh; ++p)
            for (int i = tid; i < hh_descs[p].n; i += total)
                d_V_cache[hh_descs[p].d_net_idx[i]] = hh_descs[p].d_V[i];
        for (int p = 0; p < n_iz; ++p)
            for (int i = tid; i < iz_descs[p].n; i += total)
                d_V_cache[iz_descs[p].d_net_idx[i]] = iz_descs[p].d_v[i];
        for (int p = 0; p < n_comp; ++p)
            for (int i = tid; i < comp_descs[p].n; i += total)
                d_V_cache[comp_descs[p].d_net_idx[i]] = comp_descs[p].d_V[i];
        // Record pre-step voltage — matches CPU semantics (records V before stepping)
        if (d_V_out && record_interval > 0 && (int)(t % static_cast<size_t>(record_interval)) == 0) {
            const int tr = static_cast<int>(t / static_cast<size_t>(record_interval));
            for (int i = tid; i < n_neurons; i += total)
                d_V_out[static_cast<size_t>(i) * static_cast<size_t>(n_rec) + static_cast<size_t>(tr)] = d_V_cache[i];
        }
        sync_step<MultiBlock>(grid);

        // ---- P2: accumulate synaptic currents (atomicAdd → d_I_syn) ----
        for (int s = tid; s < syn.n_synapses; s += total)
            accumulate_isyn_single(syn, s, d_V_cache, d_I_syn);
        sync_step<MultiBlock>(grid);

        // ---- Fused P3+P4+P5: gather I_syn → step pool → scatter V → detect spike ----
        // No barriers between substeps: each thread processes its own neuron only.
        const int tr_spike = (record_interval > 0)
            ? static_cast<int>(t / static_cast<size_t>(record_interval)) : 0;
        const bool record_spikes = (d_spike_buf != nullptr) && record_interval > 0 && tr_spike < n_rec;

        for (int p = 0; p < n_hh; ++p) {
            for (int i = tid; i < hh_descs[p].n; i += total) {
                const size_t nidx = hh_descs[p].d_net_idx[i];
                // P3: step
                double I = d_I_syn[nidx];
                hh_step_single(hh_descs[p], i, I, dt);
                // P4: scatter V
                const double v_new = hh_descs[p].d_V[i];
                d_V_cache[nidx] = v_new;
                // P5: detect spike (only if synapses exist or recording spikes)
                if (syn.n_synapses > 0 || record_spikes) {
                    const double vp = syn.d_V_prev ? syn.d_V_prev[nidx] : v_new;
                    const uint8_t spiked = (v_new > syn.spike_threshold && vp <= syn.spike_threshold) ? 1u : 0u;
                    if (syn.d_neuron_spiked) syn.d_neuron_spiked[nidx] = spiked;
                    if (syn.d_V_prev)        syn.d_V_prev[nidx] = v_new;
                    if (record_spikes)
                        d_spike_buf[nidx * static_cast<size_t>(n_rec) + static_cast<size_t>(tr_spike)] |= spiked;
                }
            }
        }
        for (int p = 0; p < n_iz; ++p) {
            for (int i = tid; i < iz_descs[p].n; i += total) {
                const size_t nidx = iz_descs[p].d_net_idx[i];
                double I = d_I_syn[nidx];
                iz_step_single(iz_descs[p], i, I, dt);
                const double v_new = iz_descs[p].d_v[i];
                d_V_cache[nidx] = v_new;
                if (syn.n_synapses > 0 || record_spikes) {
                    const double vp = syn.d_V_prev ? syn.d_V_prev[nidx] : v_new;
                    const uint8_t spiked = (v_new > syn.spike_threshold && vp <= syn.spike_threshold) ? 1u : 0u;
                    if (syn.d_neuron_spiked) syn.d_neuron_spiked[nidx] = spiked;
                    if (syn.d_V_prev)        syn.d_V_prev[nidx] = v_new;
                    if (record_spikes)
                        d_spike_buf[nidx * static_cast<size_t>(n_rec) + static_cast<size_t>(tr_spike)] |= spiked;
                }
            }
        }
        for (int p = 0; p < n_comp; ++p) {
            const hodgkin_huxley::CudaComposableDesc& cd = comp_descs[p];
            for (int i = tid; i < cd.n; i += total) {
                const size_t nidx = cd.d_net_idx[i];
                const double I = d_I_syn[nidx];
                composable_step_dispatch(cd, i, I, dt);
                const double v_new = cd.d_V[i];
                d_V_cache[nidx] = v_new;
                if (syn.n_synapses > 0 || record_spikes) {
                    const double vp = syn.d_V_prev ? syn.d_V_prev[nidx] : v_new;
                    const uint8_t spiked = (v_new > syn.spike_threshold && vp <= syn.spike_threshold) ? 1u : 0u;
                    if (syn.d_neuron_spiked) syn.d_neuron_spiked[nidx] = spiked;
                    if (syn.d_V_prev)        syn.d_V_prev[nidx] = v_new;
                    if (record_spikes)
                        d_spike_buf[nidx * static_cast<size_t>(n_rec) + static_cast<size_t>(tr_spike)] |= spiked;
                }
            }
        }
        sync_step<MultiBlock>(grid);

        // ---- P6: update spike ring ----
        if (syn.n_synapses > 0) {
            const int write_slot = static_cast<int>(step % static_cast<size_t>(syn.ring_size));
            for (int s = tid; s < syn.n_synapses; s += total)
                syn.d_spike_ring[s * syn.ring_size + write_slot] = syn.d_neuron_spiked[syn.d_pre[s]] ? 1u : 0u;
        }
        // P6→P7 barrier — only required when min_delay_steps == 0 (read_slot
        // could equal write_slot). For typical networks with delay >= dt, the
        // slots differ and we can elide this entire barrier.
        if (syn.min_delay_steps < 1) {
            sync_step<MultiBlock>(grid);
        }

        // ---- P7: update synapse state ----
        for (int s = tid; s < syn.n_synapses; s += total)
            update_synapse_state_single(syn, s, d_V_cache, step, dt);
        sync_step<MultiBlock>(grid);
    }
}

__global__ void simulate_all_kernel_multi(
    hodgkin_huxley::CudaHHDesc*         hh_descs,   int n_hh,
    hodgkin_huxley::CudaIzDesc*         iz_descs,   int n_iz,
    hodgkin_huxley::CudaComposableDesc* comp_descs, int n_comp,
    hodgkin_huxley::DeviceSynapseRaw    syn,
    hodgkin_huxley::CudaStimRaw         stim,
    double*   d_V_cache,
    double*   d_I_syn,
    double*   d_V_out,
    uint8_t*  d_spike_buf,
    int       n_neurons,
    size_t    num_steps,
    double    dt,
    size_t    step_start,
    int       record_interval,
    int       n_rec)
{
    simulate_all_kernel_impl<true>(
        hh_descs, n_hh, iz_descs, n_iz, comp_descs, n_comp,
        syn, stim, d_V_cache, d_I_syn, d_V_out, d_spike_buf,
        n_neurons, num_steps, dt, step_start, record_interval, n_rec);
}

__global__ void simulate_all_kernel_single(
    hodgkin_huxley::CudaHHDesc*         hh_descs,   int n_hh,
    hodgkin_huxley::CudaIzDesc*         iz_descs,   int n_iz,
    hodgkin_huxley::CudaComposableDesc* comp_descs, int n_comp,
    hodgkin_huxley::DeviceSynapseRaw    syn,
    hodgkin_huxley::CudaStimRaw         stim,
    double*   d_V_cache,
    double*   d_I_syn,
    double*   d_V_out,
    uint8_t*  d_spike_buf,
    int       n_neurons,
    size_t    num_steps,
    double    dt,
    size_t    step_start,
    int       record_interval,
    int       n_rec)
{
    simulate_all_kernel_impl<false>(
        hh_descs, n_hh, iz_descs, n_iz, comp_descs, n_comp,
        syn, stim, d_V_cache, d_I_syn, d_V_out, d_spike_buf,
        n_neurons, num_steps, dt, step_start, record_interval, n_rec);
}

} // anonymous namespace

namespace hodgkin_huxley {

// ---------------------------------------------------------------------------
// Stim upload / free
// ---------------------------------------------------------------------------

CudaStimRaw upload_stim_raw(const StimPlan& stim, size_t n_neurons, int device_id) {
    cudaSetDevice(device_id);
    CudaStimRaw r{};
    r.n_neurons = static_cast<int>(n_neurons);
    r.n_pulses  = static_cast<int>(stim.pulses.size());
    r.n_dbs     = static_cast<int>(stim.dbs.size());

    ck(cudaMalloc(reinterpret_cast<void**>(&r.d_I_const), n_neurons * sizeof(double)),
       "upload_stim_raw I_const");
    ck(cudaMemcpy(r.d_I_const, stim.I_const.data(), n_neurons * sizeof(double),
                  cudaMemcpyHostToDevice),
       "upload_stim_raw memcpy I_const");

    if (r.n_pulses > 0) {
        std::vector<DevicePulseDesc> pd(r.n_pulses);
        for (int i = 0; i < r.n_pulses; ++i) {
            pd[i].neuron_start = stim.pulses[i].neuron_start;
            pd[i].neuron_end   = stim.pulses[i].neuron_end;
            pd[i].onset_step   = stim.pulses[i].onset_step;
            pd[i].end_step     = stim.pulses[i].end_step;
            pd[i].amplitude    = stim.pulses[i].amplitude;
        }
        ck(cudaMalloc(reinterpret_cast<void**>(&r.d_pulses), r.n_pulses * sizeof(DevicePulseDesc)),
           "upload_stim_raw pulses alloc");
        ck(cudaMemcpy(r.d_pulses, pd.data(), r.n_pulses * sizeof(DevicePulseDesc),
                      cudaMemcpyHostToDevice),
           "upload_stim_raw pulses memcpy");
    }
    if (r.n_dbs > 0) {
        std::vector<DeviceDBSDesc> dd(r.n_dbs);
        for (int i = 0; i < r.n_dbs; ++i) {
            dd[i].neuron_start = stim.dbs[i].neuron_start;
            dd[i].neuron_end   = stim.dbs[i].neuron_end;
            dd[i].isi_steps    = stim.dbs[i].isi_steps;
            dd[i].pw_steps     = stim.dbs[i].pw_steps;
            dd[i].amplitude    = stim.dbs[i].amplitude;
        }
        ck(cudaMalloc(reinterpret_cast<void**>(&r.d_dbs), r.n_dbs * sizeof(DeviceDBSDesc)),
           "upload_stim_raw dbs alloc");
        ck(cudaMemcpy(r.d_dbs, dd.data(), r.n_dbs * sizeof(DeviceDBSDesc),
                      cudaMemcpyHostToDevice),
           "upload_stim_raw dbs memcpy");
    }
    return r;
}

void free_stim_raw(CudaStimRaw& r) {
    if (r.d_I_const) { cudaFree(r.d_I_const); r.d_I_const = nullptr; }
    if (r.d_pulses)  { cudaFree(r.d_pulses);  r.d_pulses  = nullptr; }
    if (r.d_dbs)     { cudaFree(r.d_dbs);     r.d_dbs     = nullptr; }
}

// ---------------------------------------------------------------------------
// simulate_all_steps — host launch wrapper
// ---------------------------------------------------------------------------

void simulate_all_steps(
    const CudaHHDesc*          hh_descs_h, int n_hh,
    const CudaIzDesc*          iz_descs_h, int n_iz,
    const CudaComposableDesc*  comp_descs_h, int n_comp,
    DeviceSynapseRaw           syn,
    CudaStimRaw                stim,
    double*                    d_V_cache,
    double*                    d_I_syn,
    double*                    d_V_out,
    uint8_t*                   d_spike_buf,
    size_t                     n_neurons,
    size_t                     num_steps,
    double                     dt,
    size_t                     step_start,
    size_t                     record_interval,
    size_t                     n_rec,
    cudaStream_t               stream)
{
    if (num_steps == 0) return;

    int device;
    ck(cudaGetDevice(&device), "simulate_all_steps: cudaGetDevice");

    // Upload pool descriptor arrays to device
    CudaHHDesc*         d_hh   = nullptr;
    CudaIzDesc*         d_iz   = nullptr;
    CudaComposableDesc* d_comp = nullptr;
    if (n_hh > 0) {
        ck(cudaMalloc(reinterpret_cast<void**>(&d_hh),   n_hh   * sizeof(CudaHHDesc)),   "d_hh alloc");
        ck(cudaMemcpy(d_hh,   hh_descs_h,   n_hh   * sizeof(CudaHHDesc),   cudaMemcpyHostToDevice), "d_hh memcpy");
    }
    if (n_iz > 0) {
        ck(cudaMalloc(reinterpret_cast<void**>(&d_iz),   n_iz   * sizeof(CudaIzDesc)),   "d_iz alloc");
        ck(cudaMemcpy(d_iz,   iz_descs_h,   n_iz   * sizeof(CudaIzDesc),   cudaMemcpyHostToDevice), "d_iz memcpy");
    }
    if (n_comp > 0) {
        ck(cudaMalloc(reinterpret_cast<void**>(&d_comp), n_comp * sizeof(CudaComposableDesc)), "d_comp alloc");
        ck(cudaMemcpy(d_comp, comp_descs_h, n_comp * sizeof(CudaComposableDesc), cudaMemcpyHostToDevice), "d_comp memcpy");
    }

    // Compute work items. Single-block launch (with stride pattern) replaces
    // grid.sync (~10-15 µs WDDM) with __syncthreads (~10 ns) but loses SM
    // parallelism — only one SM does work. The break-even depends on per-unit
    // compute cost:
    //
    //   Heavy composable kernel (~30 µs/neuron, register-spilled): single-block
    //   wins up to several thousand work items because multi-block SM utilization
    //   is poor for small per-pool counts (e.g. CTX-BG-TH: 10 neurons/pool).
    //
    //   Light HH/Iz kernel (~1 µs/neuron): multi-block SM parallelism wins above
    //   ~256 work items because compute cost shrinks faster than barrier cost.
    // Single-block (no grid.sync, ~10ns __syncthreads) wins only when the
    // network is too small to usefully fill multiple SMs. Above ~256 work
    // items, multi-block's extra SMs outweigh the grid.sync cost — especially
    // for fp64-transcendental-heavy composable models, which are throughput-
    // bound on a single SM (consumer GPUs run fp64 at ~1/64 fp32 rate).
    constexpr int kBlockSize           = 256;
    const int single_block_max_work    = 256;
    int work_items = static_cast<int>(n_neurons);
    if (syn.n_synapses > work_items) work_items = syn.n_synapses;
    if (stim.n_neurons > work_items) work_items = stim.n_neurons;
    if (work_items < 1) work_items = 1;
    const bool use_single_block = False;

    int    i_n_neurons   = static_cast<int>(n_neurons);
    size_t sz_num_steps  = num_steps;
    int    i_rec_iv      = static_cast<int>(record_interval);
    int    i_n_rec       = static_cast<int>(n_rec);

    int sb_threads = ((work_items + 31) / 32) * 32;
    if (sb_threads < 32)         sb_threads = 32;
    if (sb_threads > kBlockSize) sb_threads = kBlockSize;

    void* args[] = {
        &d_hh,         &n_hh,
        &d_iz,         &n_iz,
        &d_comp,       &n_comp,
        &syn,
        &stim,
        &d_V_cache,
        &d_I_syn,
        &d_V_out,
        &d_spike_buf,
        &i_n_neurons,
        &sz_num_steps,
        &dt,
        &step_start,
        &i_rec_iv,
        &i_n_rec
    };

    // Composable pools are dispatched per-pool to fully-specialized unrolled
    // kernels inside the kernel (composable_step_dispatch), so the launch itself
    // is no longer templated on gate/channel/substance counts.
    if (use_single_block) {
        simulate_all_kernel_single<<<1, sb_threads, 0, stream>>>(
            d_hh, n_hh, d_iz, n_iz, d_comp, n_comp, syn, stim,
            d_V_cache, d_I_syn, d_V_out, d_spike_buf,
            i_n_neurons, sz_num_steps, dt, step_start, i_rec_iv, i_n_rec);
        ck(cudaGetLastError(), "simulate_all_steps: single-block launch");
    } else {
        int supports_coop = 0;
        ck(cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, device),
           "simulate_all_steps: cudaDeviceGetAttribute cooperative");
        if (!supports_coop)
            throw std::runtime_error("simulate_all_steps: GPU does not support cooperative kernel launch");
        int sm_count = 0, max_blocks_sm = 0;
        ck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device),
           "simulate_all_steps: SM count");
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
               &max_blocks_sm, (const void*)simulate_all_kernel_multi, kBlockSize, 0),
           "simulate_all_steps: occupancy");
        const int blocks_needed = (work_items + kBlockSize - 1) / kBlockSize;
        const int total_blocks  = std::min(blocks_needed, sm_count * max_blocks_sm);
        if (total_blocks <= 0)
            throw std::runtime_error("simulate_all_steps: 0 cooperative blocks");
        ck(cudaLaunchCooperativeKernel(
               reinterpret_cast<void*>(simulate_all_kernel_multi),
               dim3(total_blocks), dim3(kBlockSize),
               args, 0, stream),
           "simulate_all_steps: cudaLaunchCooperativeKernel");
    }

    ck(cudaStreamSynchronize(stream), "simulate_all_steps: stream sync");

    // Free temp device descriptor arrays
    if (d_hh)   cudaFree(d_hh);
    if (d_iz)   cudaFree(d_iz);
    if (d_comp) cudaFree(d_comp);
}

} // namespace hodgkin_huxley
