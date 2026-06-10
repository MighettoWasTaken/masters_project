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
// Scalar physics helpers — templated on T (float or double).
// Pool state arrays stay double* on device; T is the compute type only.
// ---------------------------------------------------------------------------

template <typename T>
__device__ inline T hh_clamp01(T x) {
    return x < T(0) ? T(0) : (x > T(1) ? T(1) : x);
}

template <typename T>
__device__ inline void hh_derivs(
    T V, T m, T h, T n,
    T Cm, T gNa, T gK, T gL,
    T ENa, T EK, T EL, T I,
    T& dV, T& dm, T& dh, T& dn)
{
    const T Vp40 = V + T(40);
    const T am = (fabs(Vp40) < T(1e-7))
        ? T(1)
        : T(0.1) * Vp40 / (T(1) - exp(T(-0.1) * Vp40));
    const T bm = T(4) * exp(-(V + T(65)) / T(18));
    const T ah = T(0.07) * exp(T(-0.05) * (V + T(65)));
    const T bh = T(1) / (T(1) + exp(T(-0.1) * (V + T(35))));
    const T Vp55 = V + T(55);
    const T an = (fabs(Vp55) < T(1e-7))
        ? T(0.1)
        : T(0.01) * Vp55 / (T(1) - exp(T(-0.1) * Vp55));
    const T bn = T(0.125) * exp(T(-0.0125) * (V + T(65)));
    dV = (I - gNa*m*m*m*h*(V-ENa) - gK*n*n*n*n*(V-EK) - gL*(V-EL)) / Cm;
    dm = am*(T(1)-m) - bm*m;
    dh = ah*(T(1)-h) - bh*h;
    dn = an*(T(1)-n) - bn*n;
}

template <typename T>
__device__ inline void hh_step_single(
    hodgkin_huxley::CudaHHDesc p, int i, T I, T dt)
{
    const T Cm  = T(p.d_C_m[i]),  gNa = T(p.d_g_Na[i]), gK = T(p.d_g_K[i]);
    const T gL  = T(p.d_g_L[i]),  ENa = T(p.d_E_Na[i]), EK = T(p.d_E_K[i]);
    const T EL  = T(p.d_E_L[i]);
    T V = T(p.d_V[i]), m = T(p.d_m[i]), h = T(p.d_h[i]), n = T(p.d_n[i]);
    T k1V,k1m,k1h,k1n, k2V,k2m,k2h,k2n, k3V,k3m,k3h,k3n, k4V,k4m,k4h,k4n;
    hh_derivs<T>(V,m,h,n,Cm,gNa,gK,gL,ENa,EK,EL,I,k1V,k1m,k1h,k1n);
    hh_derivs<T>(V+T(0.5)*dt*k1V, hh_clamp01<T>(m+T(0.5)*dt*k1m), hh_clamp01<T>(h+T(0.5)*dt*k1h), hh_clamp01<T>(n+T(0.5)*dt*k1n),
                 Cm,gNa,gK,gL,ENa,EK,EL,I, k2V,k2m,k2h,k2n);
    hh_derivs<T>(V+T(0.5)*dt*k2V, hh_clamp01<T>(m+T(0.5)*dt*k2m), hh_clamp01<T>(h+T(0.5)*dt*k2h), hh_clamp01<T>(n+T(0.5)*dt*k2n),
                 Cm,gNa,gK,gL,ENa,EK,EL,I, k3V,k3m,k3h,k3n);
    hh_derivs<T>(V+dt*k3V, hh_clamp01<T>(m+dt*k3m), hh_clamp01<T>(h+dt*k3h), hh_clamp01<T>(n+dt*k3n),
                 Cm,gNa,gK,gL,ENa,EK,EL,I, k4V,k4m,k4h,k4n);
    const T d6 = dt / T(6);
    p.d_V[i] = double(V + d6*(k1V+T(2)*k2V+T(2)*k3V+k4V));
    p.d_m[i] = double(hh_clamp01<T>(m + d6*(k1m+T(2)*k2m+T(2)*k3m+k4m)));
    p.d_h[i] = double(hh_clamp01<T>(h + d6*(k1h+T(2)*k2h+T(2)*k3h+k4h)));
    p.d_n[i] = double(hh_clamp01<T>(n + d6*(k1n+T(2)*k2n+T(2)*k3n+k4n)));
}

template <typename T>
__device__ inline void iz_step_single(
    hodgkin_huxley::CudaIzDesc p, int i, T I, T dt)
{
    T vi = T(p.d_v[i]), ui = T(p.d_u[i]);
    const bool fired = vi >= T(p.threshold);
    vi = fired ? T(p.d_c[i]) : vi;
    ui = fired ? ui + T(p.d_d[i]) : ui;
    const T dv = T(0.04)*vi*vi + T(5)*vi + T(140) - ui + I;
    const T du = T(p.d_a[i]) * (T(p.d_b[i])*vi - ui);
    vi += dt * dv;
    ui += dt * du;
    p.d_v[i] = double(vi > T(100) ? T(100) : vi);
    p.d_u[i] = double(ui);
}

// Boltzmann / tau / rate helpers (mirrors cuda_composable_pool.cu), templated on T.
// Explicit template args required at call sites that cannot deduce T (e.g. synapse
// helpers that pass double and rely on the double overload).
template <typename T>
__device__ inline T sa_boltz(T x, const hodgkin_huxley::BoltzmannParams& p) {
    T arg = -(x - T(p.v_half)) / T(p.k);
    if (arg > T(500)) return T(0);
    if (arg < T(-500)) return T(1);
    return T(1) / (T(1) + exp(arg));
}
template <typename T>
__device__ inline T sa_tau(T V, const hodgkin_huxley::TauParams& p) {
    using F = hodgkin_huxley::TauParams::Form;
    switch (p.form) {
        case F::CONSTANT: return T(p.params[0]);
        case F::BOLTZMANN: {
            T arg = fmax(T(-500), fmin(T(500), -(V - T(p.params[2])) / T(p.params[3])));
            return T(p.params[0]) + T(p.params[1]) / (T(1) + exp(arg));
        }
        case F::DOUBLE_EXP_SUM: {
            T d = exp((V + T(p.params[2])) / T(p.params[3]))
                + exp(-(V + T(p.params[5])) / T(p.params[6]));
            return T(p.params[0]) + T(p.params[1]) / fmax(d, T(1e-10));
        }
        case F::OFFSET_DOUBLE_EXP: {
            T x1 = (V + T(p.params[2])) / T(p.params[3]);
            T x2 = (V + T(p.params[5])) / T(p.params[6]);
            return T(p.params[0]) + T(p.params[1])*exp(-x1*x1) + T(p.params[4])*exp(-x2*x2);
        }
        case F::SCALED_EXP: {
            T ch = cosh(fmax(T(-500), fmin(T(500), (V - T(p.params[1])) / (T(2) * T(p.params[2])))));
            return T(p.params[0]) / fmax(ch, T(1e-10));
        }
        case F::COMPOUND_AB: {
            T s = T(p.params[0])*exp((V + T(p.params[1])) / T(p.params[2]))
                + T(p.params[3])*exp((V + T(p.params[4])) / T(p.params[5]));
            return T(1) / fmax(s, T(1e-10));
        }
    }
    return T(1);
}
template <typename T>
__device__ inline T sa_rate(T V, const hodgkin_huxley::RateFuncParams& p) {
    using F = hodgkin_huxley::RateFuncParams::Form;
    switch (p.form) {
        case F::LINEAR_OVER_EXP: {
            T x = V + T(p.B), xc = x / T(p.C);
            return fabs(xc) < T(1e-6) ? T(p.A)*T(p.C)*(T(1)+xc*T(0.5)) : T(p.A)*x/(exp(xc)-T(1));
        }
        case F::EXP_DECAY:
            return T(p.A) * exp(fmax(T(-500), fmin(T(500), (V + T(p.B)) / T(p.C))));
        case F::LINEAR_OVER_EXPM1: {
            T x = V + T(p.B), xc = x / T(p.C);
            return fabs(xc) < T(1e-6) ? T(p.A)*T(p.C)*(T(1)+xc*T(0.5)) : T(p.A)*x/(T(1)-exp(-xc));
        }
        case F::SIGMOID:
            return T(p.A) / (T(1) + exp(fmax(T(-500), fmin(T(500), (V + T(p.B)) / T(p.C)))));
    }
    return T(0);
}
template <typename T>
__device__ inline T sa_clamp01(T x) {
    return x < T(0) ? T(0) : (x > T(1) ? T(1) : x);
}

// P3b composable pass: reads gates already updated by P3a, resolves DERIVED gates,
// computes channel currents, updates V, updates intracellular substances, writes back.
// Templated on exact gate/channel/substance counts AND compute type T.
// Gate kinetics (INF_TAU, ALPHA_BETA, INSTANT) are NOT recomputed here — P3a wrote them.
template <int NG, int NC, int NS, typename T>
__device__ void composable_channel_vupdate_unrolled(
    hodgkin_huxley::CudaComposableDesc p, int i, T I_in, T dt)
{
    using namespace hodgkin_huxley;
    const int stride = p.state_stride;

    T v = T(p.d_V[i]);
    // I_in carries external + synaptic current from d_I_syn (see composable_step_unrolled
    // comment about why p.d_I_ext is NOT used on the cooperative path).
    const T current = I_in;

    T gate_vals[NG];
    T x_vals[NS > 0 ? NS : 1];
    T e_nernst[NS > 0 ? NS : 1];

    // Read gates updated by P3a. DERIVED gates still hold their previous-step value
    // from d_gate_state; they'll be recomputed immediately below.
    #pragma unroll
    for (int g = 0; g < NG; ++g)
        gate_vals[g] = T(p.d_gate_state[g * stride + i]);
    #pragma unroll
    for (int s = 0; s < NS; ++s) {
        x_vals[s]   = T(p.d_substance_state[s * stride + i]);
        e_nernst[s] = T(p.d_nernst_state[s * stride + i]);
    }

    // Resolve DERIVED gates using this step's source gate values (now in gate_vals[]).
    #pragma unroll
    for (int g = 0; g < NG; ++g) {
        const auto& desc = p.d_gate_descs[g];
        if (desc.update_form == 3) {  // DERIVED
            if (desc.derived_source_gate >= 0 && desc.derived_source_gate < NG)
                gate_vals[g] = T(desc.derived_a) * (T(desc.derived_b)
                    + T(desc.derived_c) * gate_vals[desc.derived_source_gate]);
            gate_vals[g] = sa_clamp01<T>(gate_vals[g]);
        }
    }

    // Channel currents.
    T I_total = T(0);
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        const auto& ch = p.d_channel_descs[c];
        T gate_prod = T(1);
        for (int gi = 0; gi < ch.gate_ref_count; ++gi) {
            const auto& ref = p.d_channel_gate_refs[ch.gate_ref_start + gi];
            if (ref.gate_idx >= 0 && ref.gate_idx < NG)
                for (int pp = 0; pp < ref.power; ++pp) gate_prod *= gate_vals[ref.gate_idx];
        }
        const T E_rev = (ch.nernst_substance_idx >= 0 && ch.nernst_substance_idx < NS)
            ? e_nernst[ch.nernst_substance_idx] : T(ch.E_rev);
        const T I_ci = ch.is_ahp
            ? T(ch.g) * (x_vals[ch.ahp_substance_idx] / fmax(x_vals[ch.ahp_substance_idx] + T(ch.ahp_k1), T(1e-10))) * (v - E_rev)
            : T(ch.g) * gate_prod * (v - E_rev);
        I_total += I_ci;
    }

    v += dt * (-I_total + current) / T(p.C_m);

    // Intracellular substances — DECAY / DRIVEN_DECAY(_NERNST) standard forms.
    // Recomputes source-channel currents at post-update V for correct Ca2+ driving
    // force (see STN calcium post-update V memory note).
    #pragma unroll
    for (int s = 0; s < NS; ++s) {
        const auto& intr = p.d_intr_descs[s];
        T dX;
        if (intr.update_form == 0) {            // DECAY
            dX = -T(intr.k_decay) * x_vals[s];
        } else {                                // DRIVEN_DECAY (_NERNST)
            T I_src = T(0);
            for (int sc = 0; sc < intr.source_count; ++sc) {
                const int ci = p.d_intr_source_channels[intr.source_start + sc];
                if (ci < 0 || ci >= NC) continue;
                const auto& ch = p.d_channel_descs[ci];
                T gate_prod = T(1);
                for (int gi = 0; gi < ch.gate_ref_count; ++gi) {
                    const auto& ref = p.d_channel_gate_refs[ch.gate_ref_start + gi];
                    if (ref.gate_idx >= 0 && ref.gate_idx < NG)
                        for (int pp = 0; pp < ref.power; ++pp) gate_prod *= gate_vals[ref.gate_idx];
                }
                const T E_rev = (ch.nernst_substance_idx >= 0 && ch.nernst_substance_idx < NS)
                    ? e_nernst[ch.nernst_substance_idx] : T(ch.E_rev);
                I_src += ch.is_ahp
                    ? T(ch.g) * (x_vals[ch.ahp_substance_idx] / fmax(x_vals[ch.ahp_substance_idx] + T(ch.ahp_k1), T(1e-10))) * (v - E_rev)
                    : T(ch.g) * gate_prod * (v - E_rev);
            }
            dX = T(intr.epsilon) * (-I_src - T(intr.k_decay) * x_vals[s]);
        }
        x_vals[s] = fmax(T(0), x_vals[s] + dt * dX);
        if (intr.nernst_enabled)
            e_nernst[s] = T(intr.nernst_R * intr.nernst_T) / T(intr.nernst_z * intr.nernst_F)
                * log(T(intr.nernst_Ca_o) / fmax(x_vals[s], T(1e-10)));
    }

    // Write back V, substances. Write DERIVED gate values so d_gate_state stays
    // consistent for download_state() and the next step's P3b read.
    // Non-DERIVED gates are already current in d_gate_state from P3a.
    p.d_V[i] = double(v);
    p.d_synapse_g_scale[i] = 1.0;
    #pragma unroll
    for (int g = 0; g < NG; ++g) {
        if (p.d_gate_descs[g].update_form == 3)
            p.d_gate_state[g * stride + i] = double(gate_vals[g]);
    }
    #pragma unroll
    for (int s = 0; s < NS; ++s) {
        p.d_substance_state[s * stride + i] = double(x_vals[s]);
        p.d_nernst_state[s * stride + i]    = double(e_nernst[s]);
    }
}

// Per-pool dispatch for P3b (channel + V + substance update), templated on T.
// Bucket list MUST stay in sync with coop_fast_eligible() — same pools, same constraints.
template <typename T>
__device__ inline void composable_channel_vupdate_dispatch(
    const hodgkin_huxley::CudaComposableDesc& cd, int i, T I_in, T dt)
{
    const int g = cd.n_gates, c = cd.n_channels, s = cd.n_intracellulars;
    if (g == 3  && c == 3 && s == 0) { composable_channel_vupdate_unrolled<3, 3, 0, T>(cd, i, I_in, dt);  return; }
    if (g == 4  && c == 4 && s == 0) { composable_channel_vupdate_unrolled<4, 4, 0, T>(cd, i, I_in, dt);  return; }
    if (g == 5  && c == 4 && s == 0) { composable_channel_vupdate_unrolled<5, 4, 0, T>(cd, i, I_in, dt);  return; }
    if (g == 6  && c == 6 && s == 1) { composable_channel_vupdate_unrolled<6, 6, 1, T>(cd, i, I_in, dt);  return; }
    if (g == 11 && c == 7 && s == 1) { composable_channel_vupdate_unrolled<11, 7, 1, T>(cd, i, I_in, dt); return; }
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
    // Per-neuron ring: read this synapse's PREsynaptic neuron history at its delay.
    const bool arrived = syn.d_spike_ring[syn.d_pre[k] * syn.ring_size + read_slot] != 0;
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
        case 4: { double si=sa_boltz<double>(v_pre,desc.s_inf); double tau=fmax(sa_tau<double>(v_pre,desc.tau),1e-10); s=si+(s-si)*exp(-dt/tau); break; }
        case 5: { double al=sa_rate<double>(v_pre,desc.alpha),be=sa_rate<double>(v_pre,desc.beta); double rt=al+be; double si=(rt>1e-10)?al/rt:s; s=si+(s-si)*exp(-dt*rt); break; }
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

// Per-(neuron,gate) helper for P3a. Handles INF_TAU (0), ALPHA_BETA (1),
// INSTANT (2). Skips DERIVED (3) — those are cheap linear ops handled in P3b
// after all source gates are visible. One thread per gate slot; no inner loops.
template <typename T>
__device__ inline void composable_gate_update_single(
    const hodgkin_huxley::CudaComposableDesc& p, int i, int g, T v, T dt)
{
    using namespace hodgkin_huxley;
    const auto& desc = p.d_gate_descs[g];

    const T dep = (desc.dependency == 1 && desc.intracellular_idx >= 0
                   && desc.intracellular_idx < p.n_intracellulars)
        ? T(p.d_substance_state[desc.intracellular_idx * p.state_stride + i])
        : v;

    T val = T(p.d_gate_state[g * p.state_stride + i]);

    switch (desc.update_form) {
        case 0: { // INF_TAU
            const T xi  = sa_boltz<T>(dep, desc.inf);
            const T tau = fmax(sa_tau<T>(v, desc.tau), T(1e-10));
            val = xi + (val - xi) * exp(-dt * T(desc.scale) / tau);
            break;
        }
        case 1: { // ALPHA_BETA
            const T al = sa_rate<T>(v, desc.alpha), be = sa_rate<T>(v, desc.beta);
            const T rt = fmax(al + be, T(1e-10));
            const T xi = al / rt;
            val = xi + (val - xi) * exp(-dt * rt);
            break;
        }
        case 2: // INSTANT
            val = sa_boltz<T>(dep, desc.inf);
            break;
        default: return;  // DERIVED: skip, handled in P3b
    }
    p.d_gate_state[g * p.state_stride + i] = double(sa_clamp01<T>(val));
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

// Flat neuron-processing layout (model-layout fusion). One entry per neuron in
// the whole network, ordered HH → Iz → composable-by-bucket so same-type neurons
// are contiguous. Lets the step phase run as ONE flat loop over all neurons
// instead of a sequential loop per pool — critical for heterogeneous models
// with many small pools (e.g. CTX-BG-TH: 8 pools × 10 neurons), where a per-pool
// loop activates < 1 warp at a time and starves the SM of concurrent warps.
struct NeuronSlot {
    int kind;   // 0 = HH, 1 = Izhikevich, 2 = composable
    int pool;   // index into the corresponding desc array
    int local;  // neuron index within that pool
};

// One entry per (composable neuron, non-DERIVED gate) pair. Built on the host
// in simulate_all_plan_create, uploaded once, reused every step. Lets P3a run
// one gate update per thread — N_neurons*N_gates concurrent threads instead of
// N_neurons, which is the main bottleneck for high-gate-count models (STN: 11).
struct GateSlot {
    int pool;   // index into comp_descs
    int local;  // neuron index within that pool
    int gate;   // gate index (0..n_gates-1); DERIVED gates (update_form==3) excluded
};

// ---------------------------------------------------------------------------
// Main simulation kernel — templated on MultiBlock.
//
// PRE-LOOP (once per kernel invocation):
//   Scatter pool V → d_V_cache; zero d_I_syn.               [barrier]
//   Amortised over all steps — not counted in the per-step cost.
//
// Per-step phases:
//   PHASE A  (all concurrent): record V from d_V_cache;
//             stim → atomicAdd(d_I_syn); P2 atomicAdd(d_I_syn);
//             P3a gate updates                               [barrier]
//   PHASE B  (fused per-neuron): step → zero d_I_syn[nidx] →
//             scatter V (→ d_V_cache) → detect spike         [barrier]
//   PHASE C  P6 spike ring          [barrier ONLY if min_delay=0]
//             P7 synapse state update                        [barrier, end-of-step]
//
// Phase A is fully concurrent because:
//   - Record reads d_V_cache (written by Phase B of previous step; not touched by stim/P2/P3a).
//   - Stim and P2 both write d_I_syn via atomicAdd (d_I_syn zeroed by Phase B of prev step;
//     no assignment vs. atomicAdd race).
//   - P3a reads cd.d_V / d_gate_state / d_substance_state, none of which are modified by
//     stim, P2, or recording.
//
// Barrier count: 3 per step for typical networks (4 with min_delay_steps == 0).
// ---------------------------------------------------------------------------

template <bool MultiBlock, typename T>
__device__ void simulate_all_kernel_impl(
    hodgkin_huxley::CudaHHDesc*         hh_descs,   int n_hh,
    hodgkin_huxley::CudaIzDesc*         iz_descs,   int n_iz,
    hodgkin_huxley::CudaComposableDesc* comp_descs, int n_comp,
    const NeuronSlot*                   layout,
    const GateSlot*                     gate_layout,
    int                                 n_gate_slots,
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
    const T t_dt = T(dt);

    // ---- Pre-loop init (once per kernel invocation) ----
    // Populate d_V_cache from current pool voltages so Phase A of step 0 has valid
    // pre-step voltages for recording and synaptic current computation.
    // Zero d_I_syn so Phase A's atomicAdds start from 0.
    // (For chunk j>0: the previous chunk's last Phase B already left d_V_cache current
    // and d_I_syn zeroed — this is a cheap redundant write, not wrong.)
    for (int p = 0; p < n_hh; ++p)
        for (int i = tid; i < hh_descs[p].n; i += total)
            d_V_cache[hh_descs[p].d_net_idx[i]] = hh_descs[p].d_V[i];
    for (int p = 0; p < n_iz; ++p)
        for (int i = tid; i < iz_descs[p].n; i += total)
            d_V_cache[iz_descs[p].d_net_idx[i]] = iz_descs[p].d_v[i];
    for (int p = 0; p < n_comp; ++p)
        for (int i = tid; i < comp_descs[p].n; i += total)
            d_V_cache[comp_descs[p].d_net_idx[i]] = comp_descs[p].d_V[i];
    for (int i = tid; i < n_neurons; i += total)
        d_I_syn[i] = 0.0;
    sync_step<MultiBlock>(grid);

    for (size_t t = 0; t < num_steps; ++t) {
        const size_t step = step_start + t;

        // ---- PHASE A: record + stim + I_syn accumulation + gate updates ----
        // All four run concurrently — no shared writes:
        //   Record reads d_V_cache (valid from Phase B of previous step).
        //   Stim and P2 both atomicAdd to d_I_syn (zeroed at end of previous Phase B).
        //   P3a writes d_gate_state, reads cd.d_V — neither touched by the other work.

        // Record pre-step voltage (CPU-matching semantics: V before the step)
        if (d_V_out && record_interval > 0 && (int)(t % static_cast<size_t>(record_interval)) == 0) {
            const int tr = static_cast<int>(t / static_cast<size_t>(record_interval));
            for (int i = tid; i < n_neurons; i += total)
                d_V_out[static_cast<size_t>(i) * static_cast<size_t>(n_rec) + static_cast<size_t>(tr)] = d_V_cache[i];
        }
        // Stim → atomicAdd to d_I_syn (concurrent with P2; both are atomicAdd, no race)
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
                    && static_cast<uint32_t>(i) < dd.neuron_end)
                    if (static_cast<uint32_t>(step % dd.isi_steps) < dd.pw_steps)
                        I += dd.amplitude;
            }
            atomicAdd(&d_I_syn[i], I);
        }
        // P2: accumulate synaptic currents
        for (int s = tid; s < syn.n_synapses; s += total)
            accumulate_isyn_single(syn, s, d_V_cache, d_I_syn);
        // P3a: thread-per-gate update (non-DERIVED composable gates only)
        for (int gs = tid; gs < n_gate_slots; gs += total) {
            const GateSlot slot = gate_layout[gs];
            const auto& cd = comp_descs[slot.pool];
            composable_gate_update_single<T>(cd, slot.local, slot.gate,
                                             T(cd.d_V[slot.local]), t_dt);
        }
        sync_step<MultiBlock>(grid);

        // ---- PHASE B: fused per-neuron step → scatter V → reset d_I_syn → detect spike ----
        // Each thread owns exactly one neuron (nidx); it:
        //   reads  d_I_syn[nidx] (fully accumulated by Phase A barrier)
        //   reads  d_gate_state  (updated by P3a in Phase A)
        //   steps  the neuron
        //   writes d_I_syn[nidx] = 0.0  (reset for next step's Phase A atomicAdds)
        //   writes d_V_cache[nidx] = v_new (pre-step V for next step's Phase A)
        const int tr_spike = (record_interval > 0)
            ? static_cast<int>(t / static_cast<size_t>(record_interval)) : 0;
        const bool record_spikes = (d_spike_buf != nullptr) && record_interval > 0 && tr_spike < n_rec;

        for (int k = tid; k < n_neurons; k += total) {
            const NeuronSlot sl = layout[k];
            size_t nidx;
            double v_new;
            if (sl.kind == 0) {
                const auto& d = hh_descs[sl.pool];
                nidx = d.d_net_idx[sl.local];
                hh_step_single<T>(d, sl.local, T(d_I_syn[nidx]), t_dt);
                v_new = d.d_V[sl.local];
            } else if (sl.kind == 1) {
                const auto& d = iz_descs[sl.pool];
                nidx = d.d_net_idx[sl.local];
                iz_step_single<T>(d, sl.local, T(d_I_syn[nidx]), t_dt);
                v_new = d.d_v[sl.local];
            } else {
                const auto& d = comp_descs[sl.pool];
                nidx = d.d_net_idx[sl.local];
                composable_channel_vupdate_dispatch<T>(d, sl.local, T(d_I_syn[nidx]), t_dt);
                v_new = d.d_V[sl.local];
            }
            d_I_syn[nidx]   = 0.0;   // reset: this thread owns nidx, no other Phase B thread touches it
            d_V_cache[nidx] = v_new; // scatter: pre-step V for next step's Phase A
            if (syn.n_synapses > 0 || record_spikes) {
                const double vp = syn.d_V_prev ? syn.d_V_prev[nidx] : v_new;
                const uint8_t spiked = (v_new > syn.spike_threshold && vp <= syn.spike_threshold) ? 1u : 0u;
                if (syn.d_neuron_spiked) syn.d_neuron_spiked[nidx] = spiked;
                if (syn.d_V_prev)        syn.d_V_prev[nidx] = v_new;
                if (record_spikes)
                    d_spike_buf[nidx * static_cast<size_t>(n_rec) + static_cast<size_t>(tr_spike)] |= spiked;
            }
        }
        sync_step<MultiBlock>(grid);

        // ---- PHASE C: spike ring + synapse state update ----
        if (syn.n_synapses > 0) {
            const int write_slot = static_cast<int>(step % static_cast<size_t>(syn.ring_size));
            for (int i = tid; i < syn.n_neurons; i += total)
                syn.d_spike_ring[i * syn.ring_size + write_slot] = syn.d_neuron_spiked[i] ? 1u : 0u;
        }
        // Barrier only required when min_delay_steps == 0 (read_slot == write_slot possible)
        if (syn.min_delay_steps < 1)
            sync_step<MultiBlock>(grid);

        for (int s = tid; s < syn.n_synapses; s += total)
            update_synapse_state_single(syn, s, d_V_cache, step, dt);
        sync_step<MultiBlock>(grid);
    }
}

// Macro to avoid repeating the full parameter list 4 times.
#define SIM_ALL_KERNEL_PARAMS \
    hodgkin_huxley::CudaHHDesc*         hh_descs,   int n_hh,     \
    hodgkin_huxley::CudaIzDesc*         iz_descs,   int n_iz,     \
    hodgkin_huxley::CudaComposableDesc* comp_descs, int n_comp,   \
    const NeuronSlot*                   layout,                   \
    const GateSlot*                     gate_layout,              \
    int                                 n_gate_slots,             \
    hodgkin_huxley::DeviceSynapseRaw    syn,                      \
    hodgkin_huxley::CudaStimRaw         stim,                     \
    double*  d_V_cache, double* d_I_syn,                          \
    double*  d_V_out,   uint8_t* d_spike_buf,                     \
    int n_neurons, size_t num_steps, double dt,                   \
    size_t step_start, int record_interval, int n_rec

#define SIM_ALL_KERNEL_ARGS \
    hh_descs, n_hh, iz_descs, n_iz, comp_descs, n_comp, layout,  \
    gate_layout, n_gate_slots,                                    \
    syn, stim, d_V_cache, d_I_syn, d_V_out, d_spike_buf,         \
    n_neurons, num_steps, dt, step_start, record_interval, n_rec

__global__ void simulate_all_kernel_multi_f64(SIM_ALL_KERNEL_PARAMS)
{ simulate_all_kernel_impl<true,  double>(SIM_ALL_KERNEL_ARGS); }

__global__ void simulate_all_kernel_multi_f32(SIM_ALL_KERNEL_PARAMS)
{ simulate_all_kernel_impl<true,  float>(SIM_ALL_KERNEL_ARGS); }

__global__ void simulate_all_kernel_single_f64(SIM_ALL_KERNEL_PARAMS)
{ simulate_all_kernel_impl<false, double>(SIM_ALL_KERNEL_ARGS); }

__global__ void simulate_all_kernel_single_f32(SIM_ALL_KERNEL_PARAMS)
{ simulate_all_kernel_impl<false, float>(SIM_ALL_KERNEL_ARGS); }

#undef SIM_ALL_KERNEL_PARAMS
#undef SIM_ALL_KERNEL_ARGS

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

// Upload pool descriptors + decide launch config. Runs ONCE; the returned plan
// is reused across all per-chunk launches (so we never cudaMalloc/Free or sync
// inside the chunk loop — that would serialize the copy/compute pipeline and
// could free d_layout while a no-sync kernel still reads it).
SimAllPlan simulate_all_plan_create(
    const CudaHHDesc* hh_descs_h, int n_hh,
    const CudaIzDesc* iz_descs_h, int n_iz,
    const CudaComposableDesc* comp_descs_h, int n_comp,
    size_t n_neurons, int n_synapses, int stim_n_neurons,
    bool use_float32)
{
    SimAllPlan p;
    p.n_hh = n_hh; p.n_iz = n_iz; p.n_comp = n_comp;
    p.use_float32 = use_float32;
    ck(cudaGetDevice(&p.device), "simulate_all_plan_create: cudaGetDevice");

    if (n_hh > 0) {
        ck(cudaMalloc(reinterpret_cast<void**>(&p.d_hh),   n_hh   * sizeof(CudaHHDesc)),   "d_hh alloc");
        ck(cudaMemcpy(p.d_hh,   hh_descs_h,   n_hh   * sizeof(CudaHHDesc),   cudaMemcpyHostToDevice), "d_hh memcpy");
    }
    if (n_iz > 0) {
        ck(cudaMalloc(reinterpret_cast<void**>(&p.d_iz),   n_iz   * sizeof(CudaIzDesc)),   "d_iz alloc");
        ck(cudaMemcpy(p.d_iz,   iz_descs_h,   n_iz   * sizeof(CudaIzDesc),   cudaMemcpyHostToDevice), "d_iz memcpy");
    }
    if (n_comp > 0) {
        ck(cudaMalloc(reinterpret_cast<void**>(&p.d_comp), n_comp * sizeof(CudaComposableDesc)), "d_comp alloc");
        ck(cudaMemcpy(p.d_comp, comp_descs_h, n_comp * sizeof(CudaComposableDesc), cudaMemcpyHostToDevice), "d_comp memcpy");
    }

    // Flat neuron-processing layout (model-layout fusion): HH → Iz → composable
    // sorted by (n_gates, n_channels, n_intracellulars) so same-bucket neurons
    // are contiguous (minimizes warp divergence). Constant across chunks.
    std::vector<NeuronSlot> layout;
    layout.reserve(n_neurons);
    for (int q = 0; q < n_hh; ++q)
        for (int i = 0; i < hh_descs_h[q].n; ++i) layout.push_back({0, q, i});
    for (int q = 0; q < n_iz; ++q)
        for (int i = 0; i < iz_descs_h[q].n; ++i) layout.push_back({1, q, i});
    {
        std::vector<int> comp_order(n_comp);
        for (int q = 0; q < n_comp; ++q) comp_order[q] = q;
        std::sort(comp_order.begin(), comp_order.end(), [&](int a, int b) {
            const auto& A = comp_descs_h[a]; const auto& B = comp_descs_h[b];
            if (A.n_gates != B.n_gates)       return A.n_gates < B.n_gates;
            if (A.n_channels != B.n_channels) return A.n_channels < B.n_channels;
            return A.n_intracellulars < B.n_intracellulars;
        });
        for (int q : comp_order)
            for (int i = 0; i < comp_descs_h[q].n; ++i) layout.push_back({2, q, i});
    }
    if (!layout.empty()) {
        NeuronSlot* d_layout = nullptr;
        ck(cudaMalloc(reinterpret_cast<void**>(&d_layout), layout.size() * sizeof(NeuronSlot)),
           "d_layout alloc");
        ck(cudaMemcpy(d_layout, layout.data(), layout.size() * sizeof(NeuronSlot),
                      cudaMemcpyHostToDevice), "d_layout memcpy");
        p.d_layout = d_layout;
    }

    // Gate-slot array: one entry per (composable neuron, non-DERIVED gate). Lets P3a
    // run one gate-kinetics thread per slot — N_neurons*N_gates concurrent threads.
    {
        std::vector<GateSlot> gate_layout;
        for (int q = 0; q < n_comp; ++q) {
            const auto& cd = comp_descs_h[q];
            if (!cd.h_gate_descs) continue;
            for (int i = 0; i < cd.n; ++i)
                for (int g = 0; g < cd.n_gates; ++g)
                    if (cd.h_gate_descs[g].update_form != 3)  // skip DERIVED
                        gate_layout.push_back({q, i, g});
        }
        p.n_gate_slots = static_cast<int>(gate_layout.size());
        if (!gate_layout.empty()) {
            GateSlot* d_gate_layout = nullptr;
            ck(cudaMalloc(reinterpret_cast<void**>(&d_gate_layout),
                          gate_layout.size() * sizeof(GateSlot)), "d_gate_layout alloc");
            ck(cudaMemcpy(d_gate_layout, gate_layout.data(),
                          gate_layout.size() * sizeof(GateSlot),
                          cudaMemcpyHostToDevice), "d_gate_layout memcpy");
            p.d_gate_layout = d_gate_layout;
        }
    }

    // Single-block vs cooperative decision — work_items is constant across chunks.
    constexpr int kBlockSize        = 256;
    const int single_block_max_work = 256;
    int work_items = static_cast<int>(n_neurons);
    if (n_synapses     > work_items) work_items = n_synapses;
    if (stim_n_neurons > work_items) work_items = stim_n_neurons;
    if (p.n_gate_slots > work_items) work_items = p.n_gate_slots;
    if (work_items < 1) work_items = 1;
    p.use_single_block = (work_items <= single_block_max_work);

    if (p.use_single_block) {
        int sb = ((work_items + 31) / 32) * 32;
        if (sb < 32)         sb = 32;
        if (sb > kBlockSize) sb = kBlockSize;
        p.sb_threads = sb;
    } else {
        int supports_coop = 0;
        ck(cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, p.device),
           "simulate_all_plan_create: cooperative attr");
        if (!supports_coop)
            throw std::runtime_error("simulate_all_plan_create: GPU does not support cooperative launch");
        int sm_count = 0, max_blocks_sm = 0;
        ck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, p.device),
           "simulate_all_plan_create: SM count");
        const void* occ_kernel = use_float32
            ? (const void*)simulate_all_kernel_multi_f32
            : (const void*)simulate_all_kernel_multi_f64;
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
               &max_blocks_sm, occ_kernel, kBlockSize, 0),
           "simulate_all_plan_create: occupancy");
        const int blocks_needed = (work_items + kBlockSize - 1) / kBlockSize;
        p.total_blocks = std::min(blocks_needed, sm_count * max_blocks_sm);
        if (p.total_blocks <= 0)
            throw std::runtime_error("simulate_all_plan_create: 0 cooperative blocks");
    }
    return p;
}

// Enqueue one kernel launch on `stream`. No sync, no free — the caller manages
// streams/events and frees the plan once at the end.
void simulate_all_launch(
    const SimAllPlan& plan,
    DeviceSynapseRaw syn, CudaStimRaw stim,
    double* d_V_cache, double* d_I_syn, double* d_V_out, uint8_t* d_spike_buf,
    size_t n_neurons, size_t num_steps, double dt, size_t step_start,
    size_t record_interval, size_t n_rec, cudaStream_t stream)
{
    if (num_steps == 0) return;
    constexpr int kBlockSize = 256;

    // Locals whose addresses feed the cooperative-launch arg list.
    CudaHHDesc*         d_hh         = plan.d_hh;   int n_hh   = plan.n_hh;
    CudaIzDesc*         d_iz         = plan.d_iz;   int n_iz   = plan.n_iz;
    CudaComposableDesc* d_comp       = plan.d_comp; int n_comp = plan.n_comp;
    NeuronSlot*         d_layout     = static_cast<NeuronSlot*>(plan.d_layout);
    GateSlot*           d_gate_layout = static_cast<GateSlot*>(plan.d_gate_layout);
    int                 i_n_gate_slots = plan.n_gate_slots;
    int    i_n_neurons  = static_cast<int>(n_neurons);
    size_t sz_num_steps = num_steps;
    int    i_rec_iv     = static_cast<int>(record_interval);
    int    i_n_rec      = static_cast<int>(n_rec);

    void* args[] = {
        &d_hh, &n_hh, &d_iz, &n_iz, &d_comp, &n_comp, &d_layout,
        &d_gate_layout, &i_n_gate_slots,
        &syn, &stim, &d_V_cache, &d_I_syn, &d_V_out, &d_spike_buf,
        &i_n_neurons, &sz_num_steps, &dt, &step_start, &i_rec_iv, &i_n_rec
    };
    if (plan.use_single_block) {
        if (plan.use_float32) {
            simulate_all_kernel_single_f32<<<1, plan.sb_threads, 0, stream>>>(
                d_hh, n_hh, d_iz, n_iz, d_comp, n_comp, d_layout,
                d_gate_layout, i_n_gate_slots,
                syn, stim,
                d_V_cache, d_I_syn, d_V_out, d_spike_buf,
                i_n_neurons, sz_num_steps, dt, step_start, i_rec_iv, i_n_rec);
        } else {
            simulate_all_kernel_single_f64<<<1, plan.sb_threads, 0, stream>>>(
                d_hh, n_hh, d_iz, n_iz, d_comp, n_comp, d_layout,
                d_gate_layout, i_n_gate_slots,
                syn, stim,
                d_V_cache, d_I_syn, d_V_out, d_spike_buf,
                i_n_neurons, sz_num_steps, dt, step_start, i_rec_iv, i_n_rec);
        }
        ck(cudaGetLastError(), "simulate_all_launch: single-block launch");
    } else {
        void* coop_kernel = plan.use_float32
            ? reinterpret_cast<void*>(simulate_all_kernel_multi_f32)
            : reinterpret_cast<void*>(simulate_all_kernel_multi_f64);
        ck(cudaLaunchCooperativeKernel(
               coop_kernel,
               dim3(plan.total_blocks), dim3(kBlockSize), args, 0, stream),
           "simulate_all_launch: cudaLaunchCooperativeKernel");
    }
}

void simulate_all_plan_destroy(SimAllPlan& plan) {
    if (plan.d_hh)          cudaFree(plan.d_hh);
    if (plan.d_iz)          cudaFree(plan.d_iz);
    if (plan.d_comp)        cudaFree(plan.d_comp);
    if (plan.d_layout)      cudaFree(plan.d_layout);
    if (plan.d_gate_layout) cudaFree(plan.d_gate_layout);
    plan = SimAllPlan{};
}

// Thin wrapper preserving the original blocking, single-launch behavior.
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
    SimAllPlan plan = simulate_all_plan_create(
        hh_descs_h, n_hh, iz_descs_h, n_iz, comp_descs_h, n_comp,
        n_neurons, syn.n_synapses, stim.n_neurons, /*use_float32=*/true);
    simulate_all_launch(plan, syn, stim, d_V_cache, d_I_syn, d_V_out, d_spike_buf,
                        n_neurons, num_steps, dt, step_start, record_interval, n_rec, stream);
    ck(cudaStreamSynchronize(stream), "simulate_all_steps: stream sync");
    simulate_all_plan_destroy(plan);
}

} // namespace hodgkin_huxley
