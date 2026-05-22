// Placeholder for iz_pool
// cuda_iz_pool.cu (sketch)
#include "hodgkin_huxley/cuda_iz_pool.hpp"
#include <cuda_runtime.h>

namespace hodgkin_huxley
{

    __global__ void iz_step_kernel(
        double *V, double *u,
        const double *a, const double *b, const double *c, const double *d,
        const double *I_ext,
        double dt, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N)
            return;

        double v = V[i];
        double ui = u[i];

        // Izhikevich forward Euler:
        double dv = 0.04 * v * v + 5.0 * v + 140.0 - ui + I_ext[i];
        double du = a[i] * (b[i] * v - ui);

        v += dt * dv;
        ui += dt * du;

        // predicated reset on threshold — avoids divergent branches
        bool fired = (v >= 30.0);
        V[i] = fired ? c[i] : v;
        u[i] = fired ? (ui + d[i]) : ui;
    }

    __global__ void scatter_kernel(const double *d_V, const size_t *d_net_idx, double *V_buf, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        size_t dest = d_net_idx[i];
        V_buf[dest] = d_V[i];
    }

    __global__ void gather_currents_kernel(const double *I_buf, double *d_I, const size_t *d_net_idx, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        size_t src = d_net_idx[i];
        d_I[i] = I_buf[src];
    }

    // Pool method examples (allocate, step, scatter/gather)
    void CudaIzPool::step(double dt)
    {
        int N = static_cast<int>(n_);
        if (N == 0)
            return;
        int block = 256;
        int grid = (N + block - 1) / block;
        iz_step_kernel<<<grid, block, 0, stream_>>>(
            d_V_, d_u_, d_a_, d_b_, d_c_, d_d_, d_I_, dt, N);
    }

    void CudaIzPool::scatter_voltages(double *V_buf) const
    {
        // write into pinned host buffer indexed by net_idx: implement with small kernel + async memcpy if needed
        int N = static_cast<int>(n_);
        if (N == 0)
            return;
        int block = 256;
        int grid = (N + block - 1) / block;
        scatter_kernel<<<grid, block, 0, stream_>>>(d_V_, d_net_idx_, V_buf, N);
        // Note: caller must synchronize/stream ordering before CPU reads V_buf
    }

    void CudaIzPool::gather_currents(const double *I_buf)
    {
        int N = static_cast<int>(n_);
        if (N == 0)
            return;
        // copy I_buf (host pinned) into device d_I_ via cudaMemcpyAsync or small kernel reading by net_idx:
        // Option A: use cudaMemcpyAsync if I_buf is contiguous per-d_net_idx ordering (often not).
        // Option B: launch a kernel to scatter from I_buf[src] -> d_I_[i], using d_net_idx_ (shown below).
        int block = 256;
        int grid = (N + block - 1) / block;
        // we need a device-accessible I_buf; if I_buf is pinned host memory, prefer cudaMemcpyAsync to a temp device buffer then kernel.
        // For simplicity:
        // 1) cudaMemcpyAsync(host->device temp buffer), 2) use gather_currents_kernel to permute into d_I_ using d_net_idx_.
    }

} // namespace hodgkin_huxley