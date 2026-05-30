#include "hodgkin_huxley/cuda_hh_pool.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

__device__ inline double hh_clamp01(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

__device__ inline void hh_scalar_derivs(
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

    dV = (I - gNa * m * m * m * h * (V - ENa)
            - gK * n * n * n * n * (V - EK)
            - gL * (V - EL)) / Cm;
    dm = am * (1.0 - m) - bm * m;
    dh = ah * (1.0 - h) - bh * h;
    dn = an * (1.0 - n) - bn * n;
}

__global__ void hh_rk4_kernel(
    double* V, double* m, double* h, double* n,
    const double* I_ext,
    const double* C_m, const double* g_Na, const double* g_K, const double* g_L,
    const double* E_Na, const double* E_K, const double* E_L,
    int N, double dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double Cm = C_m[i];
    const double gNa = g_Na[i];
    const double gK = g_K[i];
    const double gL = g_L[i];
    const double ENa = E_Na[i];
    const double EK = E_K[i];
    const double EL = E_L[i];
    const double I = I_ext[i];

    const double V0 = V[i];
    const double m0 = m[i];
    const double h0 = h[i];
    const double n0 = n[i];

    double k1V, k1m, k1h, k1n;
    hh_scalar_derivs(V0, m0, h0, n0, Cm, gNa, gK, gL, ENa, EK, EL, I,
                     k1V, k1m, k1h, k1n);

    double k2V, k2m, k2h, k2n;
    hh_scalar_derivs(V0 + 0.5 * dt * k1V,
                     hh_clamp01(m0 + 0.5 * dt * k1m),
                     hh_clamp01(h0 + 0.5 * dt * k1h),
                     hh_clamp01(n0 + 0.5 * dt * k1n),
                     Cm, gNa, gK, gL, ENa, EK, EL, I,
                     k2V, k2m, k2h, k2n);

    double k3V, k3m, k3h, k3n;
    hh_scalar_derivs(V0 + 0.5 * dt * k2V,
                     hh_clamp01(m0 + 0.5 * dt * k2m),
                     hh_clamp01(h0 + 0.5 * dt * k2h),
                     hh_clamp01(n0 + 0.5 * dt * k2n),
                     Cm, gNa, gK, gL, ENa, EK, EL, I,
                     k3V, k3m, k3h, k3n);

    double k4V, k4m, k4h, k4n;
    hh_scalar_derivs(V0 + dt * k3V,
                     hh_clamp01(m0 + dt * k3m),
                     hh_clamp01(h0 + dt * k3h),
                     hh_clamp01(n0 + dt * k3n),
                     Cm, gNa, gK, gL, ENa, EK, EL, I,
                     k4V, k4m, k4h, k4n);

    const double dt6 = dt / 6.0;
    V[i] = V0 + dt6 * (k1V + 2.0 * k2V + 2.0 * k3V + k4V);
    m[i] = hh_clamp01(m0 + dt6 * (k1m + 2.0 * k2m + 2.0 * k3m + k4m));
    h[i] = hh_clamp01(h0 + dt6 * (k1h + 2.0 * k2h + 2.0 * k3h + k4h));
    n[i] = hh_clamp01(n0 + dt6 * (k1n + 2.0 * k2n + 2.0 * k3n + k4n));
}

__global__ void hh_smoke_kernel(double* arr, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 2.0 * i + 1.0;
}

__global__ void scatter_by_index_kernel(
    const size_t* net_idx, const double* src, double* dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[net_idx[i]] = src[i];
}

__global__ void gather_by_index_kernel(
    const size_t* net_idx, const double* src, double* dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[net_idx[i]];
}

} // namespace

namespace hodgkin_huxley {

CudaHHPool::CudaHHPool(int device_id, size_t capacity, bool fast_math)
    : HHPool(capacity, fast_math), device_id_(device_id), capacity_(capacity) {
    check_cuda(cudaSetDevice(device_id_), "CudaHHPool cudaSetDevice");
}

CudaHHPool::~CudaHHPool() {
    free_device();
}

void CudaHHPool::allocate_device() {
    if (device_ready_) return;
    check_cuda(cudaSetDevice(device_id_), "CudaHHPool cudaSetDevice");

    auto alloc = [&](double*& ptr) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr),
                              capacity_ * sizeof(double)),
                   "CudaHHPool cudaMalloc");
    };

    alloc(d_V_);
    alloc(d_m_);
    alloc(d_h_);
    alloc(d_n_);
    alloc(d_I_ext_);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_net_idx_),
                          capacity_ * sizeof(size_t)),
               "CudaHHPool cudaMalloc net_idx");
    alloc(d_C_m_);
    alloc(d_g_Na_);
    alloc(d_g_K_);
    alloc(d_g_L_);
    alloc(d_E_Na_);
    alloc(d_E_K_);
    alloc(d_E_L_);
    device_ready_ = true;
}

void CudaHHPool::free_device() {
    auto free_ptr = [](double*& ptr) {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
    };

    free_ptr(d_V_);
    free_ptr(d_m_);
    free_ptr(d_h_);
    free_ptr(d_n_);
    free_ptr(d_I_ext_);
    if (d_net_idx_) cudaFree(d_net_idx_);
    d_net_idx_ = nullptr;
    free_ptr(d_C_m_);
    free_ptr(d_g_Na_);
    free_ptr(d_g_K_);
    free_ptr(d_g_L_);
    free_ptr(d_E_Na_);
    free_ptr(d_E_K_);
    free_ptr(d_E_L_);
    device_ready_ = false;
    host_state_dirty_ = false;
}

void CudaHHPool::upload_static_parameters() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_C_m_, C_m_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload C_m");
    check_cuda(cudaMemcpyAsync(d_g_Na_, g_Na_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload g_Na");
    check_cuda(cudaMemcpyAsync(d_g_K_, g_K_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload g_K");
    check_cuda(cudaMemcpyAsync(d_g_L_, g_L_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload g_L");
    check_cuda(cudaMemcpyAsync(d_E_Na_, E_Na_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload E_Na");
    check_cuda(cudaMemcpyAsync(d_E_K_, E_K_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload E_K");
    check_cuda(cudaMemcpyAsync(d_E_L_, E_L_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload E_L");
}

void CudaHHPool::upload_state() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_V_, V_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload V");
    check_cuda(cudaMemcpyAsync(d_m_, m_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload m");
    check_cuda(cudaMemcpyAsync(d_h_, h_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload h");
    check_cuda(cudaMemcpyAsync(d_n_, n_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload n");
    host_state_dirty_ = false;
}

void CudaHHPool::upload_currents() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_I_ext_, I_ext_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload I_ext");
}

void CudaHHPool::upload_index_map() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_net_idx_, net_idx_.data(), N_ * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream_),
               "CudaHHPool upload net_idx");
}

void CudaHHPool::download_state() const {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(const_cast<double*>(V_.data()), d_V_, N_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "CudaHHPool download V");
    check_cuda(cudaMemcpyAsync(const_cast<double*>(m_.data()), d_m_, N_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "CudaHHPool download m");
    check_cuda(cudaMemcpyAsync(const_cast<double*>(h_.data()), d_h_, N_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "CudaHHPool download h");
    check_cuda(cudaMemcpyAsync(const_cast<double*>(n_.data()), d_n_, N_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "CudaHHPool download n");
}

void CudaHHPool::ensure_device_state() {
    allocate_device();
    upload_static_parameters();
    upload_state();
    upload_currents();
    upload_index_map();
    check_cuda(cudaStreamSynchronize(stream_), "CudaHHPool initial sync");
}

void CudaHHPool::ensure_host_state() const {
    if (!host_state_dirty_) return;
    const_cast<CudaHHPool*>(this)->download_state();
    check_cuda(cudaStreamSynchronize(stream_), "CudaHHPool host sync");
    host_state_dirty_ = false;
}

void CudaHHPool::scatter_voltages(double* V_buf) const {
    ensure_host_state();
    HHPool::scatter_voltages(V_buf);
}

void CudaHHPool::gather_currents(const double* I_buf) {
    HHPool::gather_currents(I_buf);
    if (!device_ready_) ensure_device_state();
    else upload_currents();
}

void CudaHHPool::step(double dt) {
    if (N_ == 0) return;
    if (!device_ready_) ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    hh_rk4_kernel<<<grid, block, 0, stream_>>>(
        d_V_, d_m_, d_h_, d_n_,
        d_I_ext_,
        d_C_m_, d_g_Na_, d_g_K_, d_g_L_,
        d_E_Na_, d_E_K_, d_E_L_,
        static_cast<int>(N_), dt);
    check_cuda(cudaGetLastError(), "CudaHHPool hh_rk4_kernel");
    host_state_dirty_ = true;
}

void CudaHHPool::scatter_voltages_device(double* d_V_buf) const {
    if (N_ == 0) return;
    if (!device_ready_) const_cast<CudaHHPool*>(this)->ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    scatter_by_index_kernel<<<grid, block, 0, stream_>>>(d_net_idx_, d_V_, d_V_buf,
                                                          static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaHHPool scatter_by_index_kernel");
}

void CudaHHPool::gather_currents_device(const double* d_I_buf) {
    if (N_ == 0) return;
    if (!device_ready_) ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    gather_by_index_kernel<<<grid, block, 0, stream_>>>(d_net_idx_, d_I_buf, d_I_ext_,
                                                         static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaHHPool gather_by_index_kernel");
}

void CudaHHPool::fill_coop_desc(CudaHHDesc& out) const {
    if (!device_ready_) const_cast<CudaHHPool*>(this)->ensure_device_state();
    out.d_V      = d_V_;
    out.d_m      = d_m_;
    out.d_h      = d_h_;
    out.d_n      = d_n_;
    out.d_I_ext  = d_I_ext_;
    out.d_C_m    = d_C_m_;
    out.d_g_Na   = d_g_Na_;
    out.d_g_K    = d_g_K_;
    out.d_g_L    = d_g_L_;
    out.d_E_Na   = d_E_Na_;
    out.d_E_K    = d_E_K_;
    out.d_E_L    = d_E_L_;
    out.d_net_idx = d_net_idx_;
    out.n        = static_cast<int>(N_);
}

void CudaHHPool::sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const {
    ensure_host_state();
    HHPool::sync_to_neurons(neurons);
}

void CudaHHPool::synchronize() {
    check_cuda(cudaStreamSynchronize(stream_), "CudaHHPool synchronize");
}

void CudaHHPool::migrate_to_device(int id) {
    if (id == device_id_) return;
    ensure_host_state();
    free_device();
    device_id_ = id;
    check_cuda(cudaSetDevice(device_id_), "CudaHHPool migrate cudaSetDevice");
    ensure_device_state();
}

double cuda_smoke_test(int device_idx) {
    if (cudaSetDevice(device_idx) != cudaSuccess) return 1.0;

    const int N = 1024;
    double* d_arr = nullptr;
    if (cudaMalloc(&d_arr, N * sizeof(double)) != cudaSuccess) return 2.0;

    hh_smoke_kernel<<<(N + 255) / 256, 256>>>(d_arr, N);
    if (cudaGetLastError() != cudaSuccess) { cudaFree(d_arr); return 3.0; }
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_arr); return 4.0; }

    double h_arr[N];
    if (cudaMemcpy(h_arr, d_arr, N * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_arr);
        return 5.0;
    }
    cudaFree(d_arr);

    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 2.0 * i + 1.0) return 6.0 + i;
    }
    return 0.0;
}

} // namespace hodgkin_huxley
