#include "hodgkin_huxley/cuda_iz_pool.hpp"

#include <algorithm>
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

__global__ void iz_euler_kernel(
    double* v, double* u, const double* I_ext,
    const double* a, const double* b, const double* c, const double* d,
    int N, double dt, double threshold)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double vi = v[i];
    double ui = u[i];
    const bool fired = vi >= threshold;
    vi = fired ? c[i] : vi;
    ui = fired ? ui + d[i] : ui;

    const double dv = 0.04 * vi * vi + 5.0 * vi + 140.0 - ui + I_ext[i];
    const double du = a[i] * (b[i] * vi - ui);

    vi = vi + dt * dv;
    ui = ui + dt * du;

    v[i] = vi > 100.0 ? 100.0 : vi;
    u[i] = ui;
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

CudaIzPool::CudaIzPool(int device_id, size_t capacity)
    : IzPool(capacity), device_id_(device_id), capacity_(capacity) {
    check_cuda(cudaSetDevice(device_id_), "CudaIzPool cudaSetDevice");
}

CudaIzPool::~CudaIzPool() {
    free_device();
}

void CudaIzPool::allocate_device() {
    if (device_ready_) return;
    check_cuda(cudaSetDevice(device_id_), "CudaIzPool cudaSetDevice");

    auto alloc = [&](double*& ptr) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr),
                              capacity_ * sizeof(double)),
                   "CudaIzPool cudaMalloc");
    };

    alloc(d_v_);
    alloc(d_u_);
    alloc(d_I_ext_);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_net_idx_),
                          capacity_ * sizeof(size_t)),
               "CudaIzPool cudaMalloc net_idx");
    alloc(d_a_);
    alloc(d_b_);
    alloc(d_c_);
    alloc(d_d_);
    device_ready_ = true;
}

void CudaIzPool::free_device() {
    auto free_ptr = [](double*& ptr) {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
    };

    free_ptr(d_v_);
    free_ptr(d_u_);
    free_ptr(d_I_ext_);
    if (d_net_idx_) cudaFree(d_net_idx_);
    d_net_idx_ = nullptr;
    free_ptr(d_a_);
    free_ptr(d_b_);
    free_ptr(d_c_);
    free_ptr(d_d_);
    device_ready_ = false;
    host_state_dirty_ = false;
}

void CudaIzPool::upload_static_parameters() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_a_, a_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload a");
    check_cuda(cudaMemcpyAsync(d_b_, b_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload b");
    check_cuda(cudaMemcpyAsync(d_c_, c_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload c");
    check_cuda(cudaMemcpyAsync(d_d_, d_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload d");
}

void CudaIzPool::upload_state() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_v_, v_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload v");
    check_cuda(cudaMemcpyAsync(d_u_, u_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload u");
    host_state_dirty_ = false;
}

void CudaIzPool::upload_currents() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_I_ext_, I_ext_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload I_ext");
}

void CudaIzPool::upload_index_map() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_net_idx_, net_idx_.data(), N_ * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream_),
               "CudaIzPool upload net_idx");
}

void CudaIzPool::download_state() const {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(const_cast<double*>(v_.data()), d_v_, N_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "CudaIzPool download v");
    check_cuda(cudaMemcpyAsync(const_cast<double*>(u_.data()), d_u_, N_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "CudaIzPool download u");
}

void CudaIzPool::ensure_device_state() {
    allocate_device();
    upload_static_parameters();
    upload_state();
    upload_currents();
    upload_index_map();
    check_cuda(cudaStreamSynchronize(stream_), "CudaIzPool initial sync");
}

void CudaIzPool::ensure_host_state() const {
    if (!host_state_dirty_) return;
    const_cast<CudaIzPool*>(this)->download_state();
    check_cuda(cudaStreamSynchronize(stream_), "CudaIzPool host sync");
    host_state_dirty_ = false;
}

void CudaIzPool::scatter_voltages(double* V_buf) const {
    ensure_host_state();
    IzPool::scatter_voltages(V_buf);
}

void CudaIzPool::gather_currents(const double* I_buf) {
    IzPool::gather_currents(I_buf);
    if (!device_ready_) ensure_device_state();
    else upload_currents();
}

void CudaIzPool::step(double dt) {
    if (N_ == 0) return;
    if (!device_ready_) ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    iz_euler_kernel<<<grid, block, 0, stream_>>>(
        d_v_, d_u_, d_I_ext_, d_a_, d_b_, d_c_, d_d_,
        static_cast<int>(N_), dt, SPIKE_THRESHOLD);
    check_cuda(cudaGetLastError(), "CudaIzPool iz_euler_kernel");
    host_state_dirty_ = true;
}

void CudaIzPool::scatter_voltages_device(double* d_V_buf) const {
    if (N_ == 0) return;
    if (!device_ready_) const_cast<CudaIzPool*>(this)->ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    scatter_by_index_kernel<<<grid, block, 0, stream_>>>(d_net_idx_, d_v_, d_V_buf,
                                                          static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaIzPool scatter_by_index_kernel");
}

void CudaIzPool::gather_currents_device(const double* d_I_buf) {
    if (N_ == 0) return;
    if (!device_ready_) ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    gather_by_index_kernel<<<grid, block, 0, stream_>>>(d_net_idx_, d_I_buf, d_I_ext_,
                                                         static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaIzPool gather_by_index_kernel");
}

void CudaIzPool::sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const {
    ensure_host_state();
    IzPool::sync_to_neurons(neurons);
}

void CudaIzPool::synchronize() {
    check_cuda(cudaStreamSynchronize(stream_), "CudaIzPool synchronize");
}

void CudaIzPool::migrate_to_device(int id) {
    if (id == device_id_) return;
    ensure_host_state();
    free_device();
    device_id_ = id;
    check_cuda(cudaSetDevice(device_id_), "CudaIzPool migrate cudaSetDevice");
    ensure_device_state();
}

} // namespace hodgkin_huxley
