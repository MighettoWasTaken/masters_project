#include "hodgkin_huxley/cuda_hh_pool.hpp"
#include "hodgkin_huxley/device.hpp"
#include <cuda_runtime.h>

namespace hodgkin_huxley {
void CudaHHPool::synchronize() { cudaDeviceSynchronize(); }
} // namespace hodgkin_huxley


// __global__ kernels must be at file scope, not inside a namespace.
// On MSVC, nvcc's generated .cudafe1.stub.c is compiled as C (not C++), so
// namespace-qualified names like hodgkin_huxley::kernel would be a syntax error.
__global__ void hh_smoke_kernel(double* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 2.0 * i + 1.0;
}

namespace hodgkin_huxley {

// 0=ok, 1=setDevice, 2=malloc, 3=kernel launch, 4=sync, 5=memcpy, 6+i=bad value at index i
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
        cudaFree(d_arr); return 5.0;
    }
    cudaFree(d_arr);

    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 2.0 * i + 1.0) return 6.0 + i;
    }
    return 0.0;
}

} // namespace hodgkin_huxley
