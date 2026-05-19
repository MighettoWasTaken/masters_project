#include "hodgkin_huxley/cuda_iz_pool.hpp"
#include <cuda_runtime.h>

namespace hodgkin_huxley {
void CudaIzPool::synchronize() { cudaDeviceSynchronize(); }
} // namespace hodgkin_huxley
