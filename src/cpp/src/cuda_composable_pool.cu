#include "hodgkin_huxley/cuda_composable_pool.hpp"
#include <cuda_runtime.h>

namespace hodgkin_huxley {
void CudaComposablePool::synchronize() { cudaDeviceSynchronize(); }
} // namespace hodgkin_huxley
