#include "hodgkin_huxley/device.hpp"

#ifdef HH_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace hodgkin_huxley {

std::string Device::str() const {
  if (type == Type::CPU)
    return "cpu";
  return "cuda:" + std::to_string(index);
}

#ifdef HH_USE_CUDA
#include <cuda_runtime_api.h>
int cuda_device_count() {
  int n = 0;
  cudaGetDeviceCount(&n);
  return n;
}
bool cuda_is_available() { return cuda_device_count() > 0; }
std::string cuda_device_name(int idx) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, idx);
  return std::string(prop.name);
}
// cuda_smoke_test is defined in cuda_hh_pool.cu
#else
int cuda_device_count() { return 0; }
bool cuda_is_available() { return false; }
std::string cuda_device_name(int) { return ""; }
double cuda_smoke_test(int) { return -1.0; }
#endif

} // namespace hodgkin_huxley
