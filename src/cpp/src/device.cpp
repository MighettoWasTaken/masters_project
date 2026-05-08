#include "hodgkin_huxley/device.hpp"

namespace hodgkin_huxley {

std::string Device::str() const {
    if (type == Type::CPU) return "cpu";
    return "cuda:" + std::to_string(index);
}

#ifdef HH_USE_CUDA
#include <cuda_runtime.h>
int cuda_device_count() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}
bool cuda_is_available() { return cuda_device_count() > 0; }
#else
int  cuda_device_count() { return 0; }
bool cuda_is_available()  { return false; }
#endif

} // namespace hodgkin_huxley
