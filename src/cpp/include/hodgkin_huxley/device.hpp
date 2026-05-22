#pragma once
#include <string>

namespace hodgkin_huxley {

struct Device {
    enum class Type { CPU, CUDA };
    Type type  = Type::CPU;
    int  index = 0;

    static Device cpu()             { return {Type::CPU,  0}; }
    static Device cuda(int idx = 0) { return {Type::CUDA, idx}; }
    bool operator==(const Device& o) const { return type == o.type && index == o.index; }
    bool operator!=(const Device& o) const { return !(*this == o); }
    std::string str() const;  // "cpu", "cuda:0", etc.
};

int         cuda_device_count();
bool        cuda_is_available();
std::string cuda_device_name(int idx);           // e.g. "NVIDIA GeForce RTX 3090"
double        cuda_smoke_test(int device_idx = 0); // True = GPU ran a kernel correctly

} // namespace hodgkin_huxley
