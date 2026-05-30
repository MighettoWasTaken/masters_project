#pragma once

#ifdef HH_USE_CUDA

#include <cstddef>

namespace hodgkin_huxley {

struct CudaHHDesc {
    double       *d_V, *d_m, *d_h, *d_n, *d_I_ext;
    const double *d_C_m, *d_g_Na, *d_g_K, *d_g_L, *d_E_Na, *d_E_K, *d_E_L;
    const size_t *d_net_idx;
    int           n;
};

struct CudaIzDesc {
    double       *d_v, *d_u, *d_I_ext;
    const double *d_a, *d_b, *d_c, *d_d;
    const size_t *d_net_idx;
    int           n;
    double        threshold;
};

} // namespace hodgkin_huxley

#endif // HH_USE_CUDA
