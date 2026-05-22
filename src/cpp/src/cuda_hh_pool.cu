#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void hh_step_kernel(
    double *V, double *m, double *h, double *n,
    const double *I_ext,
    const double *gNa, const double *gK, const double *gL,
    const double *ENa, const double *EK, const double *EL,
    const double *Cm,
    double dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    double v = V[i];
    double mi = m[i], hi = h[i], ni = n[i];

    double alpha_m = (v == -40.0) ? 1.0 : 0.1 * (v + 40.0) / (1.0 - exp(-(v + 40.0) / 10.0));
    double beta_m = 4.0 * exp(-(v + 65.0) / 18.0);
    double alpha_h = 0.07 * exp(-(v + 65.0) / 20.0);
    double beta_h = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0));
    double alpha_n = (v == -55.0) ? 0.1 : 0.01 * (v + 55.0) / (1.0 - exp(-(v + 55.0) / 10.0));
    double beta_n = 0.125 * exp(-(v + 65.0) / 80.0);

    double I_Na = gNa[i] * mi * mi * mi * hi * (v - ENa[i]);
    double I_K = gK[i] * ni * ni * ni * ni * (v - EK[i]);
    double I_L = gL[i] * (v - EL[i]);

    V[i] = v + dt * (I_ext[i] - I_Na - I_K - I_L) / Cm[i];
    m[i] = mi + dt * (alpha_m * (1.0 - mi) - beta_m * mi);
    h[i] = hi + dt * (alpha_h * (1.0 - hi) - beta_h * hi);
    n[i] = ni + dt * (alpha_n * (1.0 - ni) - beta_n * ni);
}