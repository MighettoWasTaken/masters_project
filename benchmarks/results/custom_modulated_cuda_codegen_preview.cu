__device__ __forceinline__ double custom_modulated_gate_m_inf(double V) {
    return ((0.001) + (((0.999) * ((1.0 / (((1.0) + (exp(((-5.0) + (((-0.125) * (V)))))))))))));
}

__device__ __forceinline__ double custom_modulated_gate_m_tau(double V) {
    return 2.0;
}

__device__ __forceinline__ double custom_modulated_x_DA_ode(double I_source, double x) {
    return ((((-0.0125) * (x))) + (((-0.0125) * (pow(x, 2.0)))));
}

__device__ __forceinline__ double custom_modulated_x_DA_mod_0(double dep) {
    return (1.0 / (((1.0) + (((148.4131591025766) * (exp(((-10.0) * (dep)))))))));
}