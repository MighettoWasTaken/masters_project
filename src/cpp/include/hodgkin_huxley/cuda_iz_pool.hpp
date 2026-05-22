#pragma once
#ifdef HH_USE_CUDA
#include "hodgkin_huxley/pool/pool_base.hpp"
#include "hodgkin_huxley/neuron.hpp"

namespace hodgkin_huxley
{

    class CudaIzPool : public PoolBase
    {
    public:
        explicit CudaIzPool(size_t capacity, int device_id = 0);
        ~CudaIzPool() override;

        void scatter_voltages(double *V_buf) const override; // cudaMemcpyAsync → pinned
        void gather_currents(const double *I_buf) override;  // cudaMemcpyAsync ← pinned
        void step(double dt) override;                       // launch iz kernel
        void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>> &) const override;

        bool is_cuda() const override { return true; }
        int device_id() const override { return device_id_; }
        void synchronize() override;
        bool requires_pinned_memory() const override { return true; }
        void migrate_to_device(int new_id) override;

        size_t size() const override { return n_; }

        void add(size_t net_idx, const IzNeuron::Parameters &p,
                 const IzNeuron::State &s);

    private:
        int device_id_;
        size_t n_ = 0;
        size_t capacity_;
        cudaStream_t stream_ = nullptr;

        double *d_V_ = nullptr;
        double *d_u_ = nullptr;
        double *d_a_ = nullptr;
        double *d_b_ = nullptr;
        double *d_c_ = nullptr;
        double *d_d_ = nullptr;
        double *d_I_ = nullptr;
        size_t *d_net_idx_ = nullptr;

        void alloc(size_t cap);
        void free_device();
    };

} // namespace hodgkin_huxley
#endif