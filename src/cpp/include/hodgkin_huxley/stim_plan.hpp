#pragma once

#include <cstddef>
#include <vector>

namespace hodgkin_huxley {

struct PulseEvent {
    size_t neuron_start;
    size_t neuron_end;
    size_t onset_step;
    size_t end_step;
    double amplitude;
};

struct DBSEvent {
    size_t neuron_start;
    size_t neuron_end;
    size_t isi_steps;
    size_t pw_steps;
    double amplitude;
};

struct StimPlan {
    std::vector<double>     I_const;
    std::vector<PulseEvent> pulses;
    std::vector<DBSEvent>   dbs;
};

} // namespace hodgkin_huxley
