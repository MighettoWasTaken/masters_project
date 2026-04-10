#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "hodgkin_huxley/neuron_base.hpp"
#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/izhikevich.hpp"
#include "hodgkin_huxley/network.hpp"
#include "hodgkin_huxley/regional_network.hpp"
#include "hodgkin_huxley/ion_channels.hpp"
#include "hodgkin_huxley/composable_neuron.hpp"
#include "hodgkin_huxley/dbs_stimulator.hpp"

namespace py = pybind11;
using namespace hodgkin_huxley;

PYBIND11_MAKE_OPAQUE(std::vector<GateSpec>);
PYBIND11_MAKE_OPAQUE(std::vector<ChannelSpec>);
PYBIND11_MAKE_OPAQUE(std::vector<VmInstruction>);

PYBIND11_MODULE(_core, m) {
    m.doc() = "Neural simulation library - C++ backend";

    // =========================================================================
    // Integration Method Enum (must be defined first)
    // =========================================================================
    py::enum_<IntegrationMethod>(m, "IntegrationMethod")
        .value("EULER", IntegrationMethod::EULER)
        .value("RK4", IntegrationMethod::RK4)
        .value("RK45_ADAPTIVE", IntegrationMethod::RK45_ADAPTIVE)
        .export_values();

    // =========================================================================
    // NeuronBase (abstract base class)
    // =========================================================================
    py::class_<NeuronBase>(m, "NeuronBase")
        .def_property("V", &NeuronBase::membrane_potential, &NeuronBase::set_membrane_potential,
                      "Membrane potential (mV)")
        .def_property("integration_method",
                      &NeuronBase::integration_method, &NeuronBase::set_integration_method,
                      "Integration method")
        .def("reset", &NeuronBase::reset, "Reset to resting state")
        .def("step", &NeuronBase::step, "Advance simulation by dt milliseconds",
             py::arg("dt"), py::arg("I_ext"))
        .def("simulate",
             py::overload_cast<double, double, double>(&NeuronBase::simulate),
             "Run simulation with constant current",
             py::arg("duration"), py::arg("dt"), py::arg("I_ext"))
        .def("simulate",
             py::overload_cast<double, double, const std::vector<double>&>(&NeuronBase::simulate),
             "Run simulation with time-varying current",
             py::arg("duration"), py::arg("dt"), py::arg("I_ext"))
        .def("type_name", &NeuronBase::type_name, "Get neuron type name")
        .def("__repr__", [](const NeuronBase& n) {
            return "<" + n.type_name() + "Neuron V=" + std::to_string(n.membrane_potential()) + " mV>";
        });

    // =========================================================================
    // Hodgkin-Huxley Neuron
    // =========================================================================

    // HHNeuron::Parameters
    py::class_<HHNeuron::Parameters>(m, "HHParameters")
        .def(py::init<>())
        .def_readwrite("C_m", &HHNeuron::Parameters::C_m, "Membrane capacitance (uF/cm^2)")
        .def_readwrite("g_Na", &HHNeuron::Parameters::g_Na, "Sodium conductance (mS/cm^2)")
        .def_readwrite("g_K", &HHNeuron::Parameters::g_K, "Potassium conductance (mS/cm^2)")
        .def_readwrite("g_L", &HHNeuron::Parameters::g_L, "Leak conductance (mS/cm^2)")
        .def_readwrite("E_Na", &HHNeuron::Parameters::E_Na, "Sodium reversal potential (mV)")
        .def_readwrite("E_K", &HHNeuron::Parameters::E_K, "Potassium reversal potential (mV)")
        .def_readwrite("E_L", &HHNeuron::Parameters::E_L, "Leak reversal potential (mV)")
        .def("__repr__", [](const HHNeuron::Parameters& p) {
            return "<HHParameters C_m=" + std::to_string(p.C_m) +
                   " g_Na=" + std::to_string(p.g_Na) +
                   " g_K=" + std::to_string(p.g_K) + ">";
        });

    // HHNeuron::State
    py::class_<HHNeuron::State>(m, "HHState")
        .def(py::init<>())
        .def_readwrite("V", &HHNeuron::State::V, "Membrane potential (mV)")
        .def_readwrite("m", &HHNeuron::State::m, "Na+ activation gate")
        .def_readwrite("h", &HHNeuron::State::h, "Na+ inactivation gate")
        .def_readwrite("n", &HHNeuron::State::n, "K+ activation gate")
        .def("__repr__", [](const HHNeuron::State& s) {
            return "<HHState V=" + std::to_string(s.V) +
                   " m=" + std::to_string(s.m) +
                   " h=" + std::to_string(s.h) +
                   " n=" + std::to_string(s.n) + ">";
        });

    // HHNeuron (inherits from NeuronBase)
    py::class_<HHNeuron, NeuronBase>(m, "HHNeuron")
        .def(py::init<>(), "Create a Hodgkin-Huxley neuron with default parameters")
        .def(py::init<const HHNeuron::Parameters&>(), "Create a neuron with custom parameters",
             py::arg("parameters"))
        .def(py::init<const HHNeuron::Parameters&, IntegrationMethod>(),
             "Create a neuron with custom parameters and integration method",
             py::arg("parameters"), py::arg("method"))
        .def_property_readonly("state", &HHNeuron::state, "Current state of the neuron")
        .def_property_readonly("parameters", &HHNeuron::parameters, "Neuron parameters")
        .def("set_state", &HHNeuron::set_state, "Set the neuron state", py::arg("state"))
        .def("set_parameters", &HHNeuron::set_parameters, "Set the neuron parameters",
             py::arg("parameters"))
        .def("__repr__", [](const HHNeuron& n) {
            return "<HHNeuron V=" + std::to_string(n.membrane_potential()) + " mV>";
        });

    // =========================================================================
    // Izhikevich Neuron
    // =========================================================================

    // IzhikevichNeuron::Type enum
    py::enum_<IzhikevichNeuron::Type>(m, "IzhikevichType")
        .value("REGULAR_SPIKING", IzhikevichNeuron::Type::REGULAR_SPIKING)
        .value("FAST_SPIKING", IzhikevichNeuron::Type::FAST_SPIKING)
        .value("INTRINSICALLY_BURSTING", IzhikevichNeuron::Type::INTRINSICALLY_BURSTING)
        .value("CHATTERING", IzhikevichNeuron::Type::CHATTERING)
        .value("LOW_THRESHOLD_SPIKING", IzhikevichNeuron::Type::LOW_THRESHOLD_SPIKING)
        .value("CUSTOM", IzhikevichNeuron::Type::CUSTOM)
        .export_values();

    // IzhikevichNeuron::Parameters
    py::class_<IzhikevichNeuron::Parameters>(m, "IzhikevichParameters")
        .def(py::init<>())
        .def_readwrite("a", &IzhikevichNeuron::Parameters::a, "Time scale of recovery variable")
        .def_readwrite("b", &IzhikevichNeuron::Parameters::b, "Sensitivity of u to subthreshold v")
        .def_readwrite("c", &IzhikevichNeuron::Parameters::c, "After-spike reset value of v (mV)")
        .def_readwrite("d", &IzhikevichNeuron::Parameters::d, "After-spike reset increment of u")
        .def("__repr__", [](const IzhikevichNeuron::Parameters& p) {
            return "<IzhikevichParameters a=" + std::to_string(p.a) +
                   " b=" + std::to_string(p.b) +
                   " c=" + std::to_string(p.c) +
                   " d=" + std::to_string(p.d) + ">";
        });

    // IzhikevichNeuron::State
    py::class_<IzhikevichNeuron::State>(m, "IzhikevichState")
        .def(py::init<>())
        .def_readwrite("v", &IzhikevichNeuron::State::v, "Membrane potential (mV)")
        .def_readwrite("u", &IzhikevichNeuron::State::u, "Recovery variable")
        .def("__repr__", [](const IzhikevichNeuron::State& s) {
            return "<IzhikevichState v=" + std::to_string(s.v) +
                   " u=" + std::to_string(s.u) + ">";
        });

    // IzhikevichNeuron (inherits from NeuronBase)
    py::class_<IzhikevichNeuron, NeuronBase>(m, "IzhikevichNeuron")
        .def(py::init<>(), "Create an Izhikevich neuron with default (Regular Spiking) parameters")
        .def(py::init<IzhikevichNeuron::Type>(), "Create a neuron with preset type",
             py::arg("type"))
        .def(py::init<const IzhikevichNeuron::Parameters&>(), "Create a neuron with custom parameters",
             py::arg("parameters"))
        .def_property_readonly("state", &IzhikevichNeuron::state, "Current state of the neuron")
        .def_property_readonly("parameters", &IzhikevichNeuron::parameters, "Neuron parameters")
        .def_property_readonly("u", &IzhikevichNeuron::recovery_variable, "Recovery variable u")
        .def_property_readonly("spiked", &IzhikevichNeuron::spiked, "True if neuron spiked in last step")
        .def("set_state", &IzhikevichNeuron::set_state, "Set the neuron state", py::arg("state"))
        .def("set_parameters", &IzhikevichNeuron::set_parameters, "Set the neuron parameters",
             py::arg("parameters"))
        .def_static("get_preset", &IzhikevichNeuron::get_preset,
                    "Get parameters for a preset neuron type", py::arg("type"))
        .def("__repr__", [](const IzhikevichNeuron& n) {
            return "<IzhikevichNeuron v=" + std::to_string(n.membrane_potential()) + " mV>";
        });

    // =========================================================================
    // Synapse Classes
    // =========================================================================

    py::class_<SynapseBase>(m, "SynapseBase")
        .def_property_readonly("conductance", &SynapseBase::conductance, "Current conductance")
        .def_property_readonly("reversal_potential", &SynapseBase::reversal_potential, "Reversal potential (mV)")
        .def_property_readonly("weight", &SynapseBase::weight, "Synaptic weight")
        .def_property_readonly("pre_idx", &SynapseBase::pre_idx, "Pre-synaptic neuron index")
        .def_property_readonly("post_idx", &SynapseBase::post_idx, "Post-synaptic neuron index")
        .def_property_readonly("delay", &SynapseBase::delay, "Axonal conduction delay (ms)")
        .def("type_name", &SynapseBase::type_name, "Get synapse type name")
        .def("__repr__", [](const SynapseBase& s) {
            return "<" + s.type_name() + "Synapse " +
                   std::to_string(s.pre_idx()) + "->" + std::to_string(s.post_idx()) +
                   " w=" + std::to_string(s.weight()) +
                   " delay=" + std::to_string(s.delay()) + ">";
        });


    // =========================================================================
    // Network
    // =========================================================================

    // Network::ReceptorType enum
    py::enum_<Network::ReceptorType>(m, "ReceptorType")
        .value("AMPA", Network::ReceptorType::AMPA)
        .value("NMDA", Network::ReceptorType::NMDA)
        .value("GABA_A", Network::ReceptorType::GABA_A)
        .export_values();

    // Network::NeuronType enum — internal; exposed as _NetworkNeuronType
    py::enum_<Network::NeuronType>(m, "_NetworkNeuronType")
        .value("HH", Network::NeuronType::HH)
        .value("IZHIKEVICH_RS", Network::NeuronType::IZHIKEVICH_RS)
        .value("IZHIKEVICH_FS", Network::NeuronType::IZHIKEVICH_FS)
        .value("IZHIKEVICH_IB", Network::NeuronType::IZHIKEVICH_IB)
        .value("IZHIKEVICH_CH", Network::NeuronType::IZHIKEVICH_CH)
        .value("IZHIKEVICH_LTS", Network::NeuronType::IZHIKEVICH_LTS)
        .value("IZHIKEVICH_CUSTOM", Network::NeuronType::IZHIKEVICH_CUSTOM)
        .value("COMPOSABLE", Network::NeuronType::COMPOSABLE)
        .export_values();

    auto netCls = py::class_<Network>(m, "_Network")
        .def(py::init<>(), "Create an empty network")
        .def(py::init<size_t>(), "Create a network with n HH neurons", py::arg("num_neurons"))
        .def(py::init<size_t, Network::NeuronType>(),
             "Create a network with n neurons of specified type",
             py::arg("num_neurons"), py::arg("neuron_type"))

        // Add neurons - various overloads
        .def("add_neuron",
             py::overload_cast<>(&Network::add_neuron),
             "Add a HH neuron with default parameters, returns index")
        .def("add_neuron",
             py::overload_cast<const HHNeuron::Parameters&>(&Network::add_neuron),
             "Add a HH neuron with custom parameters, returns index",
             py::arg("parameters"))
        .def("add_neuron",
             py::overload_cast<Network::NeuronType>(&Network::add_neuron),
             "Add a neuron of specified type, returns index",
             py::arg("neuron_type"))
        .def("add_neuron",
             py::overload_cast<const IzhikevichNeuron::Parameters&>(&Network::add_neuron),
             "Add an Izhikevich neuron with custom parameters, returns index",
             py::arg("parameters"))
        .def("add_neuron",
             py::overload_cast<const NeuronModelSpec&>(&Network::add_neuron),
             "Add a composable neuron from a model spec, returns index",
             py::arg("spec"))

        // Synapses — primary method (unified SynapseSpec)
        .def("add_synapse",
             py::overload_cast<size_t, size_t, double, const SynapseSpec&, double>(
                 &Network::add_synapse),
             "Add a synapse from a unified SynapseSpec, returns index",
             py::arg("pre"), py::arg("post"), py::arg("weight"),
             py::arg("spec"), py::arg("delay") = 0.0)
        // Legacy wrappers (backward compat)
        .def("add_synapse",
             py::overload_cast<size_t, size_t, double, double, double, double>(
                 &Network::add_synapse),
             "Add an exponential synapse between neurons",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("E_syn") = 0.0, py::arg("tau") = 2.0,
             py::arg("delay") = 0.0)
        .def("add_alpha_synapse", &Network::add_alpha_synapse,
             "Add an alpha-function synapse between neurons",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("E_syn") = 0.0, py::arg("tau") = 2.0,
             py::arg("delay") = 0.0)
        .def("add_double_exp_synapse", &Network::add_double_exp_synapse,
             "Add a double-exponential synapse between neurons",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("E_syn") = 0.0, py::arg("tau_rise") = 0.4, py::arg("tau_decay") = 2.5,
             py::arg("delay") = 0.0)
        .def("add_ampa_synapse", &Network::add_ampa_synapse,
             "Add an AMPA synapse (E=0, tau_r=0.5, tau_d=2.5)",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("delay") = 0.0)
        .def("add_nmda_synapse", &Network::add_nmda_synapse,
             "Add an NMDA synapse (E=0, tau_r=2.0, tau_d=67.0)",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("delay") = 0.0)
        .def("add_gaba_a_synapse", &Network::add_gaba_a_synapse,
             "Add a GABA_A synapse (E=-80, tau_r=0.4, tau_d=7.7)",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("delay") = 0.0)
        .def("add_receptor_synapse", &Network::add_receptor_synapse,
             "Add a synapse by receptor type (AMPA, NMDA, GABA_A)",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("receptor"), py::arg("delay") = 0.0)

        // Properties
        .def_property_readonly("num_neurons", &Network::num_neurons)
        .def_property_readonly("num_synapses", &Network::num_synapses)
        .def_property("fast_math", &Network::fast_math, &Network::set_fast_math,
                      "Use fast polynomial exp (~8 digits) vs full precision. Default: true.")

        // Neuron access
        .def("neuron", [](Network& net, size_t idx) -> NeuronBase& {
                return net.neuron(idx);
             },
             py::return_value_policy::reference_internal,
             "Get neuron by index (polymorphic)", py::arg("idx"))
        .def("hh_neuron", py::overload_cast<size_t>(&Network::hh_neuron),
             py::return_value_policy::reference_internal,
             "Get HH neuron by index (throws if wrong type)", py::arg("idx"))
        .def("iz_neuron", py::overload_cast<size_t>(&Network::iz_neuron),
             py::return_value_policy::reference_internal,
             "Get Izhikevich neuron by index (throws if wrong type)", py::arg("idx"))
        .def("neuron_type", &Network::neuron_type,
             "Get neuron type name at index", py::arg("idx"))
        .def("synapse", [](const Network& net, size_t idx) -> const SynapseBase& {
                return net.synapse(idx);
             },
             py::return_value_policy::reference_internal,
             "Get synapse by index (polymorphic)", py::arg("idx"))

        // Kinetic state accessors
        .def("get_kin_S", &Network::get_kin_S,
             "Get kinetic gating variable S for a synapse by index",
             py::arg("synapse_idx"))
        .def("get_kin_g", &Network::get_kin_g,
             "Get effective conductance g for a synapse by index",
             py::arg("synapse_idx"))

        // Simulation
        .def("get_potentials", &Network::get_potentials,
             "Get membrane potentials of all neurons")
        .def("reset", &Network::reset, "Reset all neurons to resting state")
        .def("step", &Network::step, "Advance simulation by dt",
             py::arg("dt"), py::arg("I_ext"))
        .def("simulate", &Network::simulate,
             "Run network simulation (returns voltage traces as nested list)",
             py::arg("duration"), py::arg("dt"), py::arg("I_ext"))

        // Recording-capable simulation
        .def("max_gate_count", &Network::max_gate_count,
             "Max gate variables across all composable neurons")
        .def("get_synapse_pre_indices",  &Network::get_synapse_pre_indices,
             "Flat presynaptic neuron index vector")
        .def("get_synapse_post_indices", &Network::get_synapse_post_indices,
             "Flat postsynaptic neuron index vector")
        .def("_simulate_into_buffers",
            [](Network& net, double dur, double dt,
               py::array_t<double, py::array::c_style | py::array::forcecast> I_ext_arr,
               py::array_t<double> V_buf,
               py::array_t<double> gate_buf,
               py::array_t<double> calcium_buf,
               py::array_t<double> u_buf,
               py::array_t<double> g_syn_buf,
               py::array_t<double> I_syn_buf,
               py::array_t<double> spike_event_buf,
               size_t interval,
               double spike_threshold) {
                // Copy numpy 2D array → vector<vector<double>> in C++ (no GIL, no Python objects)
                auto buf = I_ext_arr.unchecked<2>();
                size_t n_n = static_cast<size_t>(buf.shape(0));
                size_t n_t = static_cast<size_t>(buf.shape(1));
                std::vector<std::vector<double>> I_ext(n_n, std::vector<double>(n_t));
                for (size_t i = 0; i < n_n; ++i)
                    std::memcpy(I_ext[i].data(), &buf(i, 0), n_t * sizeof(double));

                size_t n_rec = (static_cast<size_t>(dur / dt) + interval - 1) / interval;
                size_t max_gates = (gate_buf.ndim() == 3)
                                   ? static_cast<size_t>(gate_buf.shape(1)) : 0;
                net.simulate_into_buffers(
                    dur, dt, I_ext,
                    V_buf.size()            ? V_buf.mutable_data()            : nullptr,
                    gate_buf.size()         ? gate_buf.mutable_data()         : nullptr, max_gates,
                    calcium_buf.size()      ? calcium_buf.mutable_data()      : nullptr,
                    u_buf.size()            ? u_buf.mutable_data()            : nullptr,
                    g_syn_buf.size()        ? g_syn_buf.mutable_data()        : nullptr,
                    I_syn_buf.size()        ? I_syn_buf.mutable_data()        : nullptr,
                    spike_event_buf.size()  ? spike_event_buf.mutable_data()  : nullptr,
                    interval, n_rec, spike_threshold);
            },
            py::arg("duration"), py::arg("dt"), py::arg("I_ext"),
            py::arg("V_buf"), py::arg("gate_buf"), py::arg("calcium_buf"),
            py::arg("u_buf"), py::arg("g_syn_buf"), py::arg("I_syn_buf"),
            py::arg("spike_event_buf"), py::arg("interval"),
            py::arg("spike_threshold") = 0.0)

        // Descriptor-based simulation (avoids dense I_ext matrix)
        // I_const:   (n_neurons,) float64 — per-neuron constant baseline
        // pulses:    list of (neuron_start, neuron_end, onset_step, end_step, amplitude)
        // dbs_events:list of (neuron_start, neuron_end, isi_steps, pw_steps, amplitude)
        .def("_simulate_with_descriptors",
            [](Network& net, double dur, double dt,
               py::array_t<double, py::array::c_style | py::array::forcecast> I_const_arr,
               const std::vector<std::tuple<size_t,size_t,size_t,size_t,double>>& pulses,
               const std::vector<std::tuple<size_t,size_t,size_t,size_t,double>>& dbs_events,
               py::array_t<double> V_buf,
               py::array_t<double> gate_buf,
               py::array_t<double> calcium_buf,
               py::array_t<double> u_buf,
               py::array_t<double> g_syn_buf,
               py::array_t<double> I_syn_buf,
               py::array_t<double> spike_event_buf,
               size_t interval,
               double spike_threshold) {
                StimPlan stim;
                auto ic = I_const_arr.unchecked<1>();
                stim.I_const.assign(ic.data(0), ic.data(0) + ic.shape(0));
                for (auto& t : pulses) {
                    stim.pulses.push_back({
                        std::get<0>(t), std::get<1>(t),
                        std::get<2>(t), std::get<3>(t), std::get<4>(t)
                    });
                }
                for (auto& d : dbs_events) {
                    stim.dbs.push_back({
                        std::get<0>(d), std::get<1>(d),
                        std::get<2>(d), std::get<3>(d), std::get<4>(d)
                    });
                }
                size_t n_rec = (static_cast<size_t>(dur / dt) + interval - 1) / interval;
                size_t max_gates = (gate_buf.ndim() == 3)
                                   ? static_cast<size_t>(gate_buf.shape(1)) : 0;
                net.simulate_with_descriptors(
                    dur, dt, stim,
                    V_buf.size()           ? V_buf.mutable_data()           : nullptr,
                    gate_buf.size()        ? gate_buf.mutable_data()        : nullptr, max_gates,
                    calcium_buf.size()     ? calcium_buf.mutable_data()     : nullptr,
                    u_buf.size()           ? u_buf.mutable_data()           : nullptr,
                    g_syn_buf.size()       ? g_syn_buf.mutable_data()       : nullptr,
                    I_syn_buf.size()       ? I_syn_buf.mutable_data()       : nullptr,
                    spike_event_buf.size() ? spike_event_buf.mutable_data() : nullptr,
                    interval, n_rec, spike_threshold);
            },
            py::arg("duration"), py::arg("dt"), py::arg("I_const"),
            py::arg("pulses"), py::arg("dbs_events"),
            py::arg("V_buf"), py::arg("gate_buf"), py::arg("calcium_buf"),
            py::arg("u_buf"), py::arg("g_syn_buf"), py::arg("I_syn_buf"),
            py::arg("spike_event_buf"), py::arg("interval"),
            py::arg("spike_threshold") = 0.0)

        // Python special methods
        .def("__repr__", [](const Network& net) {
            return "<Network neurons=" + std::to_string(net.num_neurons()) +
                   " synapses=" + std::to_string(net.num_synapses()) + ">";
        })
        .def("__len__", &Network::num_neurons);

    // =========================================================================
    // Backwards compatibility aliases
    // =========================================================================
    m.attr("Parameters") = m.attr("HHParameters");
    m.attr("State") = m.attr("HHState");

    // =========================================================================
    // ConnectivityPattern enum
    // =========================================================================
    py::enum_<ConnectivityPattern>(m, "ConnectivityPattern")
        .value("ALL_TO_ALL", ConnectivityPattern::ALL_TO_ALL)
        .value("ONE_TO_ONE", ConnectivityPattern::ONE_TO_ONE)
        .value("SHIFTED", ConnectivityPattern::SHIFTED)
        .value("RANDOM_SPARSE", ConnectivityPattern::RANDOM_SPARSE)
        .value("RANDOM_PERMUTATION", ConnectivityPattern::RANDOM_PERMUTATION)
        .export_values();

    // =========================================================================
    // SynapseSpec — unified synapse model specification
    // =========================================================================
    py::enum_<SynapseSpec::UpdateForm>(m, "SynapseUpdateForm")
        .value("EXP_DECAY",       SynapseSpec::UpdateForm::EXP_DECAY)
        .value("ALPHA_FUNC",      SynapseSpec::UpdateForm::ALPHA_FUNC)
        .value("DOUBLE_EXP",      SynapseSpec::UpdateForm::DOUBLE_EXP)
        .value("TANH_GATE",       SynapseSpec::UpdateForm::TANH_GATE)
        .value("BOLTZMANN_GATE",  SynapseSpec::UpdateForm::BOLTZMANN_GATE)
        .value("ALPHA_BETA",      SynapseSpec::UpdateForm::ALPHA_BETA)
        .value("CUSTOM_EXPR",     SynapseSpec::UpdateForm::CUSTOM_EXPR)
        .export_values();

    py::enum_<SynapseSpec::CurrentForm>(m, "SynapseCurrentForm")
        .value("LINEAR",      SynapseSpec::CurrentForm::LINEAR)
        .value("MG_BLOCK",    SynapseSpec::CurrentForm::MG_BLOCK)
        .value("CUSTOM_EXPR", SynapseSpec::CurrentForm::CUSTOM_EXPR)
        .export_values();

    py::class_<SynapseSpec>(m, "SynapseSpec")
        .def(py::init<>())
        .def_readwrite("name",         &SynapseSpec::name)
        .def_readwrite("update_form",  &SynapseSpec::update_form)
        .def_readwrite("current_form", &SynapseSpec::current_form)
        .def_readwrite("g",            &SynapseSpec::g)
        .def_readwrite("E_syn",        &SynapseSpec::E_syn)
        .def_readwrite("power",        &SynapseSpec::power)
        .def_readwrite("S_init",       &SynapseSpec::S_init)
        .def_readwrite("A_init",       &SynapseSpec::A_init)
        .def_readwrite("delta_S",      &SynapseSpec::delta_S)
        .def_readwrite("delta_A",      &SynapseSpec::delta_A)
        .def_readwrite("tau_S",        &SynapseSpec::tau_S)
        .def_readwrite("tau_A",        &SynapseSpec::tau_A)
        .def_readwrite("norm_factor",  &SynapseSpec::norm_factor)
        .def_readwrite("tanh_amp",     &SynapseSpec::tanh_amp)
        .def_readwrite("tanh_vh",      &SynapseSpec::tanh_vh)
        .def_readwrite("tanh_k",       &SynapseSpec::tanh_k)
        .def_readwrite("tau_decay",    &SynapseSpec::tau_decay)
        .def_readwrite("s_inf",        &SynapseSpec::s_inf)
        .def_readwrite("tau",          &SynapseSpec::tau)
        .def_readwrite("alpha",        &SynapseSpec::alpha)
        .def_readwrite("beta",         &SynapseSpec::beta)
        .def_readwrite("mg_conc",      &SynapseSpec::mg_conc)
        .def_readwrite("mg_scale",     &SynapseSpec::mg_scale)
        .def_readwrite("mg_denom",     &SynapseSpec::mg_denom)
        .def_readwrite("dS_dt_vm",     &SynapseSpec::dS_dt_vm)
        .def_readwrite("dA_dt_vm",     &SynapseSpec::dA_dt_vm)
        .def_readwrite("current_vm",   &SynapseSpec::current_vm)
        // Spike-driven presets
        .def_static("exponential",      &SynapseSpec::exponential,
                     py::arg("tau_S"), py::arg("g") = 1.0, py::arg("E_syn") = 0.0)
        .def_static("alpha_function",   &SynapseSpec::alpha_function,
                     py::arg("tau"), py::arg("g") = 1.0, py::arg("E_syn") = 0.0)
        .def_static("double_exponential", &SynapseSpec::double_exponential,
                     py::arg("tau_rise"), py::arg("tau_decay"),
                     py::arg("g") = 1.0, py::arg("E_syn") = 0.0)
        // Voltage-gated kinetic presets
        .def_static("gaba_kinetic", &SynapseSpec::gaba_kinetic)
        .def_static("nmda_kinetic", &SynapseSpec::nmda_kinetic)
        .def_static("gaba_b",       &SynapseSpec::gaba_b)
        // Receptor presets
        .def_static("ampa",  &SynapseSpec::ampa)
        .def_static("nmda",  &SynapseSpec::nmda)
        .def_static("gaba_a",&SynapseSpec::gaba_a)
        .def("__repr__", [](const SynapseSpec& s) {
            return "<SynapseSpec '" + s.name + "'>";
        });

    // Backward-compat aliases
    m.attr("KineticSynapseSpec") = m.attr("SynapseSpec");
    m.attr("KineticUpdateForm")  = m.attr("SynapseUpdateForm");
    m.attr("KineticCurrentForm") = m.attr("SynapseCurrentForm");

    // =========================================================================
    // WeightDistribution
    // =========================================================================
    py::enum_<WeightDistType>(m, "WeightDistType")
        .value("CONSTANT", WeightDistType::CONSTANT)
        .value("UNIFORM", WeightDistType::UNIFORM)
        .value("NORMAL", WeightDistType::NORMAL)
        .export_values();

    py::class_<WeightDistribution>(m, "WeightDistribution")
        .def_readwrite("type", &WeightDistribution::type)
        .def_readwrite("param1", &WeightDistribution::param1)
        .def_readwrite("param2", &WeightDistribution::param2)
        .def_static("constant", &WeightDistribution::constant, py::arg("value"))
        .def_static("uniform", &WeightDistribution::uniform,
                     py::arg("min"), py::arg("max"))
        .def_static("normal", &WeightDistribution::normal,
                     py::arg("mean"), py::arg("std"))
        .def("__repr__", [](const WeightDistribution& w) {
            std::string type_str;
            switch (w.type) {
                case WeightDistType::CONSTANT: type_str = "CONSTANT"; break;
                case WeightDistType::UNIFORM: type_str = "UNIFORM"; break;
                case WeightDistType::NORMAL: type_str = "NORMAL"; break;
            }
            return "<WeightDistribution " + type_str +
                   " p1=" + std::to_string(w.param1) +
                   " p2=" + std::to_string(w.param2) + ">";
        });

    // =========================================================================
    // RegionalNetwork
    // =========================================================================
    auto rnetCls = py::class_<RegionalNetwork>(m, "RegionalNetwork")
        .def(py::init<>())

        // Add populations (three overloads)
        .def("add_population",
             py::overload_cast<const std::string&, size_t, Network::NeuronType>(
                 &RegionalNetwork::add_population),
             "Add a population with neuron type preset",
             py::arg("name"), py::arg("count"), py::arg("neuron_type"))
        .def("add_population",
             py::overload_cast<const std::string&, size_t, const HHNeuron::Parameters&>(
                 &RegionalNetwork::add_population),
             "Add a population with custom HH parameters",
             py::arg("name"), py::arg("count"), py::arg("parameters"))
        .def("add_population",
             py::overload_cast<const std::string&, size_t, const IzhikevichNeuron::Parameters&>(
                 &RegionalNetwork::add_population),
             "Add a population with custom Izhikevich parameters",
             py::arg("name"), py::arg("count"), py::arg("parameters"))
        .def("add_population",
             py::overload_cast<const std::string&, size_t, const NeuronModelSpec&>(
                 &RegionalNetwork::add_population),
             "Add a population with a composable neuron model spec",
             py::arg("name"), py::arg("count"), py::arg("spec"))
        .def("add_population",
             py::overload_cast<const std::string&, const std::vector<NeuronModelSpec>&>(
                 &RegionalNetwork::add_population),
             "Add a heterogeneous population from a list of per-neuron specs",
             py::arg("name"), py::arg("specs"))

        // Connectivity
        .def("connect", &RegionalNetwork::connect,
             "Connect two populations with a preset pattern",
             py::arg("src"), py::arg("dst"), py::arg("pattern"),
             py::arg("synapse"), py::arg("weight"),
             py::arg("delay") = 0.0, py::arg("shift") = 1,
             py::arg("probability") = 0.1, py::arg("allow_self") = false,
             py::arg("seed") = 0)
        .def("add_connection", &RegionalNetwork::add_connection,
             "Add a single connection using local indices",
             py::arg("src"), py::arg("src_local"),
             py::arg("dst"), py::arg("dst_local"),
             py::arg("weight"), py::arg("synapse"),
             py::arg("delay") = 0.0)

        // Initial conditions
        .def("randomize_membrane_potentials",
             &RegionalNetwork::randomize_membrane_potentials,
             "Randomize membrane potentials in a population",
             py::arg("name"), py::arg("mean"), py::arg("std_dev"),
             py::arg("seed") = 0, py::arg("reset_gates") = false)

        // Population queries
        .def("population_names", &RegionalNetwork::population_names)
        .def("population_size", &RegionalNetwork::population_size, py::arg("name"))
        .def("population_start", &RegionalNetwork::population_start, py::arg("name"))
        .def("num_populations", &RegionalNetwork::num_populations)

        // Delegation
        .def_property_readonly("num_neurons", &RegionalNetwork::num_neurons)
        .def_property_readonly("num_synapses", &RegionalNetwork::num_synapses)
        .def_property("fast_math", &RegionalNetwork::fast_math,
                      &RegionalNetwork::set_fast_math)
        .def("reset", &RegionalNetwork::reset)
        .def("network", py::overload_cast<>(&RegionalNetwork::network),
             py::return_value_policy::reference_internal)
        .def("__repr__", [](const RegionalNetwork& rn) {
            return "<RegionalNetwork populations=" + std::to_string(rn.num_populations()) +
                   " neurons=" + std::to_string(rn.num_neurons()) +
                   " synapses=" + std::to_string(rn.num_synapses()) + ">";
        });

    // =========================================================================
    // Ion Channel / Composable Neuron types
    // =========================================================================

    // BoltzmannParams
    py::class_<BoltzmannParams>(m, "BoltzmannParams")
        .def(py::init<>())
        .def_readwrite("v_half", &BoltzmannParams::v_half)
        .def_readwrite("k", &BoltzmannParams::k)
        .def("__repr__", [](const BoltzmannParams& p) {
            return "<BoltzmannParams v_half=" + std::to_string(p.v_half) +
                   " k=" + std::to_string(p.k) + ">";
        });

    // TauParams
    py::enum_<TauParams::Form>(m, "TauForm")
        .value("CONSTANT", TauParams::Form::CONSTANT)
        .value("BOLTZMANN", TauParams::Form::BOLTZMANN)
        .value("DOUBLE_EXP_SUM", TauParams::Form::DOUBLE_EXP_SUM)
        .value("OFFSET_DOUBLE_EXP", TauParams::Form::OFFSET_DOUBLE_EXP)
        .value("SCALED_EXP", TauParams::Form::SCALED_EXP)
        .value("COMPOUND_AB", TauParams::Form::COMPOUND_AB)
        .export_values();

    py::class_<TauParams>(m, "TauParams")
        .def(py::init<>())
        .def_readwrite("form", &TauParams::form)
        .def("get_param", [](const TauParams& t, int i) { return t.params[i]; })
        .def("set_param", [](TauParams& t, int i, double v) { t.params[i] = v; })
        .def("__repr__", [](const TauParams& t) {
            return "<TauParams form=" + std::to_string(static_cast<int>(t.form)) + ">";
        });

    // RateFuncParams
    py::enum_<RateFuncParams::Form>(m, "RateFuncForm")
        .value("LINEAR_OVER_EXP", RateFuncParams::Form::LINEAR_OVER_EXP)
        .value("EXP_DECAY", RateFuncParams::Form::EXP_DECAY)
        .value("LINEAR_OVER_EXPM1", RateFuncParams::Form::LINEAR_OVER_EXPM1)
        .value("SIGMOID", RateFuncParams::Form::SIGMOID)
        .export_values();

    py::class_<RateFuncParams>(m, "RateFuncParams")
        .def(py::init<>())
        .def_readwrite("form", &RateFuncParams::form)
        .def_readwrite("A", &RateFuncParams::A)
        .def_readwrite("B", &RateFuncParams::B)
        .def_readwrite("C", &RateFuncParams::C)
        .def("__repr__", [](const RateFuncParams& r) {
            return "<RateFuncParams A=" + std::to_string(r.A) +
                   " B=" + std::to_string(r.B) +
                   " C=" + std::to_string(r.C) + ">";
        });

    // Opaque vector bindings — must come before any class that uses these vectors
    py::bind_vector<std::vector<GateSpec>>(m, "GateSpecVector");
    py::bind_vector<std::vector<ChannelSpec>>(m, "ChannelSpecVector");
    py::bind_vector<std::vector<VmInstruction>>(m, "VmInstructionVector");

    // VM bytecode types
    py::enum_<VmOp>(m, "VmOp")
        .value("PUSH_DEP",   VmOp::PUSH_DEP)
        .value("PUSH_CONST", VmOp::PUSH_CONST)
        .value("ADD",  VmOp::ADD)  .value("MUL",  VmOp::MUL)
        .value("NEG",  VmOp::NEG)  .value("RCP",  VmOp::RCP)
        .value("POW_INT",  VmOp::POW_INT)
        .value("POW_HALF", VmOp::POW_HALF)
        .value("POW_GEN",  VmOp::POW_GEN)
        .value("EXP",  VmOp::EXP)  .value("LOG",  VmOp::LOG)
        .value("TANH", VmOp::TANH) .value("SIN",  VmOp::SIN)
        .value("COS",  VmOp::COS)  .value("SQRT", VmOp::SQRT)
        .value("ABS",  VmOp::ABS)
        .value("PUSH_GATE", VmOp::PUSH_GATE)
        .value("PUSH_S",    VmOp::PUSH_S)
        .value("PUSH_A",    VmOp::PUSH_A)
        .export_values();

    py::class_<VmInstruction>(m, "VmInstruction")
        .def(py::init<>())
        .def_readwrite("op",      &VmInstruction::op)
        .def_readwrite("operand", &VmInstruction::operand);

    py::class_<VmExpr>(m, "VmExpr")
        .def(py::init<>())
        .def_readwrite("instructions", &VmExpr::instructions)
        .def_readwrite("constants",    &VmExpr::constants)
        .def("empty",           &VmExpr::empty)
        .def("add_instruction", &VmExpr::add_instruction,
             py::arg("op"), py::arg("operand") = 0)
        .def("add_constant",    &VmExpr::add_constant);

    // GateSpec
    py::enum_<GateSpec::UpdateForm>(m, "GateUpdateForm")
        .value("INF_TAU", GateSpec::UpdateForm::INF_TAU)
        .value("ALPHA_BETA", GateSpec::UpdateForm::ALPHA_BETA)
        .value("INSTANT", GateSpec::UpdateForm::INSTANT)
        .value("DERIVED", GateSpec::UpdateForm::DERIVED)
        .value("CUSTOM_EXPR", GateSpec::UpdateForm::CUSTOM_EXPR)
        .export_values();

    py::enum_<GateSpec::Dependency>(m, "GateDependency")
        .value("VOLTAGE", GateSpec::Dependency::VOLTAGE)
        .value("CALCIUM", GateSpec::Dependency::CALCIUM)
        .export_values();

    py::class_<GateSpec>(m, "GateSpec")
        .def(py::init<>())
        .def_readwrite("name", &GateSpec::name)
        .def_readwrite("update_form", &GateSpec::update_form)
        .def_readwrite("dependency", &GateSpec::dependency)
        .def_readwrite("scale", &GateSpec::scale)
        .def_readwrite("initial_value", &GateSpec::initial_value)
        .def_readwrite("inf", &GateSpec::inf)
        .def_readwrite("tau", &GateSpec::tau)
        .def_readwrite("alpha", &GateSpec::alpha)
        .def_readwrite("beta", &GateSpec::beta)
        .def_readwrite("derived_source_gate", &GateSpec::derived_source_gate)
        .def_readwrite("derived_a", &GateSpec::derived_a)
        .def_readwrite("derived_b", &GateSpec::derived_b)
        .def_readwrite("derived_c", &GateSpec::derived_c)
        .def_readwrite("inf_vm",    &GateSpec::inf_vm)
        .def_readwrite("tau_vm",   &GateSpec::tau_vm)
        .def_readwrite("alpha_vm", &GateSpec::alpha_vm)
        .def_readwrite("beta_vm",  &GateSpec::beta_vm)
        .def_readwrite("dxdt_vm",  &GateSpec::dxdt_vm)
        .def("__repr__", [](const GateSpec& g) {
            return "<GateSpec '" + g.name + "'>";
        });

    // ChannelSpec
    py::class_<ChannelSpec>(m, "ChannelSpec")
        .def(py::init<>())
        .def_readwrite("name", &ChannelSpec::name)
        .def_readwrite("g", &ChannelSpec::g)
        .def_readwrite("E_rev", &ChannelSpec::E_rev)
        .def_readwrite("use_calcium_nernst", &ChannelSpec::use_calcium_nernst)
        .def_readwrite("gates", &ChannelSpec::gates)
        .def_readwrite("is_ahp", &ChannelSpec::is_ahp)
        .def_readwrite("ahp_k1", &ChannelSpec::ahp_k1)
        .def_readwrite("gate_product_vm", &ChannelSpec::gate_product_vm)
        .def("__repr__", [](const ChannelSpec& c) {
            return "<ChannelSpec '" + c.name + "' g=" + std::to_string(c.g) + ">";
        });

    // CalciumSpec
    py::class_<CalciumSpec>(m, "CalciumSpec")
        .def(py::init<>())
        .def_readwrite("enabled", &CalciumSpec::enabled)
        .def_readwrite("use_nernst", &CalciumSpec::use_nernst)
        .def_readwrite("epsilon", &CalciumSpec::epsilon)
        .def_readwrite("K_Ca", &CalciumSpec::K_Ca)
        .def_readwrite("Ca_init", &CalciumSpec::Ca_init)
        .def_readwrite("Ca_o", &CalciumSpec::Ca_o)
        .def_readwrite("z", &CalciumSpec::z)
        .def_readwrite("F", &CalciumSpec::F)
        .def_readwrite("R", &CalciumSpec::R)
        .def_readwrite("T", &CalciumSpec::T)
        .def_readwrite("source_channels", &CalciumSpec::source_channels)
        .def("__repr__", [](const CalciumSpec& c) {
            return "<CalciumSpec enabled=" + std::string(c.enabled ? "true" : "false") + ">";
        });

    // NeuronModelSpec
    py::class_<NeuronModelSpec>(m, "NeuronModelSpec")
        .def(py::init<>())
        .def_readwrite("name", &NeuronModelSpec::name)
        .def_readwrite("C_m", &NeuronModelSpec::C_m)
        .def_readwrite("V_init", &NeuronModelSpec::V_init)
        .def_readwrite("gates", &NeuronModelSpec::gates)
        .def_readwrite("channels", &NeuronModelSpec::channels)
        .def_readwrite("calcium", &NeuronModelSpec::calcium)
        .def_static("hh_default", &NeuronModelSpec::hh_default,
                    "Classic Hodgkin-Huxley squid axon model (Na, K, Leak channels)")
        .def_static("izhikevich",
                    py::overload_cast<IzhikevichNeuron::Type>(&NeuronModelSpec::izhikevich),
                    "Izhikevich spiking neuron with preset type",
                    py::arg("type") = IzhikevichNeuron::Type::REGULAR_SPIKING)
        .def_static("izhikevich",
                    py::overload_cast<const IzhikevichNeuron::Parameters&>(&NeuronModelSpec::izhikevich),
                    "Izhikevich spiking neuron with custom parameters",
                    py::arg("parameters"))
        .def_readwrite("is_izhikevich", &NeuronModelSpec::is_izhikevich,
                       "True when this spec represents an Izhikevich neuron")
        .def_readwrite("iz_params", &NeuronModelSpec::iz_params,
                       "Izhikevich parameters (only valid when is_izhikevich=True)")
        .def("validate", &NeuronModelSpec::validate,
             "Validate spec — raises ValueError on structural errors")
        .def("__copy__",  [](const NeuronModelSpec& s) { return NeuronModelSpec(s); })
        .def("__deepcopy__", [](const NeuronModelSpec& s, py::dict) { return NeuronModelSpec(s); })
        .def("__repr__", [](const NeuronModelSpec& s) {
            return "<NeuronModelSpec '" + s.name + "' gates=" +
                   std::to_string(s.gates.size()) + " channels=" +
                   std::to_string(s.channels.size()) + ">";
        });

    // ComposableNeuron (inherits from NeuronBase)
    py::class_<ComposableNeuron, NeuronBase>(m, "ComposableNeuron")
        .def(py::init<const NeuronModelSpec&>(), py::arg("spec"))
        .def_property_readonly("model_spec", &ComposableNeuron::model_spec)
        .def_property_readonly("gate_states", &ComposableNeuron::gate_states)
        .def_property_readonly("calcium", &ComposableNeuron::calcium)
        .def("__repr__", [](const ComposableNeuron& n) {
            return "<ComposableNeuron " + n.type_name() +
                   " V=" + std::to_string(n.membrane_potential()) + " mV>";
        });

    // Network.add_kinetic_synapse (now delegates to add_synapse)
    netCls.def("add_kinetic_synapse",
        &Network::add_kinetic_synapse,
        py::arg("pre"), py::arg("post"), py::arg("weight"),
        py::arg("spec"), py::arg("delay") = 0.0);

    // RegionalNetwork.add_kinetic_connection
    rnetCls.def("add_kinetic_connection",
        &RegionalNetwork::add_kinetic_connection,
        py::arg("src"), py::arg("i"),
        py::arg("dst"), py::arg("j"),
        py::arg("weight"), py::arg("spec"), py::arg("delay") = 0.0);

    // =========================================================================
    // DBSStimulator
    // =========================================================================

    py::class_<DBSStimulator::Parameters>(m, "DBSParameters")
        .def(py::init<>())
        .def_readwrite("frequency",   &DBSStimulator::Parameters::frequency,
                       "Stimulation frequency in Hz (0 = off)")
        .def_readwrite("amplitude",   &DBSStimulator::Parameters::amplitude,
                       "Pulse amplitude in uA/cm^2")
        .def_readwrite("pulse_width", &DBSStimulator::Parameters::pulse_width,
                       "Pulse width in ms")
        .def("__repr__", [](const DBSStimulator::Parameters& p) {
            return "<DBSParameters freq=" + std::to_string(p.frequency) +
                   "Hz amp=" + std::to_string(p.amplitude) +
                   " PW=" + std::to_string(p.pulse_width) + "ms>";
        });

    py::class_<DBSStimulator>(m, "DBSStimulator")
        .def(py::init<>(), "Create a DBS stimulator with default parameters")
        .def(py::init<const DBSStimulator::Parameters&>(),
             "Create a DBS stimulator with given parameters",
             py::arg("params"))
        .def("generate", &DBSStimulator::generate,
             "Generate full current trace (length ≈ duration/dt + 1)",
             py::arg("duration"), py::arg("dt"))
        .def("current_at", &DBSStimulator::current_at,
             "Get current value at a specific step index",
             py::arg("step_index"), py::arg("dt"))
        .def("set_parameters", &DBSStimulator::set_parameters,
             "Update stimulator parameters (validates on assignment)",
             py::arg("params"))
        .def_property_readonly("parameters", &DBSStimulator::parameters,
                               "Current stimulator parameters")
        .def("__repr__", [](const DBSStimulator& d) {
            const auto& p = d.parameters();
            return "<DBSStimulator freq=" + std::to_string(p.frequency) +
                   "Hz amp=" + std::to_string(p.amplitude) +
                   " PW=" + std::to_string(p.pulse_width) + "ms>";
        });

    // Module version
    m.attr("__version__") = "0.7.0";
}
