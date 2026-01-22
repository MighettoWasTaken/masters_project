#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hodgkin_huxley/neuron_base.hpp"
#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/izhikevich.hpp"
#include "hodgkin_huxley/network.hpp"

namespace py = pybind11;
using namespace hodgkin_huxley;

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
    // Network
    // =========================================================================
    py::class_<Network>(m, "Network")
        .def(py::init<>(), "Create an empty network")
        .def(py::init<size_t>(), "Create a network with n HH neurons", py::arg("num_neurons"))
        .def("add_neuron",
             py::overload_cast<>(&Network::add_neuron),
             "Add a HH neuron with default parameters, returns index")
        .def("add_neuron",
             py::overload_cast<const HHNeuron::Parameters&>(&Network::add_neuron),
             "Add a HH neuron with custom parameters, returns index",
             py::arg("parameters"))
        .def("add_synapse", &Network::add_synapse,
             "Add a synaptic connection between neurons",
             py::arg("pre_idx"), py::arg("post_idx"), py::arg("weight"),
             py::arg("E_syn") = 0.0, py::arg("tau") = 2.0)
        .def_property_readonly("num_neurons", &Network::num_neurons)
        .def_property_readonly("num_synapses", &Network::num_synapses)
        .def("neuron", py::overload_cast<size_t>(&Network::neuron),
             py::return_value_policy::reference_internal,
             "Get neuron by index", py::arg("idx"))
        .def("get_potentials", &Network::get_potentials,
             "Get membrane potentials of all neurons")
        .def("reset", &Network::reset, "Reset all neurons to resting state")
        .def("step", &Network::step, "Advance simulation by dt",
             py::arg("dt"), py::arg("I_ext"))
        .def("simulate", &Network::simulate,
             "Run network simulation",
             py::arg("duration"), py::arg("dt"), py::arg("I_ext"))
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

    // Module version
    m.attr("__version__") = "0.2.0";
}
