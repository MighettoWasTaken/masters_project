#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/network.hpp"

namespace py = pybind11;
using namespace hodgkin_huxley;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Hodgkin-Huxley neuron simulation - C++ backend";

    // HHNeuron::Parameters
    py::class_<HHNeuron::Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("C_m", &HHNeuron::Parameters::C_m, "Membrane capacitance (uF/cm^2)")
        .def_readwrite("g_Na", &HHNeuron::Parameters::g_Na, "Sodium conductance (mS/cm^2)")
        .def_readwrite("g_K", &HHNeuron::Parameters::g_K, "Potassium conductance (mS/cm^2)")
        .def_readwrite("g_L", &HHNeuron::Parameters::g_L, "Leak conductance (mS/cm^2)")
        .def_readwrite("E_Na", &HHNeuron::Parameters::E_Na, "Sodium reversal potential (mV)")
        .def_readwrite("E_K", &HHNeuron::Parameters::E_K, "Potassium reversal potential (mV)")
        .def_readwrite("E_L", &HHNeuron::Parameters::E_L, "Leak reversal potential (mV)")
        .def("__repr__", [](const HHNeuron::Parameters& p) {
            return "<Parameters C_m=" + std::to_string(p.C_m) +
                   " g_Na=" + std::to_string(p.g_Na) +
                   " g_K=" + std::to_string(p.g_K) + ">";
        });

    // HHNeuron::State
    py::class_<HHNeuron::State>(m, "State")
        .def(py::init<>())
        .def_readwrite("V", &HHNeuron::State::V, "Membrane potential (mV)")
        .def_readwrite("m", &HHNeuron::State::m, "Na+ activation gate")
        .def_readwrite("h", &HHNeuron::State::h, "Na+ inactivation gate")
        .def_readwrite("n", &HHNeuron::State::n, "K+ activation gate")
        .def("__repr__", [](const HHNeuron::State& s) {
            return "<State V=" + std::to_string(s.V) +
                   " m=" + std::to_string(s.m) +
                   " h=" + std::to_string(s.h) +
                   " n=" + std::to_string(s.n) + ">";
        });

    // HHNeuron
    py::class_<HHNeuron>(m, "HHNeuron")
        .def(py::init<>(), "Create a Hodgkin-Huxley neuron with default parameters")
        .def(py::init<const HHNeuron::Parameters&>(), "Create a neuron with custom parameters",
             py::arg("parameters"))
        .def(py::init<const HHNeuron::Parameters&, IntegrationMethod>(),
             "Create a neuron with custom parameters and integration method",
             py::arg("parameters"), py::arg("method"))
        .def_property_readonly("state", &HHNeuron::state, "Current state of the neuron")
        .def_property_readonly("parameters", &HHNeuron::parameters, "Neuron parameters")
        .def_property("V", &HHNeuron::membrane_potential, &HHNeuron::set_membrane_potential,
                      "Membrane potential (mV)")
        .def_property("integration_method",
                      &HHNeuron::integration_method, &HHNeuron::set_integration_method,
                      "Integration method (EULER, RK4, or RK45_ADAPTIVE)")
        .def("set_state", &HHNeuron::set_state, "Set the neuron state", py::arg("state"))
        .def("set_parameters", &HHNeuron::set_parameters, "Set the neuron parameters",
             py::arg("parameters"))
        .def("reset", &HHNeuron::reset, "Reset to resting state")
        .def("step", &HHNeuron::step, "Advance simulation by dt milliseconds",
             py::arg("dt"), py::arg("I_ext"))
        .def("simulate",
             py::overload_cast<double, double, double>(&HHNeuron::simulate),
             "Run simulation with constant current",
             py::arg("duration"), py::arg("dt"), py::arg("I_ext"))
        .def("simulate",
             py::overload_cast<double, double, const std::vector<double>&>(&HHNeuron::simulate),
             "Run simulation with time-varying current",
             py::arg("duration"), py::arg("dt"), py::arg("I_ext"))
        .def("__repr__", [](const HHNeuron& n) {
            return "<HHNeuron V=" + std::to_string(n.membrane_potential()) + " mV>";
        });

    // Network
    py::class_<Network>(m, "Network")
        .def(py::init<>(), "Create an empty network")
        .def(py::init<size_t>(), "Create a network with n neurons", py::arg("num_neurons"))
        .def("add_neuron",
             py::overload_cast<>(&Network::add_neuron),
             "Add a neuron with default parameters, returns index")
        .def("add_neuron",
             py::overload_cast<const HHNeuron::Parameters&>(&Network::add_neuron),
             "Add a neuron with custom parameters, returns index",
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

    // Integration methods enum
    py::enum_<IntegrationMethod>(m, "IntegrationMethod")
        .value("EULER", IntegrationMethod::EULER)
        .value("RK4", IntegrationMethod::RK4)
        .value("RK45_ADAPTIVE", IntegrationMethod::RK45_ADAPTIVE)
        .export_values();

    // Module version
    m.attr("__version__") = "0.1.0";
}
