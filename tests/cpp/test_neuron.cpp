/**
 * Basic C++ tests for the Hodgkin-Huxley neuron model.
 *
 * Compile with:
 *   g++ -std=c++17 -I../../src/cpp/include test_neuron.cpp ../../src/cpp/src/*.cpp -o test_neuron
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/network.hpp"

using namespace hodgkin_huxley;

void test_neuron_creation() {
    std::cout << "Testing neuron creation... ";
    HHNeuron neuron;
    assert(std::abs(neuron.membrane_potential() - (-65.0)) < 1.0);
    std::cout << "PASSED\n";
}

void test_neuron_step() {
    std::cout << "Testing neuron step... ";
    HHNeuron neuron;
    double initial_V = neuron.membrane_potential();
    neuron.step(0.01, 10.0);  // Apply current
    assert(neuron.membrane_potential() > initial_V);
    std::cout << "PASSED\n";
}

void test_neuron_simulate() {
    std::cout << "Testing neuron simulate... ";
    HHNeuron neuron;
    auto trace = neuron.simulate(10.0, 0.01, 10.0);
    assert(trace.size() == 1000);
    std::cout << "PASSED\n";
}

void test_action_potential() {
    std::cout << "Testing action potential generation... ";
    HHNeuron neuron;
    auto trace = neuron.simulate(50.0, 0.01, 15.0);

    double max_V = *std::max_element(trace.begin(), trace.end());
    assert(max_V > 0.0);  // Action potential should exceed 0 mV
    std::cout << "PASSED (max V = " << max_V << " mV)\n";
}

void test_network_creation() {
    std::cout << "Testing network creation... ";
    Network net(3);
    assert(net.num_neurons() == 3);
    std::cout << "PASSED\n";
}

void test_network_synapse() {
    std::cout << "Testing network synapse... ";
    Network net(2);
    net.add_synapse(0, 1, 0.5);
    assert(net.num_synapses() == 1);
    std::cout << "PASSED\n";
}

void test_custom_parameters() {
    std::cout << "Testing custom parameters... ";
    HHNeuron::Parameters params;
    params.g_Na = 100.0;
    HHNeuron neuron(params);
    assert(std::abs(neuron.parameters().g_Na - 100.0) < 0.001);
    std::cout << "PASSED\n";
}

void test_reset() {
    std::cout << "Testing neuron reset... ";
    HHNeuron neuron;
    neuron.set_membrane_potential(0.0);
    neuron.reset();
    assert(std::abs(neuron.membrane_potential() - (-65.0)) < 1.0);
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Hodgkin-Huxley C++ Tests ===\n\n";

    test_neuron_creation();
    test_neuron_step();
    test_neuron_simulate();
    test_action_potential();
    test_network_creation();
    test_network_synapse();
    test_custom_parameters();
    test_reset();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
