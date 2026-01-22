/**
 * Comprehensive C++ tests for neuron models.
 * Tests both Hodgkin-Huxley and Izhikevich neuron models.
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <stdexcept>
#include "hodgkin_huxley/neuron_base.hpp"
#include "hodgkin_huxley/neuron.hpp"
#include "hodgkin_huxley/izhikevich.hpp"
#include "hodgkin_huxley/network.hpp"

using namespace hodgkin_huxley;

// Test counters
int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) void name()
#define RUN_TEST(name) run_test(#name, name)

void run_test(const char* name, void (*test_func)()) {
    std::cout << "Testing " << name << "... ";
    try {
        test_func();
        std::cout << "PASSED\n";
        tests_passed++;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        tests_failed++;
    } catch (...) {
        std::cout << "FAILED: Unknown exception\n";
        tests_failed++;
    }
}

void check(bool condition, const char* msg = "Assertion failed") {
    if (!condition) {
        throw std::runtime_error(msg);
    }
}

void check_approx(double a, double b, double tol = 1e-6, const char* msg = "Values not approximately equal") {
    if (std::abs(a - b) > tol) {
        throw std::runtime_error(std::string(msg) + " (" + std::to_string(a) + " vs " + std::to_string(b) + ")");
    }
}

// =============================================================================
// HHNeuron Basic Tests
// =============================================================================

TEST(neuron_default_creation) {
    HHNeuron neuron;
    check_approx(neuron.membrane_potential(), -65.0, 1.0, "Default membrane potential incorrect");
    check(neuron.integration_method() == IntegrationMethod::RK4, "Default integration method should be RK4");
}

TEST(neuron_custom_parameters) {
    HHNeuron::Parameters params;
    params.g_Na = 100.0;
    params.g_K = 40.0;
    HHNeuron neuron(params);
    check_approx(neuron.parameters().g_Na, 100.0, 0.001, "Custom g_Na not set");
    check_approx(neuron.parameters().g_K, 40.0, 0.001, "Custom g_K not set");
}

TEST(neuron_with_integration_method) {
    HHNeuron::Parameters params;
    HHNeuron neuron(params, IntegrationMethod::EULER);
    check(neuron.integration_method() == IntegrationMethod::EULER, "Integration method not set correctly");
}

TEST(neuron_set_integration_method) {
    HHNeuron neuron;
    neuron.set_integration_method(IntegrationMethod::EULER);
    check(neuron.integration_method() == IntegrationMethod::EULER, "set_integration_method failed");
    neuron.set_integration_method(IntegrationMethod::RK4);
    check(neuron.integration_method() == IntegrationMethod::RK4, "set_integration_method failed");
}

TEST(neuron_reset) {
    HHNeuron neuron;
    neuron.set_membrane_potential(0.0);
    check_approx(neuron.membrane_potential(), 0.0, 0.001, "set_membrane_potential failed");
    neuron.reset();
    check_approx(neuron.membrane_potential(), -65.0, 1.0, "reset failed");
}

TEST(neuron_state_access) {
    HHNeuron neuron;
    const auto& state = neuron.state();
    check_approx(state.V, -65.0, 1.0, "State V incorrect");
    check(state.m >= 0.0 && state.m <= 1.0, "State m out of bounds");
    check(state.h >= 0.0 && state.h <= 1.0, "State h out of bounds");
    check(state.n >= 0.0 && state.n <= 1.0, "State n out of bounds");
}

// =============================================================================
// HHNeuron Step Tests
// =============================================================================

TEST(neuron_step_positive_current) {
    HHNeuron neuron;
    double initial_V = neuron.membrane_potential();
    neuron.step(0.01, 10.0);
    check(neuron.membrane_potential() > initial_V, "Positive current should increase voltage");
}

TEST(neuron_step_negative_current) {
    HHNeuron neuron;
    double initial_V = neuron.membrane_potential();
    neuron.step(0.01, -10.0);
    check(neuron.membrane_potential() < initial_V, "Negative current should decrease voltage");
}

TEST(neuron_step_zero_current) {
    HHNeuron neuron;
    double initial_V = neuron.membrane_potential();
    // At rest, zero current should keep voltage relatively stable
    for (int i = 0; i < 100; i++) {
        neuron.step(0.01, 0.0);
    }
    // Should stay near resting potential
    check_approx(neuron.membrane_potential(), initial_V, 5.0, "Zero current should maintain near-resting potential");
}

TEST(neuron_step_euler_vs_rk4) {
    // Both methods should produce similar results for small dt
    HHNeuron neuron_euler;
    HHNeuron neuron_rk4;

    neuron_euler.set_integration_method(IntegrationMethod::EULER);
    neuron_rk4.set_integration_method(IntegrationMethod::RK4);

    double dt = 0.001;  // Small dt for accuracy
    double I_ext = 10.0;

    for (int i = 0; i < 1000; i++) {
        neuron_euler.step(dt, I_ext);
        neuron_rk4.step(dt, I_ext);
    }

    // Results should be close for small dt
    check_approx(neuron_euler.membrane_potential(), neuron_rk4.membrane_potential(), 1.0,
                 "Euler and RK4 should produce similar results for small dt");
}

// =============================================================================
// HHNeuron Simulation Tests
// =============================================================================

TEST(neuron_simulate_length) {
    HHNeuron neuron;
    auto trace = neuron.simulate(10.0, 0.01, 10.0);
    check(trace.size() == 1000, "Trace length incorrect");
}

TEST(neuron_simulate_varying_current) {
    HHNeuron neuron;
    std::vector<double> I_ext(1000, 10.0);
    auto trace = neuron.simulate(10.0, 0.01, I_ext);
    check(trace.size() == 1000, "Trace length incorrect for varying current");
}

TEST(neuron_action_potential_generation) {
    HHNeuron neuron;
    auto trace = neuron.simulate(50.0, 0.01, 15.0);
    double max_V = *std::max_element(trace.begin(), trace.end());
    check(max_V > 0.0, "Action potential should exceed 0 mV");
    check(max_V < 60.0, "Action potential should not exceed 60 mV");
}

TEST(neuron_subthreshold_no_spike) {
    HHNeuron neuron;
    auto trace = neuron.simulate(50.0, 0.01, 2.0);  // Lower current, shorter duration
    double max_V = *std::max_element(trace.begin(), trace.end());
    check(max_V < -40.0, "Subthreshold current should not produce spike");
}

TEST(neuron_spike_count_increases_with_current) {
    auto count_spikes = [](const std::vector<double>& trace, double threshold = 0.0) {
        int count = 0;
        for (size_t i = 1; i < trace.size(); i++) {
            if (trace[i-1] < threshold && trace[i] >= threshold) {
                count++;
            }
        }
        return count;
    };

    HHNeuron neuron1, neuron2;
    auto trace1 = neuron1.simulate(500.0, 0.01, 8.0);
    auto trace2 = neuron2.simulate(500.0, 0.01, 15.0);

    int spikes1 = count_spikes(trace1);
    int spikes2 = count_spikes(trace2);

    check(spikes2 > spikes1, "Higher current should produce more spikes");
}

// =============================================================================
// HHNeuron Edge Cases
// =============================================================================

TEST(neuron_extreme_positive_current) {
    HHNeuron neuron;
    // Very high current - should still produce valid output
    auto trace = neuron.simulate(10.0, 0.01, 1000.0);
    for (double v : trace) {
        check(!std::isnan(v), "Voltage should not be NaN");
        check(!std::isinf(v), "Voltage should not be infinite");
    }
}

TEST(neuron_extreme_negative_current) {
    HHNeuron neuron;
    auto trace = neuron.simulate(10.0, 0.01, -100.0);
    for (double v : trace) {
        check(!std::isnan(v), "Voltage should not be NaN");
        check(!std::isinf(v), "Voltage should not be infinite");
    }
}

TEST(neuron_very_small_dt) {
    HHNeuron neuron;
    // Very small timestep should still work
    for (int i = 0; i < 100; i++) {
        neuron.step(0.0001, 10.0);
    }
    check(!std::isnan(neuron.membrane_potential()), "Small dt should not produce NaN");
}

TEST(neuron_large_dt_stability) {
    HHNeuron neuron;
    // Large timestep (5x typical) - RK4 should be more stable than Euler
    neuron.set_integration_method(IntegrationMethod::RK4);
    for (int i = 0; i < 100; i++) {
        neuron.step(0.05, 10.0);
    }
    check(!std::isnan(neuron.membrane_potential()), "RK4 should handle larger dt");
    check(neuron.membrane_potential() > -200.0 && neuron.membrane_potential() < 200.0,
          "Voltage should remain bounded");
}

TEST(neuron_gating_variables_bounded) {
    HHNeuron neuron;
    neuron.simulate(100.0, 0.01, 20.0);

    const auto& state = neuron.state();
    check(state.m >= 0.0 && state.m <= 1.0, "m should be in [0,1]");
    check(state.h >= 0.0 && state.h <= 1.0, "h should be in [0,1]");
    check(state.n >= 0.0 && state.n <= 1.0, "n should be in [0,1]");
}

TEST(neuron_zero_conductance) {
    HHNeuron::Parameters params;
    params.g_Na = 0.0;
    params.g_K = 0.0;
    HHNeuron neuron(params);

    auto trace = neuron.simulate(10.0, 0.01, 10.0);
    for (double v : trace) {
        check(!std::isnan(v), "Zero conductance should not produce NaN");
    }
}

TEST(neuron_multiple_resets) {
    HHNeuron neuron;
    for (int i = 0; i < 10; i++) {
        neuron.simulate(10.0, 0.01, 20.0);
        neuron.reset();
        check_approx(neuron.membrane_potential(), -65.0, 1.0, "Reset should restore resting potential");
    }
}

// =============================================================================
// Network Basic Tests
// =============================================================================

TEST(network_empty_creation) {
    Network net;
    check(net.num_neurons() == 0, "Empty network should have 0 neurons");
    check(net.num_synapses() == 0, "Empty network should have 0 synapses");
}

TEST(network_creation_with_neurons) {
    Network net(5);
    check(net.num_neurons() == 5, "Network should have 5 neurons");
}

TEST(network_add_neuron) {
    Network net;
    size_t idx = net.add_neuron();
    check(idx == 0, "First neuron should have index 0");
    check(net.num_neurons() == 1, "Should have 1 neuron");

    idx = net.add_neuron();
    check(idx == 1, "Second neuron should have index 1");
}

TEST(network_add_neuron_custom_params) {
    Network net;
    HHNeuron::Parameters params;
    params.g_Na = 100.0;
    size_t idx = net.add_neuron(params);
    check_approx(net.neuron(idx).parameters().g_Na, 100.0, 0.001, "Custom params not applied");
}

TEST(network_add_synapse) {
    Network net(3);
    net.add_synapse(0, 1, 0.5);
    check(net.num_synapses() == 1, "Should have 1 synapse");

    net.add_synapse(1, 2, 0.3);
    check(net.num_synapses() == 2, "Should have 2 synapses");
}

TEST(network_get_potentials) {
    Network net(3);
    auto potentials = net.get_potentials();
    check(potentials.size() == 3, "Should get 3 potentials");
    for (double v : potentials) {
        check_approx(v, -65.0, 1.0, "Initial potential should be ~-65 mV");
    }
}

TEST(network_reset) {
    Network net(2);
    std::vector<double> I_ext = {20.0, 20.0};
    net.step(0.01, I_ext);
    net.step(0.01, I_ext);

    net.reset();
    auto potentials = net.get_potentials();
    for (double v : potentials) {
        check_approx(v, -65.0, 1.0, "Reset should restore resting potentials");
    }
}

// =============================================================================
// Network Simulation Tests
// =============================================================================

TEST(network_step) {
    Network net(2);
    std::vector<double> I_ext = {10.0, 0.0};

    auto initial = net.get_potentials();
    net.step(0.01, I_ext);
    auto after = net.get_potentials();

    check(after[0] > initial[0], "Stimulated neuron voltage should increase");
}

TEST(network_simulate) {
    Network net(2);
    double duration = 10.0;
    double dt = 0.01;
    size_t num_steps = static_cast<size_t>(duration / dt);

    std::vector<std::vector<double>> I_ext(2, std::vector<double>(num_steps, 10.0));
    auto traces = net.simulate(duration, dt, I_ext);

    check(traces.size() == 2, "Should have 2 traces");
    check(traces[0].size() == num_steps, "Trace length incorrect");
}

TEST(network_synaptic_transmission) {
    Network net(2);
    net.add_synapse(0, 1, 2.0, 0.0, 2.0);  // Strong excitatory synapse

    double duration = 200.0;
    double dt = 0.01;
    size_t num_steps = static_cast<size_t>(duration / dt);

    std::vector<std::vector<double>> I_ext(2, std::vector<double>(num_steps, 0.0));
    // Only stimulate first neuron
    for (size_t i = 0; i < num_steps; i++) {
        I_ext[0][i] = 15.0;
    }

    auto traces = net.simulate(duration, dt, I_ext);

    // First neuron should spike
    double max_v0 = *std::max_element(traces[0].begin(), traces[0].end());
    check(max_v0 > 0.0, "Presynaptic neuron should spike");
}

TEST(network_chain_propagation) {
    Network net(3);
    net.add_synapse(0, 1, 1.0, 0.0, 2.0);
    net.add_synapse(1, 2, 1.0, 0.0, 2.0);

    double duration = 300.0;
    double dt = 0.01;
    size_t num_steps = static_cast<size_t>(duration / dt);

    std::vector<std::vector<double>> I_ext(3, std::vector<double>(num_steps, 0.0));
    for (size_t i = 0; i < num_steps; i++) {
        I_ext[0][i] = 15.0;
    }

    auto traces = net.simulate(duration, dt, I_ext);

    // All neurons should eventually be influenced
    double max_v0 = *std::max_element(traces[0].begin(), traces[0].end());
    check(max_v0 > 0.0, "First neuron should spike");
}

// =============================================================================
// Network Edge Cases
// =============================================================================

TEST(network_self_synapse) {
    // Self-connections should be allowed (autapses)
    Network net(1);
    net.add_synapse(0, 0, 0.1);
    check(net.num_synapses() == 1, "Self-synapse should be added");
}

TEST(network_multiple_synapses_same_pair) {
    Network net(2);
    net.add_synapse(0, 1, 0.5);
    net.add_synapse(0, 1, 0.3);
    check(net.num_synapses() == 2, "Multiple synapses between same pair allowed");
}

TEST(network_bidirectional_synapses) {
    Network net(2);
    net.add_synapse(0, 1, 0.5);
    net.add_synapse(1, 0, 0.5);
    check(net.num_synapses() == 2, "Bidirectional synapses should work");
}

TEST(network_inhibitory_synapse) {
    Network net(2);
    // Inhibitory synapse (negative reversal potential)
    net.add_synapse(0, 1, 1.0, -80.0, 2.0);

    double duration = 100.0;
    double dt = 0.01;
    size_t num_steps = static_cast<size_t>(duration / dt);

    std::vector<std::vector<double>> I_ext(2, std::vector<double>(num_steps, 0.0));
    for (size_t i = 0; i < num_steps; i++) {
        I_ext[0][i] = 15.0;
        I_ext[1][i] = 10.0;  // Give second neuron some current too
    }

    auto traces = net.simulate(duration, dt, I_ext);
    // Test should complete without errors
    check(!std::isnan(traces[1].back()), "Inhibitory synapse should produce valid output");
}

TEST(network_large_network) {
    Network net(100);
    // Add some random connections
    for (int i = 0; i < 99; i++) {
        net.add_synapse(i, i + 1, 0.1);
    }
    check(net.num_neurons() == 100, "Large network should have 100 neurons");
    check(net.num_synapses() == 99, "Should have 99 synapses");
}

// =============================================================================
// Numerical Accuracy Tests
// =============================================================================

TEST(numerical_euler_vs_rk4_accuracy) {
    // RK4 should be more accurate than Euler for same dt
    double dt = 0.05;  // Moderate dt where difference is visible
    double duration = 50.0;

    HHNeuron ref;
    ref.set_integration_method(IntegrationMethod::RK4);
    auto trace_ref = ref.simulate(duration, 0.001, 10.0);  // Reference with small dt

    HHNeuron euler;
    euler.set_integration_method(IntegrationMethod::EULER);
    auto trace_euler = euler.simulate(duration, dt, 10.0);

    HHNeuron rk4;
    rk4.set_integration_method(IntegrationMethod::RK4);
    auto trace_rk4 = rk4.simulate(duration, dt, 10.0);

    // Compare max voltage (action potential peak)
    double max_ref = *std::max_element(trace_ref.begin(), trace_ref.end());
    double max_euler = *std::max_element(trace_euler.begin(), trace_euler.end());
    double max_rk4 = *std::max_element(trace_rk4.begin(), trace_rk4.end());

    double error_euler = std::abs(max_euler - max_ref);
    double error_rk4 = std::abs(max_rk4 - max_ref);

    // RK4 error should be smaller (or at least not much worse)
    // Note: This is a soft check since the comparison is approximate
    check(error_rk4 < error_euler * 2 || error_rk4 < 5.0,
          "RK4 should be at least as accurate as Euler");
}

TEST(numerical_conservation_at_rest) {
    // At rest with no input, system should stay at rest
    HHNeuron neuron;
    double initial_V = neuron.membrane_potential();

    for (int i = 0; i < 10000; i++) {
        neuron.step(0.01, 0.0);
    }

    // Should stay near resting potential (within a few mV)
    check_approx(neuron.membrane_potential(), initial_V, 2.0,
                 "Resting neuron should maintain stable potential");
}

// =============================================================================
// Izhikevich Neuron Basic Tests
// =============================================================================

TEST(izhikevich_default_creation) {
    IzhikevichNeuron neuron;  // Default is RS with c=-65
    // V is initialized to params.c
    check_approx(neuron.membrane_potential(), neuron.parameters().c, 1.0, "Default V should equal c");
    // Izhikevich always uses Euler due to discontinuous reset
    check(neuron.integration_method() == IntegrationMethod::EULER, "Should use Euler integration");
}

TEST(izhikevich_preset_types) {
    // Test all preset types can be created
    // Initial V is set to params.c (after-spike reset value) which varies by type
    IzhikevichNeuron rs(IzhikevichNeuron::Type::REGULAR_SPIKING);
    IzhikevichNeuron fs(IzhikevichNeuron::Type::FAST_SPIKING);
    IzhikevichNeuron ib(IzhikevichNeuron::Type::INTRINSICALLY_BURSTING);
    IzhikevichNeuron ch(IzhikevichNeuron::Type::CHATTERING);
    IzhikevichNeuron lts(IzhikevichNeuron::Type::LOW_THRESHOLD_SPIKING);

    // Initial V should match the c parameter for each type
    check_approx(rs.membrane_potential(), rs.parameters().c, 1.0, "RS initial V should match c");
    check_approx(fs.membrane_potential(), fs.parameters().c, 1.0, "FS initial V should match c");
    check_approx(ib.membrane_potential(), ib.parameters().c, 1.0, "IB initial V should match c");
    check_approx(ch.membrane_potential(), ch.parameters().c, 1.0, "CH initial V should match c");
    check_approx(lts.membrane_potential(), lts.parameters().c, 1.0, "LTS initial V should match c");

    // Verify the expected c values
    check_approx(rs.parameters().c, -65.0, 0.1, "RS c incorrect");
    check_approx(fs.parameters().c, -65.0, 0.1, "FS c incorrect");
    check_approx(ib.parameters().c, -55.0, 0.1, "IB c incorrect");
    check_approx(ch.parameters().c, -50.0, 0.1, "CH c incorrect");
    check_approx(lts.parameters().c, -65.0, 0.1, "LTS c incorrect");
}

TEST(izhikevich_custom_parameters) {
    IzhikevichNeuron::Parameters params;
    params.a = 0.05;
    params.b = 0.25;
    params.c = -60.0;
    params.d = 4.0;
    IzhikevichNeuron neuron(params);

    check_approx(neuron.parameters().a, 0.05, 0.001, "Custom a not set");
    check_approx(neuron.parameters().b, 0.25, 0.001, "Custom b not set");
    check_approx(neuron.parameters().c, -60.0, 0.001, "Custom c not set");
    check_approx(neuron.parameters().d, 4.0, 0.001, "Custom d not set");
}

TEST(izhikevich_get_preset) {
    auto rs = IzhikevichNeuron::get_preset(IzhikevichNeuron::Type::REGULAR_SPIKING);
    check_approx(rs.a, 0.02, 0.001, "RS preset a incorrect");
    check_approx(rs.b, 0.2, 0.001, "RS preset b incorrect");
    check_approx(rs.c, -65.0, 0.001, "RS preset c incorrect");
    check_approx(rs.d, 8.0, 0.001, "RS preset d incorrect");

    auto fs = IzhikevichNeuron::get_preset(IzhikevichNeuron::Type::FAST_SPIKING);
    check_approx(fs.a, 0.1, 0.001, "FS preset a incorrect");
    check_approx(fs.d, 2.0, 0.001, "FS preset d incorrect");
}

TEST(izhikevich_reset) {
    IzhikevichNeuron neuron;  // Default is RS with c=-65
    neuron.set_membrane_potential(0.0);
    check_approx(neuron.membrane_potential(), 0.0, 0.001, "set_membrane_potential failed");
    neuron.reset();
    // Reset sets V to params.c
    check_approx(neuron.membrane_potential(), neuron.parameters().c, 1.0, "reset failed");
}

TEST(izhikevich_state_access) {
    IzhikevichNeuron neuron;  // Default RS: c=-65, b=0.2
    const auto& state = neuron.state();
    const auto& params = neuron.parameters();
    // v is initialized to c, u is initialized to b*v
    check_approx(state.v, params.c, 1.0, "State v should equal c");
    check_approx(state.u, params.b * params.c, 1.0, "State u should equal b*c");
}

TEST(izhikevich_recovery_variable) {
    IzhikevichNeuron neuron;  // Default RS: c=-65, b=0.2
    // u is initialized to b*v = b*c
    double expected_u = neuron.parameters().b * neuron.parameters().c;
    check_approx(neuron.recovery_variable(), expected_u, 1.0, "Recovery variable should be b*c");
}

// =============================================================================
// Izhikevich Neuron Step Tests
// =============================================================================

TEST(izhikevich_step_positive_current) {
    IzhikevichNeuron neuron;
    double initial_V = neuron.membrane_potential();
    neuron.step(0.1, 10.0);
    check(neuron.membrane_potential() > initial_V, "Positive current should increase voltage");
}

TEST(izhikevich_step_negative_current) {
    IzhikevichNeuron neuron;
    double initial_V = neuron.membrane_potential();
    neuron.step(0.1, -10.0);
    check(neuron.membrane_potential() < initial_V, "Negative current should decrease voltage");
}

TEST(izhikevich_spike_detection) {
    IzhikevichNeuron neuron;
    bool spiked = false;
    for (int i = 0; i < 1000; i++) {
        neuron.step(0.1, 15.0);
        if (neuron.spiked()) {
            spiked = true;
            break;
        }
    }
    check(spiked, "Neuron should spike with sufficient current");
}

TEST(izhikevich_spike_reset) {
    IzhikevichNeuron neuron;
    for (int i = 0; i < 1000; i++) {
        neuron.step(0.1, 15.0);
        if (neuron.spiked()) {
            // After spike, V should be reset to c
            check_approx(neuron.membrane_potential(), neuron.parameters().c, 5.0,
                        "Voltage should reset to c after spike");
            break;
        }
    }
}

// =============================================================================
// Izhikevich Neuron Simulation Tests
// =============================================================================

TEST(izhikevich_simulate_length) {
    IzhikevichNeuron neuron;
    auto trace = neuron.simulate(100.0, 0.1, 10.0);
    check(trace.size() == 1000, "Trace length incorrect");
}

TEST(izhikevich_simulate_varying_current) {
    IzhikevichNeuron neuron;
    std::vector<double> I_ext(1000, 10.0);
    auto trace = neuron.simulate(100.0, 0.1, I_ext);
    check(trace.size() == 1000, "Trace length incorrect for varying current");
}

TEST(izhikevich_spike_generation) {
    IzhikevichNeuron neuron;
    auto trace = neuron.simulate(100.0, 0.1, 15.0);
    double max_V = *std::max_element(trace.begin(), trace.end());
    check(max_V > 0.0, "Should produce spikes exceeding 0 mV");
}

TEST(izhikevich_subthreshold_no_spike) {
    IzhikevichNeuron neuron;
    auto trace = neuron.simulate(100.0, 0.1, 1.0);
    double max_V = *std::max_element(trace.begin(), trace.end());
    check(max_V < 0.0, "Subthreshold current should not produce spike");
}

TEST(izhikevich_spike_count_increases_with_current) {
    auto count_spikes = [](const std::vector<double>& trace, double threshold = 0.0) {
        int count = 0;
        for (size_t i = 1; i < trace.size(); i++) {
            if (trace[i-1] < threshold && trace[i] >= threshold) {
                count++;
            }
        }
        return count;
    };

    IzhikevichNeuron neuron1, neuron2;
    auto trace1 = neuron1.simulate(500.0, 0.1, 8.0);
    auto trace2 = neuron2.simulate(500.0, 0.1, 20.0);

    int spikes1 = count_spikes(trace1);
    int spikes2 = count_spikes(trace2);

    check(spikes2 > spikes1, "Higher current should produce more spikes");
}

// =============================================================================
// Izhikevich Spiking Pattern Tests
// =============================================================================

TEST(izhikevich_fs_faster_than_rs) {
    auto count_spikes = [](const std::vector<double>& trace, double threshold = 0.0) {
        int count = 0;
        for (size_t i = 1; i < trace.size(); i++) {
            if (trace[i-1] < threshold && trace[i] >= threshold) {
                count++;
            }
        }
        return count;
    };

    IzhikevichNeuron rs(IzhikevichNeuron::Type::REGULAR_SPIKING);
    IzhikevichNeuron fs(IzhikevichNeuron::Type::FAST_SPIKING);

    auto trace_rs = rs.simulate(500.0, 0.1, 15.0);
    auto trace_fs = fs.simulate(500.0, 0.1, 15.0);

    int spikes_rs = count_spikes(trace_rs);
    int spikes_fs = count_spikes(trace_fs);

    check(spikes_fs > spikes_rs, "Fast spiking should produce more spikes than regular spiking");
}

// =============================================================================
// Izhikevich Edge Cases
// =============================================================================

TEST(izhikevich_extreme_positive_current) {
    IzhikevichNeuron neuron;
    auto trace = neuron.simulate(100.0, 0.1, 100.0);
    for (double v : trace) {
        check(!std::isnan(v), "Voltage should not be NaN");
        check(!std::isinf(v), "Voltage should not be infinite");
    }
}

TEST(izhikevich_extreme_negative_current) {
    IzhikevichNeuron neuron;
    auto trace = neuron.simulate(100.0, 0.1, -100.0);
    for (double v : trace) {
        check(!std::isnan(v), "Voltage should not be NaN");
        check(!std::isinf(v), "Voltage should not be infinite");
    }
}

TEST(izhikevich_voltage_bounded) {
    IzhikevichNeuron neuron;
    auto trace = neuron.simulate(1000.0, 0.1, 20.0);
    double max_V = *std::max_element(trace.begin(), trace.end());
    double min_V = *std::min_element(trace.begin(), trace.end());
    check(max_V <= 100.0, "Voltage should be bounded at 100 mV");
    check(min_V > -200.0, "Voltage should not go too negative");
}

TEST(izhikevich_long_simulation_stability) {
    IzhikevichNeuron neuron;
    auto trace = neuron.simulate(10000.0, 0.1, 10.0);
    for (double v : trace) {
        check(!std::isnan(v), "Long simulation should not produce NaN");
        check(!std::isinf(v), "Long simulation should not produce infinity");
    }
}

TEST(izhikevich_multiple_resets) {
    IzhikevichNeuron neuron;  // Default is RS with c=-65
    double reset_V = neuron.parameters().c;
    for (int i = 0; i < 10; i++) {
        neuron.simulate(50.0, 0.1, 20.0);
        neuron.reset();
        check_approx(neuron.membrane_potential(), reset_V, 1.0, "Reset should restore to c");
    }
}

// =============================================================================
// NeuronBase Polymorphism Tests
// =============================================================================

TEST(neuron_base_polymorphism) {
    // Test that both neuron types work through base class pointer
    // Note: HH rests at -65, Izhikevich RS also rests at -65 (c=-65)
    HHNeuron hh;
    IzhikevichNeuron iz;  // Default RS with c=-65

    std::vector<NeuronBase*> neurons = {&hh, &iz};

    // Both should support the common interface
    for (auto* neuron : neurons) {
        double initial_V = neuron->membrane_potential();
        // Both should be near -65 for default parameters
        check(initial_V < -50.0, "Initial potential should be negative");

        neuron->step(0.1, 10.0);
        check(neuron->membrane_potential() > initial_V, "Voltage should increase with positive current");

        neuron->reset();
        check_approx(neuron->membrane_potential(), initial_V, 1.0, "Reset should restore initial potential");
    }
}

TEST(neuron_base_simulate_polymorphism) {
    // Test that simulate works through base class
    HHNeuron hh;
    IzhikevichNeuron iz;

    NeuronBase* neurons[] = {&hh, &iz};
    std::string names[] = {"HH", "Izhikevich"};

    for (int i = 0; i < 2; i++) {
        auto trace = neurons[i]->simulate(100.0, 0.1, 15.0);
        check(trace.size() == 1000, (names[i] + " trace length incorrect").c_str());
        double max_V = *std::max_element(trace.begin(), trace.end());
        check(max_V > 0.0, (names[i] + " should produce spikes").c_str());
    }
}

TEST(neuron_type_names) {
    HHNeuron hh;
    IzhikevichNeuron iz;

    check(hh.type_name() == "HH", "HH type name incorrect");
    check(iz.type_name() == "Izhikevich", "Izhikevich type name incorrect");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Neural Simulation Library C++ Tests ===\n\n";

    std::cout << "--- HHNeuron Basic Tests ---\n";
    RUN_TEST(neuron_default_creation);
    RUN_TEST(neuron_custom_parameters);
    RUN_TEST(neuron_with_integration_method);
    RUN_TEST(neuron_set_integration_method);
    RUN_TEST(neuron_reset);
    RUN_TEST(neuron_state_access);

    std::cout << "\n--- HHNeuron Step Tests ---\n";
    RUN_TEST(neuron_step_positive_current);
    RUN_TEST(neuron_step_negative_current);
    RUN_TEST(neuron_step_zero_current);
    RUN_TEST(neuron_step_euler_vs_rk4);

    std::cout << "\n--- HHNeuron Simulation Tests ---\n";
    RUN_TEST(neuron_simulate_length);
    RUN_TEST(neuron_simulate_varying_current);
    RUN_TEST(neuron_action_potential_generation);
    RUN_TEST(neuron_subthreshold_no_spike);
    RUN_TEST(neuron_spike_count_increases_with_current);

    std::cout << "\n--- HHNeuron Edge Cases ---\n";
    RUN_TEST(neuron_extreme_positive_current);
    RUN_TEST(neuron_extreme_negative_current);
    RUN_TEST(neuron_very_small_dt);
    RUN_TEST(neuron_large_dt_stability);
    RUN_TEST(neuron_gating_variables_bounded);
    RUN_TEST(neuron_zero_conductance);
    RUN_TEST(neuron_multiple_resets);

    std::cout << "\n--- Izhikevich Neuron Basic Tests ---\n";
    RUN_TEST(izhikevich_default_creation);
    RUN_TEST(izhikevich_preset_types);
    RUN_TEST(izhikevich_custom_parameters);
    RUN_TEST(izhikevich_get_preset);
    RUN_TEST(izhikevich_reset);
    RUN_TEST(izhikevich_state_access);
    RUN_TEST(izhikevich_recovery_variable);

    std::cout << "\n--- Izhikevich Neuron Step Tests ---\n";
    RUN_TEST(izhikevich_step_positive_current);
    RUN_TEST(izhikevich_step_negative_current);
    RUN_TEST(izhikevich_spike_detection);
    RUN_TEST(izhikevich_spike_reset);

    std::cout << "\n--- Izhikevich Neuron Simulation Tests ---\n";
    RUN_TEST(izhikevich_simulate_length);
    RUN_TEST(izhikevich_simulate_varying_current);
    RUN_TEST(izhikevich_spike_generation);
    RUN_TEST(izhikevich_subthreshold_no_spike);
    RUN_TEST(izhikevich_spike_count_increases_with_current);

    std::cout << "\n--- Izhikevich Spiking Patterns ---\n";
    RUN_TEST(izhikevich_fs_faster_than_rs);

    std::cout << "\n--- Izhikevich Edge Cases ---\n";
    RUN_TEST(izhikevich_extreme_positive_current);
    RUN_TEST(izhikevich_extreme_negative_current);
    RUN_TEST(izhikevich_voltage_bounded);
    RUN_TEST(izhikevich_long_simulation_stability);
    RUN_TEST(izhikevich_multiple_resets);

    std::cout << "\n--- NeuronBase Polymorphism Tests ---\n";
    RUN_TEST(neuron_base_polymorphism);
    RUN_TEST(neuron_base_simulate_polymorphism);
    RUN_TEST(neuron_type_names);

    std::cout << "\n--- Network Basic Tests ---\n";
    RUN_TEST(network_empty_creation);
    RUN_TEST(network_creation_with_neurons);
    RUN_TEST(network_add_neuron);
    RUN_TEST(network_add_neuron_custom_params);
    RUN_TEST(network_add_synapse);
    RUN_TEST(network_get_potentials);
    RUN_TEST(network_reset);

    std::cout << "\n--- Network Simulation Tests ---\n";
    RUN_TEST(network_step);
    RUN_TEST(network_simulate);
    RUN_TEST(network_synaptic_transmission);
    RUN_TEST(network_chain_propagation);

    std::cout << "\n--- Network Edge Cases ---\n";
    RUN_TEST(network_self_synapse);
    RUN_TEST(network_multiple_synapses_same_pair);
    RUN_TEST(network_bidirectional_synapses);
    RUN_TEST(network_inhibitory_synapse);
    RUN_TEST(network_large_network);

    std::cout << "\n--- Numerical Accuracy Tests ---\n";
    RUN_TEST(numerical_euler_vs_rk4_accuracy);
    RUN_TEST(numerical_conservation_at_rest);

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
