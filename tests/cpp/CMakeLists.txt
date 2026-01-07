add_executable(test_neuron test_neuron.cpp)
target_link_libraries(test_neuron PRIVATE hodgkin_huxley_core)

add_test(NAME cpp_tests COMMAND test_neuron)
