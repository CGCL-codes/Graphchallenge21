#pragma once
#include "../../utils/header.h"

namespace ftxj {
    void test_benchmark_multi_gpu_graph_challenge(
    std::vector<std::vector<float>> &input,
    std::vector<std::vector<float>> &weight, 
    std::vector<std::vector<int>> &row_access,
    int batch, 
    int neuron, 
    float bias,
    int gpu_index,
    int
    );
};