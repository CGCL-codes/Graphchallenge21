#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "matrix.h"
#include "debug.h"

namespace ftxj {
    class CpuTranspose {
    public:
        static void run_and_cmp(float* input, int neuron, int batch, float* output) {
            std::vector<std::vector<float>> res(neuron, std::vector<float>(batch, 0.0));
            for(int b = 0; b < batch; ++b) {
                for(int n = 0; n < neuron; ++n) {
                    res[n][b] = input[b * neuron + n];
                    assert_msg(res[n][b] == output[n * batch + b], "error!");
                }
            }
            std::cout << "Compare with cpu result [Success]" << std::endl;
        }
    };
};