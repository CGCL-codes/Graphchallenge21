#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "matrix.h"
#include "debug.h"

namespace ftxj {
    class CpuTransposeDelete {
    public:
        static void run_and_cmp(float* input, int* old_to_new_map, int old_batch, int neuron, int new_batch, float* output) {
            std::vector<std::vector<float>> res(new_batch, std::vector<float>(neuron, 0.0));
            for(int b = 0; b < old_batch; ++b) {
                if(old_to_new_map[b] == -1) continue;
                int new_b = old_to_new_map[b];
                for(int n = 0; n < neuron; ++n) {
                    res[new_b][n] = input[n * old_batch + b];
                    if(std::abs(res[new_b][n] - output[new_b * neuron + n]) > 1e-3) {
                        std::cout << b << ", " << n << std::endl;
                        std::cout << new_b << ", " << n << std::endl;
                        std::cout << "currect = " << res[new_b][n] << ", error = " << output[new_b * neuron + n] << std::endl;
                        assert_msg(res[new_b][n] == output[new_b * neuron + n], "error!");
                    }
                }
            }
            std::cout << "Compare with cpu result [Success]" << std::endl;
        }
    };
};