#pragma once
#include <vector>
#include <iostream>
#include "matrix.h"
#include "debug.h"

namespace ftxj {
    class CpuSpmm {
    public:
        static void run_and_cmp(COOMatrix &weight, float* input, int neuron, int batch, float* output, bool T = false) {
            weight.to_row_first_ordered();
            std::vector<std::vector<float>> res(batch, std::vector<float>(neuron, 0.0));
            for(int b = 0; b < batch; ++b) {
                for(auto iter = weight.begin(); iter != weight.end(); ++iter) {
                    int row = (*iter).row;
                    int col = (*iter).col;
                    float val = (*iter).val;
                    if(T) {
                        res[b][col] += input[b * neuron + row] * val;
                    }
                    else {
                        res[b][row] += input[b * neuron + col] * val;
                    }
                }
            }
            for(int i = 0; i < batch; ++i) {
                for(int j = 0; j < neuron; ++j) {
                    float cmp = 0;
                    if(T) cmp = output[i * neuron + j];
                    else cmp = output[j * batch + i];
                    if(res[i][j] != cmp) {
                        std::cout << i << ", " << j << " cpu=" << res[i][j] << ", gpu=" << cmp << std::endl;
                        assert_msg(res[i][j] == cmp, "cpu gpu doesnot equals!");
                    }
                }
            }
            std::cout << "cpu cmp success!" << std::endl;
        }
    };
};