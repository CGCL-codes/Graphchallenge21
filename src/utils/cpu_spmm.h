#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "matrix.h"
#include "debug.h"

namespace ftxj {
    class CpuSpmm {
    public:
        static void run_and_cmp(COOMatrix &weight, float* input, int neuron, int batch, float* output, bool T = false, bool resT = true, bool inputT = true) {
            weight.to_row_first_ordered();
            std::vector<std::vector<float>> res(batch, std::vector<float>(neuron, 0.0));
            for(int b = 0; b < batch; ++b) {
                for(auto iter = weight.begin(); iter != weight.end(); ++iter) {
                    int row = (*iter).row;
                    int col = (*iter).col;
                    float val = (*iter).val;
                    float in = 0.0;
                    if(T) {
                        if(inputT) in = input[b * neuron + row];
                        else in = input[row * batch + b];
                        res[b][col] += in * val;
                        // if(b == 0 && col == 0) {
                        //     printf("%f * %f %d\n", in, val, row);
                        // }
                    }
                    else {
                        if(inputT) in = input[b * neuron + col];
                        else in = input[col * batch + b];
                        res[b][row] += in * val;
                        // if(b == 0 && row == 0) {
                        //     printf("%f * %f %d\n", in, val, col);
                        // }
                    }
                }
            }
            for(int i = 0; i < batch; ++i) {
                for(int j = 0; j < neuron; ++j) {
                    float cmp = 0;
                    if(resT) cmp = output[i * neuron + j];
                    else cmp = output[j * batch + i];
                    if(std::abs(res[i][j] - cmp) > 1e-3) {
                        std::cout << i << ", " << j << " cpu=" << res[i][j] << ", gpu=" << cmp << std::endl;
                        assert_msg(res[i][j] == cmp, "cpu gpu doesnot equals!");
                    }
                }
            }
            std::cout << "Compare with cpu result [Success]" << std::endl;
        }
    };
};