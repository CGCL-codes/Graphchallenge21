#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include "matrix.h"
#include "debug.h"

namespace ftxj {
    class CpuSpmmFuse {
    public:
        static void run_and_cmp(std::vector<COOMatrix> &weight, float* input, int neuron, int batch, int bias, float* output, int fuse_layer, bool T = false, bool resT = true, bool inputT = true) {
            for(int i = 0; i < fuse_layer; ++i) {
                weight[i].to_row_first_ordered();
            }
            std::vector<std::vector<float>> res1(batch, std::vector<float>(neuron, 0.0));
            std::vector<std::vector<float>> res2(batch, std::vector<float>(neuron, 0.0));
            for(int b = 0; b < batch; ++b) {
                if(b % 10000 == 0) std::cout << "run " << b << "..." << std::endl;
                for(auto iter = weight[0].begin(); iter != weight[0].end(); ++iter) {
                    int row = (*iter).row;
                    int col = (*iter).col;
                    float val = (*iter).val;
                    float in = 0.0;
                    if(T) {
                        if(inputT) in = input[b * neuron + row];
                        else in = input[row * batch + b];
                        res1[b][col] += in * val;
                        // if(b == 8 && col == 0) {
                        //     printf("%f * %f = %f\n", in, val, res1[b][col]);
                        // }
                    }
                    else {
                        if(inputT) in = input[b * neuron + col];
                        else in = input[col * batch + b];
                        res1[b][row] += in * val;
                        // if(b == 8 && row == 0) {
                        //     printf("%f * %f = %f\n", in, val, res1[b][row]);
                        // }
                    }
                }
                for(int j = 0; j < neuron; ++j) {
                    // res1[b][j] =  res1[b][j];  
                    // if(b == 8 && j == 0) {
                    //     printf("res1 = %f\n", res1[b][j]);
                    // }
                    res1[b][j] =   ((res1[b][j] + bias) > 32 ? 32.0 : ((res1[b][j] + bias) < 0) ? 0 : res1[b][j] + bias);
                }
            }
            for(int l = 1; l < fuse_layer; ++l) {
                for(int b = 0; b < batch; ++b) {
                    if(b % 10000 == 0) std::cout << "run l = " << l << ", b = " << b << "..." << std::endl;
                    for(auto iter = weight[l].begin(); iter != weight[l].end(); ++iter) {
                        int row = (*iter).row;
                        int col = (*iter).col;
                        float val = (*iter).val;
                        float in = 0.0;
                        if(T) {
                            in = res1[b][row];
                            res2[b][col] += in * val;
                            // if(b == 8 && col == 0) {
                            //     printf("%f * %f %d\n", in, val, row);
                            // }
                        }
                        else {
                            in = res1[b][col];
                            res2[b][row] += in * val;
                            // if(b == 8 && row == 0) {
                            //     printf("%f * %f %d\n", in, val, col);
                            // }
                        }
                    }
                    for(int j = 0; j < neuron; ++j) {
                    //    res2[b][j] =  res2[b][j];
                       res2[b][j] =  ((res2[b][j] + bias) > 32 ? 32.0 : ((res2[b][j] + bias) < 0) ? 0 : res2[b][j] + bias);
                    }
                }
                res1 = res2;
                res2 = std::vector<std::vector<float>>(batch, std::vector<float>(neuron, 0.0));
            }
            
            for(int b = 0; b < batch; ++b) {
                for(int j = 0; j < neuron; ++j) {
                    float cmp = 0;
                    if(resT) cmp = output[b * neuron + j];
                    else cmp = output[j * batch + b];
                    if(std::abs(res1[b][j] - cmp) > 1e-3) {
                        std::cout << b << ", " << j << " cpu=" << res1[b][j] << ", gpu=" << cmp << std::endl;
                        assert_msg(res1[b][j] == cmp, "cpu gpu doesnot equals!");
                    }
                }
            }

            std::cout << "Compare with cpu result [Success]" << std::endl;
        }
    };
};