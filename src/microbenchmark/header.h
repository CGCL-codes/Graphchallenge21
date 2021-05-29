#pragma once
#include "../utils/header.h"

namespace ftxj {

    void test_benchmark_succ_load_store(int, int, GpuEnv &);
    void test_benchmark_matrix_transpose(int batch, int neuron, GpuEnv &env);


    void test_benchmark_20_uiuc(COOMatrix&, UIUCMatrix &, int , GpuEnv &);
    void test_benchmark_row_succ_20_uiuc(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);
    void test_benchmark_row_succ_20_uiuc_transpose(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);
    void test_benchmark_row_succ_input_transpose_batch_parallel(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);
    void test_benchmark_rectangels_batch_parallel_kernel(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);

    void test_benchmark_graph_challenge(std::vector<std::vector<float>> &input, 
        std::vector<std::vector<float>> &weight, std::vector<std::vector<int>> &row_access, 
        int batch, int neuron, float bias,GpuEnv &env
    );

    void test_benchmark_matrix_transpose_and_delete(int batch, int neuron, GpuEnv &env);

    void test_benchmark_n16384_l2_l10_kernel(COOMatrix& coo, std::vector<float> &val, int stride, int batch, int neuron, GpuEnv &env);
    void test_benchmark_n16384_l11_kernel(COOMatrix& coo, std::vector<float> &B_val, std::vector<int> &B_index, int batch, int neuron, GpuEnv &env);

    // void test_benchmark_n16384_l11_kernel(
    //     COOMatrix& coo, 
    //     std::vector<float> &B_val, 
    //     std::vector<int> &B_index, 
    //     std::vector<int> &A_row_access,
    //     std::vector<int> &A_row_access_len,
    //     int max_input_access,
    //     int batch, int neuron, 
    //     GpuEnv &env
    // );

    void vector4_load_data_benchmark(GpuEnv &env);
    void test_shared_memory_mm(COOMatrix&, std::vector<float> &val, std::vector<int> &row_access, GpuEnv &env);
};
