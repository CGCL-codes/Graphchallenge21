#pragma once
#include "../utils/header.h"

namespace ftxj {

    void test_benchmark_succ_load_store(int, int, GpuEnv &);

    void test_benchmark_20_uiuc(COOMatrix&, UIUCMatrix &, int , GpuEnv &);
    void test_benchmark_row_succ_20_uiuc(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);
    void test_benchmark_row_succ_20_uiuc_transpose(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);
    void test_benchmark_row_succ_input_transpose_batch_parallel(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);
    void test_benchmark_rectangels_batch_parallel_kernel(COOMatrix&, std::vector<float> &, std::vector<int> &, int, int, GpuEnv &);



    void vector4_load_data_benchmark(GpuEnv &env);
    void test_shared_memory_mm(COOMatrix&, std::vector<float> &val, std::vector<int> &row_access, GpuEnv &env);
};
