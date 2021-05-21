#pragma once
#include "../utils/header.h"

namespace ftxj {
    void uiuc_test_benchmark(COOMatrix&, UIUCMatrix &matrix, GpuEnv &env);
    
    void naive_load_data_benchmark(GpuEnv &env);
    void vector4_load_data_benchmark(GpuEnv &env);

    void test_shared_memory_mm(COOMatrix&, std::vector<float> &val, std::vector<int> &row_access, GpuEnv &env);
};
