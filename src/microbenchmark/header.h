#pragma once
#include "../utils/header.h"

namespace ftxj {
    void uiuc_test_benchmark(UIUCMatrix &matrix, GpuEnv &env);
    void load_data_benchmark(GpuEnv &env);
    void test_shared_memory_mm(UIUCMatrix &matrix, GpuEnv &env);
};
