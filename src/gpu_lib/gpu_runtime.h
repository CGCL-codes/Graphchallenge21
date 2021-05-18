#pragma once

#include <cstdio>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char *file, const int line, bool fatal=false) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n",  file, line, int(result),
            cudaGetErrorString(result));\
    if (fatal) {
        exit(EXIT_FAILURE);
    }
  }
}

#define Safe_Call_Print(stmt) checkCuda(stmt, __FILE__, __LINE__)
#define Safe_Call(stmt) checkCuda(stmt, __FILE__, __LINE__, true)
