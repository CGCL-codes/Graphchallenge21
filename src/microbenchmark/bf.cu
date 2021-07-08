#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ftxj {

__global__ void bf_spmm(
    
    float* Y0, // input
    float* Y1,

    int* roffW,  // len neuron * N_SLAB - 1
    int* colsW,  // index 32 * neuron
    float* valsW, // all 32 * neuron 0.0625

    int COL_BLK, // TN, shared memory size = TN 
    int N_SLAB, //  neuron / TN
    int neuron // neuron


) {

  extern  __shared__ float shRow[];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int rid = blockIdx.x;

  __syncthreads();

  for(int i = 0; i < N_SLAB; i++) {
    __syncthreads();
    for(int j = threadIdx.x; j < COL_BLK; j++) {
      shRow[j] = 0;  
    }
    __syncthreads();
    for(int j = threadIdx.y; j < neuron; j += blockDim.y) {
      float valY = Y0[rid * neuron + j];
    //   if(valY == 0) {
    //     continue;
    //   }

      int begOffW = roffW[i * neuron + j] + threadIdx.x;
      int endOffW = roffW[i * neuron + j + 1];
      
      for(int k = begOffW; k < endOffW; k += blockDim.x) {
        int colW = colsW[k];
        float valW = valsW[k];
        // if(colW - i * COL_BLK < 0 || colW - i * COL_BLK >= 1024) {
        //   printf("bugs %d %d %d %d\n", k, i, colW, colW - i * COL_BLK);
        // }
        atomicAdd(&shRow[colW - i * COL_BLK], valY * valW);
      }
    }
    __syncthreads();
    int count = 0;
    for(size_t j = 0; j < COL_BLK; j += blockDim.x * blockDim.y) {
    //   float v = j + tid < COL_BLK ? shRow[j + tid] + bias : -1;
    //   count += __syncthreads_count(v > 0);
      if(j + tid < COL_BLK) {
        Y1[rid * neuron + i * COL_BLK + j + tid] = shRow[j + tid];
        // min(T(32), max(T(0), v));
      }
    }
  }
}

void test_benchmark_19_BF(COOMatrix &coo, BFMatrix &matrix, 
    int neuron, int batch, int TN, 
    int blockx, int blocky,
    GpuEnv &env) {

    float *nextfeat;
    float *currfeat;

    int *rowoff;

    int off_size = neuron * (neuron / TN + 1) + 1;
    
    int *rowindex;
    
    int weight_nnz = 32 * neuron;

    float *value; 

    float bias = 0;
    int mybatch = batch;

    // std::vector<std::vector<float>> input(mybatch, std::vector<float>(neuron, 0.0));
	  float * input = (float*)malloc(sizeof(float) * neuron * mybatch);
	  memset(input, 0, sizeof(float) * neuron * mybatch);

	  float * output = (float*)malloc(sizeof(float) * neuron * mybatch);
	  memset(output, 0, sizeof(float) * neuron * mybatch);


    srand (static_cast <unsigned> (time(0)));
    for(int i = 0; i < mybatch; ++i) {
      for(int j = 0; j < neuron; ++j) {
        float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/32.0));
        input[i * neuron + j] = r2;
      }
    }
  
    Safe_Call(cudaMalloc((void**)&rowoff, sizeof(int) * off_size));
    Safe_Call(cudaMemcpy(rowoff, &matrix.rowoff[0], sizeof(int) * off_size, cudaMemcpyHostToDevice));
    
    Safe_Call(cudaMalloc((void**)&rowindex, sizeof(int) * weight_nnz));
    Safe_Call(cudaMemcpy(rowindex, &matrix.rowindex[0], sizeof(int) * weight_nnz, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&value, sizeof(float) * weight_nnz));
    Safe_Call(cudaMemcpy(value, &matrix.val[0], sizeof(float) * weight_nnz, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&currfeat, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemcpy(currfeat, input, sizeof(float) * neuron * mybatch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&nextfeat, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemset(nextfeat, 0, sizeof(float) * neuron * mybatch));

    std::cout << "begin inference..." << std::endl; 
    env.add_event("uiuc_kernel_timer");
    env.event_start_record("uiuc_kernel_timer");

    dim3 block(blockx, blocky);
    dim3 grid(batch);
    bf_spmm<<<grid,block, sizeof(float) * TN, env.get_stream("uiuc_kernel_timer")>>>(
        currfeat, nextfeat,  rowoff, rowindex, value, TN, neuron / TN, neuron
    );

    env.event_stop_record("uiuc_kernel_timer");
    float time = env.get_event_time("uiuc_kernel_timer"); 

    Safe_Call(cudaMemcpy(output, nextfeat, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

    std::cout << "Kernel Exec Time [19-BF] = " << time << "ms"<< std::endl;
    std::cout << "Flops [19-BF] = " << float(2 * batch * neuron * 32) /  time * 1000 /1e12 << "TFLOPS"<< std::endl;
    
	  CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, true, true);
}

}
