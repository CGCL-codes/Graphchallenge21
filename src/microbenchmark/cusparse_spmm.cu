#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <iostream>
#include <cusparse.h>
#include <vector>

namespace ftxj {

#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}


void test_benchmark_cusparse(COOMatrix& coo, cuSPARSEMatrix &matrix, int neuron, int batch) {

	float * input = (float*)malloc(sizeof(float) * neuron * batch);
	memset(input, 0, sizeof(float) * neuron * batch);

	float * output = (float*)malloc(sizeof(float) * neuron * batch);
	memset(output, 0, sizeof(float) * neuron * batch);


    srand (static_cast <unsigned> (time(0)));
    for(int i = 0; i < batch; ++i) {
      for(int j = 0; j < neuron; ++j) {
        float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/32.0));
        input[i * neuron + j] = r2;
      }
    }

    float *A_d;
    float *B_d;
    
    int* len_d;
    int* index_d;
    float* val_d;

    Safe_Call(cudaMalloc((void**)&A_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A_d, input, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&B_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(B_d, 0, sizeof(float) * neuron * batch));


    Safe_Call(cudaMalloc((void**)&len_d, sizeof(int) * (neuron + 1)));
    Safe_Call(cudaMemcpy(len_d, matrix.len, sizeof(int) * (neuron + 1), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&index_d, sizeof(int) * (neuron * 32)));
    Safe_Call(cudaMemcpy(index_d, matrix.index, sizeof(int) * (neuron * 32), cudaMemcpyHostToDevice));
  
    Safe_Call(cudaMalloc((void**)&val_d, sizeof(float) * (neuron * 32)));
    Safe_Call(cudaMemcpy(val_d, matrix.val, sizeof(float) * (neuron * 32), cudaMemcpyHostToDevice));



    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer    = NULL;
    size_t bufferSize = 0;
    float alpha = 1.0f;
    float beta = 0.0f;

    CUSPARSE_CHECK( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format

    CUSPARSE_CHECK(cusparseCreateCsr(&matA, neuron, neuron, 32 * neuron,
                                      len_d, index_d, val_d,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, neuron, batch, neuron, A_d,
                                    CUDA_R_32F, CUSPARSE_ORDER_COL) )
                                        
    // Create dense matrix C
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, neuron, batch, neuron, B_d,
                                    CUDA_R_32F, CUSPARSE_ORDER_COL) )
    

                                        
    Safe_Call(cudaMalloc(&dBuffer, bufferSize));
              

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
   
                                CUSPARSE_CSRMM_ALG1, &bufferSize) )
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

   
    CUSPARSE_CHECK( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_MM_ALG_DEFAULT, dBuffer) )
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop); //ms 

    // destroy matrix/vector descriptors
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) )
    CUSPARSE_CHECK( cusparseDestroyDnMat(matB) )
    CUSPARSE_CHECK( cusparseDestroyDnMat(matC) )
    CUSPARSE_CHECK( cusparseDestroy(handle) )

    Safe_Call(cudaMemcpy(output, B_d, neuron * batch  * sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "kernel time = " << elapsed << "ms" << std::endl;
    std::cout << "Flops [cuSparse] = " << float(2 * batch * neuron * 32) /  elapsed * 1000 /1e12 << "TFLOPS"<< std::endl;

	CpuSpmm::run_and_cmp(coo, input, neuron, batch, output, false);

}
}
