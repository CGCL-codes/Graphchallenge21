#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

__device__ inline float __ReLU(float x) {
   return x<0.0?0.0:x>32.0?32.0:x;
};

#define MINIBATCH 8
#define UNROLL 8

__global__ void n16384_l11_kernel(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    int* __restrict__ index, 
    int* __restrict__ index_len, 
    int* __restrict__ B_index, 
    int batch, 
    int neuron, 
    float bias) {
    
    extern __shared__ float shared[];
    
    int col_gropu = threadIdx.x / 16;
	
    int last_load = ((neuron / 16) % 6) * 16 + 16 * 2;

    int start_idx = index_len[blockIdx.y];
    int load_num = index_len[blockIdx.y + 1] - index_len[blockIdx.y];

	for(int n = threadIdx.x; n < load_num * MINIBATCH; n += blockDim.x){
		int f = n / load_num;
		int k = n % load_num;
        shared[f * 160 + k] = A[(blockIdx.x * MINIBATCH + f) * neuron + index[start_idx + k]];
	}
    __syncthreads();

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //     for(int i = 0; i < load_num; ++i) {
    //         printf("load %d\n", index[start_idx + i]);
    //     }
    //     printf("load %d %f %f\n", index[start_idx + 7], shared[7], A[index[start_idx + 7]]);
    // }

    float res[MINIBATCH] = {0.0};

    for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * 128 * 32) + r * 128 + threadIdx.x];
      
        int idx = B_index[blockIdx.y * 8 * 32 + r * 8 + (threadIdx.x / 16)];
      
        for(int f = 0; f < MINIBATCH / UNROLL; ++f) {
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * 160 + idx] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * 160 + idx] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * 160 + idx] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * 160 + idx] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * 160 + idx] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * 160 + idx] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * 160 + idx] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * 160 + idx] * val;
        }
    }

    for(int f = 0; f < MINIBATCH ; ++f) {
        C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 128 + threadIdx.x] = res[f];
    }
}

void test_benchmark_n16384_l11_kernel(
    COOMatrix& coo, 
    std::vector<float> &B_val, 
    std::vector<int> &B_index, 
    std::vector<int> &A_row_access,
    std::vector<int> &A_row_access_len,
    int max_input_access,
    int batch, int neuron, 
    GpuEnv &env) {

	float *A;
    float *B;
	float *C;
	int *index;
    int* index_len; 
    int* B_index_d;

	int mybatch = batch;

	int bias = 0;

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

	float* W  = (float*)malloc(sizeof(float) * B_val.size());
	for(int i = 0; i < B_val.size(); ++i) {
		W[i] = B_val[i];
	}

    int* W_idx  = (int*)malloc(sizeof(int) * B_index.size());
	for(int i = 0; i < B_index.size(); ++i) {
		W_idx[i] = B_index[i];
	}

	int* access = (int*)malloc(sizeof(int) * A_row_access.size());
	for(int i = 0; i < A_row_access.size(); ++i) {
		access[i] = A_row_access[i];
	}

    int* access_len = (int*)malloc(sizeof(int) * A_row_access_len.size());
	for(int i = 0; i < A_row_access_len.size(); ++i) {
		access_len[i] = A_row_access_len[i];
	}


    Safe_Call(cudaMalloc((void**)&A, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemcpy(A, input, sizeof(float) * neuron * mybatch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&B, sizeof(float) * B_val.size()));
    Safe_Call(cudaMemcpy(B, W, sizeof(float) * B_val.size(), cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&C, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemset(C, 0, sizeof(float) * neuron * mybatch));

	Safe_Call(cudaMalloc((void**)&index, sizeof(int) * A_row_access.size()));
	Safe_Call(cudaMemcpy(index, access, sizeof(int) * A_row_access.size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&B_index_d, sizeof(float) * B_index.size()));
    Safe_Call(cudaMemcpy(B_index_d, W_idx, sizeof(float) * B_index.size(), cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&index_len, sizeof(float) * A_row_access_len.size()));
    Safe_Call(cudaMemcpy(index_len, access_len, sizeof(float) * A_row_access_len.size(), cudaMemcpyHostToDevice));

	env.add_event("row-succ-20-uiuc-kernel");
    env.event_start_record("row-succ-20-uiuc-kernel");

	int blocksize = 128;
	dim3 block(blocksize);
    dim3 grid((mybatch + MINIBATCH - 1) / MINIBATCH,  neuron / blocksize);

	n16384_l11_kernel<<<grid, block, sizeof(float) * (max_input_access * MINIBATCH), env.get_stream("row-succ-20-uiuc-kernel")>>>(
		A, B, C, index, index_len, B_index_d, batch, neuron, bias
	);

    env.event_stop_record("row-succ-20-uiuc-kernel");

    float time = env.get_event_time("row-succ-20-uiuc-kernel"); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

	std::cout << "Kernel Exec Time [20-uiuc-row-succ-transpose] = " << time <<  "ms" <<std::endl;
	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, true, true);
}
};