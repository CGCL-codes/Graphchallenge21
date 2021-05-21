#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>

namespace ftxj {

__device__ float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};


#define WARPSIZE 32

#define GROUPSIZE 32

#define MINIBATCH 12

#define ROW_SUCC_LEN 32
#define NNZ_PRE_COL 32
#define BATCH_BLOCK 32
#define BATCH_SIZE 1600

#define UNROLL 8

__global__ void shared_memory_mm(float* A,  float* B, float* C, int* index, float bias){
	
	__shared__ float A_tile[BATCH_BLOCK][NNZ_PRE_COL];
	// __shared__ float B_tile[ROW_SUCC_LEN][NNZ_PRE_COL];
	//load A
	int group_idx = threadIdx.x / GROUPSIZE;
	int batch_start = blockIdx.y * BATCH_BLOCK;
	int row_succ_start = blockIdx.x;
	
	for(int i = threadIdx.x; i < BATCH_BLOCK * NNZ_PRE_COL; i += blockDim.x) {
		A_tile[i % BATCH_BLOCK][i / BATCH_BLOCK] = A[index[row_succ_start * ROW_SUCC_LEN + i / BATCH_BLOCK] * BATCH_SIZE + batch_start + i % BATCH_BLOCK];
	}
	//load B
	// for(int i = threadIdx.x; i < ROW_SUCC_LEN * NNZ_PRE_COL; i += blockDim.x) {
	// 	B_tile[i / NNZ_PRE_COL][i % NNZ_PRE_COL] = B[row_succ_start * ROW_SUCC_LEN * NNZ_PRE_COL + i];
	// }
	__syncthreads();


	// if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
	// 	for(int i = 0; i < ROW_SUCC_LEN; ++i) {
	// 		for(int j = 0; j < NNZ_PRE_COL; ++j) {
	// 			printf("%f\n", B_tile[i][j]);
	// 		}
	// 	}
	// }

	register float BB[32] = {
		0.0625, 0.0625, 0.0625, 0.0625, 
		0.0625, 0.0625, 0.0625, 0.0625, 
		0.0625, 0.0625, 0.0625, 0.0625, 
		0.0625, 0.0625, 0.0625, 0.0625,
		0.0625, 0.0625, 0.0625, 0.0625,
		0.0625, 0.0625, 0.0625, 0.0625, 
		0.0625, 0.0625, 0.0625, 0.0625, 
		0.0625, 0.0625, 0.0625, 0.0625 
	};

	
	
	int B_col = threadIdx.x % ROW_SUCC_LEN;
	int A_batch = (threadIdx.x /  ROW_SUCC_LEN) * BATCH_BLOCK / 4;

	for(int r = 0; r < BATCH_BLOCK / 4; ++r) {
		register float res = bias;
		for(int i = 0; i < NNZ_PRE_COL; i += UNROLL) {
			res += A_tile[A_batch + r][i + 0] * BB[i + 0]; // bank conflict
			res += A_tile[A_batch + r][i + 1] * BB[i + 1]; // bank conflict
			res += A_tile[A_batch + r][i + 2] * BB[i + 2]; // bank conflict
			res += A_tile[A_batch + r][i + 3] * BB[i + 2]; // bank conflict
			res += A_tile[A_batch + r][i + 4] * BB[i + 2]; // bank conflict
			res += A_tile[A_batch + r][i + 5] * BB[i + 3]; // bank conflict
			res += A_tile[A_batch + r][i + 6] * BB[i + 4]; // bank conflict
			res += A_tile[A_batch + r][i + 7] * BB[i + 5]; // bank conflict
		}
		int res_col_idx = B_col >= 16 ? (row_succ_start * 16 + 512 + B_col - 16) : (row_succ_start * 16 + B_col);
		// if(res_col_idx == 528 && A_batch == 0) {
		// 	printf("(%d, %d), (%d), %f\n", blockIdx.x, blockIdx.y, threadIdx.x, res);
		// }
		C[res_col_idx * BATCH_SIZE + blockIdx.y * BATCH_BLOCK + A_batch + r] = __ReLU(res);
	}
};



void test_shared_memory_mm(COOMatrix& coo, std::vector<float> &val, std::vector<int> &row_access, GpuEnv &env) {

	float *A;
    float *B;
	float *C;
	int *index;

	int mybatch = BATCH_SIZE;
	int neuron = 1024;

	int bias = 0;

	float * input = (float*)malloc(sizeof(float) * neuron * mybatch);
	memset(input, 0, sizeof(float) * neuron * mybatch);

	float * output = (float*)malloc(sizeof(float) * neuron * mybatch);
	memset(output, 0, sizeof(float) * neuron * mybatch);

	for(int i = 0; i < mybatch; ++i) {
		for(int j = 0; j < neuron; ++j) {
			input[i * neuron + j] = 1.0;
		}
	}

	float* W  = (float*)malloc(sizeof(float) * val.size());
	for(int i = 0; i < val.size(); ++i) {
		W[i] = val[i];
	}

	int* access = (int*)malloc(sizeof(int) * row_access.size());
	for(int i = 0; i < row_access.size(); ++i) {
		access[i] = row_access[i];
	}


    Safe_Call(cudaMalloc((void**)&A, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemcpy(A, input, sizeof(float) * neuron * mybatch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&B, sizeof(float) * val.size()));
    Safe_Call(cudaMemcpy(B, W, sizeof(float) * val.size(), cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&C, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemset(C, 0, sizeof(float) * neuron * mybatch));

	Safe_Call(cudaMalloc((void**)&index, sizeof(int) * row_access.size()));
	Safe_Call(cudaMemcpy(index, access, sizeof(int) * row_access.size(), cudaMemcpyHostToDevice));

	env.add_event("naive_mm");
    env.event_start_record("naive_mm");

    dim3 block(4 * ROW_SUCC_LEN);
    dim3 grid(neuron / 32, mybatch / BATCH_BLOCK);

    shared_memory_mm<<<grid, block, sizeof(float) * (BATCH_BLOCK + ROW_SUCC_LEN) * NNZ_PRE_COL, env.get_stream("kernel_timer")>>>(
		A, B, C, index, bias
	);

    env.event_stop_record("naive_mm");

    float time = env.get_event_time("naive_mm"); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output);

    std::cout << "shared mm timer = " << time << std::endl;
	std::cout << "shared mm Flops = " << (neuron * BATCH_SIZE * 32 * 2.0) / (time / 1000.0) / 1000 / 1000 / 1000 /1000 << std::endl;
	
}
};