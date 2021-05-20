#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
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
#define BATCH_SIZE 1800

#define UNROLL 8

__global__ void shared_memory_mm(float* A,  float* B, float* C, int* index, float bias){
	__shared__ float A_tile[BATCH_BLOCK][NNZ_PRE_COL];
	__shared__ float B_tile[ROW_SUCC_LEN][NNZ_PRE_COL];
	//load A
	int group_idx = threadIdx.x / GROUPSIZE;
	int batch_start = blockIdx.y * BATCH_BLOCK;
	int row_succ_start = blockIdx.x;
	
	for(int i = threadIdx.x; i < BATCH_BLOCK * NNZ_PRE_COL; i += blockDim.x) {
		A_tile[i % BATCH_BLOCK][i / BATCH_BLOCK] = A[index[row_succ_start + i / BATCH_BLOCK] * BATCH_SIZE + batch_start + i % BATCH_BLOCK];
	}
	//load B
	for(int i = threadIdx.x; i < ROW_SUCC_LEN * NNZ_PRE_COL; i += blockDim.x) {
		B_tile[i / NNZ_PRE_COL][i % NNZ_PRE_COL] = B[row_succ_start * ROW_SUCC_LEN * NNZ_PRE_COL + i];
	}
	__syncthreads();

	float res = bias;
	
	int A_batch = threadIdx.x % BATCH_BLOCK;
	int B_col = threadIdx.x /  BATCH_BLOCK;

	for(int i = 0; i < NNZ_PRE_COL / UNROLL; i += UNROLL) {
		res += A_tile[A_batch][i] * B_tile[B_col][i]; // bank conflict
		res += A_tile[A_batch][i + 1] * B_tile[B_col][i + 1]; // bank conflict
		res += A_tile[A_batch][i + 2] * B_tile[B_col][i + 2]; // bank conflict
		res += A_tile[A_batch][i + 3] * B_tile[B_col][i + 3]; // bank conflict
		res += A_tile[A_batch][i + 4] * B_tile[B_col][i + 4]; // bank conflict
		res += A_tile[A_batch][i + 5] * B_tile[B_col][i + 5]; // bank conflict
		res += A_tile[A_batch][i + 6] * B_tile[B_col][i + 6]; // bank conflict
		res += A_tile[A_batch][i + 7] * B_tile[B_col][i + 7]; // bank conflict
	}

	int res_col_idx = B_col > 16 ? row_succ_start * 16 : row_succ_start * 16 + 512;
	
	C[res_col_idx * BATCH_SIZE + A_batch] = __ReLU(res);

};



void test_shared_memory_mm(UIUCMatrix &matrix, GpuEnv &env) {
    float *A;
    float *B;
	float *C;
	int *index;

	int mybatch = BATCH_SIZE;
	int neuron = 4096;

	int bias = -0.3;

    std::vector<std::vector<float>> input(mybatch, std::vector<float>(neuron, 1.0));
    std::vector<std::vector<float>> W(neuron, std::vector<float>(32, 0.625));
	std::vector<int> idx(neuron, 0);

	for(int i = 0; i < neuron; ++i) {
		idx[i] = i;
	}

    Safe_Call(cudaMalloc((void**)&A, sizeof(float) * input.size() * input[0].size()));
    Safe_Call(cudaMemcpy(A, &input[0][0], sizeof(float) * input.size() * input[0].size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&B, sizeof(float) * W.size() * W[0].size()));
    Safe_Call(cudaMemcpy(A, &input[0][0], sizeof(float) * W.size() * W[0].size(), cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&C, sizeof(float) * input.size() * input[0].size()));
    Safe_Call(cudaMemset(C, 0, sizeof(float) * input.size() * input[0].size()));

	Safe_Call(cudaMalloc((void**)&index, sizeof(float) * idx.size()));
	Safe_Call(cudaMemcpy(index, &idx[0], sizeof(float) * idx.size(), cudaMemcpyHostToDevice));

	env.add_event("kernel_timer");
    env.event_start_record("kernel_timer");

    dim3 block(matrix.blocksize);
    dim3 grid(neuron / 32, mybatch / BATCH_BLOCK);

    shared_memory_mm<<<grid, block, sizeof(float) * (BATCH_BLOCK + ROW_SUCC_LEN) * NNZ_PRE_COL, env.get_stream("kernel_timer")>>>(
		A, B, C, index, bias
	);

    env.event_stop_record("kernel_timer");
    float time = env.get_event_time("kernel_timer"); 
    std::cout << "shared mm timer = " << time << std::endl;
	std::cout << "shared mm Flops = " << (neuron * BATCH_SIZE * 32 * 2.0) / (time / 1000.0) / 1000 / 1000 / 1000 /1000 << std::endl;
	
}
};