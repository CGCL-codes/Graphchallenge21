#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>

namespace ftxj {

__device__ inline float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};


#define WARPSIZE 32

#define GROUPSIZE 32

#define MINIBATCH 32

#define ROW_SUCC_LEN 32
#define NNZ_PRE_COL 32
#define BATCH_BLOCK 32
#define BATCH_SIZE 1792
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


__global__ void batch_parallel(float* __restrict__ A,  float* __restrict__ B, float* __restrict__ C, int* __restrict__ index, float bias){

	register float res[8] = {0.0};
	
	// register float BB[32] = { // different thread run on same weight
	// 	0.0625, 0.0625, 0.0625, 0.0625, 
	// 	0.0625, 0.0625, 0.0625, 0.0625, 
	// 	0.0625, 0.0625, 0.0625, 0.0625, 
	// 	0.0625, 0.0625, 0.0625, 0.0625,
	// 	0.0625, 0.0625, 0.0625, 0.0625,
	// 	0.0625, 0.0625, 0.0625, 0.0625, 
	// 	0.0625, 0.0625, 0.0625, 0.0625, 
	// 	0.0625, 0.0625, 0.0625, 0.0625 
	// };
	

	int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col_idx =  blockIdx.y * 8;
	int index_idx = (col_idx / 32) * 32;
	
	
	for(int i = 0; i < 32; ++i) {
		register float a_tmp = A[index[index_idx + i] * 1024 + batch_idx];
		for(int j = 0; j < 8; ++j) {
			res[j] += a_tmp * 0.0625;
		}
	}
	if(col_idx ==516 && batch_idx ==0 ) {
		printf("%f\n", res[0]);
	}
	C[(col_idx + 0) * BATCH_SIZE + batch_idx] = __ReLU(res[0]);
	C[(col_idx + 1) * BATCH_SIZE + batch_idx] = __ReLU(res[1]);
	C[(col_idx + 2) * BATCH_SIZE + batch_idx] = __ReLU(res[2]);
	C[(col_idx + 3) * BATCH_SIZE + batch_idx] = __ReLU(res[3]);
	
	C[(col_idx + 4) * BATCH_SIZE + batch_idx] = __ReLU(res[4]);
	C[(col_idx + 5) * BATCH_SIZE + batch_idx] = __ReLU(res[5]);
	C[(col_idx + 6) * BATCH_SIZE + batch_idx] = __ReLU(res[6]);
	C[(col_idx + 7) * BATCH_SIZE + batch_idx] = __ReLU(res[7]);
};


#define BLOCK_LOAD_A_LINE 32
#define BLOCK_LOAD_B_LINE 32
#define BLOCK_REDUCE_LINE 32
#define THREAD_LOAD_A_LINE 4
#define THREAD_LOAD_B_LINE 4
#define THREAD_A_BLOCKS (BLOCK_LOAD_A_LINE / THREAD_LOAD_A_LINE)
#define THREAD_B_BLOCKS (BLOCK_LOAD_B_LINE / THREAD_LOAD_B_LINE)
#define BATCH BATCH_SIZE


__global__ void outer_product_based(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int* __restrict__ index, float bias) {

	__shared__ float A_shared_tile[BLOCK_REDUCE_LINE][BLOCK_LOAD_A_LINE];
	__shared__ float B_shared_tile[BLOCK_LOAD_B_LINE][BLOCK_REDUCE_LINE];

	float C_reg_tile[THREAD_LOAD_A_LINE][THREAD_LOAD_B_LINE] = {0.0};

	float A_reg_tile[THREAD_LOAD_A_LINE];
	
	float B_reg_tile[THREAD_LOAD_B_LINE];

	const int A_tile_idx = blockIdx.x;
	const int B_tile_idx = blockIdx.y;
	

	for(int reduce_axis = 0; reduce_axis < 32; reduce_axis += BLOCK_REDUCE_LINE) {
		// Load A, no bank conflict
		for(int i = threadIdx.x; i < BLOCK_LOAD_A_LINE * BLOCK_REDUCE_LINE; i += blockDim.x) {
			A_shared_tile[i / BLOCK_LOAD_A_LINE][i % BLOCK_LOAD_A_LINE] = A[index[B_tile_idx * 32 + i / BLOCK_LOAD_A_LINE] * BATCH + A_tile_idx * BLOCK_LOAD_A_LINE + i % BLOCK_LOAD_A_LINE];
		}
		// Load B, no bank conflict
		for(int i = threadIdx.x; i < BLOCK_LOAD_B_LINE * BLOCK_REDUCE_LINE; i += blockDim.x) {
			B_shared_tile[i / BLOCK_REDUCE_LINE][i % BLOCK_REDUCE_LINE] = B[B_tile_idx * 32 + i];
		}
		__syncthreads();
		//Compute C
		for(int r = 0; r < BLOCK_REDUCE_LINE; ++r) {
			//Load A to reg
			for(int i = 0; i < THREAD_LOAD_A_LINE; ++i) {
				A_reg_tile[i] = A_shared_tile[r][(threadIdx.x / THREAD_A_BLOCKS) * THREAD_LOAD_A_LINE + i];
			}
			//Load B to reg
			for(int i = 0; i < THREAD_LOAD_B_LINE; ++i) {
				B_reg_tile[i] = B_shared_tile[(threadIdx.x % THREAD_A_BLOCKS) * THREAD_LOAD_B_LINE + i][r];
			}

			for(int a_idx = 0; a_idx < THREAD_LOAD_A_LINE; ++a_idx) {
				for(int b_idx = 0; b_idx < THREAD_LOAD_B_LINE; ++b_idx) {
					C_reg_tile[a_idx][b_idx] += A_reg_tile[a_idx] * B_reg_tile[b_idx];
				}
			}
		}
		__syncthreads();
	}
	
	const int B_write_begin = B_tile_idx * BLOCK_LOAD_B_LINE + (threadIdx.x % THREAD_A_BLOCKS) * THREAD_LOAD_B_LINE;
	const int A_write_begin = A_tile_idx * BLOCK_LOAD_A_LINE + (threadIdx.x / THREAD_A_BLOCKS) * THREAD_LOAD_A_LINE;
	// write back C
	for (int b_idx = 0; b_idx < THREAD_LOAD_B_LINE; ++b_idx) {
		for (int a_idx = 0; a_idx < THREAD_LOAD_A_LINE; ++a_idx) {
			C[(B_write_begin + b_idx) * BATCH + A_write_begin + a_idx] = C_reg_tile[a_idx][b_idx];
		}
    }
};

__global__ void uiuc_transfer(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int* __restrict__ index, float bias) {

	extern __shared__ float shared[];
	float reduce[MINIBATCH] = {0.0};

	for(int n = threadIdx.x; n < 256; n += blockDim.x){
		int idx = index[blockIdx.y * 256 + n];
		for(unsigned int f = 0; f < MINIBATCH; f++) {
			shared[f * 256 + n] = A[(blockIdx.x * MINIBATCH + f) * 1024 + idx];
		}
	}
	__syncthreads();
	for(int r = 0; r < 32; ++r){
		float val = B[blockIdx.y * 256 * 32 + r * 256 + threadIdx.x];
		for(int f = 0; f < MINIBATCH; f++) {
			reduce[f] += shared[f * 256 + threadIdx.x] * val;
		}
	}
	for(int f = 0; f < MINIBATCH; f++) {
		C[(blockIdx.x * MINIBATCH + f) * 1024 + blockIdx.y * 256 + threadIdx.x] = reduce[f];
	}
}

__global__ void uiuc_transfer_opt(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int* __restrict__ index, float bias) {

	extern __shared__ float shared[];
	float reduce[MINIBATCH] = {0.0};

	int groupIdx = threadIdx.x / MINIBATCH;
	int lane = threadIdx.x % MINIBATCH;

	// int idx = index[blockIdx.y * 256 * 32 + threadIdx.x / MINIBATCH];

	// int idx[32];
	// for(int i = 0; i < 32; ++i) {
	// 	idx[i] = index[blockIdx.y * 256 * 32 + threadIdx.x / MINIBATCH + i];
	// }

	// for(int n = threadIdx.x; n < 256 * MINIBATCH; n += blockDim.x){
	// 	shared[n] = A[idx[n / 256] * BATCH_SIZE + blockIdx.x * MINIBATCH + lane];
	// }

	// for(int n = threadIdx.x; n < 256 * MINIBATCH; n += blockDim.x){
	// 	shared[n] = A[(blockIdx.y * 256 + n / 32) * BATCH_SIZE + blockIdx.x * 32 + lane];
	// }

	__syncthreads();
	for(int r = 0; r < 32; ++r){
		float val = B[blockIdx.y * 256 * 32 + r * 256 + threadIdx.x];
		for(int f = 0; f < MINIBATCH; f++) {
			reduce[f] += shared[f * 256 + threadIdx.x] * val;
		}
	}

	__syncthreads();

	// for(int f = 0; f < MINIBATCH; f++) {
	// 	C[(blockIdx.x * MINIBATCH + f) * 1024 + blockIdx.y * 256 + threadIdx.x] = reduce[f];
	// }

	// __shfl(0xffffffff, );
	// for(int f = 0; f < MINIBATCH; f++) {
	// 	C[(blockIdx.y * 256 + threadIdx.x) * 1024 + blockIdx.x * MINIBATCH + f] = reduce[f];
	// }
	for(int f = 0; f < MINIBATCH; ++f){
		shared[threadIdx.x * MINIBATCH + f] = reduce[f];
	}
	
	__syncthreads();

	for(int n = threadIdx.x; n < 256 * MINIBATCH; n += blockDim.x){
		C[(blockIdx.y * 256 + n / MINIBATCH) * BATCH + blockIdx.x * MINIBATCH + n % MINIBATCH] = shared[(threadIdx.x / MINIBATCH) * MINIBATCH + (n % MINIBATCH)]; 
	}
}


__global__ void uiuc_transfer_oo(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int* __restrict__ index, float bias) {

	extern __shared__ float shared[];

	float reduce[32] = {0.0};


	int groupIdx = threadIdx.x / 32;
	int groupNum = blockDim.x / 32;
	int lane = threadIdx.x % 32;

	for(int n = threadIdx.x; n < 256 * MINIBATCH; n += blockDim.x){
		int idx = index[blockIdx.y * 256 + n / 32];
		shared[n] = A[idx * BATCH_SIZE + blockIdx.x * MINIBATCH + (n / 32) * 32 + lane];
	}
	
	__syncthreads();

	for(int r = 0; r < 32; ++r){
		float val = B[blockIdx.y * 256 * 32 + r * 256 + threadIdx.x];
		for(int f = 0; f < MINIBATCH; f++) {
			reduce[f] += shared[f * 256 + threadIdx.x] * val;
		}
	}
	
	__syncthreads();

	for(int f = 0; f < MINIBATCH; ++f){
		shared[threadIdx.x * MINIBATCH + f] = reduce[f];
	}
		
	__syncthreads();
	
	for(int n = threadIdx.x; n < 256 * MINIBATCH; n += blockDim.x){
		C[(blockIdx.y * 256 + n / MINIBATCH) * BATCH + blockIdx.x * MINIBATCH + n % MINIBATCH] = shared[(threadIdx.x / MINIBATCH) * MINIBATCH + (n % MINIBATCH)]; 
	}
}



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

    // dim3 block(4 * ROW_SUCC_LEN);
    // dim3 grid(neuron / 32, mybatch / BATCH_BLOCK);

    // shared_memory_mm<<<grid, block, sizeof(float) * (BATCH_BLOCK + ROW_SUCC_LEN) * NNZ_PRE_COL, env.get_stream("kernel_timer")>>>(
	// 	A, B, C, index, bias
	// );

	dim3 block(256);
    dim3 grid(mybatch / (MINIBATCH), neuron / 256);

    // uiuc_transfer_opt<<<grid, block, sizeof(float) * (MINIBATCH * 256), env.get_stream("naive_mm")>>>(
	// 	A, B, C, index, bias
	// );

	uiuc_transfer_oo<<<grid, block, sizeof(float) * (MINIBATCH * 256), env.get_stream("naive_mm")>>>(
		A, B, C, index, bias
	);

    env.event_stop_record("naive_mm");

    float time = env.get_event_time("naive_mm"); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

	std::cout << "shared mm timer = " << time << std::endl;
	std::cout << "shared mm Flops = " << (neuron * BATCH_SIZE * 32 * 2.0) / (time / 1000.0) / 1000 / 1000 / 1000 /1000 << std::endl;

	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output);

	
}
};