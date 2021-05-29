#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

__device__ inline float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};

#define MINIBATCH 8
#define UNROLL 8

std::vector<float> val_expand(std::vector<float> &old, int stride, int number) {
	int need_add = (old.size() / stride) * number;
	std::vector<float> res;
	int now_idx = 0;
	for(int i = 0; i < old.size() + need_add; ++i) {
		if(i % (stride + number) >= stride) {
			res.push_back(0.0625);
		}
		else {
			res.push_back(old[now_idx]);
			now_idx++;
		}
	}
	return res;
}


std::vector<float> val_expand(std::vector<float> &old, int number) {
	std::vector<float> res;
	for(int i = 0; i < old.size() + number; ++i) {
		res.push_back(0.0625);
	}
	return res;
}

__global__ void rectangels_16x32_no_transpose_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
	int* __restrict__ index16x16, 
	int neuron, 
	int batch, 
	float bias) {

	extern __shared__ float shared[];

	int start_idx = index16x16[blockIdx.y];

	int col_gropu = threadIdx.x / 16;
	int last_load = ((neuron / 16) % 7) * 16 + 16;

	int load_num = (blockIdx.y + 1) == gridDim.y ? last_load : 128;

    // for(int n = threadIdx.x; n < MINIBATCH * load_num; n += blockDim.x){
	// 	// shared[(n / load_num) * 128 + n % load_num] = A[(blockIdx.x * MINIBATCH + n / load_num) * neuron + (start_idx + n % load_num) % neuron];
	// 	shared[n] = A[(blockIdx.x * MINIBATCH + n / load_num) * neuron + n % load_num];
	// }

	for(int n = threadIdx.x; n < load_num; n += blockDim.x){
		for(int f = 0; f < MINIBATCH; ++f) {
			shared[f * 128 + n] = A[(blockIdx.x * MINIBATCH + f) * neuron + (start_idx + n) % neuron];
		}
	}


	__syncthreads();
	int last_thread = (neuron % 112);
	
	if(col_gropu == 7 || ((blockIdx.y + 1) == gridDim.y && threadIdx.x >= last_thread)) return;

    float res[MINIBATCH] = {0.0};

    for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * 128 * 32) + r * 128 + threadIdx.x];
        int idx = col_gropu * 16 + r;
        for(int f = 0; f < MINIBATCH / UNROLL; ++f) {

			// if(blockIdx.x == 0 && blockIdx.y == (gridDim.y - 1) && threadIdx.x == 0 && f == 0) {
			// 	printf("%d %f * %f\n", blockIdx.y, shared[(f * UNROLL + 0) * 128 + col_gropu * 16 + r], val);
			// }
			
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * 128 + col_gropu * 16 + r] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * 128 + col_gropu * 16 + r] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * 128 + col_gropu * 16 + r] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * 128 + col_gropu * 16 + r] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * 128 + col_gropu * 16 + r] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * 128 + col_gropu * 16 + r] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * 128 + col_gropu * 16 + r] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * 128 + col_gropu * 16 + r] * val;
			
			// res[8 + f * UNROLL] += shared[(f * UNROLL + 8) * 128 + col_gropu * 16 + r] * val;
            // res[9 + f * UNROLL] += shared[(f * UNROLL + 9) * 128 + col_gropu * 16 + r] * val;
            // res[10 + f * UNROLL] += shared[(f * UNROLL + 10) * 128 + col_gropu * 16 + r] * val;
			// res[11 + f * UNROLL] += shared[(f * UNROLL + 11) * 128 + col_gropu * 16 + r] * val;
        }
    }

	// if(blockIdx.x == 0 && blockIdx.y == (gridDim.y - 1) && threadIdx.x == 0) {
	// 	printf("res = %f\n", res[0]);
	// }


    __syncthreads();
	
	// if((blockIdx.y + 1) == gridDim.y) {
	// 	int last_thread = (neuron % 112);
	// 	for(int f = 0; f < MINIBATCH & threadIdx.x < last_thread; ++f) {
	// 		C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = res[f];
	// 	}
	// }
	// else {
		for(int f = 0; f < MINIBATCH ; ++f) {
			C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = res[f];
		}
	// }
}

void test_benchmark_rectangels_batch_parallel_kernel(COOMatrix& coo, std::vector<float> &val, std::vector<int> &row_access, int batch, int neuron, GpuEnv &env) {

	float *A;
    float *B;
	float *C;
	int *index;

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

	val= val_expand(val, 146 * 16 * 32 + (128 - 32) * 32);

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

	env.add_event("row-succ-20-uiuc-kernel");
    env.event_start_record("row-succ-20-uiuc-kernel");

	int blocksize = 128;
	dim3 block(blocksize);
    dim3 grid((mybatch + MINIBATCH - 1)/ MINIBATCH, (neuron + 112 - 1) / 112);

	rectangels_16x32_no_transpose_kernel<<<grid, block, sizeof(float) * (MINIBATCH * (128 + 16)), env.get_stream("row-succ-20-uiuc-kernel")>>>(
		A, B, C, index, neuron, batch, bias
	);

    env.event_stop_record("row-succ-20-uiuc-kernel");

    float time = env.get_event_time("row-succ-20-uiuc-kernel"); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

	std::cout << "Kernel Exec Time [20-uiuc-row-succ] = " << time <<  "ms" <<std::endl;
	std::cout << "Kernel Exec Flops = " << (neuron * mybatch * 32 * 2.0) / (time / 1000.0) / 1000 / 1000 / 1000 /1000 << "TFLOPS" <<std::endl;

	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, true, true);
}
};