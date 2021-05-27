#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

__device__ inline float __ReLU(float x) {
   return x<0.0?0.0:x>32.0?32.0:x;
};

__global__ void batch_parallel_16384x32succ_kernel(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int* __restrict__ index, int neuron, int batch, float bias) {
	extern __shared__ float shared[];
	float reduce[32] = {0.0};

	for(int n = threadIdx.x; n < 32 * 32; n += blockDim.x){
		shared[n] = B[(blockIdx.y * 32 * 32) + n];
	}
	__syncthreads();
	if((blockIdx.x * blockDim.x + threadIdx.x) >= batch) return;
	
	for(int r = 0; r < 32; ++r) {
		int row_idx = index[blockIdx.y * 32 + r];
		float val = A[row_idx * batch + blockIdx.x * blockDim.x + threadIdx.x];
		for(int c = 0; c < 32; ++c){
			reduce[c] += shared[r * 32 + c] * val;
		}
	}
	__syncthreads();
	for(int c = 0; c < 16; ++c) {
		C[(blockIdx.y * 16  + c) * batch + blockIdx.x * blockDim.x + threadIdx.x] = reduce[c];
	}
	for(int c = 16; c < 32; ++c) {
		C[(neuron / 2 + blockIdx.y * 16  + c - 16) * batch + blockIdx.x * blockDim.x + threadIdx.x] = reduce[c];
	}
}

void test_benchmark_row_succ_input_transpose_batch_parallel(COOMatrix& coo, std::vector<float> &val, std::vector<int> &row_access, int batch, int neuron, GpuEnv &env) {

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

	int blocksize = 256;
	dim3 block(blocksize);
    dim3 grid((mybatch + blocksize - 1) / blocksize,  neuron / 32);

	batch_parallel_16384x32succ_kernel<<<grid, block, sizeof(float) * (32 * 32), env.get_stream("row-succ-20-uiuc-kernel")>>>(
		A, B, C, index, neuron, batch, bias
	);

    env.event_stop_record("row-succ-20-uiuc-kernel");

    float time = env.get_event_time("row-succ-20-uiuc-kernel"); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

	std::cout << "Kernel Exec Time [20-uiuc-row-succ-transpose] = " << time <<  "ms" <<std::endl;
	std::cout << "Kernel Exec Flops = " << (neuron * mybatch * 32 * 2.0) / (time / 1000.0) / 1000 / 1000 / 1000 /1000 << "TFLOPS" <<std::endl;

	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, false, false);

	
}
};