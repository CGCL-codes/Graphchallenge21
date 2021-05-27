#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

__device__ inline float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};

#define MINIBATCH 32

__global__ void rectangels_batch_parallel_kernel(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int* __restrict__ index16x16, int neuron, int batch, float bias) {

	extern __shared__ float shared[];

	for(int n = threadIdx.x; n < 128 * 32; n += blockDim.x){
		shared[n] = B[(blockIdx.y * 128 * 32) + n];
	}
	__syncthreads();

	int start_idx = index16x16[blockIdx.y];
	for(int f = 0; f < 256; ++f) {
		for(int i = threadIdx.x; i < 128; i += blockDim.x) {
			shared[i + 128 * 32] = 1.0;
			// A[(blockIdx.x * 256 + f) * neuron + (start_idx + i) % neuron];
		}
		__syncthreads();
		
		float res = 0;
		
		int idx_beg =  (threadIdx.x / 16) * 16;
		
		for(int r = 0; r < 32; ++r) {
			res += shared[r * 128 + threadIdx.x] * shared[128 * 32 + idx_beg + r];
			// if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 1 && f == 0) {
			// 	printf("%f * %f\n", shared[r * 128 + threadIdx.x], shared[128 * 32 + idx_beg + r]);
			// }
		}
		// if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 1 && f == 0) {
		// 	printf("RES = %f\n", res);
		// }
		C[(blockIdx.x * 256 + f) * neuron + blockIdx.y * 128 + threadIdx.x] = res;
		__syncthreads();
	}
}

void test_benchmark_rectangels_batch_parallel_kernel(COOMatrix& coo, std::vector<float> &val, std::vector<int> &row_access, int batch, int neuron, GpuEnv &env) {

	float *A;
    float *B;
	float *C;
	int *index;

	int mybatch = batch;

	int bias = 0;

	float * input = (float*)malloc(sizeof(float) * neuron * mybatch);
	memset(input, 1.0, sizeof(float) * neuron * mybatch);

	float * output = (float*)malloc(sizeof(float) * neuron * mybatch);
	memset(output, 0, sizeof(float) * neuron * mybatch);

	// srand (static_cast <unsigned> (time(0)));
	// for(int i = 0; i < mybatch; ++i) {
	// 	for(int j = 0; j < neuron; ++j) {
    //         float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/32.0));
	// 		input[i * neuron + j] = r2;
	// 	}
	// }


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
    dim3 grid(mybatch / (256), neuron / blocksize);

	rectangels_batch_parallel_kernel<<<grid, block, sizeof(float) * (32 * 128 + 128 + 16), env.get_stream("row-succ-20-uiuc-kernel")>>>(
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