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

__global__ void uiuc_transpose_kernel(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int* __restrict__ index, int neuron, int batch, float bias) {

	extern __shared__ float shared[];
	float reduce[MINIBATCH] = {0.0};

    int groupIdx = threadIdx.x / 32;
	int groupNum = blockDim.x / 32;
	int lane = threadIdx.x % 32;

	for(int n = threadIdx.x; n < 256 * MINIBATCH; n += blockDim.x){
		int idx = index[blockIdx.y * 256 + n / 32];
		shared[n] = A[idx * batch + blockIdx.x * MINIBATCH + lane];
	}
	__syncthreads();
    
	for(int r = 0; r < 32; ++r){
		float val = B[blockIdx.y * 256 * 32 + r * 256 + threadIdx.x];
		for(int f = 0; f < MINIBATCH; f++) {
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && f == 0) {
			// 	printf("%f * %f %d\n", shared[(threadIdx.x / 32 + r) * MINIBATCH + f], val, index[blockIdx.y * 256]);
			// }
			reduce[f] += shared[(threadIdx.x / 32 + r) * MINIBATCH + f] * val; // bank conflict!!
		}
	}
	
	__syncthreads();
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
	// 	printf("res = %f\n", reduce[0]);
	// }

	for(int f = 0; f < MINIBATCH; ++f){
		shared[threadIdx.x * MINIBATCH + f] = reduce[f];
	}
		
	__syncthreads();
	
	for(int n = threadIdx.x; n < 256 * MINIBATCH; n += blockDim.x){
		C[(blockIdx.y * 256 + n / MINIBATCH) * batch + blockIdx.x * MINIBATCH + n % MINIBATCH] = shared[(threadIdx.x / MINIBATCH) * MINIBATCH + (n % MINIBATCH)]; 
	}
}

void test_benchmark_row_succ_20_uiuc_transpose(COOMatrix& coo, std::vector<float> &val, std::vector<int> &row_access, int batch, int neuron, GpuEnv &env) {

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
    dim3 grid(mybatch / (MINIBATCH), neuron / blocksize);

	uiuc_transpose_kernel<<<grid, block, sizeof(float) * (MINIBATCH * blocksize), env.get_stream("row-succ-20-uiuc-kernel")>>>(
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