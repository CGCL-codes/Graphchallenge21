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

__global__ void n16384_l2_l11_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
	int stride,
	int neuron, 
	int batch, 
	float bias) {

	extern __shared__ float shared[];
	int start_idx1 = (blockDim.x / 16) * (blockIdx.y) * 16;
	int start_idx2 = (blockDim.x / 16) * (blockIdx.y) * 16 + stride;
	int load_num = stride > blockDim.x ? 32 * (blockDim.x / 16) : stride + 16 * (blockDim.x / 16);
	int shared_size = ((load_num + 31) / 32) * 32;
	int col_gropu = threadIdx.x / 16;
	

	for(int n = threadIdx.x; n < load_num * MINIBATCH; n += blockDim.x){
		int f = n / load_num;
		int k = n % load_num;
		int a_k = ((stride > blockDim.x) && (k >= blockDim.x)) ? (k - blockDim.x) + start_idx2 : k + start_idx1;
		// if(blockIdx.x == 0 && blockIdx.y == 0 && f == 0) {
		// 	printf("block 0 load %d\n", a_k);
		// }
		shared[f * shared_size + k] = A[(blockIdx.x * MINIBATCH + f) * neuron + (a_k) % neuron];
	}

	__syncthreads();

	int gap = stride >= blockDim.x ? blockDim.x : stride;
	
	float res[MINIBATCH] = {0.0};
	
	for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * blockDim.x * 32) + r * blockDim.x + threadIdx.x];
		int idx = col_gropu * 16 + (r >= 16? r + gap - 16 : r);
        for(int f = 0; f < MINIBATCH / UNROLL; ++f) {
			if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && f == 0) {
				printf("%d %f * %f\n", idx, shared[(f * UNROLL + 0) * shared_size + idx], val);
			}
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * shared_size + idx] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * shared_size + idx] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * shared_size + idx] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * shared_size + idx] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * shared_size + idx] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * shared_size + idx] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * shared_size + idx] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * shared_size + idx] * val;
        }
    }
	for(int f = 0; f < MINIBATCH ; ++f) {
		C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 128 + threadIdx.x] = res[f];
	}
}

void test_benchmark_n16384_l2_l10_kernel(COOMatrix& coo, std::vector<float> &val, int stride, int batch, int neuron, GpuEnv &env) {
	float *A;
    float *B;
	float *C;

	int bias = 0;

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

	float* W  = (float*)malloc(sizeof(float) * val.size());
    for(int i = 0; i < val.size(); ++i) {
		W[i] = val[i];
	}

    Safe_Call(cudaMalloc((void**)&A, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A, input, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&B, sizeof(float) * val.size()));
    Safe_Call(cudaMemcpy(B, W, sizeof(float) * val.size(), cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&C, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C, 0, sizeof(float) * neuron * batch));

	std::string event = "test_n16384_l2_l10";
	env.add_event(event);
    env.event_start_record(event);

	int blocksize = 128;
	int load_num = stride > blocksize ? 32 * (blocksize / 16) : stride + 16 * (blocksize / 16);
	int shared_size = ((load_num + 31) / 32) * 32;
	dim3 block(blocksize);
    dim3 grid((batch + MINIBATCH - 1)/ MINIBATCH, (neuron + 128 - 1) / 128);
	n16384_l2_l11_kernel<<<grid, block, sizeof(float) * (MINIBATCH * shared_size), env.get_stream(event)>>>(
		A, B, C, stride, neuron, batch, bias
	);
    env.event_stop_record(event);
    float time = env.get_event_time(event); 
	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * batch, cudaMemcpyDeviceToHost));
	std::cout << "Kernel Exec Time [n16384-l2-l10] = " << time <<  "ms" <<std::endl;
	std::cout << "Kernel Exec Flops = " << (neuron * batch * 32 * 2.0) / (time / 1000.0) / 1000 / 1000 / 1000 /1000 << "TFLOPS" <<std::endl;
	CpuSpmm::run_and_cmp(coo, input, neuron, batch, output, false, true, true);
}
};