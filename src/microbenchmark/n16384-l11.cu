#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

__device__ inline float __ReLU(float x) {
   return x<0.0?0.0:x>32.0?32.0:x;
};

#define OUT_CHANNEL 16
// batch parallel
__global__ void n16384_l11_kernel(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    int* __restrict__ index, 
    int batch, 
    int neuron, 
    float bias) {
    
    extern __shared__ float shared[];


    for(int n = threadIdx.x; n < OUT_CHANNEL * 32; n += blockDim.x){
        shared[n] = B[(blockIdx.y * OUT_CHANNEL * 32) + n];
    }
    __syncthreads();

    if((blockIdx.x * blockDim.x + threadIdx.x) >= batch) return;

    int begin_idx = blockIdx.y * OUT_CHANNEL / 16 * 32;
    for(int o_r = 0; o_r < OUT_CHANNEL / 16; ++o_r) {
        float reduce[16] = {0.0};
        int idx = begin_idx + o_r * 32;
        for(int r = 0; r < 32; ++r) {
            int row_idx = index[idx + r];
            float val = A[row_idx * batch + blockIdx.x * blockDim.x + threadIdx.x];
            // float val = 1.0;
            for(int c = 0; c < 16; c += 8) {
                // if(o_r == 0 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && c == 0) {
                //     printf("%f * %f\n", shared[o_r * 32 * 16 + r * 16 + c], val);
                // }
                reduce[c + 0] += val * shared[o_r * 32 * 16 + r * 16 + c + 0];
                reduce[c + 1] += val * shared[o_r * 32 * 16 + r * 16 + c + 1];
                reduce[c + 2] += val * shared[o_r * 32 * 16 + r * 16 + c + 2];
                reduce[c + 3] += val * shared[o_r * 32 * 16 + r * 16 + c + 3];
                
                reduce[c + 4] += val * shared[o_r * 32 * 16 + r * 16 + c + 4];
                reduce[c + 5] += val * shared[o_r * 32 * 16 + r * 16 + c + 5];
                reduce[c + 6] += val * shared[o_r * 32 * 16 + r * 16 + c + 6];
                reduce[c + 7] += val * shared[o_r * 32 * 16 + r * 16 + c + 7];
                
            }
        }
        for(int c = 0; c < 16; ++c) {
            C[(blockIdx.y * OUT_CHANNEL  + o_r * 16 + c) * batch + blockIdx.x * blockDim.x + threadIdx.x] = reduce[c];
        }
    }
}

void test_benchmark_n16384_l11_kernel(
    COOMatrix& coo, 
    std::vector<float> &B_val, 
    std::vector<int> &B_index, 
    int batch, int neuron, 
    GpuEnv &env) {

	float *A;
    float *B;
	float *C;
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

    Safe_Call(cudaMalloc((void**)&A, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemcpy(A, input, sizeof(float) * neuron * mybatch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&B, sizeof(float) * B_val.size()));
    Safe_Call(cudaMemcpy(B, W, sizeof(float) * B_val.size(), cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&C, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemset(C, 0, sizeof(float) * neuron * mybatch));

    Safe_Call(cudaMalloc((void**)&B_index_d, sizeof(float) * B_index.size()));
    Safe_Call(cudaMemcpy(B_index_d, W_idx, sizeof(float) * B_index.size(), cudaMemcpyHostToDevice));

	env.add_event("row-succ-20-uiuc-kernel");
    env.event_start_record("row-succ-20-uiuc-kernel");

	int blocksize = 256;
	dim3 block(blocksize);
    dim3 grid((mybatch + blocksize - 1) / blocksize,  neuron / OUT_CHANNEL);

	n16384_l11_kernel<<<grid, block, sizeof(float) * (OUT_CHANNEL * 32), env.get_stream("row-succ-20-uiuc-kernel")>>>(
		A, B, C, B_index_d, batch, neuron, bias
	);

    env.event_stop_record("row-succ-20-uiuc-kernel");

    float time = env.get_event_time("row-succ-20-uiuc-kernel"); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

	std::cout << "Kernel Exec Time [20-uiuc-row-succ-transpose] = " << time <<  "ms" <<std::endl;
	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, false, false);
}
};