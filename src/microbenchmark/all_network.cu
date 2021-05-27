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

__global__ void n16384l1_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
	int* __restrict__ index, 
    int* categories,
    int* active,
	int batch, 
    int neuron, 
	float bias) {

	extern __shared__ float shared[];
	int start_idx = index[blockIdx.y];
	int col_gropu = threadIdx.x / 16;
	int last_load = ((neuron / 16) % 7) * 16 + 16;
	int load_num = (blockIdx.y + 1) == gridDim.y ? last_load : 128;
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
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * 128 + idx] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * 128 + idx] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * 128 + idx] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * 128 + idx] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * 128 + idx] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * 128 + idx] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * 128 + idx] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * 128 + idx] * val;
        }
    }
    __syncthreads();
    for(int f = 0; f < MINIBATCH ; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
    }
};


__global__ void n16384_output128_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
	int* __restrict__ index, 
    int* categories,
    int* active,
	int batch, 
    int neuron, 
	float bias) {

	extern __shared__ float shared[];
	int start_idx = index[blockIdx.y];
	int col_gropu = threadIdx.x / 16;
	int load_num = 160;
	for(int n = threadIdx.x; n < load_num * MINIBATCH; n += blockDim.x){
		int f = n / load_num;
		int k = n % load_num;
        shared[f * 160 + k] = A[categories[(blockIdx.x * MINIBATCH + f)] * neuron + (start_idx + k) % neuron];
	}

	__syncthreads();
	float res[MINIBATCH] = {0.0};
	for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * 128 * 32) + r * 128 + threadIdx.x];
		int idx = col_gropu * 16 + (r >= 16? r + 16 : r);
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
    __syncthreads();
	for(int f = 0; f < MINIBATCH ; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 128 + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
	}
};

void test_benchmark_graph_challenge(
    std::vector<std::vector<float>> &input,
    std::vector<std::vector<float>> &weight, 
    std::vector<std::vector<int>> &row_access, 
    int batch, 
    int neuron, 
    float bias,
    GpuEnv &env
) {

	float *A;
    float *A_d;
	float *C;
    float *C_d;

    float **B;
    float **B_d;
	int **index;
    int **index_d;

    int *category;
    int *active;
    int *category_d;
    int *active_d;
    
    int this_round_batch = batch;
    int layer = weight.size();

    A = (float*)malloc(sizeof(float) * neuron * batch);
    C = (float*)malloc(sizeof(float) * neuron * batch);
    memset(C, 0, sizeof(float) * neuron * batch);

    for(int l = 0; l < input.size(); ++l) {
        for(int i = 0; i < input[l].size(); ++i) {
            A[l * neuron + i] = input[l][i];
        }
    }

    B = (float**) malloc(sizeof(float*) * weight.size());
    B_d = (float**) malloc(sizeof(float*) * weight.size());
    for(int l = 0; l < weight.size(); ++l) {
        B[l] = (float*) malloc(sizeof(float*) * weight[l].size());
        for(int i = 0; i < weight[l].size(); ++i) {
            B[l][i] = weight[l][i];
        }
    }

    index = (int**) malloc(sizeof(int*) * row_access.size());
    index_d = (int**) malloc(sizeof(int*) * row_access.size());
    for(int l = 0; l < row_access.size(); ++l) {
        index[l] = (int*) malloc(sizeof(int*) * row_access[l].size());
        for(int i = 0; i < row_access[l].size(); ++i) {
            index[l][i] = row_access[l][i];
        }
    }

    category = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i) {
        category[i] = i;
    }

    active = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i){
        active[i] = 1;
    }

    Safe_Call(cudaMalloc((void**)&A_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A_d, A, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&C_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C_d, 0, sizeof(float) * neuron * batch));

    Safe_Call(cudaMalloc((void**)&active_d, sizeof(int) * batch));
    Safe_Call(cudaMalloc((void**)&category_d, sizeof(int) * batch));

    for(int l = 0; l < layer; ++l) {
        Safe_Call(cudaMalloc((void**)&(B_d[l]), sizeof(float) * weight[l].size()));
        Safe_Call(cudaMemcpy(B_d[l], B[l], sizeof(float) * weight[l].size(), cudaMemcpyHostToDevice));

        Safe_Call(cudaMalloc((void**)&(index_d[l]), sizeof(float) * row_access[l].size()));
        Safe_Call(cudaMemcpy(index_d[l], index[l], sizeof(float) * row_access[l].size(), cudaMemcpyHostToDevice));
    }

    float all_time = 0;
    env.add_event("row-succ-20-uiuc-kernel");
    
    for(int l = 0; l < layer; ++l) {
        
        double need_trans_data = long(this_round_batch * neuron) / (1024.0);
        need_trans_data = need_trans_data / 1024.0 * 8;
        need_trans_data = need_trans_data / 1024.0;

        double bandwidth = 700;
        double min_time = need_trans_data / bandwidth * 1000;

        int blocksize = 128;
        auto stream = env.get_stream("row-succ-20-uiuc-kernel");
        Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * this_round_batch, stream));
        env.event_start_record("row-succ-20-uiuc-kernel");
        if(l == 0) {
            dim3 block(blocksize);
            dim3 grid((this_round_batch + MINIBATCH - 1)/ MINIBATCH, (neuron + 112 - 1) / 112);
            n16384l1_kernel<<<grid, block, sizeof(float) * (MINIBATCH * (128 + 16)), stream>>>(
                A_d, B_d[l], C_d, index_d[l], category_d, active_d, batch, neuron, bias
            );
        }
        else {
            dim3 block(blocksize);
            dim3 grid((this_round_batch + MINIBATCH - 1)/ MINIBATCH, (neuron + 128 - 1) / 128);
        	n16384_output128_kernel<<<grid, block, sizeof(float) * (MINIBATCH * (160)), stream>>>(
                A_d, B_d[l], C_d, index_d[l], category_d, active_d, batch, neuron, bias
            );
        }

        Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * this_round_batch, cudaMemcpyDeviceToHost, stream));
        env.event_stop_record("row-succ-20-uiuc-kernel");

        Safe_Call(cudaStreamSynchronize(stream));

        int feature = 0;
        for(int k = 0; k < this_round_batch; ++k) {
            if(active[k]) {
                // category[feature] = category[k];
                category[feature] = k;
                feature++;
            }
        }
        float* tmp = A_d;
        A_d = C_d;
        C_d = tmp;

        this_round_batch = feature;
        std::cout << "layer " << l  << ", batch = "<< feature << std::endl;

        Safe_Call(cudaMemcpyAsync(category_d, category, sizeof(int) * feature, cudaMemcpyHostToDevice, stream));
        float time = env.get_event_time("row-succ-20-uiuc-kernel"); 

        std::cout << "Layer "<< l << " exec Time = " << time << "," << min_time <<  "ms, Utilization = " << (min_time / time) << std::endl;
        all_time += time;
    }

	Safe_Call(cudaMemcpy(C, C_d, sizeof(float) * neuron * batch, cudaMemcpyDeviceToHost));
	std::cout << "Kernel Exec Time [20-uiuc-row-succ] = " << all_time <<  "ms" <<std::endl;
	// CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, true, true);
}
};