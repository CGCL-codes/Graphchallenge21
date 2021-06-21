#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

__device__ inline float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};

#define MINIBATCH 4
#define UNROLL 4

__global__ void n16384_l1_l11_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
    int* __restrict__ categories,
    int* __restrict__ active,
	int stride,
	int batch, 
    int neuron, 
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
		shared[f * shared_size + k] = A[categories[(blockIdx.x * MINIBATCH + f)] * neuron + (a_k) % neuron];
	}

	__syncthreads();
    int gap = stride >= blockDim.x ? blockDim.x : stride;
	float res[MINIBATCH] = {0.0};
	for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * blockDim.x * 32) + r * blockDim.x + threadIdx.x];
        int idx = col_gropu * 16 + (r >= 16? r + gap - 16 : r);
        for(int f = 0; f < MINIBATCH / UNROLL; ++f) {
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * shared_size + idx] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * shared_size + idx] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * shared_size + idx] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * shared_size + idx] * val;
            // res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * shared_size + idx] * val;
            // res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * shared_size + idx] * val;
            // res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * shared_size + idx] * val;
            // res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * shared_size + idx] * val;
        }
    }
    for(int f = 0; f < MINIBATCH ; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * blockDim.x + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
	}
};

void test_benchmark_fuse_cmp_layer1024_0_1(
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
    
    float *A_T;

	float *C;
    float *C_d;

    float **B;
    float **B_d;
	int **index;
    int **index_d;

    int *category;
    int *active;
    int *old_to_new_map;
    int *category_d;
    int *active_d;
    int *old_to_new_map_d;
    
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

    old_to_new_map = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i) {
        old_to_new_map[i] = i;
    }

    active = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i){
        active[i] = 0;
    }

    Safe_Call(cudaMalloc((void**)&A_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A_d, A, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));


    Safe_Call(cudaMalloc((void**)&A_T, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(A_T, 0, sizeof(float) * neuron * batch));

    Safe_Call(cudaMalloc((void**)&C_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C_d, 0, sizeof(float) * neuron * batch));

    Safe_Call(cudaMalloc((void**)&active_d, sizeof(int) * batch));
    Safe_Call(cudaMalloc((void**)&category_d, sizeof(int) * batch));
    Safe_Call(cudaMalloc((void**)&old_to_new_map_d, sizeof(int) * batch));

    for(int l = 0; l < layer; ++l) {
        Safe_Call(cudaMalloc((void**)&(B_d[l]), sizeof(float) * weight[l].size()));
        Safe_Call(cudaMemcpy(B_d[l], B[l], sizeof(float) * weight[l].size(), cudaMemcpyHostToDevice));

        Safe_Call(cudaMalloc((void**)&(index_d[l]), sizeof(float) * row_access[l].size()));
        Safe_Call(cudaMemcpy(index_d[l], index[l], sizeof(float) * row_access[l].size(), cudaMemcpyHostToDevice));
    }

    float all_time = 0;
    float all_time_min = 0;
    env.add_event("row-succ-20-uiuc-kernel");
    

    std::map<int, int> neuron_map = {
        {1024, 6},
        {4096, 8},
        {16384, 10}
    };
    std::map<int, int> stride_map = {
        {1, 16},
        {2, 32},
        {3, 64},
        {4, 128},
        {5, 256},
        {6, 512},
        {7, 1024},
        {8, 2048},
        {9, 4096},
        {10, 8192}
    };

    bool now_transpose = false;
    int last_feature = batch;
    int transpose_batch = 0;

    for(int l = 0; l < layer; ++l) {
        
        double need_trans_data = long(this_round_batch * neuron) / (1024.0);
        need_trans_data = need_trans_data / 1024.0 * 8;
        need_trans_data = need_trans_data / 1024.0;

        double bandwidth = 700;
        double min_time = need_trans_data / bandwidth * 1000;

        auto stream = env.get_stream("row-succ-20-uiuc-kernel");

        Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * batch, stream));

        env.event_start_record("row-succ-20-uiuc-kernel");

        if(l <= neuron_map[neuron] - 1){
            int blocksize = 128;
            dim3 block(blocksize);
            dim3 grid((this_round_batch + MINIBATCH - 1)/ MINIBATCH, (neuron + blocksize - 1) / blocksize);
            int stride = stride_map[l + 1];
            std::cout << stride << std::endl;
            int load_num = stride > blocksize ? 32 * (blocksize / 16) : stride + 16 * (blocksize / 16);
            int shared_size = ((load_num + 31) / 32) * 32;
        	n16384_l1_l11_kernel<<<grid, block, sizeof(float) * (MINIBATCH * shared_size), stream>>>(
                A_d, B_d[l], C_d, category_d, active_d, stride, this_round_batch, neuron, bias
            );
            cudaError_t err = cudaGetLastError();        
            if (err != cudaSuccess) {
                printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
        }
        Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * this_round_batch, cudaMemcpyDeviceToHost, stream));
        env.event_stop_record("row-succ-20-uiuc-kernel");
        Safe_Call(cudaStreamSynchronize(stream));

        int feature = 0;
        if(l <= neuron_map[neuron] - 1) { 
            for(int k = 0; k < this_round_batch; ++k) {
                // category[feature] = category[k];
                category[feature] = k;
                feature++;
            }
            float* tmp = A_d;
            A_d = C_d;
            C_d = tmp;
        }   
        for(int i = 0; i < batch; ++i){
            active[i] = 0;
        }

        last_feature = this_round_batch;
        this_round_batch = feature;

        std::cout << "layer " << l  << ", batch = "<< feature << std::endl;

        Safe_Call(cudaMemcpyAsync(category_d, category, sizeof(int) * feature, cudaMemcpyHostToDevice, stream));

        float time = env.get_event_time("row-succ-20-uiuc-kernel"); 
        std::cout << "Layer "<< l << " exec Time = " << time << ", " << min_time <<  "ms, Utilization = " << (min_time / time) << std::endl;
        all_time += time;
        all_time_min += min_time;
    }

	Safe_Call(cudaMemcpy(C, C_d, sizeof(float) * neuron * batch, cudaMemcpyDeviceToHost));
	std::cout << "Kernel Exec Time [20-uiuc-row-succ] = " << all_time <<  "ms" <<std::endl;
    std::cout << "Kernel Exec Upper Time = " << all_time_min <<  "ms" <<std::endl;
    
	// CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, true, true);
}
};