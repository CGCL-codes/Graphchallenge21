#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
namespace ftxj {

#define BLOCK_LOAD (256 * 10)
#define VECTOR_BLOCK_LOAD (32 * 2)

__global__ void naive_copy(float *nextfeat, float *currfeat){
    extern __shared__ float shared[];
    int i = blockIdx.x * BLOCK_LOAD; 
	for(int j = threadIdx.x; j < BLOCK_LOAD; j += blockDim.x) {
		shared[j] = currfeat[i + j];
	}
    __syncthreads();
    for(int j = threadIdx.x; j < BLOCK_LOAD; j += blockDim.x) {
        nextfeat[i + j] = shared[j] + 1;
	}
};


__global__ void uiuc_copy(float *nextfeat, float *currfeat) {
    extern __shared__ float shared[];

	int i = blockIdx.x * 12 * 16384 + blockIdx.y * 256;

	for(int j = threadIdx.x; j < 12 * 256; j += blockDim.x) {
        shared[j] =  currfeat[i + j];
	}
	__syncthreads();
    for(int j = threadIdx.x; j < 12 * 256; j += blockDim.x) {
        nextfeat[i + j] = shared[j] + 1;
	}
};

__global__ void vector4_copy(float* nextfeat, float* currfeat) {
    int idx = blockIdx.x * VECTOR_BLOCK_LOAD;
	float4* pin = reinterpret_cast<float4*>(currfeat);
	float4* pout = reinterpret_cast<float4*>(nextfeat);
	for(int i = threadIdx.x; i < VECTOR_BLOCK_LOAD; i += blockDim.x) {
		pout[idx + i] = pin[idx + i];
    }
};


void vector4_load_data_benchmark(GpuEnv &env) {
    float *nextfeat;
    float *currfeat;

    int mybatch = 60000;
	int neuron = 1024;

    std::vector<std::vector<float>> input(mybatch, std::vector<float>(neuron, 1.0));

    Safe_Call(cudaMalloc((void**)&currfeat, sizeof(float) * mybatch * neuron));
    Safe_Call(cudaMemcpy(currfeat, &input[0][0], sizeof(float) * mybatch * neuron, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&nextfeat, sizeof(float) * mybatch * neuron));
    Safe_Call(cudaMemset(nextfeat, 0, sizeof(float) * mybatch * neuron));


	env.add_event("float4 copy");
    env.event_start_record("float4 copy");

    dim3 block(64);
    dim3 grid((mybatch * neuron) / BLOCK_LOAD);

    vector4_copy<<<grid,block, 0, env.get_stream("float4 copy")>>>(
        nextfeat, currfeat
    );

    env.event_stop_record("float4 copy");
	float time2 = env.get_event_time("float4 copy");
	
    std::cout << "float4 bandwidth = " << 2 * (mybatch * (float)neuron * sizeof(float)) / (time2 / 1000) / 1024.0 / 1024.0 / 1024.0 << "GB/s" << std::endl;
    
	std::cout << "data load and write timer = " << time2 << std::endl;
}

void test_benchmark_succ_load_store(int mybatch, int neuron, GpuEnv &env) {
    float *nextfeat;
    float *currfeat;
    std::vector<std::vector<float>> input(mybatch, std::vector<float>(neuron, 1.0));

    Safe_Call(cudaMalloc((void**)&currfeat, sizeof(float) * mybatch * neuron));
    Safe_Call(cudaMemcpy(currfeat, &input[0][0], sizeof(float) * mybatch * neuron, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&nextfeat, sizeof(float) * mybatch * neuron));
    Safe_Call(cudaMemset(nextfeat, 0, sizeof(float) * mybatch * neuron));

    env.add_event("naive copy");
    env.event_start_record("naive copy");

    dim3 block(256);
    dim3 grid((mybatch * neuron) /BLOCK_LOAD);

    naive_copy<<<grid,block, BLOCK_LOAD *sizeof(float), env.get_stream("naive copy")>>>(
        nextfeat, currfeat
    );

    env.event_stop_record("naive copy");

    float time1 = env.get_event_time("naive copy"); 
	
	std::cout << "Load&Store Time [Succ] = " << time1 << "ms" << std::endl;
    std::cout << "Load&Store Bandwidth [Succ] = " << 2 * (mybatch * (float)neuron * sizeof(float)) / (time1 / 1000) / 1024.0 / 1024.0 / 1024.0 << "GB/s" << std::endl;
    
}
};