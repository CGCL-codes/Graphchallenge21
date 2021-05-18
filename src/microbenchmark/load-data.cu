#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
namespace ftxj {

#define BLOCK_LOAD (32 * 32)

__global__ void __launch_bounds__(1024,1) dummy_kernel(
	float *nextfeat, float *currfeat
){
	int i = blockIdx.x * BLOCK_LOAD; 
    float reg = 0;
	for(int j = threadIdx.x; j < BLOCK_LOAD; j += blockDim.x) {
		reg += currfeat[i + j];
	}
    nextfeat[blockIdx.x * blockDim.x + threadIdx.x] = reg;
};

void load_data_benchmark(GpuEnv &env) {
    float *nextfeat;
    float *currfeat;

    int mybatch = 60000;
	int neuron = 1024;

    std::vector<std::vector<float>> input(mybatch, std::vector<float>(neuron, 1.0));

    Safe_Call(cudaMalloc((void**)&currfeat, sizeof(float) * mybatch * neuron));
    Safe_Call(cudaMemcpy(currfeat, &input[0][0], sizeof(float) * mybatch * neuron, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&nextfeat, sizeof(float) * mybatch * neuron));
    Safe_Call(cudaMemset(nextfeat, 0, sizeof(float) * mybatch * neuron));

    env.add_event("kernel_timer");
    env.event_start_record("kernel_timer");

    dim3 block(64);
    dim3 grid((mybatch * neuron) / BLOCK_LOAD);

    dummy_kernel<<<grid,block, 0, env.get_stream("kernel_timer")>>>(
        nextfeat, currfeat
    );

    env.event_stop_record("kernel_timer");
    float time = env.get_event_time("kernel_timer"); 
    std::cout << "data load and write timer = " << time << std::endl;
}
};