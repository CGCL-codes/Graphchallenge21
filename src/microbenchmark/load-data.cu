#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
namespace ftxj {

#define BLOCK_LOAD (32 * 8)

__global__ void __launch_bounds__(1024,1) dummy_kernel(
	float *nextfeat, float *currfeat
){
	  int i = blockIdx.x * BLOCK_LOAD; 
	  for(int j = threadIdx.x; j < BLOCK_LOAD; j += blockDim.x) {
		  nextfeat[i + j] = currfeat[i + j];
	}
};

__global__ void device_copy_vector4_kernel(int* d_in, int* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
      reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
    }
  
    // in only one thread, process final elements (if there are any)
    int remainder = N%4;
    if (idx==N/4 && remainder!=0) {
      while(remainder) {
        int idx = N - remainder--;
        d_out[idx] = d_in[idx];
      }
    }
}
  
void load_data_benchmark(GpuEnv &env) {
    float *nextfeat;
    float *currfeat;

    int mybatch = 1800;
	  int neuron = 4096;

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
    std::cout << "bandwidth = " << 2 * (mybatch * (float)neuron * sizeof(float)) / (time / 1000) / 1024.0 / 1024.0 / 1024.0 << "GB/s" << std::endl;
    std::cout << "data load and write timer = " << time << std::endl;
}
};