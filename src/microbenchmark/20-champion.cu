#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ftxj {
__device__ float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
};

#define WARPSIZE 32
#define MINIBATCH 12

__global__ void __launch_bounds__(1024,1) dummy_kernel(
  float *nextfeat, float *currfeat, 
  int buffsize, int *buffdispl, int *mapdispl, unsigned short *map, 
  int *displ, unsigned short *index, float *value, 
  float bias, int neuron
){
	extern __shared__ float shared[];
	int wind = threadIdx.x % WARPSIZE;
	float reduce[MINIBATCH] = {0.0};
	for(int buff = buffdispl[blockIdx.x]; buff < buffdispl[blockIdx.x+1]; buff++){
		int mapnz = mapdispl[buff+1]-mapdispl[buff];
		for(int n = threadIdx.x; n < mapnz; n += blockDim.x){
			int ind = map[mapdispl[buff]+n];
			for(unsigned int f = 0; f < MINIBATCH; f++) {
                // if((blockIdx.y * MINIBATCH+f) * (unsigned int) neuron+ind >= neuron * 32) {
                //     printf("bugs %d\n", ind);
                // }
				shared[f*buffsize+n] = currfeat[(blockIdx.y * MINIBATCH+f) * (unsigned int) neuron+ind];
			}
    	}
		__syncthreads();
		int warp = (buff*blockDim.x+threadIdx.x)/WARPSIZE;
		for(int m = displ[warp]; m < displ[warp+1]; m++){
			int ind = index[m*WARPSIZE+wind];
			float val = value[m*WARPSIZE+wind];
			for(int f = 0; f < MINIBATCH; f++) {
				reduce[f] += shared[f*buffsize+ind] * val;
			}
		}
		__syncthreads();
	}
	int m = blockIdx.x*blockDim.x+threadIdx.x;

	for(int f = 0; f < MINIBATCH; f++)
		nextfeat[(blockIdx.y * MINIBATCH + f) * neuron + m] = (reduce[f]+bias);
    
};

void test_benchmark_20_uiuc(COOMatrix &coo, UIUCMatrix &matrix, int batch, GpuEnv &env) {
    int buffsize = matrix.buffsize;
    int neuron = matrix.neuron;
    int mybatch = batch;

    float *nextfeat;
    float *currfeat;
    int *buffdispl;
    int *mapdispl;
    unsigned short *map; 
    int *displ;
    unsigned short *index;
    float *value; 
    float bias = 0;

    // std::vector<std::vector<float>> input(mybatch, std::vector<float>(neuron, 0.0));
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
  
    Safe_Call(cudaMalloc((void**)&buffdispl, sizeof(int) * matrix.buffdispl.size()));
    Safe_Call(cudaMemcpy(buffdispl, &matrix.buffdispl[0], sizeof(int) * matrix.buffdispl.size(), cudaMemcpyHostToDevice));
    
    Safe_Call(cudaMalloc((void**)&mapdispl, sizeof(int) * matrix.mapdispl.size()));
    Safe_Call(cudaMemcpy(mapdispl, &matrix.mapdispl[0], sizeof(int) * matrix.mapdispl.size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&map, sizeof(unsigned short) * matrix.map.size()));
    Safe_Call(cudaMemcpy(map, &matrix.map[0], sizeof(unsigned short) * matrix.map.size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&displ, sizeof(int) * matrix.warpdispl.size()));
    Safe_Call(cudaMemcpy(displ, &matrix.warpdispl[0], sizeof(int) * matrix.warpdispl.size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&index, sizeof(unsigned short) * matrix.warpindex.size()));
    Safe_Call(cudaMemcpy(index, &matrix.warpindex[0], sizeof(unsigned short) * matrix.warpindex.size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&value, sizeof(float) * matrix.warpvalue.size()));
    Safe_Call(cudaMemcpy(value, &matrix.warpvalue[0], sizeof(float) * matrix.warpvalue.size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&currfeat, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemcpy(currfeat, input, sizeof(float) * neuron * mybatch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&nextfeat, sizeof(float) * neuron * mybatch));
    Safe_Call(cudaMemset(nextfeat, 0, sizeof(float) * neuron * mybatch));

    std::cout << "begin inference..." << std::endl; 
    env.add_event("uiuc_kernel_timer");
    env.event_start_record("uiuc_kernel_timer");

    dim3 block(matrix.blocksize);
    dim3 grid(neuron / matrix.blocksize, (mybatch+MINIBATCH-1)/MINIBATCH);
    dummy_kernel<<<grid,block, sizeof(float) * matrix.buffsize * MINIBATCH, env.get_stream("uiuc_kernel_timer")>>>(
        nextfeat, currfeat, buffsize, buffdispl, mapdispl, map, displ, index, value,
        bias, neuron
    );

    env.event_stop_record("uiuc_kernel_timer");
    float time = env.get_event_time("uiuc_kernel_timer"); 

    Safe_Call(cudaMemcpy(output, nextfeat, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

    std::cout << "Kernel Exec Time [20-uiuc] = " << time << "ms"<< std::endl;
    std::cout << "Flops [20-uiuc] = " << float(2 * batch * neuron * 32) /  time * 1000 /1e12 << "TFLOPS"<< std::endl;
    
	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output);
}
};