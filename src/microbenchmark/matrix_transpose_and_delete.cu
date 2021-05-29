#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

#define TILE_DIM 64
#define BLOCK_ROWS 16

__global__ void matrix_re_transpose_and_delete(
    float * __restrict__ odata, 
    float * __restrict__ idata,
    int * __restrict__ old_to_new_map,
    int neuron, int batch) {

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && x < batch; j += BLOCK_ROWS) {
        tile[(threadIdx.y + j)][threadIdx.x] = idata[(y + j) * batch + x];
    }

    __syncthreads();


    x = blockIdx.y * TILE_DIM + threadIdx.x;  // old row
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // old batch
    

    for (int j = 0; j < TILE_DIM && (y+j) < batch; j += BLOCK_ROWS) {
        if(old_to_new_map[y + j] == -1) continue;
        int tmp = old_to_new_map[y + j]; // new batch
        odata[tmp * neuron + x] = tile[threadIdx.x][threadIdx.y + j];
    }
};

void test_benchmark_matrix_transpose_and_delete(int batch, int neuron, GpuEnv &env) {

    float *A;
    float *C;
    int* old_to_new_map_d;
    
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


    int * old_to_new_map = (int*)malloc(sizeof(int) * batch);
    for(int i = 0; i < 2; ++i) {
        old_to_new_map[i] = -1;
    }
    for(int i = 2; i < batch; ++i) {
        old_to_new_map[i] = i - 2;
    }

    int new_batch = batch - 2;
    
    Safe_Call(cudaMalloc((void**)&A, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A, input, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&old_to_new_map_d, sizeof(int) * batch));
    Safe_Call(cudaMemcpy(old_to_new_map_d, old_to_new_map, sizeof(int) * batch, cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&C, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C, 0, sizeof(float) * neuron * batch));

    std::string event = "transpose_and_delete";
	env.add_event(event);
    env.event_start_record(event);

    
	dim3 grid((batch + TILE_DIM - 1) / TILE_DIM, (neuron +  TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);

	matrix_re_transpose_and_delete<<<grid, block, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
        env.get_stream(event)>>>(
            C, A, old_to_new_map_d,  neuron, batch
	);

    env.event_stop_record(event);

    float time = env.get_event_time(event); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * new_batch, cudaMemcpyDeviceToHost));


    std::cout << output[1 * neuron + 0] << ", ";
    std::cout << std::endl;
	std::cout << "Kernel Exec Time [transpose] = " << time <<  "ms" <<std::endl;
	CpuTransposeDelete::run_and_cmp(input, old_to_new_map, batch, neuron, new_batch, output);
}
};