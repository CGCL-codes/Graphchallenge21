#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>
#include <cstdlib>
namespace ftxj {

#define TILE_DIM 64
#define BLOCK_ROWS 16

__global__ void matrix_transpose(float * __restrict__ odata, float * __restrict__ idata, int neuron, int batch) {

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && (y + j) < batch; j += BLOCK_ROWS) {
        tile[(threadIdx.y + j)][threadIdx.x] = idata[(y + j) * neuron + x];
    }

    __syncthreads();


    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && x < batch; j += BLOCK_ROWS) {
        odata[(y+j) * batch + x] = tile[threadIdx.x][threadIdx.y + j];
    }    
}

void test_benchmark_matrix_transpose(int batch, int neuron, GpuEnv &env) {

    float *A;
    float *C;
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

    Safe_Call(cudaMalloc((void**)&A, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A, input, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));

	Safe_Call(cudaMalloc((void**)&C, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C, 0, sizeof(float) * neuron * batch));

    std::string event = "transpose";
	env.add_event(event);
    env.event_start_record(event);

    
	dim3 grid((neuron + TILE_DIM - 1) / TILE_DIM, (batch +  TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);

	matrix_transpose<<<grid, block, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
        env.get_stream(event)>>>(
            C, A, neuron, batch
	);

    env.event_stop_record(event);

    float time = env.get_event_time(event); 

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * batch, cudaMemcpyDeviceToHost));

	std::cout << "Kernel Exec Time [transpose] = " << time <<  "ms" <<std::endl;
    long data = neuron * batch;
    data = data * 8;
    double gb = data / 1024.0 / 1024.0 / 1024.0;
    double db = gb / time * 1000; 
	std::cout << "Kernel Bandwidth [transpose] = " << db <<  "GB/s" <<std::endl;

	CpuTranspose::run_and_cmp(input, neuron, batch, output);
}
};