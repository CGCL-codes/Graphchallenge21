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

std::vector<float> val_expand(std::vector<float> &old, int stride, int number) {
	int need_add = (old.size() / stride) * number;
	std::vector<float> res;
	int now_idx = 0;
	for(int i = 0; i < old.size() + need_add; ++i) {
		if(i % (stride + number) >= stride) {
			res.push_back(0.0625);
		}
		else {
			res.push_back(old[now_idx]);
			now_idx++;
		}
	}
	return res;
}


std::vector<float> val_expand(std::vector<float> &old, int number) {
	std::vector<float> res;
	for(int i = 0; i < old.size() + number; ++i) {
		res.push_back(0.0625);
	}
	return res;
}

__global__ void rectangels_16x32_no_transpose_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
	int* __restrict__ index16x16, 
	int neuron, 
	int batch, 
	float bias) {

	extern __shared__ float shared[];

	int start_idx = index16x16[blockIdx.y];

	int col_gropu = threadIdx.x / 16;
	int last_load = ((neuron / 16) % 7) * 16 + 16;

	int load_num = (blockIdx.y + 1) == gridDim.y ? last_load : 128;

    // for(int n = threadIdx.x; n < MINIBATCH * load_num; n += blockDim.x){
	// 	// shared[(n / load_num) * 128 + n % load_num] = A[(blockIdx.x * MINIBATCH + n / load_num) * neuron + (start_idx + n % load_num) % neuron];
	// 	shared[n] = A[(blockIdx.x * MINIBATCH + n / load_num) * neuron + n % load_num];
	// }

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

			// if(blockIdx.x == 0 && blockIdx.y == (gridDim.y - 1) && threadIdx.x == 0 && f == 0) {
			// 	printf("%d %f * %f\n", blockIdx.y, shared[(f * UNROLL + 0) * 128 + col_gropu * 16 + r], val);
			// }
			
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * 128 + col_gropu * 16 + r] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * 128 + col_gropu * 16 + r] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * 128 + col_gropu * 16 + r] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * 128 + col_gropu * 16 + r] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * 128 + col_gropu * 16 + r] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * 128 + col_gropu * 16 + r] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * 128 + col_gropu * 16 + r] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * 128 + col_gropu * 16 + r] * val;
			
			// res[8 + f * UNROLL] += shared[(f * UNROLL + 8) * 128 + col_gropu * 16 + r] * val;
            // res[9 + f * UNROLL] += shared[(f * UNROLL + 9) * 128 + col_gropu * 16 + r] * val;
            // res[10 + f * UNROLL] += shared[(f * UNROLL + 10) * 128 + col_gropu * 16 + r] * val;
			// res[11 + f * UNROLL] += shared[(f * UNROLL + 11) * 128 + col_gropu * 16 + r] * val;
        }
    }

	// if(blockIdx.x == 0 && blockIdx.y == (gridDim.y - 1) && threadIdx.x == 0) {
	// 	printf("res = %f\n", res[0]);
	// }


    __syncthreads();
	
	// if((blockIdx.y + 1) == gridDim.y) {
	// 	int last_thread = (neuron % 112);
	// 	for(int f = 0; f < MINIBATCH & threadIdx.x < last_thread; ++f) {
	// 		C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = res[f];
	// 	}
	// }
	// else {
		for(int f = 0; f < MINIBATCH ; ++f) {
			C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = res[f];
		}
	// }
}

void test_benchmark_graph_challenge(
    std::vector<COOMatrix&> coo_s, 

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
            A[l][i] = input[l][i];
        }
    }

    B = (float**) malloc(sizeof(float*) * weight.size());
    for(int l = 0; l < coo_s.size(); ++l) {
        B[l] = (float*) malloc(sizeof(float*) * weight[l].size());
        for(int i = 0; i < weight[l].size(); ++i) {
            B[l][i] = weight[l][i];
        }
    }

    index = (int**) malloc(sizeof(int*) * row_access.size());
    for(int l = 0; l < row_access.size(); ++l) {
        index[l] = (float*) malloc(sizeof(float*) * row_access[l].size());
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

    for(int l = 0; l < layer; ++i) {
        Safe_Call(cudaMalloc((void**)&(B_d[l]), sizeof(float) * weight[l].size()));
        Safe_Call(cudaMemcpy(B_d[l], B[l], sizeof(float) * weight[l].size(), cudaMemcpyHostToDevice));

        Safe_Call(cudaMalloc((void**)&(index_d[l]), sizeof(float) * index[l].size()));
        Safe_Call(cudaMemcpy(index_d[l], index[l], sizeof(float) * index[l].size(), cudaMemcpyHostToDevice));
    }

    env.add_event("row-succ-20-uiuc-kernel");
    
    for(int l = 0; l < layer; ++i) {

        Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * this_round_batch, env.get_stream("row-succ-20-uiuc-kernel")));

        env.event_start_record("row-succ-20-uiuc-kernel");
    
        int blocksize = 128;
        dim3 block(blocksize);
        dim3 grid((mybatch + MINIBATCH - 1)/ MINIBATCH, (neuron + 112 - 1) / 112);
    
        rectangels_16x32_no_transpose_kernel<<<grid, block, sizeof(float) * (MINIBATCH * (128 + 16)), env.get_stream("row-succ-20-uiuc-kernel")>>>(
            A, B, C, index, neuron, batch, bias
        );
    
        Safe_Call(cudaMemcpyAsync(active,active_d,sizeof(int)*mybatch,cudaMemcpyDeviceToHost,kernelstream));

        env.event_stop_record("row-succ-20-uiuc-kernel");
    
        float time = env.get_event_time("row-succ-20-uiuc-kernel"); 
    }

	Safe_Call(cudaMemcpy(output, C, sizeof(float) * neuron * mybatch, cudaMemcpyDeviceToHost));

	std::cout << "Kernel Exec Time [20-uiuc-row-succ] = " << time <<  "ms" <<std::endl;
	std::cout << "Kernel Exec Flops = " << (neuron * mybatch * 32 * 2.0) / (time / 1000.0) / 1000 / 1000 / 1000 /1000 << "TFLOPS" <<std::endl;

	CpuSpmm::run_and_cmp(coo, input, neuron, mybatch, output, false, true, true);
}
};