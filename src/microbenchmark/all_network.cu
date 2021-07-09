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
    for(int f = 0; f < MINIBATCH; ++f) {
        // &&  blockIdx.x * MINIBATCH + f < batch; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
    }
};

__global__ void n16384_l2_l11_kernel(
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
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * shared_size + idx] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * shared_size + idx] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * shared_size + idx] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * shared_size + idx] * val;
        }
    }
    for(int f = 0; f < MINIBATCH ; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * blockDim.x + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
	}
};

#define OUT_CHANNEL 16
__global__ void n16384_l11_kernel(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    int* __restrict__ index, 
    int* __restrict__ active,
    int batch, 
    int neuron, 
    float bias
) {
    
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
            for(int c = 0; c < 16; ++c) {
                reduce[c] += val * shared[o_r * 32 * 16 + r * 16 + c];
            }
        }
        for(int c = 0; c < 16; ++c) {
            if(C[(blockIdx.y * OUT_CHANNEL  + o_r * 16 + c) * batch + blockIdx.x * blockDim.x + threadIdx.x]
                = __ReLU(reduce[c] + bias)) {
                active[blockIdx.x * blockDim.x + threadIdx.x] = 1;
            }
        }
    }
};

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void matrix_transpose(float * __restrict__ odata, float * __restrict__ idata, int neuron, int batch) {

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && (y + j) < batch && x < neuron; j += BLOCK_ROWS) {
        tile[(threadIdx.y + j)][threadIdx.x] = idata[(y + j) * neuron + x];
    }

    __syncthreads();


    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && x < batch && y + j < neuron; j += BLOCK_ROWS) {
        odata[(y+j) * batch + x] = tile[threadIdx.x][threadIdx.y + j];
    } 
};

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
        {16384, 10},
        {65536, 12}
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
        if(l == 9) {
            Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * batch, stream));
        }
        else if(l > 9) {
            Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * batch, stream));
        }
        else {
            Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * batch, stream));
        }

        env.event_start_record("row-succ-20-uiuc-kernel");

        if(l == 0) {
            int blocksize = 128;
            dim3 block(blocksize);
            dim3 grid((this_round_batch + MINIBATCH - 1)/ MINIBATCH, (neuron + 112 - 1) / 112);
            n16384l1_kernel<<<grid, block, sizeof(float) * (MINIBATCH * (128 + 16)), stream>>>(
                A_d, B_d[l], C_d, index_d[l], category_d, active_d, this_round_batch, neuron, bias
            );
            cudaError_t err = cudaGetLastError();        
            if (err != cudaSuccess) {
                printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
        }
        else if(l <= neuron_map[neuron] - 1){
            int blocksize = 128;
            dim3 block(blocksize);
            dim3 grid((this_round_batch + MINIBATCH - 1)/ MINIBATCH, (neuron + blocksize - 1) / blocksize);
            int stride = stride_map[l + 1];
            std::cout << stride << std::endl;
            int load_num = stride > blocksize ? 32 * (blocksize / 16) : stride + 16 * (blocksize / 16);
            int shared_size = ((load_num + 31) / 32) * 32;
        	n16384_l2_l11_kernel<<<grid, block, sizeof(float) * (MINIBATCH * shared_size), stream>>>(
                A_d, B_d[l], C_d, category_d, active_d, stride, this_round_batch, neuron, bias
            );
            cudaError_t err = cudaGetLastError();        
            if (err != cudaSuccess) {
                printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
        }
        else {
            if(!now_transpose) {
                transpose_batch = last_feature;
                now_transpose = true;
                dim3 grid((neuron + TILE_DIM - 1) / TILE_DIM, (transpose_batch +  TILE_DIM - 1) / TILE_DIM);
                dim3 block(TILE_DIM, BLOCK_ROWS);
                matrix_transpose<<<grid, block, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
                    stream>>>(
                        A_T, A_d, neuron, transpose_batch
                );
                cudaError_t err = cudaGetLastError();        
   	            if (err != cudaSuccess) {
		            printf("what CUDA Error: %s\n", cudaGetErrorString(err));
      	            exit(-1);
   	            }
            }
            if(l == 22) {
                std::cout << "Begin Delete" << std::endl;
                dim3 grid((transpose_batch + TILE_DIM - 1) / TILE_DIM, (neuron +  TILE_DIM - 1) / TILE_DIM);
                dim3 block(TILE_DIM, BLOCK_ROWS);
                matrix_re_transpose_and_delete<<<grid, block, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
                    stream>>>(
                        A_d, A_T, old_to_new_map_d, neuron, transpose_batch
                );
                Safe_Call(cudaStreamSynchronize(stream));
                cudaError_t err = cudaGetLastError();        
   	            if (err != cudaSuccess) {
		            printf("what CUDA Error: %s\n", cudaGetErrorString(err));
      	            exit(-1);
   	            }

                dim3 grid2((neuron + TILE_DIM - 1) / TILE_DIM, (this_round_batch +  TILE_DIM - 1) / TILE_DIM);
                dim3 block2(TILE_DIM, BLOCK_ROWS);
                matrix_transpose<<<grid2, block2, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
                    stream>>>(
                        A_T, A_d, neuron, this_round_batch
                );
                Safe_Call(cudaStreamSynchronize(stream));
                err = cudaGetLastError();        
                if (err != cudaSuccess) {
                 printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                   exit(-1);
                }
                transpose_batch = this_round_batch;
            }
            int blocksize = 256;
            dim3 block(blocksize);
            dim3 grid((transpose_batch + blocksize - 1) / blocksize,  neuron / OUT_CHANNEL);
            n16384_l11_kernel<<<grid, block, sizeof(float) * (OUT_CHANNEL * 32), stream>>>(
                A_T, B_d[l], C_d, index_d[l], active_d, transpose_batch, neuron, bias
            );
            cudaError_t err = cudaGetLastError();        
            if (err != cudaSuccess) {
                printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }

        }

        if(l > neuron_map[neuron] - 1) {
            Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * transpose_batch, cudaMemcpyDeviceToHost, stream));
        }
        else {
            Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * this_round_batch, cudaMemcpyDeviceToHost, stream));
        }

        env.event_stop_record("row-succ-20-uiuc-kernel");

        Safe_Call(cudaStreamSynchronize(stream));

        int feature = 0;
        if(l <= neuron_map[neuron] - 1) { 
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
        }   
        else if(l == 21) {
            int neg_1 = 0;
            int have_v = 0;
            for(int k = 0; k < transpose_batch; ++k) {
                if(active[k]) {
                    old_to_new_map[k] = feature;
                    feature++;
                    have_v++;
                }
                else {
                    old_to_new_map[k] = -1;
                    neg_1++;
                }
            }
            std::cout << "begin cout : ";
            std::cout << neg_1 << ", " << have_v << std::endl;
            float* tmp = A_T;
            A_T = C_d;
            C_d = tmp;
        }   
        else {
            for(int k = 0; k < batch; ++k) {
                if(active[k]) {
                    // category[feature] = category[k];
                    category[feature] = k;
                    feature++;
                }
            }
            float* tmp = A_T;
            A_T = C_d;
            C_d = tmp;
        }

        for(int i = 0; i < batch; ++i){
            active[i] = 0;
        }

        last_feature = this_round_batch;
        this_round_batch = feature;

        std::cout << "layer " << l  << ", batch = "<< feature << std::endl;

        Safe_Call(cudaMemcpyAsync(category_d, category, sizeof(int) * feature, cudaMemcpyHostToDevice, stream));

        if(l == 21)
            Safe_Call(cudaMemcpyAsync(old_to_new_map_d, old_to_new_map, sizeof(int) * transpose_batch, cudaMemcpyHostToDevice, stream));

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