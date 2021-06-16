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
#define FUSE_LAYER 2

__global__ void fuseed_layer_l0_l11(
	float * __restrict__ A, 
	float * __restrict__ B1,
    float * __restrict__ B2,
	float * __restrict__ C, 
    int* __restrict__ categories,
    int* __restrict__ active,
    int* __restrict__ stride,

    int* __restrict__ load_num,
    int* __restrict__ shared_size,
    int* __restrict__ thread_num,
    
	int batch, 
    int neuron, 
	float bias,
    int last_layer_thread
) {
	extern __shared__ float shared[];
    int col_gropu = threadIdx.x / 16;

    int start_idx1[FUSE_LAYER] = {0};
    int start_idx2[FUSE_LAYER] = {0};

    for(int i = FUSE_LAYER - 2; i >= 0; --i) {
        start_idx1[i] = start_idx1[i + 1];
        // (thread_num[i] / 16) * (blockIdx.y) * 16;
    } 

    for(int i = FUSE_LAYER - 2; i >= 0; --i) {
        start_idx2[i] = start_idx1[i] + stride[i];
        // (thread_num[i] / 16) * (blockIdx.y) * 16 + stride[i];
    }

    // if(blockIdx.x == 1 && threadIdx.x == 0 && blockIdx.y == 0) {
    //     for(int i = 0; i < FUSE_LAYER; ++i) {
    //         printf("stride = %d, thread_num = %d, load_num = %d, start_idx1 = %d, start_idx2 = %d, shared_size = %d\n", 
    //             stride[i], thread_num[i], load_num[i], start_idx1[i], start_idx2[i], shared_size[i]);
    //     }
    // }

    // load first layer input from global memory.
    for(int n = threadIdx.x; n < load_num[0] * MINIBATCH; n += blockDim.x){
        int f = n / load_num[0];
        int k = n % load_num[0];
        int a_k = ((stride[0] > blockDim.x) && (k >= blockDim.x)) ? (k - blockDim.x) + start_idx2[0] : k + start_idx1[0];
        shared[f * shared_size[0] + k] = A[categories[(blockIdx.x * MINIBATCH + f)] * neuron + (a_k) % neuron];
    }

    for(int layer = 0; layer < FUSE_LAYER; ++layer) {
        __syncthreads();
        int gap = stride[layer] >= thread_num[layer]  ? thread_num[layer] : stride[layer];
        float res[MINIBATCH] = {0.0};
        for(int r = 0; r < 32; ++r) { // in graph challenge, all column has 32 nnzs
            if(threadIdx.x >= thread_num[layer]) break;
            float val = layer == 0 ? 
                B1[(start_idx1[layer] * 32) + r * thread_num[layer] + threadIdx.x] :
                B2[(start_idx1[layer] * 32) + r * thread_num[layer] + threadIdx.x] ;
            int idx = col_gropu * 16 + (r >= 16? r + gap - 16 : r);
            for(int f = 0; f < MINIBATCH / UNROLL; ++f) {
                res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * shared_size[layer] + idx] * val;
                res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * shared_size[layer] + idx] * val;
                res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * shared_size[layer] + idx] * val;
                res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * shared_size[layer] + idx] * val;
                res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * shared_size[layer] + idx] * val;
                res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * shared_size[layer] + idx] * val;
                res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * shared_size[layer] + idx] * val;
                res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * shared_size[layer] + idx] * val;
            }

            // if(threadIdx.x == 0 && blockIdx.x == 1 && blockIdx.y == 0) {
            //     printf("%d %f * %f = %f\n", layer, val, shared[1 * shared_size[layer] + idx], res[0]);
            // }
        }
        __syncthreads();
        
        // if(threadIdx.x == 0 && blockIdx.x == 1 && blockIdx.y == 0) {
        //     printf("l = %d,  res = %f\n", layer, res[0]);
        // }
        if(layer != FUSE_LAYER - 1) { // not last layer, write back to shared memory
            if(threadIdx.x >= thread_num[layer]) break;
            for(int f = 0; f < MINIBATCH ; ++f) {
                shared[f * thread_num[layer] + threadIdx.x] = __ReLU(res[f] + bias);
            }
        }
        if(layer == FUSE_LAYER - 1) {
            if(threadIdx.x >= thread_num[layer] || blockIdx.y * thread_num[layer] + threadIdx.x >= neuron) break;
            for(int f = 0; f < MINIBATCH ; ++f) {
                // if((blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * thread_num[layer] + threadIdx.x == 8 * 1024) {
                //     printf("real res = %f, block.x = %d, block.y = %d, thread_num = %d, thread.x = %d\n", 
                //         res[f], blockIdx.x, blockIdx.y, thread_num[layer], threadIdx.x);
                // }
                if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * thread_num[layer] + threadIdx.x] = __ReLU(res[f] + bias)) {
                    active[blockIdx.x * MINIBATCH + f] = 1;
                }
            }
        }
    }
};

void test_benchmark_fused_layer1024_0_1(
    std::vector<std::vector<float>> &input,
    std::vector<COOMatrix> &coo,
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

    int *category;
    int *category_d;

    int *active;
    int *active_d;

    int *stride;
    int *stride_d;
    
    int layer = weight.size();
    
    std::map<int, int> neuron_map = {
        {1024, 6},
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

    A = (float*)malloc(sizeof(float) * neuron * batch);
    C = (float*)malloc(sizeof(float) * neuron * batch);
    memset(C, 0, sizeof(float) * neuron * batch);
    for(int l = 0; l < input.size(); ++l) {
        for(int i = 0; i < input[l].size(); ++i) {
            float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/32.0));
            A[l * neuron + i] = r2;
            // input[l][i];
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

    category = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i) {
        category[i] = i;
    }

    active = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i){
        active[i] = 0;
    }

    stride = (int*) malloc(sizeof(int*) * FUSE_LAYER);
    for(int l = 0; l < FUSE_LAYER; ++l) {
        stride[l] = stride_map[l + 1];
    }

    Safe_Call(cudaMalloc((void**)&A_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A_d, A, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&C_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C_d, 0, sizeof(float) * neuron * batch));

    Safe_Call(cudaMalloc((void**)&active_d, sizeof(int) * batch));

    Safe_Call(cudaMalloc((void**)&category_d, sizeof(int) * batch));
    Safe_Call(cudaMemcpy(category_d, category, sizeof(int) * batch, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void**)&stride_d, sizeof(int) * FUSE_LAYER));
    Safe_Call(cudaMemcpy(stride_d, stride, sizeof(int) * FUSE_LAYER, cudaMemcpyHostToDevice));

    for(int l = 0; l < layer; ++l) {
        Safe_Call(cudaMalloc((void**)&(B_d[l]), sizeof(float) * weight[l].size()));
        Safe_Call(cudaMemcpy(B_d[l], B[l], sizeof(float) * weight[l].size(), cudaMemcpyHostToDevice));
    }

    int load_num[FUSE_LAYER] = {0};
    int shared_size[FUSE_LAYER] = {0};
    int thread_num[FUSE_LAYER] = {0};

    int* load_num_d;
    int* shared_size_d;
    int* thread_num_d;

    Safe_Call(cudaMalloc((void**)&load_num_d, sizeof(int) * FUSE_LAYER));
    Safe_Call(cudaMalloc((void**)&shared_size_d, sizeof(int) * FUSE_LAYER));
    Safe_Call(cudaMalloc((void**)&thread_num_d, sizeof(int) * FUSE_LAYER));


    int last_layer_thread = 96;
    thread_num[FUSE_LAYER - 1] = last_layer_thread; // compute by cpu
    load_num[FUSE_LAYER - 1] = stride[FUSE_LAYER - 1] > thread_num[FUSE_LAYER - 1] ? 32 * (thread_num[FUSE_LAYER - 1] / 16) : stride[FUSE_LAYER - 1] + 16 * (thread_num[FUSE_LAYER - 1] / 16);
    
    int llll = load_num[FUSE_LAYER - 1];
    
    for(int i = FUSE_LAYER - 2; i >= 0; --i) {
        thread_num[i] = load_num[i + 1];
        load_num[i] = stride[i] > load_num[i + 1] ? 32 * (thread_num[i] / 16) : stride[i] + 16 * (load_num[i + 1] / 16);
        llll = std::max(llll, load_num[i]);
    }



    for(int i = FUSE_LAYER - 1; i >= 0; --i) {
        shared_size[i] = ((load_num[i] + 31) / 32) * 32; // for no conflict
    }

    Safe_Call(cudaMemcpy(load_num_d, load_num, sizeof(int) * FUSE_LAYER, cudaMemcpyHostToDevice));
    Safe_Call(cudaMemcpy(shared_size_d, shared_size, sizeof(int) * FUSE_LAYER, cudaMemcpyHostToDevice));
    Safe_Call(cudaMemcpy(thread_num_d, thread_num, sizeof(int) * FUSE_LAYER, cudaMemcpyHostToDevice));

    int shared_size_max = ((llll + 31) / 32) * 32;
    std::cout << "using shared memory = " << float(shared_size_max) / 1024.0 * MINIBATCH * sizeof(float) << "KB" << std::endl;

    env.add_event("row-succ-20-uiuc-kernel");
    auto stream = env.get_stream("row-succ-20-uiuc-kernel");

    env.event_start_record("row-succ-20-uiuc-kernel");
    int blocksize = 128;
    dim3 block(blocksize);
    dim3 grid((batch + MINIBATCH - 1)/ MINIBATCH, (neuron + blocksize - 1) / last_layer_thread);

    fuseed_layer_l0_l11<<<grid, block, sizeof(float) * (MINIBATCH * shared_size_max), stream>>>(
        A_d, B_d[0], B_d[1], C_d, category_d, active_d, stride_d, 
        load_num_d, shared_size_d, thread_num_d,
        batch, neuron, bias, last_layer_thread
    );
    cudaError_t err = cudaGetLastError();        
    if (err != cudaSuccess) {
        printf("what CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * batch, cudaMemcpyDeviceToHost, stream));
    env.event_stop_record("row-succ-20-uiuc-kernel");

    Safe_Call(cudaStreamSynchronize(stream));
    
    int feature = 0;
    for(int k = 0; k < batch; ++k) {
        if(active[k]) {
            // category[feature] = category[k];
            category[feature] = k;
            feature++;
        }
    }
    // float* tmp = A_d;
    // A_d = C_d;
    // C_d = tmp;

    std::cout << "layer " << ", batch = "<< feature << std::endl;

    float time = env.get_event_time("row-succ-20-uiuc-kernel"); 
    std::cout << "Layer "<< " exec Time = " << time <<  "ms, Utilization = " << std::endl;
	Safe_Call(cudaMemcpy(C, C_d, sizeof(float) * neuron * batch, cudaMemcpyDeviceToHost));
    
	CpuSpmmFuse::run_and_cmp(coo, A, neuron, batch, bias, C, FUSE_LAYER, false, true, true);
}
};