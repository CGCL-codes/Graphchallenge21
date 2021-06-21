#include <cuda.h>
#include "../gpu_lib/header.h"
#include "../utils/header.h"
#include <cstdio>


namespace ftxj {

__global__ void cuda_print(int* c, int len) {
  printf("[Print BEG]\n");  
  for(int i = 0; i < len; ++i) {
    printf("%d, ", c[i]);  
    if(i % 10 == 0) {
      printf("\n");
    }
  }
  printf("[Print END]\n");  
}
__global__  void snig_inference(
    float* Y_0,
    bool* is_nonzero_row_0,
    const size_t sec_size,
    const size_t num_secs,
    const size_t num_neurons,
    int* col_w,
    int* row_w,
    float* val_w,
    float bias,
    bool* is_nonzero_row_1,
    float* Y_1
) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  //r = blockIdx.x
  //s_o = blockIdx.y
  int num_threads = blockDim.x * blockDim.y;

  // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
  //   for(int i = 0; i < 30; ++i) {
  //     printf("col %d, %d, \n", i, col_w[i]);
  //   }
  // }

  //num_secs is small enough to compute by each single thread
  bool is_all_zero = true;
  for(size_t s_i = 0; s_i < num_secs; ++s_i) {
    is_all_zero &= !is_nonzero_row_0[blockIdx.x * num_secs + s_i];
  }

  if(is_all_zero) {
    //incremental memory resetting
    //avoid calling cudaMemset
    if(is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y]) {
      for(size_t j = tid; j < sec_size; j += num_threads) {
        Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + j] = 0;
      }
      __syncthreads();
      if(tid == 0) {
        is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y] = false;
      } 
    }
    return;
  }

  //forward feeding
  extern __shared__ float results[];

  //set results to bias directly
  for(size_t k = tid; k < sec_size; k += num_threads) {
    results[k] = bias;  
  }

  //use bool array size of 2 (is_nonzero) in share memory to avoid synchronization
  //is_nonzero[1] represents whether this row is nonzero
  //if is_nonzero[1] is true, this row is nonzero
  __shared__ bool is_nonzero[2];
  if(tid == 0) {
    is_nonzero[1] = false;
  }
  __syncthreads();

  for(size_t s_i = 0; s_i < num_secs; ++s_i) {
    if(!is_nonzero_row_0[blockIdx.x * num_secs + s_i]) {
      continue;
    }
    for(size_t j = threadIdx.y + s_i * sec_size; j < (s_i + 1) * sec_size; j += blockDim.y) {
      float valY = Y_0[blockIdx.x * num_neurons + j];
      if(valY == 0) {
        continue;
      }
      int beg_w = col_w[blockIdx.y * num_neurons + j] + threadIdx.x;
      int end_w = col_w[blockIdx.y * num_neurons + j + 1];
      // printf("%d, %d\n", beg_w, end_w);
      for(int k = beg_w; k < end_w; k += blockDim.x) {
        int roww = row_w[k];
        float valw = val_w[k];
        // if(blockIdx.x == 3 && j == 526) {
        //   printf("Batch 3, idx = %d, %f * %f\n", roww, valw, valY);
        // }
        atomicAdd(&results[roww - blockIdx.y * sec_size], valY * valw);
      }
    }
  }
  __syncthreads();
  for(size_t i = tid; i < sec_size; i += num_threads) {
    float v = results[i] > 32.0 ? 32.0 : ((results[i] < 0) ? 0 : results[i]);
    // if(blockIdx.x == 3 && blockIdx.y * sec_size + i == 526) {
    //   printf("Batch 3, res = %f, %d\n", v, blockIdx.y * sec_size + i);
    // }
    Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + i] = v;
    is_nonzero[v != 0] = true;
  }

  //if one thread sets is_nonzero[1] to true
  //meaning this row is nonzero
  //toggle is_nonzero_row_1[this row] to true
  __syncthreads();
  if(tid == 0) { 
    is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y] = is_nonzero[1];
  }
};



void test_benchmark_SNIG(
    std::vector<std::vector<float>> &input,
    std::vector<SNIGMatrix> &weights, 
    int batch, 
    int neuron,
    int sec_size,
    int nnzs, 
    float bias,
    GpuEnv &env
) {

	  float *A;
    float *A_d;

    float *C;
    float *C_d;


    int** col_w_d;
    int** row_w_d;
    float** value_w_d;

    int num_secs = (neuron + sec_size - 1) / sec_size;

    printf("batch = %d, sec_size = %d, num_secs = %d\n", batch, sec_size, num_secs);

    bool *is_nonzero_row_0 = new bool [batch * num_secs];
    bool *is_nonzero_row_1 = new bool [batch * num_secs];
    
    bool *is_nonzero_row_0_d;
    bool *is_nonzero_row_1_d;

    
    int layer = weights.size();

    A = (float*)malloc(sizeof(float) * neuron * batch);
    C = (float*)malloc(sizeof(float) * neuron * batch);
    memset(C, 0, sizeof(float) * neuron * batch);
    memset(A, 0, sizeof(float) * neuron * batch);

    for(int l = 0; l < input.size(); ++l) {
      for(int sec_idx = 0; sec_idx < num_secs; ++sec_idx) {
        bool all_zero = true;  
        for(int i = sec_idx * sec_size; i < (sec_idx + 1) * sec_size; ++i) {
          A[l * neuron + i] = input[l][i];
          if(input[l][i] != 0.0) {
            all_zero = false;
          }
        }
        if(all_zero) {
          std::cout << "bugs on " << l << std::endl;
          exit(-1);
        }
        is_nonzero_row_0[l * num_secs + sec_idx] = !all_zero;
        is_nonzero_row_1[l * num_secs + sec_idx] = true;
      }
    }

    col_w_d = (int**) malloc(sizeof(int*) * layer);
    row_w_d = (int**) malloc(sizeof(int*) * layer);
    value_w_d = (float**)malloc(sizeof(float*) * layer);
    

    Safe_Call(cudaMalloc((void**)&A_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A_d, A, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void**)&C_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C_d, 0, sizeof(float) * neuron * batch));

    Safe_Call(cudaMalloc((void**)&is_nonzero_row_0_d, sizeof(bool) * num_secs * batch));
    Safe_Call(cudaMalloc((void**)&is_nonzero_row_1_d, sizeof(bool) * num_secs * batch));
    Safe_Call(cudaMemcpy(is_nonzero_row_0_d, is_nonzero_row_0, sizeof(bool) * num_secs * batch, cudaMemcpyHostToDevice));
    Safe_Call(cudaMemcpy(is_nonzero_row_1_d, is_nonzero_row_1, sizeof(bool) * num_secs * batch, cudaMemcpyHostToDevice));
    

    for(int l = 0; l < layer; ++l) {
        Safe_Call(cudaMalloc((void**)&(col_w_d[l]), sizeof(int) * (num_secs * neuron + 1)));
        Safe_Call(cudaMemcpy(col_w_d[l], weights[l].col, sizeof(int) * (num_secs * neuron + 1), cudaMemcpyHostToDevice));

        Safe_Call(cudaMalloc((void**)&(row_w_d[l]), sizeof(int) * nnzs));
        Safe_Call(cudaMemcpy(row_w_d[l], weights[l].row, sizeof(int) * nnzs, cudaMemcpyHostToDevice));
    
        Safe_Call(cudaMalloc((void**)&(value_w_d[l]), sizeof(float) * nnzs));
        Safe_Call(cudaMemcpy(value_w_d[l], weights[l].val, sizeof(float) * nnzs, cudaMemcpyHostToDevice));
    }

    float all_time = 0;
    env.add_event("SNIG_kernel");
    for(int l = 0; l < layer; ++l) {
        auto stream = env.get_stream("SNIG_kernel");
        env.event_start_record("SNIG_kernel");
        dim3 block(2, 512);
        dim3 grid(batch, num_secs);
        snig_inference<<<grid, block, sizeof(float) * sec_size, stream>>>(
            A_d, is_nonzero_row_0_d, 
            sec_size, num_secs, neuron, 
            col_w_d[l], row_w_d[l], value_w_d[l], 
            bias,
            is_nonzero_row_1_d, C_d
        );
        cudaError_t err = cudaGetLastError();        
        if (err != cudaSuccess) {
            printf("what CUDA Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        env.event_stop_record("SNIG_kernel");

        Safe_Call(cudaStreamSynchronize(stream));


        Safe_Call(cudaMemcpy(is_nonzero_row_1, is_nonzero_row_1_d, sizeof(bool) * num_secs * batch, cudaMemcpyDeviceToHost));
        int feature = 0;
        for(int b = 0; b < batch; ++b) {
          for(int s = 0; s < num_secs; ++s) {
            if(is_nonzero_row_1[b * num_secs + s]) {
              feature++;
              s = num_secs;
            }
          }
        }
        bool* tmp_bool = is_nonzero_row_0_d;
        is_nonzero_row_0_d = is_nonzero_row_1_d;
        is_nonzero_row_1_d = tmp_bool;

        float* tmp_input = A_d;
        A_d = C_d;
        C_d = tmp_input;
        float time = env.get_event_time("SNIG_kernel"); 

        std::cout << "layer = " << l  << ", batch = "<< feature << ", time = " << time << std::endl;

        all_time += time;
    }
	Safe_Call(cudaMemcpy(C, C_d, sizeof(float) * neuron * batch, cudaMemcpyDeviceToHost));
	std::cout << "Kernel Exec Time [SNIG] = " << all_time <<  "ms" <<std::endl;
}
};
