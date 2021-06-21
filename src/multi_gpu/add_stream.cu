#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
// This application demonstrates how to use CUDA API to use mutiple GPUs
// Function to add the elements of two arrays
 
//Mutiple-GPU Plan Structure
typedef struct
{
    //Host-side input data
    float *h_x, *h_y;
	  
    //Result copied back from GPU
	  float *h_yp;
    //Device buffers
    float *d_x, *d_y;
 
    //Stream for asynchronous command execution
    cudaStream_t stream;
 
} TGPUplan;


 
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
 
int main(void)
{
  int N = 1<<20; // 1M elements
  
  //Get the numble of CUDA-capble GPU
  int N_GPU;
  cudaGetDeviceCount(&N_GPU);
  printf("gpu count : %d\n",N_GPU);
 
  //Arrange the task of each GPU
  int Np = (N + N_GPU - 1) / N_GPU;
  
  //Create GPU plans
  TGPUplan plan[N_GPU];
 
  //Initializing 
  for(int i = 0; i < N_GPU; i++)
  {
    cudaSetDevice(i);
    cudaStreamCreate(&plan[i].stream);
 
    cudaMalloc((void **)&plan[i].d_x, Np * sizeof(float));
    cudaMalloc((void **)&plan[i].d_y, Np * sizeof(float));
    plan[i].h_x = (float *)malloc(Np * sizeof(float));
    plan[i].h_y = (float *)malloc(Np * sizeof(float));
    plan[i].h_yp = (float *)malloc(Np * sizeof(float));
 
	  for(int j = 0; j < Np; j++)
    {
      plan[i].h_x[j] = 1.0f;
      plan[i].h_y[j] = 2.0f;
    }
  }
 
  int blockSize = 256;
  int numBlock = (Np + blockSize - 1) / blockSize;
 

    // double iStart,iElaps;
    // iStart=cpuSecond();

  clock_t start, finish;
  start = clock();

  for(int i = 0; i < N_GPU; i++)
  {
    //Set device
    cudaSetDevice(i);
 
    //Copy input data from CPU
    cudaMemcpyAsync(plan[i].d_x, plan[i].h_x, Np * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
    cudaMemcpyAsync(plan[i].d_y, plan[i].h_y, Np * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
    //Run the kernel function on GPU
    add<<<numBlock, blockSize, 0, plan[i].stream>>>(Np, plan[i].d_x, plan[i].d_y);
    
    //Read back GPU results
    cudaMemcpyAsync(plan[i].h_yp, plan[i].d_y, Np * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream);
  }
  finish = clock();
  float duration = (double)(finish - start) / CLOCKS_PER_SEC;  
     printf("GPU Kernel time: %f\n",duration);
    // cudaDeviceSynchronize();
    // iElaps=cpuSecond()-iStart;
    // printf("GPU Kernel time: %f\n",iElaps);
  //Process GPU results
  float y[N];
  for(int i = 0; i < N_GPU; i++)
  {
    //Set device
    cudaSetDevice(i);
 
    //Wait for all operations to finish
    cudaStreamSynchronize(plan[i].stream);
 
    //Get the final results
	  for(int j = 0; j < Np; j++)
		  if(Np * i + j < N)
			   y[Np * i + j]=plan[i].h_yp[j];
	  
    //shut down this GPU
    cudaFree(plan[i].d_x);
    cudaFree(plan[i].d_y);
    free(plan[i].h_x);
    free(plan[i].h_y);
  	cudaStreamDestroy(plan[i].stream); //Destroy the stream
  }
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  return 0;

}