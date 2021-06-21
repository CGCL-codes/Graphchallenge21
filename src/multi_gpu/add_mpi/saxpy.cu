#include <stdio.h>

__global__ void add_kernel(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void handle(int gpu_number)
{
  int N_GPU;
  cudaGetDeviceCount(&N_GPU);
  //printf("gpu count : %d\n",N_GPU);
 
   //Arrange the task of each GPU
  int N = ((1<<30)+N_GPU - 1)/N_GPU;

  cudaSetDevice(gpu_number);

  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  float time_elapsed=0;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);    //创建Event
  cudaEventCreate(&stop);
  cudaEventRecord( start,0);    //记录当前时间

  // Perform SAXPY on 1M elements
  add_kernel<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaEventRecord(stop,0);    //记录当前时间
  cudaEventSynchronize(start);    //Waits for an event to complete.
  cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
  cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  cudaEventDestroy(start);    //destory the event
  cudaEventDestroy(stop);
  printf("card%d 执行时间：%f(ms)\n",gpu_number,time_elapsed);
}