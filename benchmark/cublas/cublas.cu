#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>


#include <sys/time.h>

using namespace std;


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


int main()
{
    srand(time(0));
    int M = 2000;              //矩阵A的行，矩阵C的行
    int N = 16384;               //矩阵A的列，矩阵B的行
    int K = 16384;               //矩阵B的列，矩阵C的列

    float *h_A = (float*)malloc(sizeof(float)*M*N);
    float *h_B = (float*)malloc(sizeof(float)*N*K);
    float *h_C = (float*)malloc(sizeof(float)*M*K);

    for (int i = 0; i < M*N; i++) {
        h_A[i] = i;
        // cout << h_A[i] << "  ";
        // if ((i + 1) % N == 0)
        //     cout << endl;        
    }
    //  cout << endl;

    for (int i = 0; i < N*K; i++) {
        h_B[i] =i;
        // cout << h_B[i] << "  ";
        // if ((i + 1) % K == 0)
        //     cout << endl;
    }
    cout << endl;

    double iStart, iElaps;

    float *d_A, *d_B, *d_C,*d_CT;
    cudaMalloc((void**)&d_A, sizeof(float)*M*N);
    cudaMalloc((void**)&d_B, sizeof(float)*N*K);
    cudaMalloc((void**)&d_C, sizeof(float)*M*K);
    cudaMemcpy(d_A, h_A, M*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*K * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1;
    float beta = 0;

    //C=A*B
    cublasHandle_t handle;
    
    cublasCreate(&handle);
    
    // clock_t start = clock();//MNK Bt*At
    
    iStart = seconds();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cublasSgemm(handle,
        CUBLAS_OP_N,  
        CUBLAS_OP_N,   
        K,                    //矩阵B的列数
        M,                    //矩阵A的行数
        N,                    //矩阵A的列数
        &alpha,           
        d_B,            
        K,                    
        d_A,         
        N,         
        &beta,          
        d_C,           
        K);
    
    CHECK(cudaGetLastError()) ;

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);


    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 1000.0f;

    iElaps = seconds() - iStart;

    // clock_t end = clock();
    // double sum_time = double(double(end - start)/CLOCKS_PER_SEC) * 1000;
    

    printf("time= %lf\n", elapsed);

    // cout<<"inference time: "<< sum_time <<endl; 
    float teps = (2 *(long) M * N * K) / elapsed;
    cout << "TEPS = " << teps << endl; 

    cudaMemcpy(h_C, d_C, M*K * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1; i++)
    {
        cout << h_C[i] << "  ";
        // if ((i+1)%K==0)
        //     cout << endl;
    }
    cout << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}