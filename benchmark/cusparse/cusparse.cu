#include <stdio.h>
#include <iostream>
#include <cusparse.h>
#include <vector>
#include "vars.h"

char *dataset;

int neuron; // 神经元数量（每一层固定的神经元数量，1024/4096/16384/65536）
int layer; // 网络深度（120/480/1920）
int batch; // 固定大小，60000
int input; // 60000 * （1024/4096/16384/65536）的稀疏矩阵nnzs
float bias; // 固定值

int blocksize;

long totnz;
int **csrdispl;     // csr offset 数组，每层一个
INDPREC **csrindex; // csr index 数组，每层一个
VALPREC **csrvalue; // csr value 数组，每层一个

int **csrdispl_d;     // csr offset 数组，每层一个
INDPREC **csrindex_d; // csr index 数组，每层一个
VALPREC **csrvalue_d; // csr value 数组，每层一个

FEATPREC *currfeat; // 当前层处理的特征
FEATPREC *nextfeat; // 下一层要处理的特征（当前层的输出结果）

FEATPREC *currfeat_d; // 当前层处理的特征
FEATPREC *nextfeat_d; // 下一层要处理的特征（当前层的输出结果）

int *active;   
int *active_d;     

double timeio;
double timetot;
double timeinfer;
double timebalance = 0.0;
double timekernel = 0.0;
double timecopy = 0.0;

int *numbatch;
int *batchdispl;
int mybatch; // 当前 id 分到的 输入batch数量

__device__ float __ReLU(float x){
    return x<0.0?0.0:x>32.0?32.0:x;
    //return x;
};
 

float ReLU(float x){
    //return x;
    return x<0.0?0.0:x>32.0?32.0:x;
 };

__global__ void __launch_bounds__(256,4) dummy_kernel(FEATPREC *nextfeat, float bias, int neuron, int mybatch, int *active){
    int testbatch = gridDim.x * blockDim.x + threadIdx.x;
    if(testbatch > 60000) return;
    for(int n = 0; n < neuron; n++) {
        nextfeat[testbatch * neuron + n] = __ReLU(nextfeat[testbatch * neuron + n] + bias);
        if(nextfeat[testbatch * neuron + n]) {
            atomicAdd(active + testbatch, 1);
        }
    }
};

void setup_gpu() {
    OR_FATAL(cudaSetDevice(0));
    int deviceCount;
    OR_FATAL(cudaGetDeviceCount(&deviceCount));
    //printf("\n");
    //printf("Device Count: %d\n",deviceCount);
    int dev = 0;
    cudaDeviceProp deviceProp;
    OR_FATAL(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("Device %d name: %s\n",dev,deviceProp.name);
    // printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    // printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    // printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    // printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    // printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    // printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    // printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    // printf("Warp size: %d\n",deviceProp.warpSize);
    // printf("\n");

    csrdispl_d = new int*[layer];
    csrindex_d = new INDPREC*[layer];
    csrvalue_d = new VALPREC*[layer];
    
    for(int l = 0; l < layer; l++){
        OR_FATAL(cudaMalloc((void**)&csrdispl_d[l], sizeof(int) * (neuron+1)));
        OR_FATAL(cudaMemcpy(csrdispl_d[l], csrdispl[l], sizeof(int) * (neuron+1), cudaMemcpyHostToDevice));

        OR_FATAL(cudaMalloc((void**)&csrindex_d[l], sizeof(INDPREC) * csrdispl[l][neuron]));
        OR_FATAL(cudaMemcpy(csrindex_d[l], csrindex[l], sizeof(INDPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));

        OR_FATAL(cudaMalloc((void**)&csrvalue_d[l], sizeof(VALPREC) * csrdispl[l][neuron]));
        OR_FATAL(cudaMemcpy(csrvalue_d[l], csrvalue[l], sizeof(VALPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));
    }
    OR_FATAL(cudaMalloc((void**)&currfeat_d, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(currfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));

    OR_FATAL(cudaMalloc((void**)&nextfeat_d, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(nextfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));

    OR_FATAL(cudaMemcpy(currfeat_d, currfeat, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyHostToDevice));

    OR_FATAL(cudaMalloc((void**)&active_d, sizeof(int) * mybatch));
    OR_FATAL(cudaMemset(active_d, 0, sizeof(int)*mybatch));

    OR_FATAL(cudaMemcpy(active_d, active,sizeof(int) * mybatch,cudaMemcpyHostToDevice));
}

double kernel_spmm(int l) {
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer    = NULL;
    size_t bufferSize = 0;
    float alpha = 1.0f;
    float beta = 0.0f;

    CUSPARSE_CHECK( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    
    OR_FATAL(cudaMemcpy(currfeat_d, currfeat, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemset(nextfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(active_d, 0, sizeof(int) * mybatch));

    CUSPARSE_CHECK(cusparseCreateCsr(&matA, neuron, neuron, csrdispl[l][neuron],
                                      csrdispl_d[l], csrindex_d[l], csrvalue_d[l],
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CUSPARSE_CHECK( cusparseCreateDnMat(&matB, neuron, mybatch, neuron, currfeat_d,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
                                        
    // Create dense matrix C
    CUSPARSE_CHECK( cusparseCreateDnMat(&matC, neuron, mybatch, neuron, nextfeat_d,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    

                                        
    OR_FATAL(cudaMalloc(&dBuffer, bufferSize));
              

    CUSPARSE_CHECK( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
   
                                 CUSPARSE_CSRMM_ALG1, &bufferSize) )
   
                                 cudaEvent_t start, stop;
                                 cudaEventCreate(&start);
                                 cudaEventCreate(&stop);
                                 cudaEventRecord(start, 0);
   
   
                                 CUSPARSE_CHECK( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_MM_ALG_DEFAULT, dBuffer) )
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop); //ms 
    elapsed /= 1000.0f; // s

    // destroy matrix/vector descriptors
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) )
    CUSPARSE_CHECK( cusparseDestroyDnMat(matB) )
    CUSPARSE_CHECK( cusparseDestroyDnMat(matC) )
    CUSPARSE_CHECK( cusparseDestroy(handle) )

    
    // dim3 block(blocksize);
    // dim3 grid(mybatch / blocksize);

    // dummy_kernel<<<grid,block>>>(nextfeat_d, bias, neuron, mybatch, active_d);

    OR_FATAL(cudaMemcpy(nextfeat, nextfeat_d, neuron * mybatch  * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < mybatch; ++i) {
        active[i] = 0;
    }
    
    for(int i = 0; i < mybatch; ++i) {
       for(int j = 0; j < neuron; ++j) {
            if(nextfeat[i * neuron + j] =  ReLU(nextfeat[i * neuron + j] + bias))
                active[i] += 1;
        }
    }

    int feature = 0;
    for(int i = 0; i < mybatch; ++i) {
        if(active[i]) {
            for(int j = 0; j < neuron; ++j) {
                nextfeat[feature * neuron + j] = nextfeat[i * neuron + j];
            }
            feature++;
        }
    }
    //printf("featrure = %d\n", feature);

    mybatch = feature;
    FEATPREC *tempfeat = currfeat;
    currfeat = nextfeat;
    nextfeat = tempfeat;
    return double(elapsed);
}

int main(int argc, char* argv[]) {

    dataset = "/home/xinjie/ye/SpDNN_Challenge2020/dataset";
    char *chartemp;
    neuron = 16384;
    layer = 1;
    batch = 2000;
    input = 6374505;
    bias = 0;
    // int degree = 2;

    // dataset = getenv("DATASET");
	// char *chartemp;
	// chartemp = getenv("NEURON");
	// neuron = atoi(chartemp);
	// chartemp = getenv("LAYER");
	// layer = atoi(chartemp);
	// chartemp = getenv("BATCH");
	// batch = atoi(chartemp);
	// chartemp = getenv("INPUT");
	// input = atoi(chartemp);
	// chartemp = getenv("BIAS");
	// bias = atof(chartemp);
    // chartemp = getenv("BLOCKSIZE");
    // blocksize = atoi(chartemp);
    
    mybatch = batch;
  
    csrdispl = new int*[layer];
    csrindex = new INDPREC*[layer];
    csrvalue = new VALPREC*[layer];
    currfeat = new FEATPREC[neuron*(long)mybatch];
    nextfeat = new FEATPREC[neuron*(long)mybatch];
  
    active = new int [mybatch];
    
    // int count = 1;

    // for(int l = 0; l < layer; ++l) {
    //     csrdispl[l] = new int [neuron + 1];
    //     csrdispl[l][0] = 0;
    //     csrindex[l] = new int [degree * neuron];
    //     csrvalue[l] = new float [degree * neuron];
    //     for(int n = 1; n < neuron + 1; ++n) {
    //         csrdispl[l][n] = csrdispl[l][n - 1] + degree; 
    //         for(int m = 0; m < degree; ++m) {
    //             csrindex[l][(n - 1) * degree + m] = m;
    //             csrvalue[l][(n - 1) * degree + m] = count;
    //             count++;
    //         }
    //     }
    // }

    // printf("================weight===========================\n");
    // for(int j = 1; j < neuron + 1; ++j) {
    //     int len = csrdispl[0][j] - csrdispl[0][j-1];
    //     for(int i = 0; i < len; ++i) {
    //         printf("%d, ", csrindex[0][csrdispl[0][j - 1] + i]);
    //     }
    //     printf("\n");
    // }
    // printf("================weight===========================\n");

    // count = 0;
    // printf("================input===========================\n");
    // for(int j = 0; j < mybatch; ++j) {
    //     for(int i = 0; i < neuron; ++i) {
    //         // if(i == (j + 1) % neuron || i == (j + 2) % neuron) {
    //         //    currfeat[i * neuron + j] = i;
    //         // }
    //         // else {
    //         //    currfeat[j * neuron + i] = 0;
    //         // }
    //         currfeat[i * mybatch + j] = ++count;
    //         printf("%f, ", currfeat[i * mybatch + j]);
    //     }
    //     printf("\n");
    // }
    // printf("================input===========================\n");

    
    
    printf("\n");
    printf("READING WEIGHTS\n");
    readweights();
    printf("READING INPUT\n");
    readinput();

    for(int k = 0; k < mybatch; k++){
      active[k] = neuron;
    }
    

    setup_gpu();

    printf("INFERENCE......\n");
    double spmm_times = 0; 
    clock_t total_start = clock();
    for(int i = 0; i < 1; ++i) {
        auto t = kernel_spmm(i);
        spmm_times += double(t);
    }
    clock_t end_start = clock();
    auto gemm_time = double(spmm_times);
    auto all_time = double(end_start - total_start)  / CLOCKS_PER_SEC;
    
    printf("Inference time : %lfs, %lfs, %f TTEPS\n", gemm_time, all_time, long((long)batch * (long)neuron * 32 * layer) / gemm_time / 1e12);
	return 0;
}


void readweights(){
    totnz = 0;
    for(int l = 0; l < layer; l++){
        int rownz[neuron];
        for(int n = 0; n < neuron; n++)
            rownz[n] = 32;
        csrdispl[l] = new int[neuron+1];
        csrdispl[l][0] = 0;
        for(int n = 1; n < neuron+1; n++)
            csrdispl[l][n] = csrdispl[l][n-1]+rownz[n-1];
        totnz += csrdispl[l][neuron];
        csrindex[l] = new INDPREC[csrdispl[l][neuron]];
        csrvalue[l] = new VALPREC[csrdispl[l][neuron]];
    }

    printf("weights: %ld (%f GB)\n",totnz,totnz*(sizeof(INDPREC)+sizeof(VALPREC))/1.0e9);
    
    char filename[500];
    sprintf(filename,"%s/neuron%d.bin",dataset,neuron);
    printf("open filename = %s\n", filename);
    FILE *weightf = fopen(filename,"rb");
    for(int l = 0; l < layer; l++){
        int *row = new int[csrdispl[l][neuron]];
        int *col = new int[csrdispl[l][neuron]];
        float *val = new float[csrdispl[l][neuron]];
        fread(row, sizeof(int), csrdispl[l][neuron], weightf);
        fread(col, sizeof(int), csrdispl[l][neuron], weightf);
        fread(val, sizeof(int), csrdispl[l][neuron],weightf);
        int rownz[neuron];
        for(int n = 0; n < neuron; n++)
            rownz[n] = 0;
        for(int n = 0; n < csrdispl[l][neuron]; n++){
            csrindex[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = col[n]-1;
            csrvalue[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = val[n];
            rownz[row[n]-1]++;
        }
        delete[] row;
        delete[] col;
        delete[] val;
    }
    fclose(weightf);
}


void readinput(){
    char filename[500];
    printf("features: %ld (%f GB)\n",neuron*(long)batch*2,neuron*(long)batch*2*sizeof(FEATPREC)/1.0e9);
    sprintf(filename, "%s/sparse-images-%d.bin", dataset, neuron);
    FILE *inputf = fopen(filename,"rb");
    int *row = new int[input];
    int *col = new int[input];
    float *val = new float[input];
    fread(row,sizeof(int),input,inputf);
    fread(col,sizeof(int),input,inputf);
    fread(val,sizeof(float),input,inputf);
    for(long n = 0; n < neuron * (long)batch; n++)
        currfeat[n] = 0.0;
    for(int n = 0; n < input; n++) {
        if(col[n] - 1 < batch) {
            currfeat[(col[n] - 1) * (long)neuron + row[n] - 1] = val[n];
        }
    }
    fclose(inputf);
    delete[] row;
    delete[] col;
    delete[] val;
}