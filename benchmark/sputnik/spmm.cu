#include "sputnik/sputnik.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include "vars.h"
#include <chrono>

typedef std::chrono::duration<double> Duration;
typedef std::chrono::system_clock sc;

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
INDPREC **csrindex_dT; // csr index 数组，每层一个
VALPREC **csrvalue_d; // csr value 数组，每层一个

FEATPREC *currfeat; // 当前层处理的特征
FEATPREC *nextfeat; // 下一层要处理的特征（当前层的输出结果）

FEATPREC *currfeat_d; // 当前层处理的特征
FEATPREC *nextfeat_d; // 下一层要处理的特征（当前层的输出结果）


Duration timekernel;
float time_kernel;

int *active;   

float * bias_a;
float *bias_d;   

double timeio;
double timetot;
double timeinfer;
double timebalance = 0.0;
double timecopy = 0.0;

int *numbatch;
int *batchdispl;
int mybatch; // 当前 id 分到的 输入batch数量

float elapsedTime;

cudaEvent_t kernelstart, kernelstop;
cudaStream_t kernelstream;


int last_feature = 0;

void setup_gpu() {
    printf("set up\n");
    OR_FATAL(cudaSetDevice(0));
    int deviceCount;
    OR_FATAL(cudaGetDeviceCount(&deviceCount));
    int dev = 0;
    cudaDeviceProp deviceProp;
    OR_FATAL(cudaGetDeviceProperties(&deviceProp, dev));
    csrdispl_d = new int*[layer];
    csrindex_d = new INDPREC*[layer];
    csrindex_dT = new INDPREC*[layer];

    csrvalue_d = new VALPREC*[layer];


    cudaEventCreate(&kernelstart);
    cudaEventCreate(&kernelstop);
    cudaStreamCreate(&kernelstream);

    int *temp = new int [neuron];
    for(int i = 0; i < neuron; ++i) {
        temp[i] = i;
    }

    for(int l = 0; l < layer; l++){
        OR_FATAL(cudaMalloc((void**)&csrdispl_d[l], sizeof(int) * (neuron+1)));
        OR_FATAL(cudaMemcpy(csrdispl_d[l], csrdispl[l], sizeof(int) * (neuron+1), cudaMemcpyHostToDevice));

        OR_FATAL(cudaMalloc((void**)&csrindex_d[l], sizeof(INDPREC) * csrdispl[l][neuron]));
        OR_FATAL(cudaMemcpy(csrindex_d[l], csrindex[l], sizeof(INDPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));

        OR_FATAL(cudaMalloc((void**)&csrindex_dT[l], sizeof(int) * neuron));
        OR_FATAL(cudaMemcpy(csrindex_dT[l], temp, sizeof(int) * neuron, cudaMemcpyHostToDevice));

        OR_FATAL(cudaMalloc((void**)&csrvalue_d[l], sizeof(VALPREC) * csrdispl[l][neuron]));
        OR_FATAL(cudaMemcpy(csrvalue_d[l], csrvalue[l], sizeof(VALPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));
    }
    OR_FATAL(cudaMalloc((void**)&currfeat_d, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(currfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemcpy(currfeat_d, currfeat, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyHostToDevice));


    OR_FATAL(cudaMalloc((void**)&nextfeat_d, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(nextfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));

    OR_FATAL(cudaMalloc((void**)&bias_d, sizeof(float) * mybatch));
    OR_FATAL(cudaMemset(bias_d, 0, sizeof(float) * mybatch));
    OR_FATAL(cudaMemcpy(bias_d, bias_a, sizeof(float) * mybatch, cudaMemcpyHostToDevice));

}

void kernel_spmm(int l) {
    
    // cudaError_t CudaSpmmBiasRelu(int m, int k, int n, int nonzeros,
    //     const int* __restrict__ row_indices,
    //     const float* __restrict__ values,
    //     const int* __restrict__ row_offsets,
    //     const int* __restrict__ column_indices,
    //     const float* __restrict__ dense_matrix,
    //     const float* __restrict__ bias,
    //     float* __restrict__ output_matrix,
    //     cudaStream_t stream);


    OR_FATAL(cudaMemcpy(currfeat_d, currfeat, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemset(nextfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));

    auto x = sc::now();
    
    cudaEventRecord(kernelstart,kernelstream);

    OR_FATAL(sputnik::CudaSpmmBiasRelu(neuron, neuron, mybatch, 32 * neuron, 
        csrindex_dT[l],
        csrvalue_d[l], 
        csrdispl_d[l],
        csrindex_d[l], 
        currfeat_d, 
        bias_d,  
        nextfeat_d, 
        kernelstream
    ));

    cudaEventRecord(kernelstop,kernelstream);
    Duration d = (sc::now() - x);
    OR_FATAL(cudaMemcpyAsync(nextfeat, nextfeat_d, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyDeviceToHost, kernelstream));


    cudaEventElapsedTime(&elapsedTime,kernelstart,kernelstop);
    time_kernel += elapsedTime/1.0e3;
    timekernel += d;
    //OR_FATAL(cudaMemcpy(nextfeat, nextfeat_d, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyDeviceToHost));
    for(int i = 0; i < mybatch; ++i) {
		active[i] = 0;
    }
    
    if(last_feature != mybatch) {
        int feature = 0;
        for(int i = 0; i < mybatch; ++i) {
            for(int j = 0; j < neuron; ++j) {
            if(nextfeat[j * mybatch + i]) {
                    active[i] = 1;
                }
            }
        }
        for(int i = 0; i < mybatch; ++i) {
            if(active[i]) {
                feature ++;
            }   
        }
        last_feature = mybatch;
        if(feature != mybatch) {
            for(int j = 0; j < neuron; ++j) {
                int curr_feature = 0;
                for(int i = 0; i < mybatch; ++i) {
                    if(active[i]) {
                        nextfeat[j * feature + curr_feature] = nextfeat[j * mybatch + i];
                        curr_feature++;
                    }
                }   
            }
            mybatch = feature;
        }
    }

    FEATPREC *tempfeat = currfeat;
    currfeat = nextfeat;
    nextfeat = tempfeat;
}

int main(int argc, char* argv[]) {
	dataset = "/home/xinjie/SpDNN_Challenge2020/dataset";
	char *chartemp;
	neuron = 4096;
	layer = 120;
	batch = 60000;
	input = 25019051;
	bias = -0.35;
  
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
    
    mybatch = batch;

    csrdispl = new int*[layer];
    csrindex = new INDPREC*[layer];
    csrvalue = new VALPREC*[layer];
    currfeat = new FEATPREC[neuron*(long)mybatch];
    nextfeat = new FEATPREC[neuron*(long)mybatch];
    active = new int [mybatch];
    bias_a = new float [neuron];
        
    printf("\n");
    printf("READING WEIGHTS\n");
    readweights();
    printf("READING INPUT\n");
    readinput();    

    for(int i = 0; i < neuron; ++i) {
        bias_a[i] = bias;
    }

    // dataset = "/home/xinjie/SpDNN_Challenge2020/dataset";
    // neuron = 4;
    // layer = 1;
    // batch = 4;
    
	// input = 6;
    // bias = 0;
	// mybatch = batch;
    // int degree = 1;
	// int count = 1;

	// for(int l = 0; l < layer; ++l) {
	// 	csrdispl[l] = new int [neuron + 1];
	// 	csrdispl[l][0] = 0;
	// 	csrindex[l] = new int [degree * neuron];
	// 	csrvalue[l] = new float [degree * neuron];
	// 	for(int n = 1; n < neuron + 1; ++n) {
	// 		csrdispl[l][n] = csrdispl[l][n - 1] + degree; 
	// 		for(int m = 0; m < degree; ++m) {
	// 			csrindex[l][(n - 1) * degree + m] = (n-1);
	// 			csrvalue[l][(n - 1) * degree + m] = 1;
	// 			count++;
	// 		}
	// 	}
	// }

	// printf("================weight===========================\n");
	// for(int j = 1; j < neuron + 1; ++j) {
	// 	int len = csrdispl[0][j] - csrdispl[0][j-1];
	// 	for(int i = 0; i < len; ++i) {
	// 		printf("%d, ", csrindex[0][csrdispl[0][j - 1] + i]);
	// 	}
	// 	printf("\n");
	// }
	// printf("================weight===========================\n");

	// count = 0;
	// printf("================input===========================\n");
	// for(int i = 0; i < neuron; ++i) {
	// 	for(int j = 0; j < mybatch; ++j) {
	// 		currfeat[i * mybatch + j] = count++;
	// 		printf("%f, ", currfeat[i * mybatch + j]);
	// 	}
	// 	printf("\n");
	// }
    // printf("================input===========================\n");
    

    setup_gpu();

    printf("INFERENCE......\n");
    for(int i = 0; i < layer; ++i) {
        kernel_spmm(i);
        printf("my batch %d\n", mybatch);
    }
    printf("my batch %d\n", mybatch);
    printf("INFERENCE TIME: %f s, %f s\n",timekernel.count(), time_kernel);

    printf("INFERENCE THRP: %e EDGES/s (%f TTLOPS)\n", totnz /timekernel.count() * batch, totnz / timekernel.count()*batch /1e12);
    printf("INFERENCE THRP: %e EDGES/s (%f TTLOPS)\n", totnz /time_kernel * batch, totnz / time_kernel*batch /1e12);

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
            currfeat[(row[n]-1)*(long)batch+col[n]-1] = val[n];
        }
    }
    fclose(inputf);
    delete[] row;
    delete[] col;
    delete[] val;
}