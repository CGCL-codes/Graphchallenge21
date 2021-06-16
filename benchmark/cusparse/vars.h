  
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

//using namespace std;

void readweights();
void readinput();

void setup_gpu();
void final_gpu();
void infer_gpu(int);

//#define BALANCE 30 //BALANCE LAYER 0 FOR EVERY LAYER COMMENT OUT FOR TURN OFF
//#define OUTOFCORE //COMMENT THIS OUT IF YOU HAVE ENOUGH MEMORY
//#define OVERLAP //WORKS ONLY WHEN OUTOFCORE IS ENABLED
#define INDPREC int
#define VALPREC float
#define FEATPREC float



inline void checkCuda(cudaError_t result, const char *file, const int line, bool fatal=false) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n",  file, line, int(result),
            cudaGetErrorString(result));\
    if (fatal) {
        exit(EXIT_FAILURE);
    }
  }
}

#define OR_PRINT(stmt) checkCuda(stmt, __FILE__, __LINE__);
#define OR_FATAL(stmt) checkCuda(stmt, __FILE__, __LINE__, true);

#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}