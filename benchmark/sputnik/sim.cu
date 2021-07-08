#include "vars.h"
#include <cuda.h>
#include <cuda_runtime.h>
	
inline void checkCuda(cudaError_t result, const char *file, const int line, bool fatal=false) {
	if (result != cudaSuccess) {
	  fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n",  file, line, int(result),
			  cudaGetErrorString(result));\
	  if (fatal) {
		  exit(EXIT_FAILURE);
	  }
	}
  }
  
#define OR_FATAL(stmt) checkCuda(stmt, __FILE__, __LINE__, true)

extern int neuron;
extern int layer;
extern int batch;
extern int input;
extern float bias;


extern Duration timekernel;

extern int **csrdispl;
extern INDPREC **csrindex;
extern VALPREC **csrvalue;

extern FEATPREC *currfeat;
extern FEATPREC *nextfeat;
extern FEATPREC *nextfeat_tmp;
extern int *active;
extern int *categories;
extern int *globalcategories;

extern int myid;
extern int numproc;
extern int numthreads;

extern int *numbatch;
extern int *batchdispl;
extern int mybatch;

extern double timebalance;
extern double timecopy;

int **csrdispl_d;
INDPREC *indbuff_d;
VALPREC *valbuff_d;
#ifdef OUTOFCORE
int  weightsizemax;
#ifdef OVERLAP
INDPREC *indstream_d;
VALPREC *valstream_d;
#endif
#else
INDPREC **csrindex_d;
VALPREC **csrvalue_d;
#endif

FEATPREC *currfeat_d;
FEATPREC *nextfeat_d;
int *active_d;
int *categories_d;

int blocksize;
int numblocks;
int numwarp;
int buffsize;

#ifdef BALANCE
int numfeature;
FEATPREC *sendbuff;
FEATPREC *recvbuff;
MPI_Request *catrecvrequests;
MPI_Request *catsendrequests;
MPI_Request *featrecvrequests;
MPI_Request *featsendrequests;
#endif

cudaEvent_t copystart, copystop;
cudaEvent_t kernelstart, kernelstop;
cudaStream_t copystream;
cudaStream_t kernelstream;
float elapsedTime;


__device__ __forceinline__ float __ReLU(float x){
	 return x < 0.0 ? 0.0 : x > 32.0 ? 32.0 : x;
};


float ReLU(float x){
	return x<0.0?0.0:x>32.0?32.0:x;
};

void kernel_serial(FEATPREC *nextfeat, FEATPREC *currfeat, int *wdispl, INDPREC *windex, VALPREC *wvalue, float bias, int *active) {
	int weight_block_size = 4;
	int batch_per_round = 1;
	printf("run kernel\n");
	for(int weight_block = 0; weight_block < neuron / weight_block_size; ++weight_block) { // 这个循环划分到不同 SM 上
		
		int current_block_index_begin = wdispl[weight_block * weight_block_size];
		
		int current_block_index_end = wdispl[(weight_block + 1) * (weight_block_size)];
		
		int current_block_index_size = current_block_index_end - current_block_index_begin;

		//printf("run kernel SM %d %d %d %d\n", weight_block, current_block_index_begin, current_block_index_end, current_block_index_size);
		float shared [current_block_index_size] = {0}; // shared memory 固定长度
		
		for(int idx = 0; idx < current_block_index_size; ++idx) {
			shared[idx] = wvalue[current_block_index_begin + idx];
		}

		// 同步，写 shared memory 与 读 shared memory
		
		for(int batch_idx = 0; batch_idx < (mybatch + batch_per_round - 1) / batch_per_round; ++batch_idx) { // batch 分段执行
			for(int col = weight_block * weight_block_size; col < weight_block * weight_block_size + weight_block_size; ++ col) {  // thread y
				//最大 thread 个数：batch_block_num * weight_per_block * reg_num;
				
				// wrap 任务，划分 reg_num * batch_block_num 个线程
				float reg [batch_per_round] = {0}; // 对于 batch 来说，连续访问，空间局部性好；

 				for(int idx = wdispl[col]; idx < wdispl[col + 1]; ++idx) { // 同一个 idx 的会更新同一块区域
					for(int i = 0; i < batch_per_round; ++i) { // thread x 
						if(batch_idx * batch_per_round + i < mybatch) {
							reg[i] += shared[idx - current_block_index_begin] * currfeat[windex[idx] * mybatch + batch_idx * batch_per_round + i];
							//printf("%d %f\n", i, reg[i]);
						}
					}
				}
				// 写回分段后 batch 的结果
				int batch_id_begin = batch_idx * batch_per_round;
				for(int i = batch_id_begin; i < batch_id_begin + batch_per_round; ++i) {
					if(i < mybatch) {
						if(nextfeat[mybatch * col + i] = ReLU(reg[i - batch_id_begin] + bias)) {
							active[i] += 1;
						}
					}	
				}
			
			}
		}
	}
}

void infer_serial(int l) {
	VALPREC *csr_val = csrvalue[l];
    INDPREC *csr_index = csrindex[l];
	int *csr_bias = csrdispl[l];

	for(int i = 0; i < mybatch; ++i) {
		active[i] = 0;
	}

	kernel_serial(nextfeat, currfeat, csr_bias, csr_index, csr_val, bias, active);
	
	int feature = 0;
	
	for(int i = 0; i < mybatch; ++i) {
        if(active[i]) {
			feature ++;
		}
	}
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
	FEATPREC *tempfeat = currfeat;
    currfeat = nextfeat;
    nextfeat = tempfeat;
	printf("real count = %d\n", feature);
}

__device__ inline void __float4Timesfloat(float4 a, float b, float4& c) {
	c.x += a.x * b;
	c.y += a.y * b;
	c.z += a.z * b;
	c.w += a.w * b;
}


__device__ inline void __float4AddfloatReLU(float4 a, float b, float4& c) {
	c.x = __ReLU(a.x + b);
	c.y = __ReLU(a.y + b);
	c.z = __ReLU(a.z + b);
	c.w = __ReLU(a.w + b);
}

__device__ inline void __float2Timesfloat(float2 a, float b, float2& c) {
	c.x += a.x * b;
	c.y += a.y * b;
}


__device__ inline void __float2AddfloatReLU(float2 a, float b, float2& c) {
	c.x = __ReLU(a.x + b);
	c.y = __ReLU(a.y + b);
}

template <typename T>
__device__ __forceinline__ T Load(const T* address) {
  return __ldg(address);
}

inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__device__ __forceinline__ void FMA(float x1, float x2, float* out) {
	out[0] += x1 * x2;
}

static __device__ __forceinline__ void FMA(float4 x2, float x1, float4 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
    out[0].z += x1 * x2.z;
    out[0].w += x1 * x2.w;
  }


__global__ void sputnik_kernel(FEATPREC *nextfeat, FEATPREC *currfeat, int *wdispl, INDPREC *windex, VALPREC *wvalue, float bias, int *active, int mybatch){
	const int col_index = blockIdx.x * 4 + threadIdx.y; // 每个 block 处理 4 列，每个 thread 只处理一列
	const int feat_index = blockIdx.y * 32; // 每个 block 处理 32 个 feat
	if(feat_index > mybatch) return; 
	
	const int row_offset = Load(wdispl + col_index); // 要处理的行的 开始位置
	const int nnzs = Load(wdispl + col_index + 1) - row_offset; // 一定等于 32
	
	//-------------
	const int sparse_tile_size = 4 * 32 / 4; // 处理的 4 行，每行 32 个数，每个数站4个位
	__shared__ int4 column_indices_tile_array[sparse_tile_size]; // 存储 所需要的下标
	__shared__ float4 values_tile_array[sparse_tile_size]; // 存储 所需要的值
	
	float4* values_tile = values_tile_array + 32 / 4 * threadIdx.y; // 这一个 thread 所要处理的 列 起始地址，每一行 8 个
	int4* column_indices_tile = column_indices_tile_array + 32 / 4 * threadIdx.y; 
	
	float4* sparse_values_p = reinterpret_cast<float4*>(wvalue + row_offset) + threadIdx.x; // load 数据时，每个 thread 相应的指针（原始数据位置）
	int4* sparse_indexs_p = reinterpret_cast<int4*>(windex + row_offset) + threadIdx.x;

	float4 *sparse_values_pp = sparse_values_p;
	int4* sparse_indexs_pp = sparse_indexs_p;
	
	float4* values_tile_p = values_tile + threadIdx.x; // load 数据时，每个 thread 相应的指针（目标数据位置）
	int4* column_indices_tile_p = column_indices_tile + threadIdx.x; 

	//-------------
	const int dense_tile_size = 32 * 32 / 8 / 4; // 每个 block 处理 一项 feat的 32 个位置
	__align__(16) float4 dense_matrix_tile_array[dense_tile_size]; // 一个 thread 处理 4 项 feat
	float4* dense_tile_array = reinterpret_cast<float4*>(dense_matrix_tile_array); // 指针

	float4* dense_value_p = reinterpret_cast<float4*>(currfeat + feat_index) + threadIdx.x; // 原始数据 开始位置


	//-------------
	const int output_tile_size = 32 / 8; // 每个 thread 得到 4 个 结果
	__align__(16) float output_matrix_tile_array[output_tile_size] = {0};


	
	__syncthreads();
	#pragma unroll
	for(int i = 0; i < 32 / blockDim.x / 4; ++i) {
		*(values_tile_p) = Load(sparse_values_pp);
		*(column_indices_tile_p) = Load(sparse_indexs_pp) * (mybatch / 4);

		sparse_values_pp += blockDim.x;
		sparse_indexs_pp += blockDim.x;
		values_tile_p += blockDim.x;
		column_indices_tile_p += blockDim.x;
	}
	
	__syncthreads();
	#pragma unroll
	for(int i = 0; i < 32 / 4; ++i) { // 每个 feat 加载 32 项
		int* col_offset = reinterpret_cast<int*>(column_indices_tile + i);
		for(int k = 0; k < 4; ++k) { // 加载 与 稀疏数据位置相对应的 稠密数据 位置
			int offset = col_offset[k];
			float4* dense_value_pp = dense_value_p + offset; // offset偏移了feat的多少项
			for(int j = 0; j < 32 / blockDim.x / 4; ++j) {
				int off = (i * 4 + k) * (32 / blockDim.x / 4) + j;
				dense_tile_array[off] = Load(dense_value_pp); // 每一项加载了连续存储的 4 个 feat
				dense_value_pp += blockDim.x;
			}
		}
	}

	float* sparse_value = reinterpret_cast<float*>(values_tile); // 开始计算 MAC，要处理的 列
	
	for(int i = 0; i < 32; ++i) { // 每一列 32 个元素
		float* dense_value = reinterpret_cast<float*>(dense_tile_array + i); // bug here!!! 简单写法
		#pragma unroll
		for(int k = 0; k < 4; ++k) { // 每个元素与 4 个 feat 做操作
			#pragma unroll
			for(int j = 0; j < 32 / blockDim.x / 4; ++j) {
				float* outputs = output_matrix_tile_array + j * 4 + k; // maybe bug here!!!
				FMA(dense_value[k], sparse_value[i], outputs);
			}
		}
	}

	for(int i = 0; i < 32 / blockDim.x / 4; ++i) {
		for(int j = 0; j < 4; ++j) {
			if(nextfeat[col_index * mybatch + feat_index + threadIdx.x * 4 + j] = __ReLU(output_matrix_tile_array[i * 4 + j] + bias)) {
				active[feat_index + threadIdx.x * 4 + j] = 1; // bug here!!!
			}
		} 
	}
}


__global__ void sputnik_kernel2(FEATPREC *nextfeat, FEATPREC *currfeat, int *wdispl, INDPREC *windex, VALPREC *wvalue, float bias, int *active, int mybatch){
	const int col_index = blockIdx.x * 32 + threadIdx.y; // 每个 block 处理 4 列，每个 thread 只处理一列
	const int row_offset = Load(wdispl + col_index); // 要处理的行的 开始位置
	const int nnzs = Load(wdispl + col_index + 1) - row_offset; // 一定等于 32
	
	//-------------
	const int sparse_tile_size = 32 * 32; // 处理的 4 行，每行 32 个数，每个数站4个位
	__shared__ int column_indices_tile_array[sparse_tile_size]; // 存储 所需要的下标
	__shared__ float values_tile_array[sparse_tile_size]; // 存储 所需要的值
	
	float* values_tile = values_tile_array + 32 * threadIdx.y; // 这一个 thread 所要处理的 列 起始地址，每一行 8 个
	int* column_indices_tile = column_indices_tile_array + 32 * threadIdx.y; 
	
	float4* sparse_values_p = reinterpret_cast<float4*>(wvalue + row_offset) + threadIdx.x; // load 数据时，每个 thread 相应的指针（原始数据位置）
	int4* sparse_indexs_p = reinterpret_cast<int4*>(windex + row_offset) + threadIdx.x;

	float4 *sparse_values_pp = sparse_values_p;
	int4* sparse_indexs_pp = sparse_indexs_p;
	
	float4* values_tile_p = reinterpret_cast<float4*>(values_tile) + threadIdx.x; // load 数据时，每个 thread 相应的指针（目标数据位置）
	int4* column_indices_tile_p = reinterpret_cast<int4*>(column_indices_tile) + threadIdx.x; 

	//-------------
	const int dense_tile_size = 32 * 32 / 8; // 每个 block 处理 一项 feat的 32 个位置
	__align__(16) float dense_matrix_tile_array[dense_tile_size]; // 一个 thread 处理 4 项 feat
	float4* dense_tile_array = reinterpret_cast<float4*>(dense_matrix_tile_array); // 指针

	//-------------
	const int output_tile_size = 32 / 8; // 每个 thread 得到 4 个 结果
	__align__(16) float output_matrix_tile_array[output_tile_size] = {0};
	
	__syncthreads();
	#pragma unroll
	for(int i = 0; i < 32 / blockDim.x / 4; ++i) {
		*(values_tile_p) = Load(sparse_values_pp);
		*(column_indices_tile_p) = Load(sparse_indexs_pp) * (mybatch / 4);

		sparse_values_pp += blockDim.x;
		sparse_indexs_pp += blockDim.x;
		values_tile_p += blockDim.x;
		column_indices_tile_p += blockDim.x;
	}
	#pragma unroll
	for(int f = 0; f < 2; ++f) {
		int feat_index = blockIdx.y * 32 * 2 + f * 32;
		if(feat_index > mybatch) return; 
		__syncthreads();
		float4* dense_value_p = reinterpret_cast<float4*>(currfeat + feat_index) + threadIdx.x; // 原始数据 开始位置
		#pragma unroll
		for(int i = 0; i < 32; ++i) { // 每个 feat 加载 32 项
			int* col_offset = reinterpret_cast<int*>(column_indices_tile + i);
			#pragma unroll
			for(int k = 0; k < 1; ++k) { // 加载 与 稀疏数据位置相对应的 稠密数据 位置
				float4* dense_value_pp = reinterpret_cast<float4*>(dense_value_p + col_offset[k]); // offset偏移了feat的多少项
				#pragma unroll
				for(int j = 0; j < 1; ++j) {
					int off = (i * 1 * 1) + k * 1 + j;
					dense_tile_array[off] = Load(dense_value_pp); // 每一项加载了连续存储的 4 个 feat
					dense_value_pp += blockDim.x;
				}
			}
		}

		float* sparse_value = reinterpret_cast<float*>(values_tile); // 开始计算 MAC，要处理的 列
		float4* dense_value = reinterpret_cast<float4*>(dense_tile_array); // bug here!!! 简单写法
		#pragma unroll
		for(int i = 0; i < 32; ++i) { // 每一列 32 个元素
			float* lhs_values = (sparse_value + i);
			#pragma unroll
			for(int k = 0; k < 1; ++k) { // 每个元素与 4 个 feat 做操作
				#pragma unroll
				for(int j = 0; j < 1; ++j) {
					float4* outputs = reinterpret_cast<float4*>(output_matrix_tile_array + j * 4 * 1); // maybe bug here!!!
					int rhs_offset = j * 1 * 1 + k * 1 + i;
					FMA(dense_value[rhs_offset], lhs_values[k], outputs);
				}
			}
		}

		#pragma unroll
		for(int i = 0; i < 32 / blockDim.x / 4; ++i) {
			#pragma unroll
			for(int j = 0; j < 4; ++j) {
				if(
					nextfeat[col_index * mybatch + feat_index + threadIdx.x * 4 + j] = __ReLU(output_matrix_tile_array[i * 4 + j] + bias)
				){
					active[feat_index + threadIdx.x * 4 + j] = 1; // bug here!!!
				}
				output_matrix_tile_array[i * 4 + j] = 0;
			} 
		}
	}
}


__global__ void dummy_kernel(FEATPREC *nextfeat, FEATPREC *currfeat, int *wdispl, INDPREC *windex, VALPREC *wvalue, float bias, int *active, int mybatch){
	extern __shared__ float shared[];

	const int weight_block_size = blockDim.x / 32;

	const int batch_per_round = 32 * 4;

	const int weight_block = blockIdx.x;

	const int current_block_index_begin = wdispl[weight_block * weight_block_size];
	
	const int current_block_index_end = wdispl[(weight_block + 1) * (weight_block_size)];
	
	const int current_block_index_size = current_block_index_end - current_block_index_begin;
	
	int* shared_index = (int*)(shared + 32 * weight_block_size);
	
	for(int idx = threadIdx.x; idx < current_block_index_size; idx += blockDim.x) {
		shared[idx] = wvalue[current_block_index_begin + idx];
		shared_index[idx] = windex[current_block_index_begin + idx];
	}
	__syncthreads();

	int col = threadIdx.x / 32 + blockIdx.x * weight_block_size;
	int thread_idx = threadIdx.x % 32;

	for(int batch_idx = blockIdx.y * mybatch / (batch_per_round) / gridDim.y; 
		batch_idx < (blockIdx.y + 1) * (mybatch + batch_per_round - 1) / (batch_per_round) / gridDim.y;
		batch_idx += 1) { // 展开此循环
			float4 reg[1]  = {0};
			for(int idx = wdispl[col]; idx < wdispl[col + 1]; idx += 4) { // 同一个的 idx 会更新同一块区域
				float4 weight_reg = reinterpret_cast<float4*>(shared)[(idx - current_block_index_begin) / 4]; 
				int4 weight_idx_reg = reinterpret_cast<int4*>(shared_index)[(idx - current_block_index_begin) / 4]; 

				if((batch_idx + 0) * batch_per_round + thread_idx * 4 + 1 < mybatch) {
					for(int unroll = 0; unroll < 1; ++unroll) {
						int feature_id = (batch_idx + unroll) * batch_per_round;
						auto feat_p0 = reinterpret_cast<float4*>(currfeat 
							+ weight_idx_reg.x * mybatch + feature_id);
						auto feat_p1 = reinterpret_cast<float4*>(currfeat 
							+ weight_idx_reg.y * mybatch + feature_id);
						auto feat_p2 = reinterpret_cast<float4*>(currfeat 
							+ weight_idx_reg.z * mybatch + feature_id);
						auto feat_p3 = reinterpret_cast<float4*>(currfeat 
							+ weight_idx_reg.w * mybatch + feature_id);
						//__float4Timesfloat(feat_p[thread_idx], shared[idx - current_block_index_begin], reg);
						__float4Timesfloat(feat_p0[thread_idx], weight_reg.x, reg[unroll]);
						__float4Timesfloat(feat_p1[thread_idx], weight_reg.y, reg[unroll]);
						__float4Timesfloat(feat_p2[thread_idx], weight_reg.z, reg[unroll]);
						__float4Timesfloat(feat_p3[thread_idx], weight_reg.w, reg[unroll]);
					}
				}
			}
			// 写回分段后 batch 的结果
			if((batch_idx + 0) * batch_per_round + thread_idx * 4 + 1 < mybatch) {
				for(int unroll = 0; unroll < 1; ++unroll) {
					int feature_id = (batch_idx + unroll) * batch_per_round;
					auto feat_p = reinterpret_cast<float4*>(nextfeat + mybatch * col + feature_id);

					__float4AddfloatReLU(reg[unroll], bias, feat_p[thread_idx]);
					
					if(feat_p[thread_idx].x) {
						active[(batch_idx + unroll) * batch_per_round + thread_idx * 4] =  1;
					}
					if(feat_p[thread_idx].y) {
						active[(batch_idx + unroll) * batch_per_round + thread_idx * 4 + 1] =  1;
					}
					if(feat_p[thread_idx].z) {
						active[(batch_idx + unroll) * batch_per_round + thread_idx * 4 + 2] =  1;
					}
					if(feat_p[thread_idx].w) {
						active[(batch_idx + unroll) * batch_per_round + thread_idx * 4 + 3] =  1;
					}
				}
			}
        }
};

void setup_gpu(){
	mybatch = batch;
	cudaSetDevice(myid % 2); // 每个 node 两张 GPU
	cudaFuncSetAttribute(dummy_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);

	OR_FATAL(cudaEventCreate(&kernelstart));
	OR_FATAL(cudaEventCreate(&kernelstop));
	OR_FATAL(cudaEventCreate(&copystart));
	OR_FATAL(cudaEventCreate(&copystop));
	OR_FATAL(cudaStreamCreate(&copystream));
	OR_FATAL(cudaStreamCreate(&kernelstream));

	OR_FATAL(cudaMallocHost((void**)&active,sizeof(int)*mybatch));
	OR_FATAL(cudaMalloc((void**)&active_d,sizeof(int)*mybatch));
	
	for(int k = 0; k < mybatch; k++){
		active[k] = neuron;
	}

	OR_FATAL(cudaMemset(active_d,0,sizeof(int)*mybatch));
	
	csrdispl_d = new int*[layer];
	csrindex_d = new INDPREC*[layer];
	csrvalue_d = new VALPREC*[layer];
	

	for(int l = 0; l < layer; l++){
		OR_FATAL(cudaMalloc((void**)&csrdispl_d[l], sizeof(int) * (neuron+1)));
		OR_FATAL(cudaMemcpy(csrdispl_d[l], csrdispl[l], sizeof(int) * (neuron+1), cudaMemcpyHostToDevice));
		
		OR_FATAL(cudaMalloc((void**)&csrindex_d[l],sizeof(INDPREC) * csrdispl[l][neuron]));
		OR_FATAL(cudaMalloc((void**)&csrvalue_d[l],sizeof(VALPREC) * csrdispl[l][neuron]));
		
		OR_FATAL(cudaMemcpy(csrindex_d[l], csrindex[l], sizeof(INDPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));
		OR_FATAL(cudaMemcpy(csrvalue_d[l], csrvalue[l], sizeof(VALPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));
	}

	OR_FATAL(cudaMalloc((void**)&indbuff_d,sizeof(INDPREC)* neuron * 32));
	OR_FATAL(cudaMalloc((void**)&valbuff_d,sizeof(VALPREC)* neuron * 32));

	OR_FATAL(cudaMalloc((void**)&currfeat_d, sizeof(FEATPREC) * mybatch * neuron));
	OR_FATAL(cudaMalloc((void**)&nextfeat_d, sizeof(FEATPREC) * mybatch * neuron));

	OR_FATAL(cudaMemset(currfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));
	OR_FATAL(cudaMemset(nextfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));

	OR_FATAL(cudaMemcpy(currfeat_d, currfeat, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyHostToDevice));
}

void infer_gpu(int l){

	int weight_block_size = 1;

	int batch_per_round = 32;

	int batch_block_size = mybatch / (batch_per_round * 4) + 1;
	
	int shared_memory_size = 32 * weight_block_size * sizeof(float);

	dim3 block(8, 32);
	dim3 grid(neuron / 32, (mybatch / (32 * 2) + 1));

	indbuff_d = csrindex_d[l];
	valbuff_d = csrvalue_d[l];
	
	auto startkernel = sc::now();
	
	OR_FATAL(cudaMemsetAsync(active_d, 0, sizeof(int) * mybatch, kernelstream));
	OR_FATAL(cudaEventRecord(kernelstart,kernelstream));

	sputnik_kernel2<<<grid, block, 0, kernelstream>>>(nextfeat_d, currfeat_d, csrdispl_d[l], indbuff_d, valbuff_d,bias, active_d, mybatch);
	OR_FATAL(cudaEventRecord(kernelstop,kernelstream));

	OR_FATAL(cudaMemcpyAsync(active, active_d, sizeof(int) * mybatch, cudaMemcpyDeviceToHost, kernelstream));
	OR_FATAL(cudaStreamSynchronize(kernelstream));

	timekernel += sc::now() - startkernel;	

	int feature = 0;

	// for(int i = 0; i < mybatch; ++i) {
	// 	active[i] = 0;
	// }
	
	// OR_FATAL(cudaMemcpy(nextfeat, nextfeat_d, neuron * mybatch  * sizeof(float), cudaMemcpyDeviceToHost));

	// for(int i = 0; i < mybatch; ++i) {
	// 	for(int j = 0; j < neuron; ++j) {
	// 	if(nextfeat[j * mybatch + i]) {
	// 			active[i] = 1;
	// 		}
	// 	}
	// }
		
	for(int i = 0; i < mybatch; ++i) {
        if(active[i]) {
			feature ++;
        }
    }
    
    int alignment_bias = (feature % 4 == 0) ? 0 : (4 - feature % 4);
    feature += alignment_bias;

	if(feature != mybatch) {
		OR_FATAL(cudaMemcpy(nextfeat, nextfeat_d, neuron * mybatch  * sizeof(float), cudaMemcpyDeviceToHost));
		for(int j = 0; j < neuron; ++j) {
			int curr_feature = 0;
			for(int i = 0; i < mybatch + alignment_bias; ++i) {
				if(i < mybatch && active[i]) {
					nextfeat_tmp[j * feature + curr_feature] = nextfeat[j * mybatch + i];
					curr_feature++;
                }
                if(i >= mybatch) {
					nextfeat_tmp[j * feature + curr_feature] = 0;
                    curr_feature++;
                }
			}
        }
		OR_FATAL(cudaMemcpy(nextfeat_d, nextfeat_tmp, neuron * feature  * sizeof(float), cudaMemcpyHostToDevice));
	}
	mybatch = feature;
	FEATPREC *tempfeat_d = currfeat_d;
	currfeat_d = nextfeat_d;
	nextfeat_d = tempfeat_d;
	printf("real count = %d\n", feature);
};