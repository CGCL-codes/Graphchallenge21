#pragma once
#include <vector>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include "gpu_runtime.h"
#include <string>

namespace ftxj {
    class GpuEnv {
        std::vector<cudaStream_t> streams;
        std::vector<cudaEvent_t> start_event;
        std::vector<cudaEvent_t> stop_event;
        std::vector<std::string> event_name;
        
    public:
        void set_up(int gpu_id, bool print_device_info = true) {
            Safe_Call(cudaSetDevice(gpu_id));
            if(print_device_info) {
                int deviceCount;
                Safe_Call(cudaGetDeviceCount(&deviceCount));
                printf("\n");
                printf("Device Count: %d\n",deviceCount);
                int dev = 0;
                
                cudaDeviceProp deviceProp;
                Safe_Call(cudaGetDeviceProperties(&deviceProp, dev));
                printf("Device %d name: %s\n",dev,deviceProp.name);
                printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
                printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
                printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
                printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
                printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
                printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
                printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
                printf("Warp size: %d\n",deviceProp.warpSize);
                printf("\n");
            }
        }

        void add_event(std::string name = "non") {
            cudaStream_t stream;
            cudaEvent_t start, stop;

            streams.push_back(stream);
            start_event.push_back(start);
            stop_event.push_back(stop);
            event_name.push_back(name);

            Safe_Call(cudaEventCreate(&start_event[start_event.size() - 1]));
            Safe_Call(cudaEventCreate(&stop_event[stop_event.size() - 1]));
            Safe_Call(cudaStreamCreate(&streams[streams.size() - 1]));
        }
    };
};