run_multi_gpu:multi_gpu_cu.o multi_gpu_cpp.o
	mpicxx multi_gpu_cu.o multi_gpu_cpp.o -L/usr/local/cuda/lib64 -lcudart -o run_multi_gpu
multi_gpu_cu.o:./microbenchmark/multi_gpu/multi_gpu.cu
	nvcc -c ./microbenchmark/multi_gpu/multi_gpu.cu -o multi_gpu_cu.o
multi_gpu_cpp.o:multi_gpu.cpp
	mpicxx -c multi_gpu.cpp -o multi_gpu_cpp.o
clean:
	rm  -f  *.o