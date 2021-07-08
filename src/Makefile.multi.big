run_multi_gpu_big:run_multi_gpu_big_cu.o run_multi_gpu_big_cpp.o
	mpicxx run_multi_gpu_big_cu.o run_multi_gpu_big_cpp.o -L/usr/local/cuda/lib64 -lcudart -o run_multi_gpu_big
run_multi_gpu_big_cu.o:./microbenchmark/multi_gpu/multi_gpu_big.cu
	nvcc -c ./microbenchmark/multi_gpu/multi_gpu_big.cu -o run_multi_gpu_big_cu.o
run_multi_gpu_big_cpp.o:multi_gpu.cpp
	mpicxx -c multi_gpu.cpp -o run_multi_gpu_big_cpp.o
clean:
	rm  -f  *.o