test:saxpy.o mpi_call.o
	mpicxx mpi_call.o saxpy.o -L/usr/local/cuda/lib64 -lcudart -o test
saxpy.o:saxpy.cu
	nvcc -c saxpy.cu -o saxpy.o
mpi_call.o:mpi_call.cpp
	mpicxx -c mpi_call.cpp -o mpi_call.o
clean:
	rm  -f  *.o