#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include "vars.h"

int main(int argc, char *argv[]) {
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int task_count;
    int rank;
    int len;
    int ret;

    ret = MPI_Init(&argc, &argv);
    if (MPI_SUCCESS != ret) {
        printf("start mpi fail\n");
        MPI_Abort(MPI_COMM_WORLD, ret);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &task_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);

    printf("task_count = %d, my rank = %d on %s\n", task_count, rank, hostname);
    
    float esp_time_cpu;
	clock_t start_cpu, stop_cpu;

    start_cpu = clock();// start timing

    handle(rank);//在此调用用cuda写的函数
    stop_cpu = clock();// end timing

	esp_time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;

	printf("The time by host:\t%f(ms)\n", esp_time_cpu);

    MPI_Finalize();

    return 0;
}