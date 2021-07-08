#include "utils/header.h"
#include "reorder/header.h"
#include "inspector/header.h"
// #include "gpu_lib/header.h"
#include "microbenchmark/multi_gpu/header.h"
#include "fuse/header.h"
#include <functional>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
using namespace ftxj;


std::string get_weight_file_name(int neuron, int layer) {
    std::string weight_file_dir = "../data/neuron";
    std::string neuron_str = std::to_string(neuron);
    weight_file_dir += neuron_str + "/n" + neuron_str + "-l" + std::to_string(layer + 1) + ".tsv";
    return weight_file_dir;
}

void dense_reorder(std::vector<std::vector<float>> &input, Reorder &reorder_class) {
    // std::vector<std::vector<float>> old = input;
    for(int i = 0; i < input.size(); ++i) {
        std::vector<float> tmp(input[i].size());
        for(int j = 0; j < input[i].size(); ++j) {
            auto new_j = reorder_class.reorder(j);
            tmp[new_j] = input[i][j];
        }
        input[i] = tmp;
    }
}

void read_input(std::vector<std::vector<float>> &input, int neuron, int batch) {
    std::string input_file_name = "../data/sparse-images-";
    input_file_name += std::to_string(neuron) + ".tsv";
    std::ifstream input_file(input_file_name);
    if(!input_file){
        std::cout << "FILE:" << input_file_name << " does not exists.\n";
        exit(-1);
    }
    int b, n;
    float val;
    long read_num = 0;
    while(input_file >> b >> n >> val) {
        if(b <= batch) {
            read_num++;
            input[b - 1][n - 1] = val;
            if(val != 1.00) {
                printf("read input %d, %f\n", b, val);
            }
        }
    }
    std::cout << "Read Input success! read_numeber = " << read_num << std::endl;
}

int main(int argc, char* argv[]) {
    
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int task_count;
    int rank;
    int len;
    int ret;


    int neuron = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int layer = atoi(argv[3]);
    // int nnzs = atoi(argv[4]);

    ret = MPI_Init(&argc, &argv);
    if (MPI_SUCCESS != ret) {
        printf("start mpi fail\n");
        MPI_Abort(MPI_COMM_WORLD, ret);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &task_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);

    if(rank == 0)
        printf("task_count = %d, my rank = %d on %s\n", task_count, rank, hostname);


    // int neuron = 16384;
    // int batch = 60000;
    // int layer = 1920;

    std::map<int, int> hash_map = {
        {65536, 4096},
        {16384, 1024},
        {4096, 256},
        {1024, 64}
    };

    std::map<int, float> bias_map = {
        {65536, -0.45},
        {16384, -0.4},
        {4096, -0.35},
        {1024, -0.3}
    };

    std::map<int, float> type_1 = {
        {65536, 12},
        {16384, 10},
        {4096, 8},
        {1024, 6}
    };

    std::vector<std::vector<float>> input(batch, std::vector<float>(neuron));
    std::vector<std::vector<float>> weight; 
    std::vector<std::vector<int>> row_access; 


    std::cout << "GPU[" << rank << "] " << "[BEGIN]..." << std::endl;
    read_input(input, neuron, batch);
    std::cout << "GPU[" << rank << "] " << "Read Input success!" << std::endl;
    HashReorder hash_reorder_t(hash_map[neuron], neuron);
    dense_reorder(input, hash_reorder_t);

    for(int l = 0; l < layer; ++l) {
        auto weight_file = get_weight_file_name(neuron, l);
        COOMatrix coo(weight_file, 1, false);
        std::cout << "GPU[" << rank << "] " << "["<< weight_file << "] to COO success!" << std::endl;
        coo.reorder(hash_reorder_t);
        std::cout << "GPU[" << rank << "] " << "Reorder success!" << std::endl;
        CSRCSCMatrix csr_csc(coo);
        csr_csc.transpose();
        BlockContainer blocks(csr_csc, SparseMatrixBlockGen::naive_method);
        std::cout << "GPU[" << rank << "] " << "Structural Info success!" << std::endl;
        MaxInReuseBSchedule schedule(blocks);
        if(l == 0) {
            schedule.schedule(16, 7);
        }
        else if(l < type_1[neuron]) {
            schedule.schedule_output_parallel(128, 1, false);
        }        
        else {
            schedule.schedule(128, 1);
        }
        std::cout << "GPU[" << rank << "] " << "Schedule succ" << std::endl;
        auto data = schedule.get_data(neuron);
        weight.push_back(data.value);
        row_access.push_back(data.row_access);
    }
    int gpu_id = 0;
    if(rank == 0) gpu_id = 0;
    if(rank == 1) gpu_id = 1;
    if(rank == 2) gpu_id = 2;
    if(rank == 3) gpu_id = 3;
    test_benchmark_multi_gpu_graph_challenge(input, weight, row_access, (batch + 1) / 2, neuron, bias_map[neuron], gpu_id, rank);
    std::cout << "GPU[" << rank << "] " <<"[END]..." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}