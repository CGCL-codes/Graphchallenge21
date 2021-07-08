#include "utils/header.h"
#include "reorder/header.h"
#include "inspector/header.h"
#include "gpu_lib/header.h"
#include "microbenchmark/header.h"
#include "fuse/header.h"

#include <functional>
#include <algorithm>
using namespace ftxj;

int main(int argc, char* argv[]) {

    std::cout << "begin" << std::endl;
    std::map<int, int> stride_map = {
        {1, 16},
        {2, 32},
        {3, 64},
        {4, 128},
        {5, 256},
        {6, 512},
        {7, 1024},
        {8, 2048},
        {9, 4096},
        {10, 8192}
    };

    int neuron = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int l = atoi(argv[3]);
    int hash_type = atoi(argv[4]);


    int TN = atoi(argv[5]);
    int blockx = atoi(argv[6]);
    int blocky = atoi(argv[7]);

    std::string file_name = "../data/neuron"+ 
        std::to_string(neuron) + "/n" + std::to_string(neuron) +"-l" + std::to_string(l) + ".tsv";

    COOMatrix coo(file_name, 1, false);
    COOMatrix coo_cpu(file_name, 1, false);
    std::cout << "read coo success" << std::endl;


    if(hash_type == 0) {
    }

    if(hash_type == 1) {
        HashReorder hash_reorder_t(64, neuron, REORDER::ROW_REORDER);
        coo.reorder(hash_reorder_t);
        coo_cpu.reorder(hash_reorder_t);
    }
    
    if(hash_type == 2) {
        HashReorder hash_reorder_t(64, neuron, REORDER::COL_REORDER);
        coo.reorder(hash_reorder_t);
        coo_cpu.reorder(hash_reorder_t);
    }

    if(hash_type == 3) {
        HashReorder hash_reorder_t(64, neuron, REORDER::ALL_REORDER);
        coo.reorder(hash_reorder_t);
        coo_cpu.reorder(hash_reorder_t);
    }
    
    std::cout << "reorder success" << std::endl;
    BFMatrix bf(coo, neuron, TN);
    std::cout << "BF success" << std::endl;

    GpuEnv env(0);
    // test_benchmark_succ_load_store(batch, neuron, env);
    // test_benchmark_matrix_transpose(batch, neuron, env); 
    // test_benchmark_matrix_transpose_and_delete(batch, neuron, env);
    // return 0;

    test_benchmark_19_BF(
        coo,  bf, 
        neuron, batch, TN, 
        blockx, blocky,
        env
    );
    return 0;
}