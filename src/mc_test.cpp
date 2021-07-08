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


    std::map<int, int> hash_map = {
        {65536, 4096},
        {16384, 1024},
        {4096, 256},
        {1024, 64}
    };
    int neuron = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int l = atoi(argv[3]);
    int hash_type = atoi(argv[4]);
    
    std::string file_name = "../data/neuron"+ 
        std::to_string(neuron) + "/n" + std::to_string(neuron) +"-l" + std::to_string(l) + ".tsv";
    COOMatrix coo(file_name, 1, false);
    std::cout << "read coo success" << std::endl;

    if(hash_type == 0) {
    }

    if(hash_type == 1) {
        HashReorder hash_reorder_t(hash_map[neuron], neuron, REORDER::ROW_REORDER);
        coo.reorder(hash_reorder_t);
    }
    
    if(hash_type == 2) {
        HashReorder hash_reorder_t(hash_map[neuron], neuron, REORDER::COL_REORDER);
        coo.reorder(hash_reorder_t);
    }

    if(hash_type == 3) {
        HashReorder hash_reorder_t(hash_map[neuron], neuron, REORDER::ALL_REORDER);
        coo.reorder(hash_reorder_t);
    }
    
    std::cout << "reorder success" << std::endl;
    
    CSRCSCMatrix csr_csc(coo);
    std::cout << "coo to csr_csc success" << std::endl;

    UIUCMatrix uiuc(csr_csc, 256, neuron);
    std::cout << "uiuc success" << std::endl;

    GpuEnv env(0);

    test_benchmark_20_uiuc(coo, uiuc, batch, env);
    return 0;


    return 0;
}