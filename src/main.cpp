#include "utils/header.h"
#include "reorder/header.h"
#include "inspector/header.h"
#include "gpu_lib/header.h"
#include "microbenchmark/header.h"


#include <functional>

using namespace ftxj;

int main() {

    std::cout << "begin" << std::endl;


    COOMatrix coo("../data/neuron1024/n1024-l120.tsv", 1, true);
    // COOMatrix coo("../data/uiuc-paper-example.txt", 0, true);
    std::cout << "coo success" << std::endl;

    HashReorder hash_reorder(64, 1024);
    coo.reorder(hash_reorder);
    // coo.save_matrix("reorder_matrix.txt")
    std::cout << "reorder success" << std::endl;

    
    CSRCSCMatrix csr_csc(coo);
    std::cout << "csr_csc success" << std::endl;

    // csr_csc.print_csc();
    // std::cout << "---------------" << std::endl;
    // csr_csc.print_csr();
    

    // UIUCMatrix uiuc(csr_csc);
    // std::cout << "uiuc success" << std::endl;

    // uiuc.print_buffdispl();
    // uiuc.print_mapdispl();
    // uiuc.print_map();
    // uiuc.print_warpdispl();
    // uiuc.print_warpindex();
    
    BlockContainer blocks(csr_csc, SparseMatrixBlockGen::naive_method);
    std::cout << "block container success" << std::endl;

    MaxInReuseBSchedule schedule(blocks);
    schedule.schedule();
    std::cout << "block schedule succ" << std::endl;
    
    auto data = schedule.get_data();

    // schedule.print_schedule();


    GpuEnv env(0);
    
    // vector4_load_data_benchmark(env);
    // vector4_load_data_benchmark(env);

    // uiuc_test_benchmark(coo, uiuc, env);
    test_shared_memory_mm(coo, data.value, data.row_access, env);

    return 0;
}