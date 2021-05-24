#include "utils/header.h"
#include "reorder/header.h"
#include "inspector/header.h"
#include "gpu_lib/header.h"
#include "microbenchmark/header.h"
#include "fuse/header.h"

#include <functional>

using namespace ftxj;

int main() {

    std::cout << "begin" << std::endl;


    COOMatrix coo("../data/neuron16384/n16384-l120.tsv", 1, true);
    // COOMatrix coo_2("../data/neuron16384/n16384-l119.tsv", 1, true);


    // HashReorder hash_reorder_t(1024, 16384);
    // coo.reorder(hash_reorder_t);
    // coo_2.reorder(hash_reorder_t);

    // std::vector<std::vector<int>> block_cols(16384/16);
    // for(int b = 0; b < 16384 / 16; ++b) {
    //     for(int j = 0; j < 16; ++j) {
    //         block_cols[b].push_back(b * 16 + j);
    //     }
    // }

    // FuseLayer fuse(coo, block_cols);

    // fuse.print_need_access();

    // fuse.fuse(coo_2);

    // fuse.print_need_access();
    // return 0;

    // COOMatrix coo("../data/uiuc-paper-example.txt", 0, true);
    std::cout << "coo success" << std::endl;

    HashReorder hash_reorder(1024, 16384);
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
    naive_load_data_benchmark(env);

    // uiuc_test_benchmark(coo, uiuc, env);
    // test_shared_memory_mm(coo, data.value, data.row_access, env);

    return 0;
}