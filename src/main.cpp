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


    int neuron = 16384;
    int batch = 1900;

    COOMatrix coo("../data/neuron16384/n16384-l1.tsv", 1, false);
    // COOMatrix coo_2("../data/neuron16384/n16384-l119.tsv", 1, true);
    // std::cout << "read coo success" << std::endl;

    HashReorder hash_reorder_t(1024, neuron);
    coo.reorder(hash_reorder_t);
    // std::cout << "reorder success" << std::endl;

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
    
    CSRCSCMatrix csr_csc(coo);
    std::cout << "coo to csr_csc success" << std::endl;

    // UIUCMatrix uiuc(csr_csc, 256, neuron);
    // std::cout << "uiuc success" << std::endl;

    // GpuEnv env(0);
    // test_benchmark_20_uiuc(coo, uiuc,  batch, env);

    // uiuc_test_benchmark(coo, uiuc, env);
    // uiuc.print_buffdispl();
    // uiuc.print_mapdispl();
    // uiuc.print_map();
    // uiuc.print_warpdispl();
    // uiuc.print_warpindex();
    
    csr_csc.transpose();
    BlockContainer blocks(csr_csc, SparseMatrixBlockGen::naive_method);
    std::cout << "block container success" << std::endl;

    MaxInReuseBSchedule schedule(blocks);
    schedule.schedule();
    std::cout << "block schedule succ" << std::endl;
    
    auto data = schedule.get_data();

    // test_benchmark_row_succ_20_uiuc(coo, data.value, data.row_access, batch, neuron, env);
    // test_benchmark_row_succ_20_uiuc_transpose(coo, data.value, data.row_access, batch, neuron, env);
    // test_benchmark_row_succ_20_uiuc_transpose_no_conflict(coo, data.value, data.row_access, batch, neuron, env);
    


    schedule.print_schedule();


    // GpuEnv env(0);
    
    // vector4_load_data_benchmark(env);
    // test_benchmark_succ_load_store(batch, neuron, env);

    // test_shared_memory_mm(coo, data.value, data.row_access, env);

    return 0;
}