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


    int neuron = 16384;
    int batch = 1918;


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
    int l = atoi(argv[1]);
    // int l = 5;
    std::string file_name = "../data/neuron16384/n16384-l" + std::to_string(l) + ".tsv";
    COOMatrix coo(file_name, 1, false);
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

    GpuEnv env(0);
    // test_benchmark_succ_load_store(batch, neuron, env);
    // test_benchmark_matrix_transpose(batch, neuron, env); 
    test_benchmark_matrix_transpose_and_delete(batch, neuron, env);
    return 0;

    // test_benchmark_20_uiuc(coo, uiuc,  batch, env);
    // return 0;

    // uiuc_test_benchmark(coo, uiuc, env);
    // uiuc.print_buffdispl();
    // uiuc.print_mapdispl();
    // uiuc.print_map();
    // uiuc.print_warpdispl();
    // uiuc.print_warpindex();
    
    csr_csc.transpose();
    BlockContainer blocks(csr_csc, SparseMatrixBlockGen::naive_method);
    std::cout << "block container success" << std::endl;
    // blocks.print();

    MaxInReuseBSchedule schedule(blocks);
    
    // schedule.schedule_output_parallel(128, 1, false);
    schedule.schedule(128, 1);

    std::cout << "block schedule succ" << std::endl;
    
    // auto data = schedule.get_data2(neuron);
    auto data = schedule.get_data(neuron);
    

    // std::cout << "data size = " << data.value.size() << std::endl;
    // std::cout << "data access size = " << data.row_access.size() << std::endl;
    
    // std::cout << "data load idx len = ";
    // for(int i = 0; i < data.load_idx_row_len.size(); ++i) {
    //     std::cout << data.load_idx_row_len[i] << ", ";
    // }
    // std::cout << std::endl;
    
    // std::cout << "data row access = ";
    // for(int i = 0; i < data.row_access.size(); ++i) {
    //     std::cout << data.row_access[i] << ", ";
    // }
    // std::cout << std::endl;

    // std::cout << "data value access = ";
    // for(int i = 0; i < data.value_access.size(); ++i) {
    //     std::cout << data.value_access[i] << ", ";
    // }
    // std::cout << std::endl;

    // schedule.print_schedule();

    // test_benchmark_row_succ_20_uiuc(coo, data.value, data.row_access, batch, neuron, env);
    // test_benchmark_row_succ_20_uiuc_transpose(coo, data.value, data.row_access, batch, neuron, env);
    // test_benchmark_row_succ_20_uiuc_transpose_no_conflict(coo, data.value, data.row_access, batch, neuron, env);
    // test_benchmark_rectangels_batch_parallel_kernel(coo, data.value, data.row_access, batch, neuron, env);
    // test_benchmark_n16384_l2_l10_kernel(coo, data.value, stride_map[l], batch, neuron, env);
    test_benchmark_n16384_l11_kernel(coo, data.value, data.row_access, batch, neuron, env);



    // GpuEnv env(0);
    
    // vector4_load_data_benchmark(env);

    // test_shared_memory_mm(coo, data.value, data.row_access, env);

    return 0;
}