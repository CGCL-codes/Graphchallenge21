#include "utils/header.h"
#include "reorder/header.h"
#include "inspector/header.h"
#include "fuse/header.h"
#include <functional>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace ftxj;


std::string get_weight_file_name(int neuron, int layer) {
    std::string weight_file_dir = "../data/neuron";
    std::string neuron_str = std::to_string(neuron);
    weight_file_dir += neuron_str + "/n" + neuron_str + "-l" + std::to_string(layer + 1) + ".tsv";
    return weight_file_dir;
}

int main(int argc, char* argv[]) {

    if(argc != 7) {
        std::cout << "Usage: exe neuron layer TB1 TN1 TB2 TN2" << std::endl;
        return 0;
    }
    int neuron = atoi(argv[1]);
    int layer = atoi(argv[2]);


    int TB1 = atoi(argv[3]);
    int TN1 = atoi(argv[4]);
    int TB2 = atoi(argv[5]);
    int TN2 = atoi(argv[6]);


    std::map<int, int> hash_map = {
        {65536, 4096},
        {16384, 1024},
        {4096, 256},
        {1024, 64}
    };

    std::map<int, float> type_1 = {
        {65536, 12},
        {16384, 10},
        {4096, 8},
        {1024, 6}
    };
   HashReorder hash_reorder_t(hash_map[neuron], neuron);

    std::cout << "[BEGIN]..." << std::endl;

    auto weight_file = get_weight_file_name(neuron, layer);
    COOMatrix coo(weight_file, 1, false);
    std::cout << "["<< weight_file << "] to COO success!" << std::endl;
    coo.reorder(hash_reorder_t);
    std::cout << "Reorder success!" << std::endl;
    coo.cost_analysis(TB1, TN1, TB2, TN2);
    return 0;
}