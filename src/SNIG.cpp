#include "utils/header.h"
#include "reorder/header.h"
#include "inspector/header.h"
#include "gpu_lib/header.h"
#include "microbenchmark/header.h"
#include "fuse/header.h"
#include <functional>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace ftxj;





size_t get_sec_size(const size_t num_neurons) {

    //only for the same GPUs
    //
    //get tuned shared memory size
    //num_neurons must be divisible by shared memory (a.k.a. sec_size)
    //only for double float
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    size_t sec_size{0};

    size_t max_num_per_block = props.sharedMemPerBlock / sizeof(float);
    if(num_neurons <= max_num_per_block) {
        sec_size = num_neurons;
    }
    else{
        int max_divisor = 2;
        while((num_neurons % max_divisor != 0) || 
            (max_num_per_block < (num_neurons / max_divisor))) {
        ++max_divisor;
        }
        sec_size = num_neurons / max_divisor;
    }
    return sec_size;
}

std::string get_weight_file_name(int neuron, int layer) {
    std::string weight_file_dir = "../data/neuron";
    std::string neuron_str = std::to_string(neuron);
    weight_file_dir += neuron_str + "/n" + neuron_str + "-l" + std::to_string(layer + 1) + ".tsv";
    return weight_file_dir;
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

    if(argc != 5) {
        std::cout << "Usage: exe neuron batch layer nnzs" << std::endl;
        return 0;
    }
    int neuron = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int layer = atoi(argv[3]);
    int nnzs = atoi(argv[4]);
    int sec_size = get_sec_size(neuron);

    std::cout << "[Config] sec size = " << sec_size << std::endl;
    std::map<int, float> bias_map = {
        {65536, -0.45},
        {16384, -0.4},
        {4096, -0.35},
        {1024, -0.3}
    };

    std::vector<std::vector<float>> input(batch, std::vector<float>(neuron));
    std::cout << "[BEGIN]..." << std::endl;
    read_input(input, neuron, batch);
    std::cout << "Read Input success!" << std::endl;
    std::vector<SNIGMatrix> weights;
    
    for(int l = 0; l < layer; ++l) {
        auto weight_file = get_weight_file_name(neuron, l);
        SNIGMatrix snig_weight(weight_file, 32 * neuron, sec_size, neuron);
        weights.push_back(snig_weight);
        std::cout << "["<< weight_file << "] to SNIG Matrix success!" << std::endl;
    }

    GpuEnv env(0);
    test_benchmark_SNIG(input, weights, batch, neuron, sec_size, nnzs, bias_map[neuron], env);
    
    std::cout << "[END]..." << std::endl;
    return 0;
}