#include "utils/matrix.h"


using namespace ftxj;

int main() {

    COOMatrix coo("../data/neuron1024/n1024-l1.tsv", 1);
    HashReorder hash_reorder(64, 1024);
    coo.reorder(hash_reorder);
    coo.save_matrix("tmp.txt");
    return 0;
}