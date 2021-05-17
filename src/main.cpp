#include "utils/matrix.h"
#include "inspector/header.h"


using namespace ftxj;

int main() {

    COOMatrix coo("../data/neuron1024/n1024-l120.tsv", 1);
    coo.save_matrix("tmp1.txt");
    HashReorder hash_reorder(64, 1024);
    
    coo.reorder(hash_reorder);

    coo.save_matrix("tmp.txt");

    CSRCSCMatrix csr_csc(coo);

    BlockContainer blocks(csr_csc, SparseMatrixBlockGen::naive_method);
    
    return 0;
}