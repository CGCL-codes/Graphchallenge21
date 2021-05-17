#pragma once
#include <vector>
#include <string>

#include "matrix_block.h"
#include "matrix_block_gen.h"
#include  "utils/matrix.h"

namespace ftxj {

    using namespace std;
    
    class BlockContainer {
        std::vector<MatrixBlockBase> blocks_;
        CSRCSCMatrix &csr_csc;
        
    public:
    
        void add_block()
    };


};