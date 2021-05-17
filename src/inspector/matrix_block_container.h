#pragma once
#include <vector>
#include <string>
#include <function>

#include "matrix_block.h"
#include "matrix_block_gen.h"
#include "utils/matrix.h"

namespace ftxj {

    using namespace std;
    
    class BlockContainer {
        std::vector<MatrixBlockBase> blocks_;
        CSRCSCMatrix &csr_csc;
        
    public:
    
        BlockContainer(COOMatrix &matrix, std::function<std::vector<std::pair<MatrixPos, MatrixPos>>(CSRCSCMatrix &csr_csc)> func) {
            auto pos_s = func(matrix);
            
        }
    };


};