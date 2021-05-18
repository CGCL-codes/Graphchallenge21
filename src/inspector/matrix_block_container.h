#pragma once
#include <vector>
#include <string>
#include <functional>

#include "../utils/header.h"

#include "matrix_block.h"
#include "matrix_block_gen.h"

namespace ftxj {

    using namespace std;
    
    class BlockContainer {
        std::vector<MatrixBlockBase> blocks_;
        CSRCSCMatrix &csr_csc;
        
    public:
    
        BlockContainer(CSRCSCMatrix &matrix, std::vector<std::pair<MatrixPos, MatrixPos>> (*func)(CSRCSCMatrix &)) 
            : csr_csc(matrix) {
            auto pos_s = func(csr_csc);
            for(int i = 0; i < pos_s.size(); ++i) {
                std::cout << i << ", beg = ";
                pos_s[i].first.print();
                std::cout << ", end = ";
                pos_s[i].second.print();
                std::cout << "\n";
            }
        }
    };


};