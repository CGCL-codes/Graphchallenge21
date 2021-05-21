#pragma once
#include "matrix_block_container.h"
#include <iostream>
#include <fstream>

namespace ftxj {
    class GpuBlock {
        int block_idx_;
        int block_idy_;
    public:
        BlockContainer blocks_;
        GpuBlock(int x, int y, BlockContainer blocks) : blocks_(blocks)  {
            block_idx_ = x;
            block_idy_ = y;
        }

        // std::vector<int> 
        void file_gen() {

        }

        void print() {
            std::cout << "(";
            if(block_idx_ == -1) {
                std::cout << "{...}, "; 
            }
            else {
                std::cout << block_idx_ << ", ";
            }
            if(block_idy_ == -1) {
                std::cout << "{...})"; 
            }
            else {
                std::cout << block_idy_ << ")";
            }
            std::cout << "\n";
            blocks_.print_unique();
        }
    };
};