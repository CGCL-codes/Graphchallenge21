#pragma once
#include "matrix_block_container.h"

namespace ftxj {
    class GpuBlock {
        int block_idx_;
        int block_idy_;
        BlockContainer blocks_;
    public:
        GpuBlock(int x, int y, BlockContainer blocks) {
            block_idx_ = x;
            block_idy_ = y;
            blocks_ = blocks;
        }
    };
};