#pragma once
#include <vector>

namespace ftxj {
    struct MatrixPos{
        int row_idx;
        int col_idx;
        MatrixPos(int r, int c) {row_idx = r; col_idx = c;}
        MatrixPos() {}
    };
};