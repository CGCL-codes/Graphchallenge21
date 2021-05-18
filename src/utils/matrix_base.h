#pragma once
#include <vector>
#include <iostream>
namespace ftxj {
    struct MatrixPos{
        int row_idx;
        int col_idx;
        MatrixPos(int r, int c) {row_idx = r; col_idx = c;}
        MatrixPos() {}
        void print() {
            std::cout << "(" << row_idx << "," << col_idx << ")";
        }
    };
};