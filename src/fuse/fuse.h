#pragma once

#include "../utils/header.h"
#include <vector>
#include <algorithm>

namespace ftxj {
    class FuseLayer {
        int fuse_numbers_;
        std::vector<std::set<int>> input_need_access_;
        std::vector<COOMatrix> fused_matrix_;
    public:
        FuseLayer(COOMatrix outer_matrix, std::vector<std::vector<int>> block_cols) {
            fuse_numbers_ = 1;
            input_need_access_ = std::vector<std::set<int>>(block_cols.size(), std::set<int>());
            for(int i = 0; i < block_cols.size(); ++i) {
                for(int j = 0; j < block_cols[i].size(); ++j) {
                    int need_access_col = block_cols[i][j];
                    for(auto x = outer_matrix.begin(); x != outer_matrix.end(); ++x) {
                        if((*x).col == need_access_col) {
                            input_need_access_[i].insert((*x).row);
                        }
                    }
                }
            }
        }

        void print_need_access() {
            for(int b = 0; b < input_need_access_.size(); ++b) {
                std::cout << "block b = " << b << ",size = "<< input_need_access_[b].size() <<" : ";
                for(auto x : input_need_access_[b]) {
                    std::cout << x << ",";
                }
                std::cout << std::endl;
            }
        }

        void fuse(COOMatrix outer_matrix) {
            fuse_numbers_ += 1;
            fused_matrix_.push_back(outer_matrix);
            std::vector<std::set<int>> old_access = input_need_access_;
            input_need_access_.clear();
            input_need_access_ = std::vector<std::set<int>>(old_access.size(), std::set<int>());
            for(int b = 0; b < old_access.size(); ++b) {
                for(auto row : old_access[b]) {
                    for(auto x = outer_matrix.begin(); x != outer_matrix.end(); ++x) {
                        if((*x).col == row) {
                            input_need_access_[b].insert((*x).row);
                        }
                    }
                }
            }
        }
    };
}