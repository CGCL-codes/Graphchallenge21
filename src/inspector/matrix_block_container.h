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
        std::vector<std::pair<MatrixPos, MatrixPos>> pos_s;
        CSRCSCMatrix &csr_csc;
        

        bool same_col_;
    public:
        std::vector<int> access_row_idx;
        BlockContainer(CSRCSCMatrix &matrix, std::vector<std::pair<MatrixPos, MatrixPos>> &poss, bool same_col)
            : csr_csc(matrix) {
            pos_s = poss;
            
            for(int i = 0; i < pos_s.size(); ++i) {
                for(int r = pos_s[i].first.row_idx; r <= pos_s[i].second.row_idx; ++r) {
                    access_row_idx.push_back(r);
                }
            }

            same_col_ = same_col;
        }

        BlockContainer(CSRCSCMatrix &matrix, std::vector<std::pair<MatrixPos, MatrixPos>> (*func)(CSRCSCMatrix &), bool same_col = false) 
            : csr_csc(matrix) {
            pos_s = func(csr_csc);
            
            for(int i = 0; i < pos_s.size(); ++i) {
                for(int r = pos_s[i].first.row_idx; r <= pos_s[i].second.row_idx; ++r) {
                    access_row_idx.push_back(r);
                }
            }
            same_col_ = same_col;
            // for(int i = 0; i < pos_s.size(); ++i) {
            //     std::cout << i << ", beg = ";
            //     pos_s[i].first.print();
            //     std::cout << ", end = ";
            //     pos_s[i].second.print();
            //     std::cout << "\n";
            // }
        }

        void print() {
            for(int i = 0; i < pos_s.size(); ++i) {
                std::cout << i << ", beg = ";
                pos_s[i].first.print();
                std::cout << ", end = ";
                pos_s[i].second.print();
                std::cout << "\n";
            }
        }

        std::vector<BlockContainer> split_by_col() {
            std::vector<BlockContainer> res;
            int pre_col = 0;
            int pre_idx = 0;
            for(int i = 0; i < pos_s.size(); ++i) {
                if(pos_s[i].first.col_idx != pre_col) {
                    auto tmp = std::vector<std::pair<MatrixPos, MatrixPos>>(pos_s.begin() + pre_idx, pos_s.begin() + i);
                    res.push_back(BlockContainer(csr_csc, tmp, true));
                    std::cout << "-----------------------------------------------" << std::endl;
                    res[res.size() - 1].print();
                    pre_idx = i;
                    pre_col = pos_s[i].first.col_idx;
                }
            }
            auto tmp = std::vector<std::pair<MatrixPos, MatrixPos>>(pos_s.begin() + pre_idx, pos_s.end());
            res.push_back(BlockContainer(csr_csc, tmp, true));
            return res;    
        }

        std::pair<int, int> get_col_idx() {
            if(same_col_) {
                return {pos_s[0].first.col_idx, pos_s[0].second.col_idx};
            }
            return {-1, -1};
        }
    };


};