#pragma once
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <set>

#include "../utils/header.h"

#include "matrix_block.h"
#include "matrix_block_gen.h"

namespace ftxj {

    using namespace std;
    
    class BlockContainer {
        std::vector<std::pair<MatrixPos, MatrixPos>> pos_s;
        CSRCSCMatrix &csr_csc;
        

        bool same_col_;

        void shrink_to_unique_value(std::vector<int> &vec) {
            std::set<int> s;
            for(auto i : vec) {
                s.insert(i);
            }
            vec.clear();
            for(auto i : s) {
                vec.push_back(i);
            }
        }
        
        void shrink_to_unique_value(std::vector<std::pair<int, int>> &vec) {
            auto cmp = [](const std::pair<int, int> &l, const std::pair<int, int> &r) {
                if(l.first < r.first) return true;
                return l.second < r.second;
            };
            std::set<std::pair<int, int>, decltype(cmp)> s(cmp);

            for(auto i : vec) {
                s.insert(i);
            }
            vec.clear();
            for(auto i : s) {
                vec.push_back(i);
            }
        }


    public:
        std::vector<int> access_row_idx;
        std::vector<int> access_col_idx;

        std::vector<std::pair<int, int>> access_unique_row_range;
        std::vector<std::pair<int, int>> access_unique_col_range;

        bool same_col() {
            
        }


        void print_unique() {
            std::cout << "row = {";
            for(auto x : access_unique_row_range) {
                if(x.second - x.first <= 3) {
                    for(int i = 0; i <= x.second - x.first; ++i) {
                        std::cout << x.first + i << ", "; 
                    }
                }
                else {
                    std::cout << "(" << x.first << ", " << x.second << "), ";
                }
            }
            std::cout << "}\n";
            std::cout << "col = {";
            for(auto x : access_unique_col_range) {
                    if(x.second - x.first <= 3) {
                    for(int i = 0; i <= x.second - x.first; ++i) {
                        std::cout << x.first + i << ", "; 
                    }
                }
                else {
                    std::cout << "(" << x.first << ", " << x.second << "), ";
                }
            }
            std::cout << "}\n";
        }

        void compute_access_col_idx() {
            access_col_idx.clear();
            for(int i = 0; i < pos_s.size(); ++i) {
                for(int r = pos_s[i].first.col_idx; r <= pos_s[i].second.col_idx; ++r) {
                    access_col_idx.push_back(r);
                }
                access_unique_col_range.push_back({pos_s[i].first.col_idx, pos_s[i].second.col_idx});
            }
            shrink_to_unique_value(access_unique_col_range);
            shrink_to_unique_value(access_col_idx);
        }


        void compute_access_row_idx() {
            access_row_idx.clear();
            for(int i = 0; i < pos_s.size(); ++i) {
                for(int r = pos_s[i].first.row_idx; r <= pos_s[i].second.row_idx; ++r) {
                    access_row_idx.push_back(r);
                }
                access_unique_row_range.push_back({pos_s[i].first.row_idx, pos_s[i].second.row_idx});
            }
            shrink_to_unique_value(access_unique_row_range);
            shrink_to_unique_value(access_row_idx);
        }

        BlockContainer(CSRCSCMatrix &matrix, std::vector<std::pair<MatrixPos, MatrixPos>> &poss)
            : csr_csc(matrix) {
            pos_s = poss;
            compute_access_row_idx();
            compute_access_col_idx();
            same_col_ = access_unique_col_range.size() == 1;
        }


        BlockContainer(const BlockContainer &c)
            : csr_csc(c.csr_csc)  {
            pos_s = c.pos_s;
            compute_access_row_idx();
            compute_access_col_idx();
            same_col_ = access_unique_col_range.size() == 1;
        }

        BlockContainer(CSRCSCMatrix &matrix, std::vector<std::pair<MatrixPos, MatrixPos>> (*func)(CSRCSCMatrix &)) 
            : csr_csc(matrix) {
            pos_s = func(csr_csc);
            
            compute_access_row_idx();
            compute_access_col_idx();

            same_col_ = access_unique_col_range.size() == 1;
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
                    res.push_back(BlockContainer(csr_csc, tmp));
                    std::cout << "-----------------------------------------------" << std::endl;
                    res[res.size() - 1].print();
                    pre_idx = i;
                    pre_col = pos_s[i].first.col_idx;
                }
            }
            auto tmp = std::vector<std::pair<MatrixPos, MatrixPos>>(pos_s.begin() + pre_idx, pos_s.end());
            res.push_back(BlockContainer(csr_csc, tmp));
            return res;    
        }

        std::pair<int, int> get_col_idx() {
            if(same_col_) {
                return {pos_s[0].first.col_idx, pos_s[0].second.col_idx};
            }
            return {-1, -1};
        }

        static BlockContainer merge(std::vector<BlockContainer> &need_merge) {
            //check validity
            auto addr = &(need_merge[0].csr_csc);
            for(auto iter : need_merge) {
                assert_msg(&(iter.csr_csc) == addr, "just same matrix block can merge");
            }
            std::vector<std::pair<MatrixPos, MatrixPos>> res_pos;
            for(auto block : need_merge) {
                for(auto pos : block.pos_s) {
                    res_pos.push_back(pos);
                }
            }
            BlockContainer res(need_merge[0].csr_csc, res_pos);
            return res;
        }
    };
};