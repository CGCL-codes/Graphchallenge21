#pragma once
#include <vector>
#include "utils/string.h"
#include "matrix_block.h"
#include  "utils/matrix.h"

namespace ftxj {

    class SparseMatrixBlockGen {

        static int row_line_succ_max(MatrixPos start_pos, CSRCSCMatrix &csr_csc) {
            int row_idx = start_pos.row_idx;
            int col_idx = start_pos.col_idx;
            int res  = 0;
            for(auto iter = csr_csc.row_iter_begin_at(row_idx, col_idx); 
                iter != csr_csc.row_iter_end_at(row_idx); ++iter) {
                if((*iter).col == col_idx + res) {
                    res++;
                }
                else {
                    return res;
                }
            }
            return res;
        }


        static int col_line_succ_max(MatrixPos start_pos, CSRCSCMatrix &csr_csc) {
            int row_idx = start_pos.row_idx;
            int col_idx = start_pos.col_idx;
            int res  = 0;
            for(auto iter = csr_csc.col_iter_begin_at(row_idx, col_idx); 
                iter != csr_csc.col_iter_end_at(col_idx); ++iter) {
                if((*iter).row == row_idx + res) {
                    res++;
                }
                else {
                    return res;
                }
            }
            return res;
        }


        static std::pair<int, int> rectangels_max(MatrixPos start_pos, CSRCSCMatrix &csr_csc) {
            int row_max = row_line_succ_max(start_pos, csr_csc);
            
            int now_max_row = 0;
            int now_max_col = 0;
            int now_max = 0;

            int res_row = 0;
            int res_col = 0;
            
            for(int i = 0; i < row_max; ++i) {
                now_max_row = i + 1;
                int col_max = col_line_succ_max({start_pos.row, start_pos.col + i}, csr_csc);
                now_max_col = std::min(col_max, now_max_col);
                int tmp_area = now_max_col * now_max_row;
                if(tmp_area > now_max) {
                    now_max = tmp_area;
                    res_row = now_max_row;
                    res_col = now_max_col;
                }
            }
            return {res_row, res_col};
        }

    public:

        static std::vector<std::pair<MatrixPos, MatrixPos>> naive_method(CSRCSCMatrix &csr_csc) {
            MatrixPos start_pos {0, 0};
            std::vector<std::pair<MatrixPos, MatrixPos>> res;
            int end_len = 0;
            
            int col_each_big_block = 1;


            int now_lookup_col = 0;
            int now_lookup_row = 0;

            auto col_iter = csr_csc.col_iter_begin_at(now_lookup_row, now_lookup_col);
            for(; col_iter != csr_csc.col_iter_end(); col_iter = col_iter.next_ncol(col_each_big_block)) {
                while(col_iter != csr_csc.col_iter_end_at(now_lookup_col)) {
                    auto row_idx = *col_iter.row;
                    auto col_idx = *col_iter.col;
                    auto end_pos = rectangels_max({row_idx, col_idx});
                    int tmp_col_len = end_pos.col_idx - col_idx; // 多少行长
                    int tmp_row_len = end_pos.row_idx - row_idx; // 多少列长
                    if(tmp_col_len != 0 || tmp_row_len != 0) {
                        if(tmp_col_len != end_len && end_len != 0) {
                            std::cout << "TODO fix this bug" << std::endl;
                            exit(-1);
                        }
                        col_iter += tmp_row_len; 
                    }
                    else {
                        col_iter++;
                    }
                }
            }
            return res;
        }
    };

};