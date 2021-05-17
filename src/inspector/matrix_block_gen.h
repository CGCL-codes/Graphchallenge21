#pragma once
#include <vector>
#include "utils/string.h"

namespace ftxj {

    enum MatrixBlockProp {
        Random,                 // 随机
        Col_Line,               // 列连续
        Row_Line,               // 行连续
        Col_Stride_Line,
        Row_Stride_Line,
        Rectangles,             // 稠密块
        Pad_Rectangles,         // 填充少量的 0 可以变成稠密块的
        Stride_Rectangles,      // 忽略固定间隔后，是稠密块 
    };


    struct SparseMatrixBlock {
        int stride_;
        MatrixPos begin_pos_;
        MatrixPos end_pos_;
        MatrixBlockProp type_;
    };


    class SparseMatrixBlockGen {
        String file_path;

        CSRCSCMatrix csr_csc;

        
        int row_line_succ_max(MatrixPos start_pos) {
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


        int col_line_succ_max(MatrixPos start_pos) {
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


        std::pair<int, int> rectangels_max(MatrixPos start_pos) {
            int row_max = row_line_succ_max(start_pos);
            
            int now_max_row = 0;
            int now_max_col = 0;
            int now_max = 0;

            int res_row = 0;
            int res_col = 0;
            
            for(int i = 0; i < row_max; ++i) {
                now_max_row = i + 1;
                int col_max = col_line_succ_max({start_pos.row, start_pos.col + i});
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


        SparseMatrixBlock gen_one_block(MatrixPos start_pos) {

        }
    public:
        SparseMatrixBlockGen(CSRCSCMatrix &matrix) : csr_csc(matrix) {
            MatrixPos start_pos {0, 0};
            int end_len = 0;
            for(; start_pos.col_idx < csr_csc.col_number; ) {
                for(; start_pos.row_idx < csr_csc.row_number; ) {
                    auto end_pos = rectangels_max(start_pos);
                    int tmp_len = end_pos.col_idx - start_pos.col_idx; // 多少行长
                    if(tmp_len != end_len && end_len != 0) {
                        std::cout << "TODO fix this bug" << std::endl;
                        exit(-1);
                    }
                    end_len = tmp_len;
                    start_pos.row_idx = end_pos.row_idx + 1;
                }
            }
        }


    };

};