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
    struct MatrixPos{
        int row_idx;
        int col_idx;
    };

    struct SparseMatrixBlock {
        int stride_;
        MatrixPos begin_pos_;
        MatrixPos end_pos_;
        MatrixBlockProp type_;
    };


    class SparseMatrixBlockGen {
        String file_path;
        bool col_line_block_judge();        
        bool row_line_block_judge();        
        bool col_stride_block_judge();        
        bool row_stride_block_judge();        
        bool rectangels_block_judge();
        bool pad_rectangels_judge();
        bool stride_rectangels_judge();

        SparseMatrixBlock gen_one_block(MatrixPos start_pos);
        public:
            void block_gen();            
    };

};