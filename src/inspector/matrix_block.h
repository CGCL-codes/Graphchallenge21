#pragma once
#include <vector>
#include <string>

#include "type.h"

namespace ftxj {
    using namespace std;
    
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

    class MatrixBlockBase {
        const MatrixBlockProp type_;
        const string name;
        MatrixPos begin_pos_;
        MatrixPos end_pos_;

    };

    class LineBlock : public MatrixBlockBase {
        std::vector<SparseDataType> data_;
        int stride;
        int row_idx_;
        int col_idx_;
    };

    class ColLineBlock : public LineBlock {

    };

    class RowLineBlock : public LineBlock {
        
    };

    class ColStrideLineBlock : public LineBlock {

    };

    class RowStrideLineBlock : public LineBlock {

    };
};