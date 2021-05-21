#pragma once
#include <vector>
#include <string>

#include "../utils/type.h"
#include "../utils/debug.h"
#include "../utils/matrix.h"

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

    class MatrixBlockBase {
    public:
        MatrixBlockProp type_;
        string name_;
        MatrixPos begin_pos_;
        MatrixPos end_pos_;
        CSRCSCMatrix &csr_csc_;
        MatrixBlockBase(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : csr_csc_(matrix), begin_pos_(begin_pos), end_pos_(end_pos) {
        }
        virtual std::string get_block_type() = 0;
    };
    
    class RandomBlock : public MatrixBlockBase {
        COOMatrix sub_matrix;
        void fill_data() {

        }
    public:
        RandomBlock(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : MatrixBlockBase(matrix, begin_pos, end_pos) {
                fill_data();
                type_ = Random;
        }
        std::string get_block_type() {
            return "Random";
        }
    };

    class LineBlock : public MatrixBlockBase {
    public:
        std::vector<SparseDataType> data_;
        LineBlock(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : MatrixBlockBase(matrix, begin_pos, end_pos)  {
            
        }
    };

    class RectangleBlock : public MatrixBlockBase {
        std::vector<std::vector<SparseDataType>> data_;
        void allocate_data() {
            data_ = std::vector<std::vector<SparseDataType>>(end_pos_.row_idx - begin_pos_.row_idx, 
                std::vector<SparseDataType>(end_pos_.col_idx - end_pos_.col_idx, 0)
            );
        }
        void fill_data() {
            auto iter = csr_csc_.row_iter_begin_at(begin_pos_.row_idx, begin_pos_.col_idx);
            for(int i = 0; i < end_pos_.row_idx - begin_pos_.col_idx; ++i) {
                for(int j = 0; j < end_pos_.col_idx - begin_pos_.col_idx; ++j) {
                    data_[i].push_back((*iter).val);
                    ++iter;
                }
                iter = csr_csc_.row_iter_begin_at(begin_pos_.row_idx + 1, begin_pos_.col_idx);
            }
        }
    public:
        RectangleBlock(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : MatrixBlockBase(matrix, begin_pos, end_pos)  {
            allocate_data();
            fill_data();
            type_ = Rectangles;
        }
        std::string get_block_type() {
            return "Rectangles";
        }
    };

    class RowLineBlock : public LineBlock {
        void allocate_data() {
            assert_msg(end_pos_.row_idx == begin_pos_.row_idx, "row line block allocate data error");
            data_ = std::vector<SparseDataType>(end_pos_.col_idx - begin_pos_.col_idx, 0);
        }
        void fill_data() {
            auto iter = csr_csc_.col_iter_begin_at(begin_pos_.row_idx, begin_pos_.col_idx);
            for(int j = 0; j < end_pos_.col_idx - begin_pos_.col_idx; ++j) {
                data_[j] = ((*iter).val);
                ++iter;
            }
        }
    public:
        RowLineBlock(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : LineBlock(matrix, begin_pos, end_pos)  {
            allocate_data();
            fill_data();
            type_ = Row_Line;
        }
        
        std::string get_block_type() {
            return "Row_Line";
        }
        
        int get_line_len() {
            return end_pos_.col_idx - begin_pos_.col_idx;
        }
    };

    class ColLineBlock : public LineBlock {
        void allocate_data() {
            assert_msg(end_pos_.col_idx == begin_pos_.col_idx, "col line block allocate data error");
            data_ = std::vector<SparseDataType>(end_pos_.row_idx - begin_pos_.row_idx, 0);
        }
        void fill_data() {
            auto iter = csr_csc_.row_iter_begin_at(begin_pos_.row_idx, begin_pos_.col_idx);
            for(int j = 0; j < end_pos_.row_idx - begin_pos_.row_idx; ++j) {
                data_[j] = ((*iter).val);
                ++iter;
            }
        }
    public:
        ColLineBlock(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : LineBlock(matrix, begin_pos, end_pos)  {
            allocate_data();
            fill_data();
        }
        
        std::string get_block_type() {
            return "ColLineBlock";
        }
    };

    // class RowLineBundle {
    //     std::vector<RowLineBlock> data_;
    //     std::vector<int> row_line_index;
    //     std::vector<int> col_begins;
    // public:
    //     RowLineBundle(BlockContainer &container) {

    //     }
    // };

    // TODO implement these type
    // class StrideBlockBase {
    //     int stride_;
    // public:
    //     void set_stride(int stride) {
    //         stride_ = stride;
    //     }
    //     int get_stride() {
    //         return stride_;
    //     }
    // };

    // class ColStrideLineBlock : public ColLineBlock, StrideBlockBase {
    // public:
    //     std::string get_block_type() {
    //         return "ColStrideLineBlock";
    //     }
    // };

    // class RowStrideLineBlock : public RowLineBlock, StrideBlockBase {
    // public:
    //     std::string get_block_type() {
    //         return "RowStrideLineBlock";
    //     }
    // };

    // class StrideRectanglesBlock : public RectangleBlock, StrideBlockBase {
    // public:
    //     std::string get_block_type() {
    //         return "StrideRectanglesBlock";
    //     }
    // };

    // class PadRectanglesBlock : public RectangleBlock {
    // public:
    //     std::string get_block_type() {
    //         return "PadRectanglesBlock";
    //     }
    // };


};