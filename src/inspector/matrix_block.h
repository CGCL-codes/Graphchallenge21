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
        const MatrixBlockProp type_;
        const string name_;
        MatrixPos begin_pos_;
        MatrixPos end_pos_;
        CSRCSCMatrix &csr_csc_;
    public:
        MatrixBlockBase(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : csr_csc_(matrix), begin_pos_(begin_pos), end_pos_(end_pos) {
        }

        virtual void fill_data(MatrixPos &beg, MatrixPos &end) = 0;
        virtual std::string get_block_type() = 0;
    };
    
    class RandomBlock : public MatrixBlockBase {
        COOMatrix sub_matrix;
    public:
        RandomBlock(CSRCSCMatrix &matrix, const MatrixPos &begin_pos, const MatrixPos &end_pos) 
            : MatrixBlockBase(matrix, begin_pos, end_pos) {
                fill_data(begin_pos_, end_pos_);
        }
        void fill_data(MatrixPos &beg, MatrixPos &end) {

        }
        std::string get_block_type();
    };

    class LineBlock : public MatrixBlockBase {
        std::vector<SparseDataType> data_;
        virtual void allocate_data() = 0;
        void set_data(MatrixElmIterator &iter, MatrixPos &beg, MatrixPos &end) {
            begin_pos_ = beg;
            end_pos_ = end;
            int idx = 0;
            for(; idx < data_.size(); ++iter, ++idx) {
                data_.push_back(iter->data);
            }
        }
    };

    class RectangleBlock : public MatrixBlockBase {
        std::vector<std::vector<SparseDataType>> data_;
        void allocate_data(MatrixPos &beg, MatrixPos &end) {
            data_ = std::vector<std::vector<SparseDataType>>(end.row_idx - beg.row_idx, 
                std::vector<SparseDataType>(end.col_idx - beg.col_idx, 0)
            );
        }
        void set_data(MatrixElmIterator &iter, MatrixPos &beg, MatrixPos &end) {
            begin_pos_ = beg;
            end_pos_ = end;
            for(int i = 0; i < end.row_idx - beg.col_idx; ++i) {
                for(int j = 0; j < end.col_idx - beg.col_idx; ++j) {
                    data_[i].push_back(iter->data);
                    iter++;
                }
                iter = iter.next_iterator();
            }
        }
    public:
        void RectangleBlock(SparseMatrix &matrix, MatrixPos &beg, MatrixPos &end) {
            allocate_data(beg, end);
            RowIterator iter = matrix.row_begin_at(beg.row_idx, beg.col_idx);
            set_data(iter, beg, end);
        }
        std::string get_block_type() {
            return "RectangleBlock";
        }
    };

    class RowLineBlock : public LineBlock {
        void allocate_data(MatrixPos &beg, MatrixPos &end) {
            assert_msg(end.row_idx == beg.row_idx, "row line block allocate data error");
            data_ = std::vector<SparseDataType>(end.col_idx - beg.col_idx, 0);
        }
    public:
        void RowLineBlock(SparseMatrix &matrix, MatrixPos &beg, MatrixPos &end) {
            allocate_data(beg, end);
            RowIterator iter = matrix.row_begin_at(beg.row_idx, beg.col_idx);
            set_data(iter, beg, end);
        }
        
        int get_line_len() {
            return 0; // TODO, fix bugs
        }
        
        SparseDataType get_values() {
            return 0; // TODO, fix bugs
        }

    };

    class ColLineBlock : public LineBlock {
        void allocate_data(MatrixPos &beg, MatrixPos &end) {
            assert_msg(end.col_idx == beg.col_idx, "col line block allocate data error");
            data_ = std::vector<SparseDataType>(end.row_idx - beg.row_idx, 0);
        }
    public:
        void ColLineBlock(SparseMatrix &matrix, MatrixPos &beg, MatrixPos &end) {
            allocate_data(beg, end);
            ColIterator iter = matrix.col_begin_at(beg.row_idx, beg.col_idx);
            set_data(iter, beg, end);
        }
        
        std::string get_block_type() {
            return "ColLineBlock";
        }
    };

    class StrideBlockBase {
        int stride_;
    public:
        void set_stride(int stride) {
            stride_ = stride;
        }
        int get_stride() {
            return stride_;
        }
    };

    class ColStrideLineBlock : public ColLineBlock, StrideBlockBase {
    public:
        std::string get_block_type() {
            return "ColStrideLineBlock";
        }
    };

    class RowStrideLineBlock : public RowLineBlock, StrideBlockBase {
    public:
        std::string get_block_type() {
            return "RowStrideLineBlock";
        }
    };

    class StrideRectanglesBlock : public RectangleBlock, StrideBlockBase {
    public:
        std::string get_block_type() {
            return "StrideRectanglesBlock";
        }
    };

    class PadRectanglesBlock : public RectangleBlock {
    public:
        std::string get_block_type() {
            return "PadRectanglesBlock";
        }
    };


};