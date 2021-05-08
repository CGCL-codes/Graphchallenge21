#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h> 

namespace ftxj {

    class MatrixElmIterator {
    public:

    };

    class ColIterator {

    };
    
    class SparseMatrix {
    public:
        struct Elm {
            int row;
            int col;
            SparseDataType val;
        };
    };


    class COOMatrix : public SparseMatrix {
        std::vector<Elm> coo_values_;
    public:

        int nnzs;
        int row_number;
        int col_number;

        bool row_first;
        bool col_first;
        
        class ElmIterator {
            std::vector<Elm>::iterator iter_;
        public:
            ElmIterator(std::vector<Elm>::iterator &iter) : iter_(iter) {
            }
            
            ElmIterator& operator++() {
                iter_++;
                return *this;
            }
            
            bool operator !=(ElmIterator& that) const {
                return iter_ != that.iter_;
            }

            Elm operator*() const {
                return *iter_;
            }
        };
        
        ElmIterator begin() {
            return ElmIterator(coo_values_.begin());
        }

        ElmIterator end() {
            return ElmIterator(coo_values_.end());
        }

        
        void to_col_first_ordered() {
            if(col_first) return;
            std::sort(coo_values_.begin(), coo_values_.end(), [](const Elm& e1, const Elm&e2)->bool {
                if(e1.col != e2.col) return e1.col < e2.col;
                else return e1.row < e2.row;
            });
            row_first = false;
            col_first = true;
        }

        void to_row_first_ordered() {
            if(row_first) return;
            std::sort(coo_values_.begin(), coo_values_.end(), [](const Elm& e1, const Elm&e2)->bool {
                if(e1.row != e2.row) return e1.row < e2.row;
                else return e1.col < e2.col;
            });
            row_first = true;
            col_first = false;
        }

        COOMatrix(std::string coo_input_file, int begin_node_idx) {
            std::ifstream input_file(filename);
            if(!input_file){
                std::cout << "File " << filename << " does not exists.\n";
                exit(-1);
            }
            SparseDataType val;
            int u, v;
            while(input_file >> u >> v >> val) {
                coo_values_.push_back({u - begin_node_idx, v - begin_node_idx, val});
                row_number = std::max(row_number, u + 1 - begin_node_idx);
                col_number = std::max(col_number, v + 1 - begin_node_idx);
            }
            nnzs = coo_values_.size();
            row_first = false;
            col_first = false;
        }
    };

    class CSRCSCMatrix : public SparseMatrix {
        std::vector<SparseDataType> csr_values_;
        std::vector<SparseDataType> csc_values_;
        std::vector<int> csr_index_;
        std::vector<int> csc_index_;
        std::vector<int> csr_len_;
        std::vector<int> csc_len_;

        void coo2csr(COOMatrix &coo_matrix) {
            coo_matrix.to_row_first_ordered();
            int begin_row = 0;
            csr_len_.push_back(0);
            for(auto iter = coo_matrix.begin(); iter < coo_matrix.end(); ++iter) {
                csr_index_.push_back((*iter).col);
                csr_values_.push_back((*iter).val);
                if((*iter).row != begin_row) {
                    csr_len_.push_back(csr_index_.size() - 1);
                }
            }
            csr_len_.push_back(csr_index_.size());
        }

        void coo2csc(COOMatrix &coo_matrix) {
            coo_matrix.to_col_first_ordered();
            int begin_col = 0;
            csc_len_.push_back(0);
            for(auto iter = coo_matrix.begin(); iter < coo_matrix.end(); ++iter) {
                csc_index_.push_back((*iter).row);
                csc_values_.push_back((*iter).val);
                if((*iter).col != begin_col) {
                    csc_len_.push_back(csc_index_.size() - 1);
                }
            }
            csc_len_.push_back(csr_index_.size());
        }


    public:
        
        enum FileType {
            COO_FILE,
            CSR_FILE,
            CSC_FILE,
            BIN_FILE
        };

        class RowIterator {
            int idx_;
            int row_idx_;
        public:
            RowIterator(int row_idx, int idx) : row_idx_(row_idx), idx_(idx) {
            
            }
            
            RowIterator& operator++() {
                idx_++;
                return *this;
            }
            
            bool operator !=(RowIterator& that) const {
                return row_idx_ != that.row_idx_ || idx_ != that.idx_;
            }

            Elm operator*() const {
                return {row_idx_, csr_index_[csr_len_[row_idx_] + idx_], csr_values_[csr_len_[row_idx_] + idx_]};
            }

            RowIterator& next_row() {
                idx_ = 0;
                row_idx_++;
                return *this;
            }
        };

        class ColIterator {
            int idx_;
            int col_idx_;
        public:
            ColIterator(int col_idx) : col_idx_(col_idx), idx_(0) {
            
            }
            
            RowIterator& operator++() {
                idx_++;
                return *this;
            }
            
            bool operator !=(ColIterator& that) const {
                return col_idx_ != that.col_idx_ || idx_ != that.idx_;
            }

            Elm operator*() const {
                return {csc_index_[csc_len_[col_idx_] + idx_], col_idx_, csc_values_[csc_len_[col_idx_] + idx_]};
            }

            ColIterator& next_col() {
                idx_ = 0;
                col_idx_++;
                return *this;
            }
        };


        RowIterator row_iter_begin_at(int row, int col) {
            RowIterator iter(row, 0);
            for(; iter != row_iter_end_at(row); ++iter) {
                if((*iter).col == col) {
                    return iter;
                }
            }
            return row_iter_end_at(row);
        }

        RowIterator row_iter_begin_at(int row) {
            return RowIterator iter(row, 0);
        }

        RowIterator row_iter_end_at(int row) {
            int len = csr_len_[row + 1] - csr_len_[row];
            return RowIterator iter(row, len);
        }


        ColIterator col_iter_begin_at(int row, int col) {
            ColIterator iter(col, 0);
            for(; iter != col_iter_end_at(col); ++iter) {
                if((*iter).row == row) {
                    return iter;
                }
            }
            return col_iter_end_at(col);
        }

        ColIterator col_iter_begin_at(int col) {
            return ColIterator iter(col, 0);
        }

        ColIterator col_iter_end_at(int col) {
            int len = csc_len_[col + 1] - csc_len_[col];
            return ColIterator iter(col, len);
        }



        CSRCSCMatrix(std::string &filename, int begin_node_idx, FileType type = COO_FILE) {
            switch(type) {
            case COO_FILE:
                COOMatrix coo(filename, begin_node_idx);
                init_from_COO(coo);
                break;
            case BIN_FILE:
                break;
            }
        }
        
        void init_from_COO(COOMatrix &coo_matrix) {
            coo2csr(coo_matrix);
            coo2csc(coo_matrix);
        }
    };
};