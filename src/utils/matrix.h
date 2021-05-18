#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h> 

#include "../reorder/reorder.h"

#include "type.h"


namespace ftxj {
    class SparseMatrix {
    public:
        struct Elm {
            int row;
            int col;
            SparseDataType val;
        };
        int row_number;
        int col_number;
        int nnzs;
    };


    class COOMatrix : public SparseMatrix {
        std::vector<Elm> coo_values_;
    public:
        bool row_first;
        bool col_first;
        
        class ElmIterator {
            std::vector<Elm>::iterator iter_;
        public:
            ElmIterator(const std::vector<Elm>::iterator &iter) : iter_(iter) {
            }
            
            ElmIterator& operator++() {
                iter_++;
                return *this;
            }
            
            bool operator !=(const ElmIterator& that) const {
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
            std::ifstream input_file(coo_input_file);
            if(!input_file){
                std::cout << "File:" << coo_input_file << " does not exists.\n";
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
        
        COOMatrix() {
            nnzs = 0;
            row_number = 0;
            col_number = 0;
            row_first = false;
            col_first = false;
        }

        void reorder(Reorder &reorder_class) {
            for(int i = 0; i < coo_values_.size(); ++i) {
                auto pos = reorder_class.new_pos({coo_values_[i].row, coo_values_[i].col});
                coo_values_[i].row = pos.row_idx;
                coo_values_[i].col = pos.col_idx;
            }
        }

        void save_matrix(const std::string &filename) {
            std::ofstream ofile(filename);
            for(int i = 0; i < coo_values_.size(); ++i) {
                ofile << coo_values_[i].row << " " << coo_values_[i].col << " " <<coo_values_[i].val << "\n";
            }
            ofile.close();
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
            for(auto iter = coo_matrix.begin(); iter != coo_matrix.end(); ++iter) {
                csr_index_.push_back((*iter).col);
                csr_values_.push_back((*iter).val);
                if((*iter).row != begin_row) {
                    csr_len_.push_back(csr_index_.size() - 1);
                }
            }
            csr_len_.push_back(csr_index_.size());
            row_number = csr_len_.size() - 1;
        }

        void coo2csc(COOMatrix &coo_matrix) {
            coo_matrix.to_col_first_ordered();
            int begin_col = 0;
            csc_len_.push_back(0);
            for(auto iter = coo_matrix.begin(); iter != coo_matrix.end(); ++iter) {
                csc_index_.push_back((*iter).row);
                csc_values_.push_back((*iter).val);
                if((*iter).col != begin_col) {
                    csc_len_.push_back(csc_index_.size() - 1);
                }
            }
            csc_len_.push_back(csr_index_.size());
            row_number = csc_len_.size() - 1;
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
            CSRCSCMatrix &self_;
        public:
            RowIterator(int row_idx, int idx, CSRCSCMatrix &self) : self_(self), row_idx_(row_idx), idx_(idx) {
            
            }
            
            RowIterator& operator++() {
                idx_++;
                return *this;
            }
            
            RowIterator& operator = (const RowIterator &t) {
                idx_ = t.idx_;
                row_idx_ = t.row_idx_;
                self_ = t.self_;
            }

            bool operator !=(const RowIterator& that) const {
                return row_idx_ != that.row_idx_ || idx_ != that.idx_;
            }

            Elm operator*() const {
                return {
                    row_idx_, 
                    self_.csr_index_[self_.csr_len_[row_idx_] + idx_], 
                    self_.csr_values_[self_.csr_len_[row_idx_] + idx_]
                    };
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
            CSRCSCMatrix &self_;
        public:
            ColIterator(int col_idx, int idx, CSRCSCMatrix &self) : col_idx_(col_idx), idx_(idx), self_(self) {
            
            }

            ColIterator& operator = (const ColIterator &t) {
                idx_ = t.idx_;
                col_idx_ = t.col_idx_;
                self_ = t.self_;
            }
            
            ColIterator& operator++() {
                idx_++;
                return *this;
            }
            
            ColIterator& operator+=(int x) {
                idx_ += x;
                return *this;
            }
            
            bool operator !=(const ColIterator& that) const {
                return col_idx_ != that.col_idx_ || idx_ != that.idx_;
            }

            Elm operator*() const {
                return {self_.csc_index_[self_.csc_len_[col_idx_] + idx_], col_idx_, self_.csc_values_[self_.csc_len_[col_idx_] + idx_]};
            }

            ColIterator& next_col() {
                idx_ = 0;
                col_idx_++;
                return *this;
            }
            
            ColIterator& next_ncol(int itm) {
                idx_ = 0;
                col_idx_ += itm;
                return *this;
            }
        };

        
        RowIterator row_iter_begin_at(int row, int col) {
            RowIterator iter(row, 0, *this);
            for(; iter != row_iter_end_at(row); ++iter) {
                if((*iter).col == col) {
                    return iter;
                }
            }
            return row_iter_end_at(row);
        }

        RowIterator row_iter_begin_at(int row) {
            RowIterator iter(row, 0, *this);
            return  iter;
        }

        RowIterator row_iter_end_at(int row) {
            int len = csr_len_[row + 1] - csr_len_[row];
            RowIterator iter(row, len, *this);
            return iter;
        }


        ColIterator col_iter_begin_at(int row, int col) {
            ColIterator iter(col, 0, *this);
            for(; iter != col_iter_end_at(col); ++iter) {
                if((*iter).row == row) {
                    return iter;
                }
            }
            return col_iter_end_at(col);
        }

        ColIterator col_iter_begin_at(int col) {
            ColIterator iter(col, 0, *this);
            return iter;
        }

        ColIterator col_iter_end_at(int col) {
            int len = csc_len_[col + 1] - csc_len_[col];
            ColIterator iter(col, len, *this);
            return iter;
        }

        ColIterator col_iter_end() {
            ColIterator iter(col_number - 1, 0, *this);
            return iter;
        }


        CSRCSCMatrix(std::string &filename, int begin_node_idx, FileType type = COO_FILE) {
            switch(type) {
            case COO_FILE: {
                COOMatrix coo(filename, begin_node_idx);
                init_from_COO(coo);
                break;
            }
            case BIN_FILE:
                break;
            }
        }

        CSRCSCMatrix(COOMatrix &coo_matrix) {
            init_from_COO(coo_matrix);
        }

        void init_from_COO(COOMatrix &coo_matrix) {
            coo2csr(coo_matrix);
            coo2csc(coo_matrix);
        }
    };
};