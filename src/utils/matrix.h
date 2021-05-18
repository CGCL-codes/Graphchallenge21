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

        COOMatrix(std::string coo_input_file, int begin_node_idx, bool T = false) {
            std::ifstream input_file(coo_input_file);
            if(!input_file){
                std::cout << "File:" << coo_input_file << " does not exists.\n";
                exit(-1);
            }
            SparseDataType val;
            int u, v;
            if(T) {
                while(input_file >> u >> v >> val) {
                    coo_values_.push_back({u - begin_node_idx, v - begin_node_idx, val});
                    row_number = std::max(row_number, u + 1 - begin_node_idx);
                    col_number = std::max(col_number, v + 1 - begin_node_idx);
                }
            }
            else {
                while(input_file >> v >> u >> val) {
                    coo_values_.push_back({u - begin_node_idx, v - begin_node_idx, val});
                    row_number = std::max(row_number, u + 1 - begin_node_idx);
                    col_number = std::max(col_number, v + 1 - begin_node_idx);
                }
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
                    begin_row = (*iter).row;
                }
            }
            csr_len_.push_back(csr_index_.size());
            col_number = csr_len_.size() - 1;
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
                    begin_col = (*iter).col;
                }
            }
            csc_len_.push_back(csr_index_.size());
            row_number = csc_len_.size() - 1;
        }


    public:
        void print_len() {
            std::cout << "csr:";
            for(int i = 0; i != row_number + 1; ++i) {
                std::cout << csr_len_[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "csc:";
            for(int i = 0; i != col_number + 1; ++i) {
                std::cout << csc_len_[i] << " ";
            }
            std::cout << std::endl;
        }
        void print_one_row(int row) {
            for(int i = csr_len_[row]; i != csr_len_[row + 1]; ++i) {
                std::cout << csr_index_[i] << " ";
            }
            std::cout << std::endl;
        }

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
                if(col_idx_ >= self_.col_number) {
                    std::cout << "de* out of memory" << std::endl;
                    exit(-1);
                }
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
                if((*iter).col >= col) {
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
                if((*iter).row >= row) {
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
            ColIterator iter(col_number, 0, *this);
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

    class UIUCMatrix : public SparseMatrix {
        // 20 champion UIUC format
    public:
        int blocksize = 4;
        int neuron = 16;
        int buffsize = 6;
        int WRAPSIZE = 2;

        // int blocksize = 256;
        // int neuron = 1024;
        // int buffsize = 24 *1024/sizeof(float)/12;
        // int WRAPSIZE = 32;
        std::vector<int> buffdispl;
        std::vector<int> mapdispl;
        std::vector<unsigned short> map;

        std::vector<int> warpdispl;
        std::vector<unsigned short> warpindex;
        std::vector<float> warpvalue;
        void print_buffdispl() {
            for(auto i : buffdispl) {
                std::cout << i << ", ";
            }
            std::cout << std::endl;
        }

        void print_mapdispl() {
            for(auto i : mapdispl) {
                std::cout << i << ", ";
            }
            std::cout << std::endl;
        }


        void print_map() {
            for(auto i : map) {
                std::cout << i << ", ";
            }
            std::cout << std::endl;
        }


        void print_warpdispl() {
            for(auto i : warpdispl) {
                std::cout << i << ", ";
            }
            std::cout << std::endl;
        }

        
        void print_warpindex() {
            for(auto i : warpindex) {
                std::cout << i << ", ";
            }
            std::cout << std::endl;
        }
        UIUCMatrix(CSRCSCMatrix &csr_csc) {
            int numblocks = neuron / blocksize;
            int numwarp = blocksize / WRAPSIZE;
            buffdispl = std::vector<int>(numblocks + 1);
            
            std::vector<int> numbuff(numblocks, 0);
            
            buffdispl[0] = 0;
            for(int b = 0; b < numblocks; ++b) {
                std::vector<int> temp(neuron, 0);
                for(int m = b * blocksize; m < (b + 1) * blocksize; ++m) {
                    auto iter = csr_csc.row_iter_begin_at(m);
                    for(; iter != csr_csc.row_iter_end_at(m) ; ++iter) {
                        temp[(*iter).col]++;
                    }
                }
                int footprint = 0;
                for(int n = 0; n < neuron; n++){
                    if(temp[n]) footprint++;
                }
                numbuff[b] = (footprint + buffsize - 1)/buffsize;
            }
            for(int b = 0; b < numblocks; b++) {
                buffdispl[b + 1] = buffdispl[b] + numbuff[b];
            }
            
            std::vector<int> warpnz(buffdispl[numblocks] * numwarp, 0);
            std::vector<int> mapnz(buffdispl[numblocks], 0);

            for(int b = 0; b < numblocks; b++) {
                std::vector<int> temp(neuron, 0);
                for(int m = b * blocksize; m < (b + 1) * blocksize; ++m) {
                    auto iter = csr_csc.row_iter_begin_at(m);
                    for(; iter != csr_csc.row_iter_end_at(m) ; ++iter) {
                        temp[(*iter).col]++;
                    }
                }
                int footprint = 0;
                for(int n = 0; n < neuron; n++){
                    if(temp[n]) {
                        int buff = footprint / buffsize;
                        mapnz[buffdispl[b] + buff]++;
                        temp[n] = buff;
                        footprint++;
                    }
                }
                for(int buff = 0; buff < numbuff[b]; buff++) {
                    for(int warp = 0; warp < numwarp; warp++){
                        int tempnz[WRAPSIZE] = {0};
                        for(int t = 0; t < WRAPSIZE; t++) {
                            auto iter = csr_csc.row_iter_begin_at(b*blocksize+warp*WRAPSIZE+t);
                            for(; iter != csr_csc.row_iter_end_at(b*blocksize+warp*WRAPSIZE+t); ++iter)
                                if(temp[(*iter).col]==buff) tempnz[t]++;
                        }
                        int warpmax = 0;
                        for(int t = 0; t < WRAPSIZE; t++) {
                            if(tempnz[t]>warpmax) warpmax = tempnz[t];
                        }
                        warpnz[(buffdispl[b]+buff)*numwarp+warp] = warpmax;
                    }
                }
            }

            warpdispl = std::vector<int>(buffdispl[numblocks] * numwarp + 1);
            warpdispl[0] = 0;
            for(int warp = 0; warp < buffdispl[numblocks]*numwarp; warp++) {
                warpdispl[warp+1] = warpdispl[warp] + warpnz[warp];
            }
            
            warpindex = std::vector<unsigned short>(warpdispl[buffdispl[numblocks] * numwarp] * WRAPSIZE, 0);
            warpvalue = std::vector<float>(warpdispl[buffdispl[numblocks] * numwarp] * WRAPSIZE, 0.0);
            mapdispl = std::vector<int>(buffdispl[numblocks] + 1, 0);
            
            for(int buff = 0; buff < buffdispl[numblocks]; buff++) {
                mapdispl[buff+1] = mapdispl[buff] + mapnz[buff];
            }

            map = std::vector<unsigned short>(mapdispl[buffdispl[numblocks]], 0);

            mapnz = std::vector<int>(buffdispl[numblocks], 0);

            for(int b = 0; b < numblocks; b++) {
                std::vector<int> temp(neuron, 0);
                for(int m = b * blocksize; m < (b + 1) * blocksize; ++m) {
                    auto iter = csr_csc.row_iter_begin_at(m);
                    for(; iter != csr_csc.row_iter_end_at(m) ; ++iter) {
                        temp[(*iter).col]++;
                    }
                }
                int footprint = 0;
                for(int n = 0; n < neuron; n++) {
                    if(temp[n]){
                        int buff = footprint/buffsize;
                        map[ mapdispl[buffdispl[b]+buff] + mapnz[buffdispl[b]+buff] ] = n;
                        mapnz[buffdispl[b]+buff]++;
                        temp[n] = footprint;
                        footprint++;
                    }
                }
                for(int buff = 0; buff < numbuff[b]; buff++) {
                    for(int warp = 0; warp < numwarp; warp++){
                        int tempnz[WRAPSIZE] = {0};
                        for(int t = 0; t < WRAPSIZE; t++) {
                            auto iter = csr_csc.row_iter_begin_at(b*blocksize+warp*WRAPSIZE+t);
                            for(; iter != csr_csc.row_iter_end_at(b*blocksize+warp*WRAPSIZE+t); ++iter) {
                                if(temp[(*iter).col] / buffsize == buff){
                                    int ind = (warpdispl[(buffdispl[b]+buff)*numwarp+warp]+tempnz[t]) * WRAPSIZE +t;
                                    warpindex[ind] = temp[(*iter).col] % buffsize;
                                    warpvalue[ind] = 0.625;
                                    tempnz[t]++;
                                }
                            }
                        }
                    }
                }
            }   
        }

    };
};