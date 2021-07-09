#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h> 
#include <set>

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
    public:
        std::vector<Elm> coo_values_;
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

        int get_input_cost(std::set<int> R_signature, int TB, int TN) {
            return R_signature.size() * TB;
        }

        int get_output_cost(std::set<int> C_signature, int TB, int TN) {
            return C_signature.size() * TB;
        }

        int get_weight_cost(std::vector<int> number_distribution, int begin, int end) {
            int res = 0;
            for(int i = begin; i < end; ++i) {
                res += number_distribution[i];
            }
            return res;
        }

        int get_balance_cost(std::vector<int> number_distribution, int begin, int end) {
            int max = 0;
            int min = 100000;
            for(int i = begin; i < end; ++i) {
                max = std::max(max, number_distribution[i]);
                min = std::min(min, number_distribution[i]);
            }
            return max - min;
        }

        int get_random_cost(std::set<int> C_signature) {
            int beg = (*C_signature.begin()) - 1;
            int success = 0;
            for(auto i : C_signature) {
                if(i == beg + 1) {
                    success += 1;
                }
                beg = i;
            }
            return C_signature.size() - success;
        }

        void cost_analysis(int tb1, int tn1, int tb2, int tn2) {
            std::vector<std::set<int>> row_signature_1(row_number / tn1);
            std::vector<std::set<int>> col_signature_1(col_number / tn1);
            std::vector<std::set<int>> row_signature_2(row_number / tn2);
            std::vector<std::set<int>> col_signature_2(col_number / tn2);
            std::vector<int> len_signature(row_number, 0);

            // std::cout << "[Execution Model One Cost Analysis]" << std::endl;

            // std::cout << "nnzs = " << coo_values_.size() << std::endl;
            
            // std::cout << "signature size = " << row_number / tn1 << ", " \
            << col_number / tn1 << ", " << row_number / tn2 << ", " \
            << col_number / tn2 << std::endl; 
            

            for(int i = 0; i < coo_values_.size(); ++i) {
                if(i == 1000) std::cout << i << std::endl;
                int row_sec_1 = coo_values_[i].row / tn1;
                int col_sec_1 = coo_values_[i].col / tn1;
                int row_sec_2 = coo_values_[i].row / tn2;
                int col_sec_2 = coo_values_[i].col / tn2;
                // if(col_sec_1 == 0) {
                //     std::cout << coo_values_[i].row << ",";
                // }
                row_signature_1[row_sec_1].insert(coo_values_[i].col);
                col_signature_1[row_sec_1].insert(coo_values_[i].row);
                
                len_signature[coo_values_[i].col] ++;

                row_signature_2[row_sec_2].insert(coo_values_[i].col);
                col_signature_2[row_sec_2].insert(coo_values_[i].row);
            }

            // std::cout << "[Signature Success]" << std::endl;




            int cost_input_1 = 0;
            int cost_input_2 = 0;

            int cost_output_1 = 0;
            int cost_output_2 = 0;

            int cost_weight_1 = 0;
            int cost_weight_2 = 0;

            int cost_balance_1 = 0;
            int cost_balance_2 = 0;

            int balance_cost_1 = 0;
            int balance_cost_2 = 0;

            int random_cost_1 = 0;
            int random_cost_2 = 0;

            for(int i = 0; i < col_number / tn1; ++i) {
                cost_input_1 += get_input_cost(row_signature_1[i], tb1, tn1);

                // for(auto iter : row_signature_1[i]) {
                //     std::cout << iter << ", "; 
                // }
                // std::cout << "[In]" << cost_input_1 << std::endl;

                cost_weight_1 += get_weight_cost(len_signature, i * tn1, (i + 1) * tn1);
                cost_output_1 += get_output_cost(col_signature_1[i], tb1, tn1);


                // for(auto iter : col_signature_1[i]) {
                //     std::cout << iter << ", "; 
                // }
                // std::cout << "[Out]" << cost_output_1 << std::endl;
                // exit(1);


                balance_cost_1 += get_balance_cost(len_signature, i * tn1, (i + 1) * tn1);
                random_cost_1 += get_random_cost(row_signature_1[i]);
                // std::cout << "[Random]" << random_cost_1 << std::endl;
            } 

            for(int i = 0; i < col_number / tn2; ++i) {
                cost_weight_2 += get_weight_cost(len_signature, i * tn2, (i + 1) * tn2);
                cost_output_2 += get_output_cost(col_signature_2[i], tb2, tn2);
                cost_input_2 += get_input_cost(row_signature_2[i], tb2, tn2);
            }
            // std::cout << "[Input]" << cost_input_1 << std::endl;
            // std::cout << "[Input]" << cost_input_2 << std::endl;

            // std::cout << "[Out]" << cost_output_1 << std::endl;
            // std::cout << "[Out]" << cost_output_2 << std::endl;

            cost_input_1 = cost_input_1 / (col_number / tn1);
            cost_weight_1 = cost_weight_1 / (col_number / tn1);
            cost_output_1 = cost_output_1 / (col_number / tn1);
            balance_cost_1 = balance_cost_1 / (col_number / tn1);
            random_cost_1 = random_cost_1 / (col_number / tn1);


            cost_input_2 = cost_input_2 / (col_number / tn2);
            cost_weight_2 = cost_weight_2 / (col_number / tn2);
            cost_output_2 = cost_output_2 / (col_number / tn2);

            // std::cout << "[2 Input]" << cost_input_2 << std::endl;
            // std::cout << "[2 Weight]" << cost_weight_2 << std::endl;
            // std::cout << "[2 Output]" << cost_output_2 << std::endl;

            // std::cout << std::endl;
            // std::cout << std::endl;
            // std::cout << std::endl;
            

            // std::cout << "[1 Input]" << cost_input_1 << std::endl;
            // std::cout << "[1 Weight]" << cost_weight_1 << std::endl;
            // std::cout << "[1 Output]" << cost_output_1 << std::endl;
            // std::cout << "[1 Balance]" << balance_cost_1 << std::endl;
            // std::cout << "[1 Random]" << random_cost_1 << std::endl;
            // std::cout << "[1 Random Ratio]" << (float)random_cost_1 * tb1 / (float)cost_input_1 << std::endl;


            // std::cout << std::endl;
            // std::cout << std::endl;
            // std::cout << std::endl;

            // std::cout << "[1/2 Input]" << (float)cost_input_1 / (float)(cost_input_2) << std::endl;
            // std::cout << "[1/2 Weight]" << (float)cost_weight_1 / (float)(cost_weight_2) << std::endl;
            // std::cout << "[1/2 Output]" << (float)cost_output_1 / (float)(cost_output_2)<< std::endl;
            // std::cout << "[1/2 Total]" << (float)(cost_input_1 +cost_output_1)  / (float)(cost_output_2 + cost_input_2)<< std::endl;
            // std::cout << "[1/2 Total]" << (float)(cost_input_1 +cost_weight_1+cost_output_1)  / (float)(cost_output_2 + cost_input_2 + cost_weight_2)<< std::endl;



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
            row_number = 0;
            col_number = 0;
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

        void add_edge(int row, int col, float val) {
            coo_values_.push_back({row, col, val});
            row_number = std::max(row_number, row + 1);
            col_number = std::max(col_number, col + 1);
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
        void transpose() {
            std::vector<SparseDataType> csr_values_tmp = csr_values_;
            csr_values_ = csc_values_;
            csc_values_ = csr_values_tmp;

            std::vector<int> csr_index_tmp = csr_index_;
            csr_index_ = csc_index_;
            csc_index_ = csr_index_tmp;

            std::vector<int> csr_len_tmp = csr_len_;
            csr_len_ = csc_len_;
            csc_len_ = csr_len_tmp;
        }

        void print_csr() {
            for(int i = 0; i < csr_len_.size() - 1; ++i) {
                int len = csr_len_[i + 1] - csr_len_[i];
                for(int j = 0; j < len; ++j) {
                    std::cout << csr_index_[csr_len_[i] + j] + 1 << "\t" << i + 1 << "\t" << csr_values_[csr_len_[i] + j] << "\n";
                }
            }
        }

        void print_csc() {
            for(int i = 0; i < csc_len_.size() - 1; ++i) {
                int len = csc_len_[i + 1] - csc_len_[i];
                for(int j = 0; j < len; ++j) {
                    std::cout << csc_index_[csc_len_[i] + j] + 1 << "\t" << i + 1 << "\t" << csc_values_[csc_len_[i] + j] << "\n";
                }
            }
        }
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
                return *this;
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
                return *this;
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
        // int blocksize = 4;
        // int neuron = 16;
        // int buffsize = 6;
        // int WRAPSIZE = 2;

        int blocksize;
        int neuron;
        int buffsize = 24 *1024/sizeof(float)/12;
        int WRAPSIZE = 32;

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
        UIUCMatrix(CSRCSCMatrix &csr_csc, int block_size, int n) {
            blocksize = block_size;
            neuron = n;

            int numblocks = neuron / blocksize; // 4, 16, 4
            int numwarp = blocksize / WRAPSIZE; // 2, 4, 2
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
                numbuff[b] = (footprint + buffsize - 1)/buffsize; // buffsize = 6
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
                        int buff = footprint / buffsize; // buffsize = 6
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
                            for(; iter != csr_csc.row_iter_end_at(b*blocksize+warp*WRAPSIZE+t); ++iter) {
                                if(temp[(*iter).col]==buff) tempnz[t]++;
                            }
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
                                    warpvalue[ind] = 0.0625;
                                    tempnz[t]++;
                                }
                            }
                        }
                    }
                }
            }   
        }

    };


    class BFMatrix : public SparseMatrix{
    public:
        int TN;
        int neuron;
        int* rowoff;
        int* rowindex;
        float* val;

        BFMatrix(COOMatrix &coo_matrix, int n, int tn) : TN(tn) {
            neuron = n;
            rowoff = new int [neuron * (neuron / TN + 1) + 1];
            for(int i = 0; i < neuron * (neuron / TN + 1) + 1; ++i) {
                rowoff[i] = 0;
            }
            rowoff[0] = 0;
            rowindex = new int [32 * neuron];
            val = new float [32 * neuron];
            init_with_coo_file(coo_matrix);
        }

        void init_with_coo_file(COOMatrix &coo_matrix) {
            int last_col = 0;
            int len = 0;
            coo_matrix.to_col_first_ordered();
            int sec_number = neuron / TN;
            std::vector<int> sec_inc(sec_number, 0);
            for(int i = 0; i < coo_matrix.coo_values_.size(); ++i) {
                int row = coo_matrix.coo_values_[i].row;
                int col = coo_matrix.coo_values_[i].col;
                float v = coo_matrix.coo_values_[i].val;
                int sec_idx = row / TN;

                rowoff[sec_idx * neuron + col + 1]++;

                val[sec_idx * 32 * TN + sec_inc[sec_idx]] = v;
                rowindex[sec_idx * 32 * TN + sec_inc[sec_idx]] = row;
                sec_inc[sec_idx] = sec_inc[sec_idx] + 1;
            }


            for(int i = 1; i < neuron * (neuron / TN + 1) + 1; ++i) {
                rowoff[i] += rowoff[i - 1];
            }
            coo_matrix.to_col_first_ordered();
            // for(int i = 0; i < neuron * (neuron / TN + 1) + 1; ++i) {
            //     if(i % neuron == 0) printf("%d \n", rowoff[i]);
            //     // printf("%d, ", rowoff[i]);  
            // }
        }
    };


    
    class cuSPARSEMatrix : public SparseMatrix{
    public:
        int neuron;
        int* len;
        int* index;
        float* val;

        cuSPARSEMatrix(COOMatrix &coo_matrix, int n) {
            neuron = n;
            len = new int [neuron + 1];
            for(int i = 0; i < neuron + 1; ++i) {
                len[i] = 0;
            }
            index = new int [32 * neuron];
            val = new float [32 * neuron];
            init_with_coo_file(coo_matrix);
        }

        void init_with_coo_file(COOMatrix &coo_matrix) {
            coo_matrix.to_row_first_ordered();
            int l = 0;
            for(int i = 0; i < coo_matrix.coo_values_.size(); ++i) {
                int r = coo_matrix.coo_values_[i].row;
                int c = coo_matrix.coo_values_[i].col;
                float v = coo_matrix.coo_values_[i].val;
                len[r + 1]++;
                val[l] = v;
                index[l++] = c;
            }
            for(int i = 1; i < neuron + 1; ++i) {
                len[i] += len[i - 1];
            }
            coo_matrix.to_col_first_ordered();
        }
    };


    class SNIGMatrix : public SparseMatrix{
    std::vector<Elm> coo_values_;
    public:
        int sec_size;
        int num_secs;
        int nnzs;
        int neuron;
        int* row;
        int* col;
        float* val;
        void to_col_first_ordered() {
            std::sort(coo_values_.begin(), coo_values_.end(), [](const Elm& e1, const Elm&e2)->bool {
                if(e1.col != e2.col) return e1.col < e2.col;
                else return e1.row < e2.row;
            });
        }
        SNIGMatrix(std::string file_name, int nn, int sec_size_, int n) : sec_size(sec_size_) {
            nnzs = nn;
            neuron = n;
            num_secs = neuron / sec_size;
            col = new int [num_secs * neuron + 1];
            col[0] = 0;
            row = new int [nnzs];
            val = new float [nnzs];
            init_with_coo_file(file_name);
        }

        void init_with_coo_file(std::string &coo_input_file) {
            std::ifstream input_file(coo_input_file);
            if(!input_file){
                std::cout << "File:" << coo_input_file << " does not exists.\n";
                exit(-1);
            }
            int u_f, v_f;
            float val_f;
            // std::vector<int> ori_row;
            // std::vector<int> ori_col;
            // std::vector<float> ori_val;

            // while(input_file >> u_f >> v_f >> val_f) {
            // while(input_file >> v_f >> u_f >> val_f) {
            //     ori_row.push_back(u_f - 1);
            //     ori_col.push_back(v_f - 1);
            //     ori_val.push_back(val_f);
            // }

            while(input_file >> v_f >> u_f >> val_f) {
                 coo_values_.push_back({u_f - 1, v_f - 1, val_f});
            }


            // for(int col_idx = 0; col_idx < neuron; ++col_idx) {
            //     for(int sec_idx = 0; sec_idx < num_secs; ++sec_idx) {
            //         int len = 0;
            //         for(int iter = 0; iter < ori_col.size(); ++iter) {
            //             if(ori_col[iter] == col_idx && ori_row[iter] / sec_size == sec_idx) {
            //                 row[add_nums] = ori_row[iter];
            //                 val[add_nums] = ori_val[iter];
            //                 add_nums++;
            //                 len++;
            //             }
            //         }
            //         col[col_idx * num_secs + sec_idx + 1] = col[col_idx * num_secs + sec_idx] + len;
            //     }
            // }

            // for(int sec_idx = 0; sec_idx < num_secs; ++sec_idx) {//ori_row有序
            //     for(int col_idx = 0; col_idx < neuron; ++col_idx) {
            //         int len = 0;
            //         for(int iter = 0; iter < ori_col.size(); ++iter) {
            //             if(ori_col[iter] == col_idx && ori_row[iter] / sec_size == sec_idx) {
            //                 row[add_nums] = ori_row[iter];
            //                 val[add_nums] = ori_val[iter];
            //                 add_nums++;
            //                 len++;
            //             }
            //         }
            //         col[sec_idx * neuron + col_idx + 1] = col[sec_idx * neuron + col_idx ] + len;
            //     }
            // }

            this->to_col_first_ordered();
            for(int sec_idx = 0; sec_idx < num_secs; ++sec_idx){
                int col_increment = 0;
                for(int iter = 0; iter < coo_values_.size(); iter+=32) {
                    int len = 0;
                    for(int idx = iter; idx < iter + 32; ++idx){
                        if(coo_values_[idx].row / sec_size == sec_idx)
                            len++;
                    }
                    col[sec_idx * neuron + col_increment + 1] = col[sec_idx * neuron + col_increment ] + len;
                    col_increment++;
                }
            }

            
            for(int iter = 0; iter < coo_values_.size(); iter+=32) {// iter/32为colidx
                for(int sec_idx = 0; sec_idx < num_secs; ++sec_idx){
                    int add_nums = 0;
                    int col_sum = col[iter / 32 + sec_idx * neuron] - col[0];
                    for(int idx = iter; idx < iter + 32; ++idx){
                        if(coo_values_[idx].row / sec_size == sec_idx) {
                            row[col_sum + add_nums] = coo_values_[idx].row;
                            val[col_sum + add_nums] = coo_values_[idx].val;
                            add_nums++;
                            }
                    }
                }
            }

            // for(int i = 0; i < num_secs * neuron + 1; ++i) {
            //     printf("%d, ", col[i]);
            //     if(i % 100 == 0) {
            //         printf("\n");
            //     }
            // }
            // printf("\n");
        }
    };
};

