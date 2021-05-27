#pragma once

#include "../utils/header.h"
#include "reorder.h"

namespace ftxj {
    class HashReorder : public Reorder {
        
        int buckets_num_;
        int max_domain_;
        int buckets_width;
        REORDER type_;

        int hash(int v) {
            if(v >= max_domain_) {
                std::cout << "ERROR: Hasing Function error!" << std::endl;
                exit(-1);
            }
            int col = v % buckets_num_;
            int row = v / buckets_num_;
            return row + col * buckets_width;
        }
    public:
        HashReorder(int buckets_num, int max_domain, REORDER type = ALL_REORDER) 
            : buckets_num_(buckets_num), max_domain_(max_domain), type_(type) {
            buckets_width = max_domain / buckets_num_;
        }

        int reorder(int r) {
            return hash(r);
        }

        MatrixPos new_pos(const MatrixPos &old_pos) {
            MatrixPos n_pos = old_pos;
            if(type_ == COL_REORDER || type_ == ALL_REORDER) {
                n_pos.col_idx = hash(n_pos.col_idx);
            }
            if(type_ == ROW_REORDER || type_ == ALL_REORDER) {
                n_pos.row_idx = hash(n_pos.row_idx);
            }
            return n_pos;
        }
    };
}