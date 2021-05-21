#pragma once
#include <vector>
#include "gpu_block.h"
#include "gpu_run_config.h"
#include "matrix_block_container.h"
// #include "code_gen_basic.h"

#include <map>

namespace ftxj {
    class BlockSchedule {
    protected:
        friend class MaxInReuseBSchedule;
        GpuRunConfig config_;
        std::vector<GpuBlock> schedule_result_;
        BlockContainer &original_data_blocks_;
    public:
        BlockSchedule(BlockContainer &all_data_blocl) : original_data_blocks_(all_data_blocl) {
        }
        virtual void schedule() = 0; 
    };


    class MaxInReuseBSchedule : public BlockSchedule {
        int cal_reuse_time(std::vector<int> a, std::vector<int> b) {
            if(a.size() != b.size()) {
                std::cout << "doesnot support this kind " << std::endl;
                exit(-1);
            }
            int res = 0;
            int i = 0, j = 0;
            while(true) {
                if(i >= a.size() || j >= b.size()) break;
                if(a[i] == b[j]) {
                    res++;
                    i++, j++;
                }
                else if(a[i] < b[j]) {
                    i++;
                }
                else {
                    j++;
                }
            }
            return res;
        }

        void print_reuse_number(std::vector<BlockContainer> &col_container) {
            for(int i = 0; i < col_container.size(); ++i) {
                for(int j = 0; j < col_container.size(); ++j) {
                    if(i == j) continue;
                    int res = cal_reuse_time(col_container[i].access_row_idx, col_container[j].access_row_idx);
                    std::cout << "(" << col_container[i].get_col_idx().first 
                        << ", " << col_container[j].get_col_idx().first 
                        << ", " << res 
                        << ", " << col_container[i].access_row_idx.size() 
                        << ") " << std::endl;
                }
            }
        }

        std::vector<std::vector<int>> greedy_search(std::vector<BlockContainer> &col_container) {
            std::vector<int> visit(col_container.size(), false);
            
            std::vector<std::vector<int>> res;

            for(int i = 0; i < col_container.size(); ++i) {
                if(visit[i] == true) continue;
                visit[i] = true;
                res.push_back(std::vector<int>(1, i));
                for(int j = 0; j < col_container.size(); ++j) {
                    if(visit[j]) continue;
                    int r = cal_reuse_time(col_container[i].access_row_idx, col_container[j].access_row_idx);
                    if(r > 0) {
                        visit[j] = true;
                        res[res.size() - 1].push_back(j);
                    }
                }
            }
            return res;
        }

        struct Affine {
            int slope;
            int bias;
        };

        // std::vector<Affine> affine_test() {
        //     int size = schedule_result_.size();
        //     int need_col = schedule_result_[0].
        //     for(int i = 0; i < size; ++i) {

        //     }
        // }

    public:
        MaxInReuseBSchedule(BlockContainer &all_data_blocl) : BlockSchedule(all_data_blocl) {

        }
        void schedule() {
            std::vector<BlockContainer> col_container = original_data_blocks_.split_by_col();
            std::vector<std::vector<int>> combs = greedy_search(col_container);
            for(int j = 0; j < combs.size(); ++j) {
                auto comb = combs[j];
                std::vector<BlockContainer> need_merge;
                for(int i = 0; i < comb.size(); ++i) {
                    need_merge.push_back(col_container[comb[i]]);
                }
                BlockContainer res_block(BlockContainer::merge(need_merge));
                GpuBlock gpu_block(-1,  j, res_block);
                schedule_result_.push_back(gpu_block);
            }
        }

        struct LineBlock {
            std::vector<float> value;
            std::vector<int> row_access;
        };

        LineBlock get_data() {
            LineBlock res;
            for(auto x : schedule_result_) {
                auto need_merge = x.blocks_.get_line_block_data();
                res.value.insert(res.value.end(), need_merge.value.begin(), need_merge.value.end());
                res.row_access.insert(res.row_access.end(), need_merge.row_access.begin(), need_merge.row_access.end());
            }
            return res;
        }

        void dummy_print_COO() {
            COOMatrix res;

        }

        // std::string gen_col_block_start_address_code(Context &context) {

        // }

        void print_schedule() {
            for(auto x : schedule_result_) {
                x.print();
            }
        }
    };
};