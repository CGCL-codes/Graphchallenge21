#pragma once
#include <vector>
#include "gpu_block.h"
#include "gpu_run_config.h"
#include "matrix_block_container.h"

namespace ftxj {
    class BlockSchedule {
        GpuRunConfig config_;
        std::vector<GpuBlock> schedule_result_;
        BlockContainer original_data_blocks_;
    public:
        virtual void schedule() = 0; 
    };


    class MaxInReuseBSchedule : public BlockSchedule {
    public:
        void schedule() {
            
        }
    };
};