#pragma once

namespace ftxj {
    class MatrixTask {
        std::vector<MatrixBlockBase> task_blocks_;
    };

    class BlockDistributor {
    public:
        virtual std::vector<MatrixTask> run() = 0; 
    };

    class OneDimSplitDistributor : public BlockDistributor{
    public:
        // assign all blocks in the same row/col block to one Task
        std::vector<MatrixTask> run() {

        }
    };
};