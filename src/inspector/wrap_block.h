#pragma once

namespace ftxj {
    // AC = AB * BC 
    // dense = dense * sparse
    class WrapBlock {
        private:
        int wrap_id_;
            int block_id_;

            int batch_dim_;
            int output_channel_dim_;
            int input_channel_dim_;

            int batch_offset_;
            int output_channel_offset_;
            int input_channel_offset_;

            int write_dst_;

        public:

        static const int WRAP_SIZE = 32;
    };
};