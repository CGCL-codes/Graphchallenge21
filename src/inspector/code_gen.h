#pragma once

#include "utils/string.h"
#include <map>
#include <vector>

namespace ftxj {
    
    enum VariableType {
        Reg,
        Global,
        Shared,
        Constant
    };

    enum LoadType {
        ThreadShared,
        ThreadExclusove
    };

    enum CmpOp {
        Equal = 0,
        NotEqual,
        Less,
        Greater
    };

    std::vector<std::string> CmpOp2String = {
        "==", "!=", "<", ">"
    };

    class Variable {
        std::string name_;
        int len_;
        VariableType type_;
    public:
        std::string get_name() {
            return name_;
        }
        std::string access_index() {

        }
    };

    class Code {
        std::map<std::string, int> symbol_;
        std::vector<std::string> code_txt_;
    public:
        void push_code(std::string &code) {
            code_txt_.push_back(code);
        }
    };



    class CodeGen {
        Code code;

        std::vector<WrapBlock> wrap_block_schedule; 
        
        Schedule group_task_schedule;

        Variable dense_matrix_block_mem;
        Variable sparse_matrix_block_mem;
        Variable output_matrix_block_mem;

        
        
        const std::string threadIdx_x = "threadIdx.x";
        const std::string blockIdx_x = "blockIdx.x";
        const std::string blockDim_x = "blockDim.x";
        const std::string groupIdx = "groupIdx";
        

        Variable emit_reg_variable_declare_statement(std::string &name, std::string type) {
            std::string instr;
            instr += "register " + type + " " + name + " = 0";
            code.push_code(instr);
            Variable tmp;
            return tmp;
        }

        std::string gen_bianry_op(std::string &left, std::string &right, std::string &op) {
            return left + " " + op + " " + right;
        }

        std::string gen_fma(std::string &res, std::string &left, std::string &right) {
            return res + " += " + left + " * " + right;
        }

        std::string gen_memory_access(std::string name, int bias) {
            return name + "[" + std::to_string(bias) + "]";
        }
        
        std::string gen_memory_access(std::string name, std::string bias) {
            return name + "[" + bias + "]";
        }

        std::string gen_assign(std::string &left, std::string &right) {
            return left + "=" + right;
        }

        void emit_for_loop_begin(std::string expr_1, std::string expr_2, std::string expr_3, std::string iter_var = "i") {
            std::string instr;
            instr += "for(int " + iter_var + " = " + expr_1 + "; ";
            instr += "i < " + expr_2 + "; "
            instr += "i += " + expr_3 + ") {";
            code.push_code(instr);
        }

        void emit_for_loop_end() {
            std::string instr;
            instr += "}";
            code.push_code(instr);
        }

        void emit_if_statement_begin(std::string &expr1, std::string &expr2, CmpOp &op) {
            std::string instr;
            instr += "if(" + expr1 + CmpOp2String[op] + expr2 + "){";
            code.push_code(instr);
        }

        void emit_if_statement_end(std::string &expr1, std::string &expr2, CmpOp &op) {
            std::string instr;
            instr += "}";
            code.push_code(instr);
        }

    public:
        void emit_inline_asm(std::string &asm_code) {
            std::string instr;
            instr += "asm(" + asm_code + ")";
            code.push_code(instr);
        }

        void emit_inline_asm_note(std::string &note) {
            std::string instr;
            instr += "//" + note;
            emit_inline_asm(instr);
        }

        void emit_unroll_one2many_compute() {
            Variable reg("dense_tmp", 1);
            emit_load_dense_to_reg(reg, bias);
            gen_fma(output_matrix, sparse_matrix, dense_matrix);
        }
        
        void emit_thread_parallel_for_loop_begin(int low, int high, std::string iter_var = "i") {
            std::string expr_1, expr_2, expr_3;
            
            expr_1 = std::to_string(low) + threadIdx_x;
            expr_2 = std::to_string(high);
            expr_3 = blockDim_x;

            emit_for_loop_begin(expr_1, expr_2, expr_3, iter_var);
        }

        void emit_load_dense_to_reg(Variable &reg_array, int bias) {

        }

        void emit_load_sparse_to_reg(Variable &reg, std::string &subscript) {

        }

        void emit_load_dense_to_reg(Variable &reg_array, int bias, int size) {
            
            emit_thread_parallel_for_loop_begin(bias, bias + size);

            std::string instr;

            auto reg_access = emit_memory_access(reg_array.get_name(), bias);

            emit_for_loop_end(code);
        }

        void emit_block_start_control(int block_id) {
            emit_if_statement_begin(blockIdx_x, std::to_string(block_id), Equal);
        }

        void emit_block_end_control() {
            emit_if_statement_end();
        }

        void emit_group_start_control(int block_id, int group_id) {
            std::string notes = "B" + std::to_string(block_id) + "G" + std::to_string(group_id);
            emit_inline_asm_note(notes);
            emit_if_statement_begin(groupIdx, std::to_string(group_id), Equal);
        }

        void emit_group_end_control() {
            std::string notes = "END";
            emit_inline_asm_note(notes);
            emit_if_statement_end();
        }

        void emit_feature_axis_parallel_row_line_unroll(int block_id, int group_id, RowLineBlock &row_line_block) {
            std::string tmp_reg_name = "tmp_dense_feature";
            Variable tmp = emit_reg_variable_declare_statement(tmp_reg_name, "float");
            
            std::string feature_idx = group_task_schedule.get_feature_begin(block_id, group_id);
            std::string stride = group_task_schedule.get_feature_stride(block_id, group_id);
            std::string max_feature = group_task_schedule.get_feature_number(block_id, group_id);

            std::string loop_iter_name = "feature_idx";

            emit_for_loop_begin(feature_idx, max_feature, stride, loop_iter_name);
            {
                emit_load_dense_to_reg(tmp, loop_iter_name);
                for(int i = 0; i < row_line_block.get_line_len(); ++i) {
                    auto output_idx = gen_memory_access(output_matrix_block_mem, i);
                    gen_fma(output_idx, tmp, row_line_block.get_value(i));
                }
            }
            emit_for_loop_end();

        }


        void emit_feature_axis_parallel_row_line_no_unroll(int block_id, int group_id, RowLineBlock &row_line_block) {
            std::string tmp_reg_name = "tmp_dense_feature";
            Variable tmp = emit_reg_variable_declare_statement(tmp_reg_name, "float");
            
            std::string feature_idx = group_task_schedule.get_feature_begin(block_id, group_id);
            std::string stride = group_task_schedule.get_feature_stride(block_id, group_id);
            std::string max_feature = group_task_schedule.get_feature_number(block_id, group_id);

            std::string loop_iter_name = "feature_idx";
            
            emit_for_loop_begin(feature_idx, max_feature, stride, loop_iter_name);
            {
                emit_load_dense_to_reg(tmp, feature_idx);
            
                std::string expr1 = "0";
                std::string expr2 = row_line_block.get_line_len();
                std::string expr3 = "++";
                std::string iter_name = "i";
                
                emit_for_loop_begin(expr1, expr2, expr3, iter_name);
                {
                    auto output_value = gen_memory_access(output_matrix_block_mem, iter_name);
                    auto sparse_value = gen_memory_access(sparse_matrix_block_mem, iter_name);

                    gen_fma(output_value, tmp, sparse_value);
                }
                emit_for_loop_end();
            }
            emit_for_loop_end();
        }

        void emit_output_axis_parallel_row_line_unroll(int block_id, int group_id, RowLineBlock &row_line_block, Variable &tmp) {
            
            std::string feature_idx = group_task_schedule.get_feature_begin(block_id, group_id);
            std::string stride = group_task_schedule.get_feature_stride(block_id, group_id);
            std::string max_feature = group_task_schedule.get_feature_number(block_id, group_id);

            std::string loop_iter_name = "feature_idx";

            emit_load_sparse_to_reg(tmp, loop_iter_name);

            emit_for_loop_begin(feature_idx, max_feature, stride, loop_iter_name);
            {
                for(int i = 0; i < row_line_block.get_line_len(); ++i) {
                    auto output_idx = gen_memory_access(output_matrix_block_mem, i);
                    gen_fma(output_idx, tmp, row_line_block.get_value(i));
                }
            }
            emit_for_loop_end();
        }


        void emit_output_axis_parallel_row_line_no_unroll(int block_id, int group_id, RowLineBlock &row_line_block) {
            
        }



        void emit_output_axis_parallel_row_line() {
            
        }

        void emit_group_code(int block_id, int group_id, MatrixBlockBase* matrix_block) {
            switch(matrix_block->get_block_type()) {
                case Random:
                break;
                case Rectangles:
                break;
                case Row_Line:
                break;
            }
        }

        void run(int block_num, int thread_group_num) {
            for(int i = 0; i < block_num; ++i) {
                emit_block_start_control(i);
                for(int j = 0; j < thread_group_num; ++j) {
                    emit_group_start_control(i, j);
                    WrapBlock wb = wrap_block_schedule(i, j);
                    for(auto mb = wb.begin_block(); mb != wb.end_block(); ++mb) {
                        MatrixBlockBase* matrix_block = *mb;
                        emit_group_code(block_id, group_id, matrix_block);
                    }
                    emit_group_end_control();
                }
                emit_block_end_control();
            }
        }
    };

};