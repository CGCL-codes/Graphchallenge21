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

    class Variable {
        std::string name_;
        int len_;
        VariableType type_;
    public:
        std::string get_name() {
            return name_;
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
        
        Variable dense_matrix_block_mem;
        Variable sparse_matrix_block_mem;
        Variable output_matrix_block_mem;

        std::map<std::pair<int, int>, int>
        
        const std::string threadIdx_x = "threadIdx.x";
        const std::string blockDim_x = "blockDim.x";

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

    public:
        void emit_

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

        void emit_load_dense_to_reg(Variable &reg_array, int bias, int size) {
            
            emit_thread_parallel_for_loop_begin(bias, bias + size);

            std::string instr;

            auto reg_access = emit_memory_access(reg_array.get_name(), bias);

            emit_for_loop_end(code);
        }
         
    };

};