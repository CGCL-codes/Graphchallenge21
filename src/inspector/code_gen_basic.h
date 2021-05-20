#pragma once

#include "utils/string.h"
#include <map>
#include <vector>
#include <stack>
#include <map>

namespace ftxj {

    enum VariableSpace {
        Reg,
        Global,
        Shared,
        Constant
    };

    enum VariableLink {
        InputDense,
        InputSparse,
        Output,
        Others
    };

    enum VariableType {
        f16,
        f32,
        i16,
        i32
    };

    class Statement {
        Code &code;
        int line_number;
        public:
        virtual std::string gen_statement() = 0;
        void emit_statement() {
            code.push_code(gen_statement());
        }
    };

    class VariableDecl : public Statement {
        bool is_const;
        bool is_restrict; // solve pointer aliasing problem, https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing
        bool is_pointer;
        bool is_extern;

        VariableSpace space_;
        VariableLink link_;
        VariableType type_;
        std::string name_;

        static std::map<VariableSpace, std::string> variable_space2string = {
            {Reg, "register"},
            {Global, ""},
            {Shared, "__shared__"},
            {Constant, "__constant__"}
        };

        static std::map<VariableType, std::string> variable_type2string = {
            {f16, "half"},
            {f32, "float"},
            {i16, "short"},
            {i32, "int"}
        };

        public:
        VariableDecl(VariableType type, std::string name, VariableSpace space, bool is_p) : 
            type_(type), name_(name), space_(space) {
            is_const = false;
            is_restrict = false; 
            is_pointer = is_p;
            is_extern = false;
        }
        
        void set_extern() {
            is_extern = true;
        }
        void set_restrict() {
            is_restrict = true;
        }
        void set_pointer() {
            is_pointer = true;
        }
        void set_const() {
            is_const = true;
        }

        std::string get_name() {
            return name_;
        }

        std::string gen_statement() {
            std::string instr;
            if(is_extern) instr += "extern ";
            if(is_const) instr += "const ";
            instr += variable_space2string[space_] + " ";
            instr += variable_type2string[type_] + " ";
            if(is_pointer) instr += "* ";
            if(is_restrict) instr += "__restrict__ ";
            instr += name_;
            return instr;
        }
        
        ArrayAccess operator[](Operation &i) {
            Variable* var = new Variable(this);
            return ArrayAccess(var, &i);
        }
    };

    class VariableArrayDecl : public Statement {
        VariableDecl *var;
        std::vector<ConstantVar> dim_len_;
        public:

        VariableArrayDecl(VariableType type, std::string name, VariableSpace space, bool is_p, std::vector<ConstantVar> &dim_len) {
            var = new VariableDecl(type, name, space,is_p);
            dim_len_ = dim_len;
        }


        VariableArrayDecl(VariableDecl *v, std::vector<ConstantVar> &dim_len) 
            : var(v), dim_len_(dim_len)  
        {

        }

        std::string gen_statement() {
            std::string instr;
            instr += var->gen_statement();
            if(var->is_extern()) {
                return instr;
            }
            for(auto dim : dim_len_) {
                instr += "[" + dim.gen_statement() + "]";
            }
            return instr;
        }
    };

    class VariableArrayInit : public Statement {
        VariableArrayDecl* array_decl_;
        ConstantVar* value;
        public:
        VariableArrayInit(VariableArrayDecl* array_decl, ConstantVar* var) : 
            array_decl_(array_decl), value(var) {

        }

        std::string gen_statement() {
            std::string insrt;
            instr += array_decl_->gen_statement();
            instr += "= {" + var->gen_statement() + "}";
            return instr;
        }
    }

    class ForLoopScope : public Statement {
        Statement* expr1;
        Statement* expr2;
        Statement* expr3;
        Variable* iter;
        public:
        ForLoopScope(Statement* expr_1, Statement* expr_2, Statement* expr_3, Variable* iter_) : {
            expr1 = expr_1;
            expr2 = expr_2;
            expr3 = expr_3;
            iter = iter_;
        }

        std::string gen_statement() {
            std::string instr;
            instr += "for(int " + iter->gen_statement() + " = " + expr1->gen_statement() + "; ";
            instr += iter->gen_statement() + " < " + expr2->gen_statement() + "; ";
            instr += iter->gen_statement() + " += " + expr3->gen_statement() + "){";
            return instr;  
        }
    };

    class ScopeEnd : public Statement {
        public:
        std::string gen_statement() {
            std::string instr;
            instr += "}";
            return instr;
        }
    };

    class WrapSync : public Statement {
        public:
        std::string gen_statement() {
            return "__syncthreads()";  
        }
    };

    class Operation : public Statement {
        Operation* left_;
        Operation* right_;
        std::string op_;
        public:
        Operation(std::string op, Operation* l, Operation* r) : left_(l), right_(r), op_(op) {}
        Operation operator+(const Operation &that) {
            Operation res("+", this, &that);
            return res;
        }

        Operation operator*(const Operation &that) {
            Operation res("*", this, &that);
            return res;
        }

        
        Operation operator<(const Operation &that) {
            Operation res("<", this, &that);
            return res;
        }

        Operation operator>(const Operation &that) {
            Operation res(">", this, &that);
            return res;
        }

        Operation operator==(const Operation &that) {
            Operation res("==", this, &that);
            return res;
        }

        Operation operator!=(const Operation &that) {
            Operation res("!=", this, &that);
            return res;
        }

        Operation operator=(const Operation &that) {
            Operation res("=", this, &that);
            return res;
        }

        Operation operator+=(const Operation &that) {
            Operation res("+=", this, &that);
            return res;
        }

        std::string gen_statement() {
            std::string instr;
            instr += left->gen_statement();
            instr += op;
            instr += right->gen_statement();
        }
    };

    class Variable : public Operation {
        VariableDecl* father_;
        public:
        
        Variable(VariableDecl* vdecl) : father_(vdecl) {}

        std::string gen_statement() {
            return father_->get_name();  
        }
        
        ArrayAccess operator[](Operation &i) {
            ArrayAccess ret(this, &i);
            return ret;
        }
    };

    class ConstantVar : public Operation {
        std::string name_;
        public:
        ConstantVar(std::string name) : name_(name) {}
        ConstantVar(int i) : name_(std::to_string(i)) {}
        
        std::string gen_statement() {
            return name_;  
        }
    };

    class GpuConstant {
        public:
        static ConstantVar ThreadIdx_x("threadIdx.x");
        static ConstantVar ThreadIdx_y("threadIdx.y");
        static ConstantVar ThreadIdx_z("threadIdx.z");
        
        static ConstantVar BlockIdx_x("blockIdx.x");
        static ConstantVar BlockIdx_y("blockIdx.y");
        static ConstantVar BlockIdx_z("blockIdx.z");

        static ConstantVar BlockDim_x("blockDim.x");
        static ConstantVar BlockDim_y("blockDim.y");
        static ConstantVar BlockDim_z("blockDim.z");
        
        static ConstantVar GridDim_x("gridDim.x");
        static ConstantVar GridDim_y("gridDim.y");
        static ConstantVar GridDim_z("gridDim.z");

        static ConstantVar WrapSize(32);
        static ConstantVar SubWrap(16);
        static ConstantVar MinWrap(8);
    };

    class ArrayAccess : public Operation {
        Variable* var;
        Operation* op;
        public:
        ArrayAccess(Variable* v, Operation* p) : var(v), op(p) {}
        std::string gen_statement() {
            return var->gen_statement() + "[" + op->gen_statement() +"]";
        }
    }

    class VaribaleInit : public Statement {
        VariableDecl* decl_;
        Operation* init_statement_;
        public:
        std::string gen_statement() {
            return decl_->gen_statement() + " = " + init_statement_->gen_statement();  
        }
    };

    class ASMInline : public Statement {
        std::string inline_code_;
        public:
        ASMInline(std::string inline_code) : inline_code_(inline_code) {}
        std::string gen_statement() {
            std::string instr;
            instr += "asm(" + inline_code_ + ")";
            return instr;
        }
    };

    class GpuGlobalFunction {
        
        int launch_bound_thread;
        int launch_bound_memory;
        std::string name_;
        std::vector<VariableDecl*> param_list_;
        public:

        GpuGlobalFunction(
            std::string name, std::vector<VariableDecl*> &param_list, 
            int launch_bound_1, int launch_bound_2) : 
        
            param_list_(param_list), name_(name), 
            launch_bound_thread(launch_bound_1), launch_bound_memory(launch_bound_2)
        {

        }
        std::string gen_statement() {
            std::string instr;
            instr += "__global__ void " + "__launch_bounds__(" + 
                std::to_string(launch_bound_thread) + ", " +
                std::to_string(launch_bound_memory) + ") " +
                name_ + "(";
            for(int i = 0; i < param_list_.size(); ++i) {
                instr += param_list_[i]->gen_statement();
                if(i < param_list_.size() - 1) {
                    instr += ", ";
                }
            }
            instr += "){";
            return instr;
        }


    };

    class AtomicAdd : public Statement {
        Operation* addr_;
        Operation* val_;
        public:
        AtomicAdd(Operation* addr, Operation* val) : addr_(addr), val_(val) {

        }
        std::string gen_statement() {
            std::string instr;
            instr += "atomicAdd(" + addr_.gen_statement() + ", " + val_.gen_statement() + ")";
        }

    };

    class IfScope : public Statement {
        Operation* val_;
        public:
        IfScope(Operation* val) : val_(val){}
        std::string gen_statement() {
            std::string instr;
            instr += "if(" + val_->gen_statement() + ")" + "{";
            return instr;
        }
    };

    class GroupScope : public Statement {
        int group_id_;
        int block_id_;
        public:
        GroupScope(int block_id, int group_id) : block_id_(block_id), group_id_(group_id) {}
        std::stride emit_statement() {
            std::string instr;
            instr += "if(groupId == " + std::to_string(group_id) + "){\n";
            instr += "asm(//" + "B" + std::to_string(block_id) + "G" + std::to_string(group_id) + ")";
            return instr;
        }
    };

    class BlockScope : public Statement {
        int block_id_;
        public:
        BlockScope(int block_id) : block_id_(block_id) {}
        std::stride emit_statement() {
            std::string instr;
            instr += "if(blockIdx.x == " + std::to_string(block_id_) + "){\n";
            return instr;
        }
    };

    class Context {
    public:
        std::string get_global_A_name();
        std::string get_global_B_name();
        std::string get_global_C_name();
    };
}