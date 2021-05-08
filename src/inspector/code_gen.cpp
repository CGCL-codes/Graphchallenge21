#include "code_gen_basic.h"


void generate_20champion_code {
    std::vector<VariableDecl*> param_list_;

    VariableDecl nextfeat(f32, "nextfeat", Global, true);
    param_list_.push_back(&nextfeat);

    VariableDecl currfeat(f32, "currfeat", Global, true);
    param_list_.push_back(&currfeat);

    VariableDecl buffsize(i32, "buffsize", Global, false);
    param_list_.push_back(&buffsize);
 
    VariableDecl buffdispl(i32, "buffdispl", Global, true);
    param_list_.push_back(&buffdispl);

    VariableDecl mapdispl(i32, "mapdispl", Global, true);
    param_list_.push_back(&mapdispl);

    VariableDecl map(i16, "map", Global, true);
    param_list_.push_back(&map);

     
    VariableDecl displ(i32, "displ", Global, true);
    param_list_.push_back(&displ);

    VariableDecl index(i16, "index", Global, true);
    param_list_.push_back(&index);

    VariableDecl value(f32, "value", Global, true);
    param_list_.push_back(&value);

    VariableDecl bias(f32, "bias", Global, false);
    param_list_.push_back(&bias);

    VariableDecl neuron(i32, "neuron", Global, false);
    param_list_.push_back(&neuron);

    VariableDecl categories(i32, "categories", Global, true);
    param_list_.push_back(&categories);

    VariableDecl active(i32, "active", Global, true);
    param_list_.push_back(&active);

    GpuGlobalFunction dummy_kernel("dummy_kernel", param_list_, 1024, 1);
    dummy_kernel.emit_statement();
    ScopeEnd dummy_kernel_end;

    VariableArrayDecl shared

    dummy_kernel_end.emit_statement();

    VariableDecl wind();
    
    ConstantVar WARPSIZE();
    ConstantVar ThreadIdx_x();

    Operation tmp = ThreadIdx_x % WARPSIZE;    

    VaribaleInit wind_init(wind, tmp);

    buffdispl[ThreadIdx_x];
}