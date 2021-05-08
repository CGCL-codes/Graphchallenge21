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

    VariableDecl shared(f32, "shared", Shared, false);
    shared.set_extern();

    VariableArrayDecl shared_array(shared, {});
    shared_array.emit_statement();

    VariableDecl wind(i32, "wind", Reg, false);
    ConstantVar WARPSIZE("WARPSIZE");
    ConstantVar ThreadIdx_x("threadIdx.x");
    Operation tmp = ThreadIdx_x % WARPSIZE;    
    VaribaleInit wind_init_statement(wind, tmp);
    wind_init_statement.emit_statement();


    ArrayAccess line95_1 = buffdispl[ThreadIdx_x];
    ArrayAccess line95_2 = buffdispl[ThreadIdx_x + 1];
    ConstantVar ConstOne(1);
    VariableDecl iter_var_1(i32, "buff", Global, false);
    Variable iter_var(iter_var_1);
    ForLoopScope forloop_1(line95_1, line95_2, ConstOne, iter_var);
    forloop_1.emit_statement();



    ScopeEnd forloop_1_end;
    forloop_1_end.emit_statement();


    ScopeEnd dummy_kernel_end;
    dummy_kernel_end.emit_statement();
}


void generate_ramdom_block_code(Schedule &block_schedule) {
    VariableArrayDecl output_tile(f32, "output_tile", Reg, false, {8});
    ConstantVar floatZero(0.0);

    VariableArrayInit output_tile_init(&output_tile, floatZero);
    output_tile_init.emit_statement();
    
    VariableDecl dense_tile(f32, "dense_value", Reg, false);
    VaribaleInit dense_tile_init(dense_tile, floatZero);

    for(int b = 0; b < blockSize; ++b) {
        BlockScope block_scope(b);
        block_scope.emit_statement();
        for(int t = 0; t < threadSize; ++t) {
            MatrixBlockBase* base_block = block_schedule.get_block(b, t);
            if(base_block->get_block_type() == "Random") {

            }
            else if() {

            }
        }
        ScopeEnd block_scope_end;
        block_scope_end.emit_statement();
    }

}