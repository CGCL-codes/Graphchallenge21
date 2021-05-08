#pragma once
namespace ftxj {

class Reorder {
public:
    virtual MatrixPos new_pos(MatrixPos &old_pos);
};

class HashReorder : public Reorder {
    
public:
    MatrixPos new_pos(MatrixPos &old_pos) {

    }
};


};