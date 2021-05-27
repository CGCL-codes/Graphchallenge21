#pragma once

#include "../utils/matrix_base.h"

namespace ftxj {

enum REORDER {
    COL_REORDER,
    ROW_REORDER,
    ALL_REORDER
};

class Reorder {
public:
    virtual MatrixPos new_pos(const MatrixPos &old_pos) = 0;
    virtual int reorder(int r) = 0;

};

};