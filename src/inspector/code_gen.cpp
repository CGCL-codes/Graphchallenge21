# AC = AB * BC

START_LINE = """
ROW_SUCC_LEN, COL_SUCC_LEN, 
for(int i = 0; i < ROW_SUCC_LEN; ++i) {
    for(int j = 0; j < COL_SUCC_LEN; ++j) {
        AC[(A_Offset + threadIdx.x) * C_Dim + C_Offset] += AB[] * BC[];
    }
}
"""

