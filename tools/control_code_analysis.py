file = "../3rd_party/20-graphchallenge/SpDNN_Challenge2020/singlegpu/kernel.sass"
control_line = 0
with open(file) as f:
    line = f.readline()
    x = 1
    gap = 0
    while line:
        idx = line.rfind('/*')
        if idx != -1:
            if control_line == 1:
                hex = "0x" + line[idx + 9 : idx + 11]
                # print(hex)
                stalls = int(hex, 16) 
                stalls = (stalls >> 1) & 0x0f
                # if yield_code == 0:
                #     print(x, gap)
                #     gap = 0
                # else:
                #     gap = gap + 1
                print(x, stalls)
            control_line = (control_line + 1) % 2
        line = f.readline()
        x = x + 1