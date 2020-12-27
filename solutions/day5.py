#!/usr/bin/env python
import re
import sys
import math

Gff_file = open(sys.argv[1],'r')
seat_list = []
for line in Gff_file:
    line = line.rstrip('\n')
    lw_r = 0
    hg_r = 127
    lw_c = 0
    hg_c = 7
    row_val = 128
    col_val = 8
    for i in list(line[:7]):
        #print(line[:7])
        row_val = row_val/2
        if row_val > 1:
            #print(i, end='')
            if i == 'F':
                hg_r = hg_r - row_val
            else:
                lw_r = lw_r + row_val
        else:
            #print(i)
            if i == 'F':
                row = lw_r
            else:
                row = hg_r
    for i in list(line[7:]):
        col_val = col_val/2
        if col_val > 1:
            #print(i, end='')
            if i == 'L':
                hg_c = hg_c - col_val
            else:
                lw_c = lw_c + col_val
        else:
            #print(i)
            if i == 'L':
                col = lw_c
            else:
                col = hg_c
    seatID = (row * 8) + col
    seat_list.append(seatID)
#print(max(seat_list))
for i in range(int(max(seat_list))):
    if i not in seat_list:
        if i-1 in seat_list and i+1 in seat_list:
            print(i)

Gff_file.close()