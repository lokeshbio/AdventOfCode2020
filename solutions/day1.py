#!/usr/bin/env python

import re
import sys
import math

Gff_file = open(sys.argv[1],'r')

count = 0
list = []
for line in Gff_file:
    line = line.rstrip('\n')
    list.append(int(line))
    '''
        if count > 0:
            for i in range(len(list) - 1):
                if list[i] + list[-1] == 2020:
                    print(list[i] * list[-1])
                    sys.exit()
    '''
    if count > 1:
        for i in range(len(list) - 2):
            for j in range(len(list) - 1):
                if list[i] + list[j] + list[-1] == 2020:
                    print(list[i] * list[j] * list[-1])
                    sys.exit()
    count += 1