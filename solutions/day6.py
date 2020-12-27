#!/usr/bin/env python
import re
import sys
import math
import json

Gff_file = open(sys.argv[1],'r')
main_line = ''
dict_list = {}
grp_count = 0
que_count = 0
que_dict = {}

for line in Gff_file:
    if line == '\n':
        tmp_list = re.split(',', main_line)
        tmp = tmp_list[0]
        grp_count = 0
        str_list = list(tmp)
        if len(tmp_list) > 1:
            for que in str_list:
                for i in range(1,len(tmp_list)):
                    if que in tmp_list[i]:
                        grp_count += 1
                if grp_count + 1 == len(tmp_list):
                    que_dict[que] = 1
                grp_count = 0
            que_count += len(que_dict)
        else:
            for que in str_list:
                que_dict[que] = 1
            que_count += len(que_dict)
        main_line = ''
        que_dict = {}
    else:
        line = line.rstrip('\n')
        if main_line == '':
            main_line = line
        else:
            main_line = main_line + ',' + line


print(que_count)
Gff_file.close()
'''
        for tmp in tmp_list:
            str_list = list(tmp)
            for que in str_list:
                que_dict[que] = 1
        que_count += len(que_dict)

        main_line = ''
        que_dict = {}
    else:
        line = line.rstrip('\n')
        if main_line == '':
            main_line = line
        else:
            main_line = main_line + ',' + line

grp_count += 1
tmp_list = re.split(',', main_line)
for tmp in tmp_list:
    str_list = list(tmp)
    for que in str_list:
        que_dict[que] = 1
que_count += len(que_dict)
'''