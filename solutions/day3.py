#!/usr/bin/env python

import re
import sys
import math

count = 0
str_dict = {}
Gff_file = open(sys.argv[1],'r')
for line in Gff_file:
	line = line.rstrip('\n')
	str_dict[count] = list(line)
	count += 1
Gff_file.close()

def tree(a,b):
	count = 0
	req_index = 0
	tree_count = 0
	while count <= max(str_dict.keys()):
		if (count > 0) and (count % b == 0):
			str_list = str_dict[count]
			req_index += a
			if req_index > (len(str_list) - 1):
				req_index = req_index % len(str_list)
			if str_list[req_index] == '#':
				tree_count += 1
		count += 1
	return tree_count

print(tree(3,1))
print('part 2: ', tree(1,1)*tree(3,1)*tree(5,1)*tree(7,1)*tree(1,2))

'''
count = 0
req_index = 0
tree_count = 0
for line in Gff_file:
    line = line.rstrip('\n')
    str_list = list(line)
    if count > 0:
        req_index += 3
        if req_index > (len(str_list) - 1):
            req_index = req_index % len(str_list)
        if str_list[req_index] == '#':
            tree_count += 1
    count += 1
print(tree_count)


        if count > 0:
            for i in range(len(list) - 1):
                if list[i] + list[-1] == 2020:
                    print(list[i] * list[-1])
                    sys.exit()

    if count > 1:
        for i in range(len(list) - 2):
            for j in range(len(list) - 1):
                if list[i] + list[j] + list[-1] == 2020:
                    print(list[i] * list[j] * list[-1])
                    sys.exit()
    count += 1
'''