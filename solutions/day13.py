#!/usr/bin/env python
import re
import sys
import math
import copy

p1 = re.compile(',')


ln = 0

Gff_file = open(sys.argv[1],'r')
for line in Gff_file:
	line = line.rstrip('\n')
	if ln != 0:
		ID_list = re.split(p1, line)
	ln += 1
Gff_file.close()


tim_stmp = 379786358533323
count = 0
while count < len(ID_list):
	main_dict = {}
	count = 0
	tim_stmp += 1
	for idx,id in enumerate(ID_list):
		if id != 'x':
			if (tim_stmp + idx) % int(id) == 0:
				count += 1
			else: 
				break	
		else:
			count += 1

print(tim_stmp)
'''
for i in range(max(new_dict.keys()) +1):
			for j in range(row_len):
				print(new_dict[i][j], end = '')
			print()
'''

	
