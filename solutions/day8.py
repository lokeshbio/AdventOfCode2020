#!/usr/bin/env python
import re
import sys
import math

p1 = re.compile(' ')

main_dict = {}
count = 0
acc_sum = 0

Gff_file = open(sys.argv[1],'r')
for line in Gff_file:
	line = line.rstrip('\n')
	tmp_list = re.split(p1, line)
	main_dict[count] = {}
	main_dict[count][tmp_list[0]] = int(tmp_list[1])
	count += 1

#print(max(main_dict.keys()))
P_list = ['acc', 'jmp', 'nop']

i = 0
track_index = []
while i < max(main_dict.keys()):
	if i in track_index:
		break
	else:
		track_index.append(i)
		mat_index = P_list.index(list(main_dict[i].keys())[0])
		if mat_index == 0:
			acc_sum += main_dict[i]['acc']
		elif mat_index == 1:
			if main_dict[i]['jmp'] > 0:
				i += main_dict[i]['jmp'] - 1
			else:
				i += main_dict[i]['jmp'] - 1
		i += 1

print(acc_sum)

Gff_file.close()