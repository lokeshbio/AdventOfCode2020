#!/usr/bin/env python
import re
import sys
import math
import itertools

p1 = re.compile(' ')

main_list = []
count = 0
acc_sum = 0

Gff_file = open(sys.argv[1],'r')
for line in Gff_file:
	line = line.rstrip('\n')
	main_list.append(int(line))

for i in range(len(main_list) - 24):
	tar_val = main_list[i + 25]
	tmp_list = main_list[i: i + 25]
	count = 0
	for numbers in itertools.combinations(tmp_list,2):
		if sum(numbers) == tar_val:
			count += 1
			break
	if count == 0:
		main_val = tar_val
		break

for i in range(len(main_list)):
	for j in range(i+1,len(main_list)-1):
		if sum(main_list[i:j]) == main_val:
			print(min(main_list[i:j]) + max(main_list[i:j]))

Gff_file.close()