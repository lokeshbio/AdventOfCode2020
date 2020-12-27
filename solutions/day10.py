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

count1 = 1
count3 = 1
main_list = sorted(main_list)
for i in range(len(main_list) -1):
	if main_list[i+1] - main_list[i] == 1:
		count1 += 1
	elif main_list[i+1] - main_list[i] == 3:
		count3 += 1
print(count1 * (count3))
'''
i = 0
while i < max(main_list):
	i += 1
	if i in main_list:
		count1 += 1
	elif i+1 in main_list:
		i += 1
	elif i+2 in main_list:
		count3 += 1
		i += 2
	else:
		print(count1 * count3)
		break



print(count1)
print(count3)
print(count1 * (count3))
'''
Gff_file.close()