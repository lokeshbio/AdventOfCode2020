#!/usr/bin/env python
import re
import sys
import math

main_list = []
acc_sum = 0

p1 = re.compile(r'^(.*)\((.+?)\)(.*)$')
p2 = re.compile(' ')
p3 = re.compile('[\*\+]')
def eval_own(sub_str):
	tmp_list = re.split(p2, sub_str)
	tar_val = eval(''.join(tmp_list[:3]))
	for i in range(3, len(tmp_list), 2):
		tar_val = eval(str(tar_val) + tmp_list[i] + tmp_list[i+1])
	return tar_val

Gff_file = open(sys.argv[1],'r')
for line in Gff_file:
	line = line.rstrip('\n')
	while re.search(p1, line) is not None:
		front = re.search(p1, line).group(1)
		capt = re.search(p1, line).group(2)
		later = re.search(p1, line).group(3)
		line = front + str(eval_own(capt)) + later
	acc_sum +=  eval_own(line)

print(acc_sum)
		
	