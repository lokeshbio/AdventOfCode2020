#!/usr/bin/env python
import re
import sys
import math

p1 = re.compile('\: ')
p2 = re.compile(' or ')
p3 = re.compile('your')
p5 = re.compile('\,')
p6 = re.compile('\d')
rule_dict = {}


def pop_rule(line):
	empty_list = []
	tmp_list1 = re.split(p1, line)
	tmp_list2 = re.split(p2, tmp_list1[1])
	for num_rule in tmp_list2:
		tmp_list3 = re.split('-', num_rule)
		empty_list.extend(range(int(tmp_list3[0]), int(tmp_list3[1])+1))
	return tmp_list1[0], empty_list 

Gff_file = open(sys.argv[1],'r')
count = 0
not_val = []
rul_list = []
for line in Gff_file:
	line = line.rstrip()
	if line:
		ln += 1
		if re.match(p3, line) is not None:
			count += 1
		elif count == 0:
			rul_nam, rul_list = pop_rule(line)
			rule_dict[rul_nam] = rul_list
		elif count == 1:
			my_ticket = re.split(p5,line)
			for rul_nam in rule_dict:
				rul_list.extend(rule_dict[rul_nam])
			rul_set = set(rul_list)
			uni_rul = list(rul_set)
			count += 1
		elif re.match(p6, line) is not None:
			nei_tic = re.split(p5,line)
			nei_tic = [int(i) for i in nei_tic] 
			for i in range(len(nei_tic)):
				if nei_tic[i] not in uni_rul:
					not_val.append(nei_tic[i])
					continue
Gff_file.close()
print(sum(not_val))




