#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy
from itertools import chain

main_dict = {}
Gff_file = open(sys.argv[1],'r')

p1 = re.compile(r'^(\d+?): ')
p2 = re.compile('\"(.)\"')
p3 = re.compile(' \| ')
p4 = re.compile('^[ab]')

rule_dict = {}
chk_list = []
fin_list = []
for line in Gff_file:
	line = line.rstrip('\n')
	if re.search(p1, line) is not None:
		num = re.search(p1, line).group(1)
		if re.search(p2, line) is not None:
			rule_dict[num] = re.search(p2, line).group(1)
			fin_list.append(rule_dict[num])
		else:
			line = re.sub(p1, '', line)
			if re.search(p3, line) is not None:
				tmp_list = re.split(p3, line)
				rule_dict[num] = [i.split(' ', 1) for i in tmp_list]
			else:
				rule_dict[num] = re.split(' ', line)
	elif re.search(p4, line) is not None:
		chk_list.append(line)
Gff_file.close()

'''
for rule in rule_dict:
	if isinstance(rule_dict[rule], list):
		print(rule_dict[rule])
		sys.exit()
'''

def flatten(l):
    if isinstance(l, list):
        for e in l:
            yield from flatten(e)
    else:
        yield l

print(fin_list)
sys.exit()

count = 0
key_list = ['0']
main_list = []
while count < 1:
	for key in key_list:
		if key in fin_list:
			continue
		if isinstance(rule_dict[key][0], list): # OR rules options
			tmp_list = main_list
			tmp_list1 = copy.deepcopy(tmp_list)
			main_list = []
			for rule in rule_dict[key]:
				for idx,id in enumerate(tmp_list):
					tmp_list1[idx] = list(chain.from_iterable(rule if item == key else [item] for item in id))
				main_list.extend(tmp_list1)
		else:
			if len(main_list) == 0:
				main_list = [rule_dict[key]]
			for idx,id in enumerate(main_list):
				main_list[idx] = list(chain.from_iterable(rule_dict[key] if item == key else [item] for item in id))
	
	key_list = list(set(flatten(main_list)))
	if key_list == fin_list:
		count += 1
