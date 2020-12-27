#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy
from itertools import cycle, islice, dropwhile

in_list = list('364297581')
in_list = [int(i) for i in in_list] 
round = 0
new_list = copy.deepcopy(in_list)
while round < 100:
	cut_list = []
	old_list = copy.deepcopy(new_list)
	new_list = []
	idx = 0
	for id in old_list:
		if idx == 0:
			src = id
		if idx > 0 and idx <= 3:
			cut_list.append(id)
		if idx > 3:
			new_list.append(id)
		idx += 1
	new_list.append(src)
	if src == min(new_list):
		dest = max(new_list)
		dest_idx = new_list.index(dest)
	else: 
		for i in range(1, src):
			if src-i in new_list:
				 dest_idx = new_list.index(src-i)
				 break
	new_list[dest_idx+1:dest_idx+1] = cut_list
	round += 1

print(new_list)