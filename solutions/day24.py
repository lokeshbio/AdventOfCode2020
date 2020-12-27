#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy

main_list = ['e', 'se', 'ne', 'w', 'sw', 'nw']
def dir_list(line):
	line_list = []
	while line != '':
		if line[:2] in main_list:
			line_list.append(line[:2])
			line = line[2:]
		else:
			line_list.append(line[:1])
			line = line[1:]
	return line_list

Gff_file = open(sys.argv[1],'r')

co_ord = {}
for line in Gff_file:
	line = line.rstrip('\n')
	req_list = dir_list(line)
	x = 0
	y = 0
	for dir in req_list:
		if dir == 'e':
			x += 2
		elif dir == 'se':
			x += 1
			y -= 1
		elif dir == 'ne':
			x += 1
			y += 1
		elif dir == 'w':
			x -= 2
		elif dir == 'sw':
			x -= 1
			y -= 1
		else:
			x -= 1
			y += 1
	ord = str(x) + ' ,' + str(y)
	if ord in co_ord:
		if co_ord[ord] == 0:
			co_ord[ord] = 1
		else:
			co_ord[ord] = 0
	else:
		co_ord[ord] = 1
		
print(sum(co_ord.values()))