#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy

main_dict = {}
Gff_file = open(sys.argv[1],'r')
ln = 0
for line in Gff_file:
	line = line.rstrip()
	main_dict[ln] = list(line)
	ln += 1
Gff_file.close()

cube_dict = {}
for i in range(ln):
	cube_dict[i] = {}
	for j in range(len(main_dict[0])):
		cube_dict[i][j] = {}
		cube_dict[i][j][0] = main_dict[i][j]

def init_neig(fun_dict):
	d2 = copy.deepcopy(fun_dict)
	for x in fun_dict:
		for y in fun_dict[x]:
			for z in fun_dict[x][y]:
				x_list = [x-1, x, x+1]
				y_list = [y-1, y, y+1]
				z_list = [z-1, z, z+1]
				a = [x_list, y_list, z_list]
				for i,j,k in list(itertools.product(*a)):
					if i not in d2:
						d2[i] = {}
					if j not in d2[i]:
						d2[i][j] = {}
					if k not in d2[i][j]:
						d2[i][j][k] = '.'
	return d2

def get_neig(fun_dict, x, y, z):
	neig_list = []
	x_list = [x-1, x, x+1]
	y_list = [y-1, y, y+1]
	z_list = [z-1, z, z+1]
	a = [x_list, y_list, z_list]
	tmp_list = list(itertools.product(*a))
	tmp_list.pop(14)
	for i,j,k in tmp_list:
		neig_list.append(fun_dict[i][j][k])
	return neig_list

cycle = 0
new_dict = copy.deepcopy(cube_dict)
while cycle < 6:
	cycle += 1
	old_dict = copy.deepcopy(new_dict)
	old_dict_1 = init_neig(old_dict)
	new_dict = copy.deepcopy(old_dict_1)
	for x in old_dict:
		for y in old_dict[x]:
			for z in old_dict[x][y]:
				neig_list = get_neig(old_dict_1, x, y, z)
				act_count = neig_list.count('#')
				if old_dict_1[x][y][z] == '#':
					if not (act_count == 2 or act_count == 3):
						new_dict[x][y][z] = '.'
				else:
					if act_count == 3:
						new_dict[x][y][z] = '#'

act_sum = 0
for x in new_dict:
	for y in new_dict[x]:
		for z in new_dict[x][y]:
			if new_dict[x][y][z] == '#':
				act_sum += 1

print(act_sum)

