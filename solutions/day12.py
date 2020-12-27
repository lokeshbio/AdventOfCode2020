#!/usr/bin/env python
import re
import sys
import math
import itertools
from itertools import cycle


dir_lst = ['E', 'S', 'W', 'N']
cir_lst = cycle(dir_lst)
turn_lst = ['R', 'L']

shp_xy = [0,0]
wp_xy = [10,1]
face = 'E'
new_wp = [0,0]

Gff_file = open(sys.argv[1],'r')
for line in Gff_file:
	line = line.rstrip('\n')
	tmp_list = list(line)
	if tmp_list[0] in turn_lst:
		if tmp_list[0] == 'L':
			rot_deg = int((360 - int(''.join(tmp_list[1:])))/90)
		else:
			rot_deg = int(int(''.join(tmp_list[1:]))/90)
		if rot_deg == 1:
			new_wp[0] = wp_xy[1]
			new_wp[1] = -1 * wp_xy[0]
			wp_xy = new_wp
		elif rot_deg == 2:
			new_wp[0] = -1 * wp_xy[0]
			new_wp[1] = -1 * wp_xy[1]
			wp_xy = new_wp
		else:
			new_wp[0] = -1 * wp_xy[1]
			new_wp[1] = wp_xy[0]
			wp_xy = new_wp
	else:
		if tmp_list[0] in dir_lst:
			move = tmp_list[0]
			if move == 'E':
				wp_xy[0] += int(''.join(tmp_list[1:]))
			elif move == 'W':
				wp_xy[0] -= int(''.join(tmp_list[1:]))
			elif move == 'N':
				wp_xy[1] += int(''.join(tmp_list[1:]))
			else:
				wp_xy[1] -= int(''.join(tmp_list[1:]))
		else:
			shp_xy[0] = shp_xy[0] + (wp_xy[0]*int(''.join(tmp_list[1:])))
			shp_xy[1] = shp_xy[1] + (wp_xy[1]*int(''.join(tmp_list[1:])))


print(abs(shp_xy[0]) + abs(shp_xy[1]))
'''		


main_list = sorted(main_list)
for i in range(len(main_list) -1):
	if main_list[i+1] - main_list[i] == 1:
		count1 += 1
	elif main_list[i+1] - main_list[i] == 3:
		count3 += 1
print(count1 * (count3))

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