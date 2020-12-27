#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy

main_dict = {}
Gff_file = open(sys.argv[1],'r')

p1 = re.compile(r'^Tile (\d+)\:$')
p2 = re.compile('[\#|\.]')

def extr_bord(imag_dict):
	bord_dict = {}
	tile = str(list(main_dict.keys())[0])
	bord_dict['top'] = {}
	bord_dict['bot'] = {}
	bord_dict['lef'] = {}
	bord_dict['rgh'] = {}
	row_len = len(imag_dict[tile].keys())
	col_len = len(imag_dict[tile][0])
	bord_dict['top'][tile] = ''.join(imag_dict[tile][0])
	bord_dict['bot'][tile] = ''.join(imag_dict[tile][row_len-1])
	bord_dict['lef'][tile] = []
	bord_dict['rgh'][tile] = []
	for i in range(row_len):
		bord_dict['lef'][tile].append(imag_dict[tile][i][0])
		bord_dict['rgh'][tile].append(imag_dict[tile][i][col_len-1])
	bord_dict['lef'][tile] = ''.join(bord_dict['lef'][tile])
	bord_dict['rgh'][tile] = ''.join(bord_dict['rgh'][tile])
	return bord_dict

tar_dict = {}
tar_dict['top'] = {}
tar_dict['bot'] = {}
tar_dict['lef'] = {}
tar_dict['rgh'] = {}
ln = 0
for line in Gff_file:
	line = line.rstrip('\n')
	if re.search(p1, line) is not None:
		tile = re.search(p1, line).group(1)
		main_dict[tile] = {}
	elif re.search(p2, line) is not None:
		main_dict[tile][ln] = list(line)
		ln += 1
	else:
		ln = 0
		temp_dict = extr_bord(main_dict)
		tar_dict['top'][tile] = temp_dict['top'][tile]
		tar_dict['bot'][tile] = temp_dict['bot'][tile]
		tar_dict['lef'][tile] = temp_dict['lef'][tile]
		tar_dict['rgh'][tile] = temp_dict['rgh'][tile]
		main_dict = {}

Gff_file.close()
tile_list = sorted(list(tar_dict['top'].keys()))
dim_list = sorted(list(tar_dict.keys()))

tar_list = []
for tile1 in tile_list:
	count = 0
	for dim1 in dim_list:
		frame1 = tar_dict[dim1][tile1]
		for tile2 in tile_list:
			for dim2 in dim_list:
				if tile1 != tile2:
					frame2 = tar_dict[dim2][tile2]
					if (frame1 == frame2) or (frame1 == frame2[::-1]): 
						count += 1
	if count < 3:
		tar_list.append(tile1)

print(eval('*'.join(tar_list)))
