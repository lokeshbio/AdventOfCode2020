#!/usr/bin/env python
import re
import sys
import math

def DecToBin(n):  
    return bin(n).replace("0b", "")

def BinToDec(n):
	return int(n,2)

p1 = re.compile('mask')
p2 = re.compile(r'\[(.+)\]')
p3 = re.compile(' = ')
mem_dict = {}

Gff_file = open(sys.argv[1],'r')
for line in Gff_file:
	line = line.rstrip('\n')
	tmp_list = re.split(p3, line)
	if re.match(p1, line):
		mask = list(tmp_list[1])
	else:
		mem_key = re.search(p2, tmp_list[0]).group(1)
		dec_val = int(tmp_list[1])
		bin_val = DecToBin(dec_val)
		bin_val = '0'*(36-len(bin_val)) + bin_val
		bin_list = list(bin_val)
		for idx,id in enumerate(mask):
			if id != 'X':
				bin_list[idx] = id
		new_bin = ''.join(bin_list)
		new_dec = BinToDec(new_bin)
		mem_dict[mem_key] = new_dec

Gff_file.close()

tar_val = 0
for id in mem_dict.keys():
	tar_val += mem_dict[id]

print(tar_val)
 