#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy
from itertools import chain

main_dict = {}
Gff_file = open(sys.argv[1],'r')

count = 0

p1 = re.compile('^P')
p2 = re.compile('^\d')

pla_dict = {}
for line in Gff_file:
	line = line.rstrip('\n')
	if re.search(p1, line) is not None:
		count += 1
		pla_dict[count] = []
	elif re.search(p2, line) is not None:
		pla_dict[count].append(int(line))

Gff_file.close()

count = 0
while count == 0:
	card1, card2 = pla_dict[1].pop(0), pla_dict[2].pop(0)
	if card1 > card2:
		pla_dict[1].extend([card1, card2])
	else:
		pla_dict[2].extend([card2, card1])
	if not pla_dict[1]:
		count = 2
	if not pla_dict[2]:
		count = 1

tar_val = 0
for idx, val in enumerate(pla_dict[count]):
	tar_val += (len(pla_dict[count]) - idx) * val

print(tar_val)	 