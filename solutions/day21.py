#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy
from itertools import chain
'''
main_dict = {}
Gff_file = open(sys.argv[1],'r')

count = 0

p1 = re.compile(r' \(contains (.+)\)')
p2 = re.compile('\, ')
p3 = re.compile(' ')
main_alr = []

for line in Gff_file:
	line = line.rstrip('\n')
	alrgns = re.search(p1, line).group(1)
	alr_lst = re.split(p2, alrgns)
	main_alr.append(alr_lst)
	line = re.sub(p1, '', line)
	for alr in alr_lst:
		if alr in main_dict:
			main_dict[alr] &= set(re.split(p3, line))
		else:
			main_dict[alr] = set(re.split(p3, line))


all_alrgs = set(e for a in main_dict.values() for e in a)
print( sum(i not in all_alrgs for ingr in main_alr for i in ingr) )
'''
input = open("day21.txt","rt").read().splitlines()
alrgs = {}
ingrs = []
for e in input:
  a,b = e.rstrip(")").split("(contains ") # ingredients, allergens
  ingr = a.split()
  ingrs.append(ingr)
  for a in b.split(", "):
    if a not in alrgs: alrgs[a] = set(ingr)
    else:              alrgs[a] &= set(ingr)
#all_alrgs = set(e for a in alrgs.values() for e in a)
print(set(list(e for a in alrgs.values() for e in a)))
#print( sum(i not in all_alrgs for ingr in ingrs for i in ingr) )	




