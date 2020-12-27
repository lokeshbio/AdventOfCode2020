#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy

pub_door = 3469259
pub_card = 13170438

'''
cal = 1
for i in range(1, 100000000):
	cal = (cal * 7) % 20201227
	if cal == pub_door:
		door_loop = i
		break
print(i)
print(cal)
'''
#sub_num = 17807724
cal = 1
for x in range(13739269):
	cal = (cal * pub_card) % 20201227

print(cal)
