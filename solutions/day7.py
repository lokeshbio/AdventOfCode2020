#!/usr/bin/env python
import re
import sys
import math

p1 = re.compile(r' (\d shiny gold) bag')
p2 = re.compile(r'^(.+?) bags.+$')

main_dict = {}
old_count = 0
new_count = 0
used_list = []
while new_count == 0 or new_count > old_count:
    old_count = new_count
    Gff_file = open(sys.argv[1],'r')
    if old_count == 0:
            for line in Gff_file:
                line = line.rstrip('\n')
                if re.search(p1, line) is not None:
                    new = re.sub(p2, r'\1', line)
                    main_dict[new] = re.search(p1, line).group(1)
            Gff_file.close()
            new_count = len(main_dict)
    else:
        for new in list(main_dict.keys()):
            if new not in used_list:
                used_list.append(new)
                temp = ' (\d ' + new + ') bag'
                Gff_file = open(sys.argv[1],'r')
                for line in Gff_file:
                    line = line.rstrip('\n')
                    if re.search(temp, line) is not None:
                        new = re.sub(p2, r'\1', line)
                        main_dict[new] = re.search(temp, line).group(1)
                Gff_file.close()
                new_count = len(main_dict)
print(new_count)