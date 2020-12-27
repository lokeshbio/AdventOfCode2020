#!/usr/bin/env python
import re
import sys
import math
import json

Gff_file = open(sys.argv[1],'r')
main_line = ''
dict_list = {}
count = 0
for line in Gff_file:
    if line == '\n':
        main_line = '{' + main_line + '}'
        res = eval(main_line)
        temp = 'a' + chr(count)
        if len(res) > 6:
            if len(res) == 7 and 'cid' not in res:
                dict_list[temp] = res
                count += 1
            elif len(res) == 8:
                dict_list[temp] = res
                count += 1
        main_line = ''
    else:
        line = '\"' + line
        line = re.sub('\:', '\":\'', line)
        line = re.sub(' ', '\',\"', line)
        line = re.sub('\n', '\'', line)
        if main_line == '':
            main_line = line
        else:
            main_line = main_line + ',' + line
main_line = '{' + main_line + '}'
res = eval(main_line)
if len(res) > 6:
    if len(res) == 7 and 'cid' not in res:
        dict_list[temp] = res
        count += 1
    elif len(res) == 8:
        dict_list[temp] = res
        count += 1
#print(len(dict_list))
pas_list = list(dict_list.keys())
col_list = ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']
for pas in pas_list:
    if not (len(dict_list[pas]['byr']) == 4 and int(dict_list[pas]['byr']) >= 1920 and int(dict_list[pas]['byr']) <= 2002):
        dict_list.pop(pas)
    elif not (len(dict_list[pas]['iyr']) == 4 and int(dict_list[pas]['iyr']) >= 2010 and int(dict_list[pas]['iyr']) <= 2020):
        dict_list.pop(pas)
    elif not (len(dict_list[pas]['eyr']) == 4 and int(dict_list[pas]['eyr']) >= 2020 and int(dict_list[pas]['eyr']) <= 2030):
        dict_list.pop(pas)
    elif (re.search('[cm|in]', dict_list[pas]['hgt']) is None):
        dict_list.pop(pas)
    elif re.search('cm', dict_list[pas]['hgt']) is None:
        if not(int(dict_list[pas]['hgt'][:-2]) >= 59 and int(dict_list[pas]['hgt'][:-2]) <= 76):
            dict_list.pop(pas)
    elif re.search('in', dict_list[pas]['hgt']) is None:
        if not(int(dict_list[pas]['hgt'][:-2]) >= 150 and int(dict_list[pas]['hgt'][:-2]) <= 193):
            dict_list.pop(pas)
    elif dict_list[pas]['ecl'] not in col_list:
        dict_list.pop(pas)
    elif not(re.match('\#[a-f0-9]{6}', dict_list[pas]['hcl']) or len(dict_list[pas]['hcl']) != 7):
        dict_list.pop(pas)
    elif not(re.match('\d{9}', dict_list[pas]['pid']) or len(dict_list[pas]['pid']) != 9):
        dict_list.pop(pas)

print(len(dict_list))
Gff_file.close()