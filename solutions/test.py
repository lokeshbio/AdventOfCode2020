#!/usr/bin/env python
import re
import sys
import math
import itertools
import copy
from itertools import chain
'''
def rule_maker(raw_rules):
    rules = {}
    for rule in raw_rules:
        key, value = rule.split(': ')
        if value[0] == '"':
            rules[int(key)] = value[1:-1]
        else:
            values = value.split(' | ')
            temp_v = []
            for v in values:
                temp_v.append([int(vv) for vv in v.split(' ')])
            rules[int(key)] = temp_v
    return rules

def match_rule(expr, stack):
    if len(stack) > len(expr):
        return False
    elif len(stack) == 0 or len(expr) == 0:
        return len(stack) == 0 and len(expr) == 0

    c = stack.pop()
    if isinstance(c, str):
        if expr[0] == c:
            return match_rule(expr[1:], stack.copy())
    else:
        for rule in rules[c]:
            if match_rule(expr, stack + list(reversed(rule))):
                return True
    return False

def count_messages(rules, messages):
    total = 0
    for message in messages:
        if match_rule(message, list(reversed(rules[0][0]))):
            total += 1
    return total

with open("day19.txt") as fp:
    raw_rules, message = fp.read().split('\n\n')
    raw_rules = raw_rules.splitlines()
    message = message.splitlines()

rules = rule_maker(raw_rules)

print(f"Part 1: {count_messages(rules, message)}")
rules[8] = [[42], [42, 8]]
rules[11] = [[42, 31], [42, 11, 31]]
print(f"Part 2: {count_messages(rules, message)}")
'''
'''day4
class Validator:
    def __init__(self, passport):
        self.passport = passport

    def check_field_count(self):
        return len(self.passport) == 8 or (len(self.passport) == 7 and 'cid' not in self.passport)

    def check_year(self, key, start, end):
        return len(self.passport[key]) == 4 and int(self.passport[key]) >= start and int(self.passport[key]) <= end

    def check_byr(self):
        return self.check_year('byr', 1920, 2002)

    def check_iyr(self):
        return self.check_year('iyr', 2010, 2020)

    def check_eyr(self):
        return self.check_year('eyr', 2020, 2030)

    def check_hcl(self):
        return self.passport['hcl'][0] == "#" and self.passport['hcl'][1:].isalnum()

    def check_ecl(self):
        return self.passport['ecl'] in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']

    def check_pid(self):
        return len(self.passport['pid']) == 9

    def check_hgt(self):
        if self.passport['hgt'][-2:] == "cm":
            return int(self.passport['hgt'][:-2]) >= 150 and int(self.passport['hgt'][:-2]) <= 193
        elif self.passport['hgt'][-2:] == "in":
            return int(self.passport['hgt'][:-2]) >= 59 and int(self.passport['hgt'][:-2]) <= 76

    def is_valid(self):
        return (self.check_field_count() and self.check_byr() and self.check_iyr() and self.check_eyr() 
            and self.check_hcl() and self.check_ecl() and self.check_pid() and self.check_hgt())


def get_passports(inp):
    passports = []
    passport = {}
    for line in inp:
        if line != "\n":
            line = line.rstrip().split(" ")
            line = [field.split(":") for field in line]
            for field in line:
                passport[field[0]] = field[1]
        else:
            passports.append(passport)
            passport = {}
    passports.append(passport)
    return passports


with open('day4.txt') as inp:
    passports = get_passports(inp)
    validators = [Validator(passport) for passport in passports]
    part_1_count = 0
    part_2_count = 0
    for validator in validators:
        if validator.check_field_count(): 
            part_1_count += 1
        if validator.is_valid(): 
            part_2_count += 1                        

    print(part_1_count) 
    print(part_2_count)                
'''
'''day7
from collections import defaultdict

class RuleParser():

    def __init__(self):
        self.containment_tree = defaultdict(lambda: defaultdict(list))


    def parse_row(self, row):
        if row.strip()=="": return
        row = row.replace(' bags', '').replace(' bag', '').replace('.','').strip()
        outer, contents = row.split(' contain ')
        content_rules = contents.split(', ')
        for contained_rule in content_rules:
            words = contained_rule.split(' ')
            n = 0 if words[0] == "no" else int(words[0])
            colour = " ".join(words[1:])
            colour = colour if n > 0 else "no other"
            self.containment_tree[outer][colour]=n


    def find_in_subtree(self, target_color):
        outers = set()
        def search_subtree(for_colour):
            for outer in self.containment_tree:
                if for_colour in self.containment_tree[outer]:
                    outers.add(outer)
                    search_subtree(outer)
        search_subtree(target_color)
        return len(outers)


    def _n_in_subtree(self, outer_bag):
        total_children = 1
        for inner_bag in parser.containment_tree[outer_bag]:
            n_this_colour = parser.containment_tree[outer_bag][inner_bag]
            n_children = self._n_in_subtree(inner_bag)
            total_children += n_this_colour * n_children
        return total_children


    def n_in_children(self, outer_bag):
        return self._n_in_subtree(outer_bag)-1


parser = RuleParser()

with open ("day7.txt", "r") as input:
    for row in input:
        parser.parse_row(row)

goal_colour = 'shiny gold'
part_1 = parser.find_in_subtree(goal_colour)
print(f"Part 1 solution: {part_1} colours can ultimately contain a {goal_colour} bag")

part_2 = parser.n_in_children(goal_colour)
print(f"Part 2 solution: a {goal_colour} bag has to contain {part_2} other bags")
'''
'''day8
class Program:
	def __init__(self):
		self.instruction_index = 0
		self.prev_instruction_indexes = []
		self.acc_value = 0
		self.is_loop = False

	def is_infinite_loop(self, instructions, is_debug=False):
		self.instruction_index = 0
		self.prev_instruction_indexes = []
		self.acc_value = 0
		self.is_loop = False
		while not self.is_at_end(instructions):
			instruction = instructions[self.instruction_index]
			if self.instruction_index in self.prev_instruction_indexes:
				self.is_loop = True
				break
			self.prev_instruction_indexes.append(self.instruction_index)
			self.instruction_index, self.acc_value = instruction.execute(self.instruction_index, self.acc_value)
		
		if is_debug:
			print(f"Acc Value before end: {self.acc_value:<5} Is an Infinite Loop: {self.is_loop}")
		return self.is_loop
	
	def fix(self, instructions, is_debug=False):
		for i, instruction in enumerate(instructions):
			instruction_type = type(instruction)
			if instruction_type == Instruction:
				new_instruction = InstructionJmp(instruction.argument)
			elif instruction_type == InstructionJmp:
				new_instruction = Instruction(instruction.argument)
			else:
				continue
			instructions[i] = new_instruction

			if not self.is_infinite_loop(instructions):
				if is_debug:
					print("Acc Value at end of Fixed Instructions:", self.acc_value)
				return instructions

			instructions[i] = instruction

	def is_at_end(self, instructions):
		return self.instruction_index >= len(instructions)

class Instruction:
	def __init__(self, argument):
		self.argument = argument
	
	def execute(self, instruction_index, acc_value):
		return instruction_index + 1, acc_value

class InstructionAcc(Instruction):
	def execute(self, instruction_index, acc_value):
		return instruction_index + 1, acc_value + self.argument

class InstructionJmp(Instruction):
	def execute(self, instruction_index, acc_value):
		return instruction_index + self.argument, acc_value

instruction_classes = {
	"acc": InstructionAcc,
	"jmp": InstructionJmp,
}
with open("day8.txt", "r") as file:
	instructions = []
	for line in file:
		components = line.split()
		operation = components[0]
		argument = int(components[1])

		instruction_class = instruction_classes.get(operation, Instruction)
		instruction = instruction_class(argument)
		instructions.append(instruction)

program = Program()

# Part 1
print("Part 1")
program.is_infinite_loop(instructions, is_debug=True)

print()

# Part 2
print("Part 2")
program.fix(instructions, is_debug=True)
'''
'''day17
import itertools
from copy import deepcopy

ic = [ list(x) for x in open('day17.txt').read().splitlines() ]

def spaceviewer(hyperspace): # Take a look at the current state of the system
    w_ = 0
    for w in hyperspace:
        # print(w)
        z_ = 0
        print('w='+str(w_))
        for z in w:
            # print(z)
            print('w='+str(w_)+', z='+str(z_))
            z_+=1
            [print(y) for y in z]
        w_+=1

def generate_plane(space):
    plane = space[0]
    empty = [ [ '.' for i in range(len(plane[0])) ] for j in range(len(plane)) ]
    space.insert(0, empty)
    space.append(deepcopy(empty))
    return space
space = generate_plane([ic])

def expand_space(space):
    space = generate_plane(generate_plane(space))
    space_ = deepcopy(space)
    empty = [ '.' for i in range(len(space[0][0])+4) ]
    for k in range(len(space)):
        for j in range(len(space[k])):
            space_[k][j] = ['.','.'] + space[k][j] + ['.','.']
        space_[k].insert(0,deepcopy(empty))
        space_[k].insert(0,deepcopy(empty))
        space_[k].append(deepcopy(empty))
        space_[k].append(deepcopy(empty))
        # print(empty)
        # print(len(space_[k]))
    return space_

def generate_space(z_,y_,x_):
    return [[['.' for x in range(x_)] for y in range(y_)] for z in range(z_)]

def expand_hyperspace(hyperspace):
    # hyperspace
    for w in range(len(hyperspace)):
        print()
        print(len(hyperspace[w]), len(hyperspace[w][0]), len(hyperspace[w][0][0]))
        hyperspace[w] = expand_space(hyperspace[w])
        print(len(hyperspace[w]), len(hyperspace[w][0]), len(hyperspace[w][0][0]))
    empty_space = generate_space(len(hyperspace[0]),len(hyperspace[0][0]),len(hyperspace[0][0][0]))
    hyperspace.insert(0,deepcopy(empty_space))
    hyperspace.insert(0,deepcopy(empty_space))
    hyperspace.append(deepcopy(empty_space))
    hyperspace.append(deepcopy(empty_space))
    return hyperspace

empty_space = generate_space(len(space),len(space[0]),len(space[0][0]))
hyperspace = [deepcopy(empty_space), deepcopy(empty_space), space, deepcopy(empty_space), deepcopy(empty_space)]
print(hyperspace)
spaceviewer(hyperspace)
# print(expand_space(hyperspace))
# spaceviewer(expand_hyperspace(hyperspace))

def search(i,j,k,w,space):
    c = 0
    # print('search',w,k,j,i)
    for w_ in range(w-1,w+2):
        for k_ in range(k-1,k+2):
            for j_ in range(j-1,j+2):
                for i_ in range(i-1,i+2):
                    if (w_ >= 0 and w_ < len(space) and k_ >= 0 and k_ < len(space[w_]) and j_ >= 0 and j_ < len(space[w_][k_]) and i_ >= 0 and i_ < len(space[w_][k_][j_])) and (w_ != w or k_ != k or j_ != j or i_ != i):
                        # print(w_,k_,j_,i_,"  ",len(space),len(space[w_]),len(space[w_][k_]),len(space[w_][k_][j_]))
                        # print(space[w_][k_][j_][i_])
                        if space[w_][k_][j_][i_] == '#':
                            c+=1
    return c

# print(len(space_),len(space_[0]),len(space_[0][0]))
def update_state(c, space):
    space = deepcopy(expand_hyperspace(space))
    # spaceviewer(space)
    space_ = deepcopy(space)
    change = False
    for w in range(0,len(space_)):
        for k in range(0,len(space_[0])):
            for j in range(0,len(space_[0][0])):
                for i in range(0,len(space_[0][0][0])):
                    adj = search(i,j,k,w,space)
                    if space[w][k][j][i] == '#' and adj not in [2,3]:
                        space_[w][k][j][i] = '.'
                        change = True
                    elif space[w][k][j][i] == '.' and adj == 3:
                        space_[w][k][j][i] = '#'
                        change = True
    if change == True and c < 5:
        c+=1
        # spaceviewer(space_)
        space_ = update_state(c, space_)
    return space_

hyperspace_ = update_state(0,hyperspace)

spaceviewer(hyperspace_)
print(sum([sum([ sum([ y.count('#') for y in z ]) for z in w]) for w in hyperspace_ ]))
'''
'''
import re

rules = {}
myticket = []
nearbytickets = []

with open("day16.txt") as f:
    sect = "rules"
    for line in f:
        line = line.rstrip()
        if line == "your ticket:":
            sect = "myticket"
            continue
        if line == "nearby tickets:":
            sect = "nearby"
            continue
        if sect == "myticket" and line != '':
            myticket = list(map(int, line.split(',')))
        elif sect == "nearby" and line != '':
            nearbytickets.append(list(map(int, line.split(','))))
        elif sect == "rules" and line != '':
            [name, r1begin, r1end, r2begin, r2end] = re.split("^(.+): ([0-9]+)-([0-9]+) or ([0-9]+)-([0-9]+)$", line)[1:6]
            rules[name] = { "range1": [int(r1begin), int(r1end)],
                            "range2": [int(r2begin), int(r2end)] }

def makerulefuncs():
    for r in rules:
        funcbody = "lambda x: (x >= {} and x <=  {}) or (x >= {} and x <= {})"
        rules[r]["inside"] = eval(funcbody.format(rules[r]["range1"][0],
                                                  rules[r]["range1"][1],
                                                  rules[r]["range2"][0],
                                                  rules[r]["range2"][1]))
makerulefuncs()

def part1():
    def findinvalid(tickets):
        invalid = 0
        for t in tickets:
            for number in t:
                valid = False
                for rule in rules:
                    valid = valid or rules[rule]["inside"](number)
                if not valid:
                    invalid += number
        return invalid
    print(findinvalid(nearbytickets))
### end of part 1 ###

def part2():    
    def deleteinvalid():
        invalid = 0
        for near in nearbytickets[:]:
            for number in near:
                valid = False
                for rule in rules:
                    valid = valid or rules[rule]["inside"](number)
                if not valid:
                    nearbytickets.remove(near)
    
    def mappings(tickets):
        candidates = {}
        found = {}
        for r in rules.keys():
            candidates[r] = []
            found[r] = -1
        found["cols_taken"] = []
        takenby = {}
        for rule in rules.keys():
            for i in range(0, len(tickets[0])):
                for nb in tickets:
                    if rules[rule]["inside"](nb[i]):
                        if i not in candidates[rule]:
                            candidates[rule].append(i)
                    elif not rules[rule]["inside"](nb[i]):
                        if  i in candidates[rule]:
                            candidates[rule].remove(i)
                        break
        
        impossible = False
        while candidates and not impossible:
            for rule in sorted(candidates.keys(), key=(lambda x: len(candidates[x]))):
                if len(candidates[rule]) == 1:
                    col = candidates[rule][0]
                    if col in found["cols_taken"]:
                        print("Cannot assign ", col, " to column ", col,
                              " because it is already taken by ", takenby[col], ".")
                        impossible = True
                        break
                    found[rule] = col
                    takenby[col] = rule
                    found["cols_taken"].append(col)
                    del candidates[rule]
                    break
                else:
                    for col in found["cols_taken"]:
                        candidates[rule].remove(col)
                    break
        found["solved"] = not impossible
        return found
    
    deleteinvalid()
    found = mappings(nearbytickets)
    if found["solved"]:
        prod = 1
        for r in ["departure location", "departure station", "departure platform",
                  "departure track", "departure date", "departure time"]:
            prod *= myticket[found[r]]
        print(prod)
    else:
        print("Solution seems impossible.")
### end of part 2 ###

print("Part 1:")
part1()
print("Part 2:")
part2()
'''
'''
import re

class K:
	__slots__ = ('value')

	def __init__(self, value):
		self.value = value

	def __sub__(self, other):
		return K(self.value + other.value)

	def __add__(self, other):
		return K(self.value * other.value)

	def __mul__(self, other):
		return K(self.value + other.value)


fin = open('day18.txt')
exprs = fin.read().splitlines()

table1 = str.maketrans('+*', '-+')
table2 = str.maketrans('+*', '*+')
regexp = re.compile(r'(\d+)')
total1 = 0
total2 = 0

for expr in exprs:
	expr = regexp.sub(r'K(\1)', expr)

	expr1 = expr.translate(table1)
	total1 += eval(expr1).value

	expr2 = expr.translate(table2)
	total2 += eval(expr2).value

print('Part 1:', total1)
print('Part 2:', total2)
'''
'''
import re

datafile = 'day19.txt'

with open(datafile) as fh:
    txt = fh.read()
    rulestxt, datatxt = txt.split('\n\n')

data = [y for y in (x.strip() for x in datatxt.split('\n')) if y]

def make_rules(lines):
    D = {}
    for line in lines:
        if not line:
            continue
        k, v = line.strip().split(':')
        v = v.replace('"', '')
        if '|' in v:
            v = '(?: ' + v + ' )'
        D[k] = v.split()
    return D

rules = make_rules(rulestxt.split('\n'))

def rules_to_re(rules):
    L = rules['0'].copy()
    while any(x.isdigit() for x in L):
        i, k = next((i,x) for (i, x) in enumerate(L) if x.isdigit())
        L[i:i+1] = rules[k].copy()
    L.insert(0, '^')
    L.append('$')
    return re.compile(''.join(L))

rules_re_1 = rules_to_re(rules)
part_1 = sum(bool(rules_re_1.match(x)) for x in data)

rules_2 = make_rules(rulestxt.split('\n'))
rules_2['8'] = ['(?:', '42', ')+']
rules_2['11'] = [
    '(?:',
    '(?:', '(?:', '42', ')', '{1}', '(?:', '31', ')', '{1}', ')', '|',
    '(?:', '(?:', '42', ')', '{2}', '(?:', '31', ')', '{2}', ')', '|',
    '(?:', '(?:', '42', ')', '{3}', '(?:', '31', ')', '{3}', ')', '|',
    '(?:', '(?:', '42', ')', '{4}', '(?:', '31', ')', '{4}', ')', '|',
    '(?:', '(?:', '42', ')', '{5}', '(?:', '31', ')', '{5}', ')', '|',
    '(?:', '(?:', '42', ')', '{6}', '(?:', '31', ')', '{6}', ')', '|',
    '(?:', '(?:', '42', ')', '{7}', '(?:', '31', ')', '{7}', ')', '|',
    '(?:', '(?:', '42', ')', '{8}', '(?:', '31', ')', '{8}', ')', '|',
    '(?:', '(?:', '42', ')', '{9}', '(?:', '31', ')', '{9}', ')',
    ')'
]

rules_re_2 = rules_to_re(rules_2)
part_2 = sum(bool(rules_re_2.match(x)) for x in data)
print(part_2)
'''

'''
import re
import numpy as np
import itertools
import networkx as nx
from collections import deque
from functools import reduce


def all_regex_matches(pattern, line, pos=0):
    collect = []
    while True:
        if (pos > len(line)):
            break
        m = pattern.search(line[pos:])

        found = False
        if (m):
            start, stop = m.span()
            start += pos
            stop += pos
            count = sum([1 if c == '1' else 0 for c in line[start:stop]])
            collect.append([start, count])
            found = True
            pos = start+1
        if (not found):
            break
    return collect


def find(L, x):
    for (i, s) in enumerate(L):
        if (s == x):
            return i
    return -1


def get_top(T):
    return T[0, :]


def get_bottom(T):
    return T[-1, :]


def get_left(T):
    return T[:, 0]


def get_right(T):
    return T[:, -1]


def to_int(row):
    a = [str(s) for s in list(row)]
    return int("".join(a), 2)


def valid_graph(G, grid):
    for (src, dst) in G.edges():
        src_i = src[0]
        src_j = src[1]
        dst_i = dst[0]
        dst_j = dst[1]
        if (not grid[src_i][src_j] is None) and (not grid[dst_i][dst_j] is None):
            edge_data = G.get_edge_data(src, dst)
            node = grid[src_i][src_j]
            if (edge_data['color'] == 'red'):
                above = grid[dst_i][dst_j]
                x = node[0]
                y = above[2]
                if (x != y):
                    return False
            if (edge_data['color'] == 'blue'):
                l = grid[dst_i][dst_j]
                x = node[3]
                y = l[1]
                if (x != y):
                    return False
    return True


def serialize(cur_soln):
    s = []
    for x in cur_soln:
        s.append({"tile_id": int(x[1]), "tile_rotation": int(x[2])})

    return s


def parse(s, truncate=False):
    tile_no = None
    tile = []
    tiles = dict()
    for line in s:
        if (line.startswith("Tile")):
            tile_no = line[5:len(line)-1]
            tile = []
            continue
        if (line == ""):
            if truncate:
                tile.pop()
                tiles[tile_no] = tile[1:]
            else:
                tiles[tile_no] = tile
            continue
        else:
            row = list(map(lambda x: 1 if x == '#' else 0, list(line)))
            if (len(row) > 2 and truncate):
                row = row[1:len(row)-1]
            tile.append(row)
    if (tile != [] and tile_no != None):
        if (truncate):
            tile.pop()
            tiles[tile_no] = tile[1:]
        else:
            tiles[tile_no] = tile

    return tiles


def all_orientations(tile):
    rotated = []
    v = np.array(tile)
    for rotations in range(3):
        mat = np.rot90(v, k=rotations)
        rotated.append([(rotations, 0), mat])
        for flip in range(1, 3):
            flp = np.flip(mat, flip-1)
            rotated.append([(rotations, flip-1), flp])
    return rotated


def all_sides(tile):
    sides = []
    for rotation in tile:
        matrix = rotation[-1]
        top = to_int(get_top(matrix))
        right = to_int(get_right(matrix))
        bottom = to_int(get_bottom(matrix))
        left = to_int(get_left(matrix))

        sides.append([top, right, bottom, left])
    return sides


def get_match_table(permutation_table):
    match_table = dict()
    for (t1, t2) in itertools.combinations(permutation_table, 2):
        if (t1 == t2):
            continue
        if (t1 not in match_table):
            match_table[t1] = set()

        if (t2 not in match_table):
            match_table[t2] = set()

        t1_sides = map(lambda x: set(x), permutation_table[t1])
        t2_sides = map(lambda x: set(x), permutation_table[t2])

        for (side1, side2) in itertools.product(t1_sides, t2_sides):
            if (len(side1.intersection(side2)) > 0):
                match_table[t2].add(t1)
                match_table[t1].add(t2)
                break
    return match_table


def build_orienttion_table(tiles):
    table = dict()
    for tile in tiles:
        table[tile] = all_orientations(tiles[tile])
    return table


def build_permutation_table(tiles):
    table = dict()
    for tile in tiles:
        table[tile] = all_sides(all_orientations(tiles[tile]))
    return table


def bootstrap(tiles, match_table=None):
    possible_corners = []
    if (not match_table):
        match_table = get_match_table(tiles)

    for tile in match_table.keys():
        if (len(match_table[tile]) == 2):
            possible_corners.append(tile)

    bootstrapped = []
    for tile_id in possible_corners:
        b = all_sides(all_orientations(tiles[tile_id]))
        for i, matrix in enumerate(b):
            bootstrapped.append((matrix, tile_id, i))

    return bootstrapped


def iterative_traversal(G, placement_order, tiles, permutation_table, match_table, bootstrap=None):
    pos = 0
    current_soln = []
    num_rotations = 9
    backtracking = False
    alternative_stack = deque()
    while True:
        if (pos < 0):
            return -1

        if (backtracking):
            if (len(alternative_stack) < 1):
                return -1

            pos -= 1
            current_soln.pop()
            alternatives = alternative_stack.pop()

            if (len(alternatives) < 1):
                continue

            (next_matrix, mat_id, rot_id) = alternatives.pop()
            alternative_stack.append(alternatives)
            y = [next_matrix, mat_id, rot_id]
            current_soln.append(y)
            backtracking = False
            pos += 1

        # If we're not backtracking, immediately check the current soln
        grid = [[None] * len(G.nodes()) for _ in range(len(G.nodes()))]
        for i in range(len(current_soln)):
            p_i, p_j = placement_order[i]
            pnn = current_soln[i][0]
            grid[p_i][p_j] = pnn
            if (not valid_graph(G, grid)):
                backtracking = True
                continue

        # Here, the current solution is good, move forward
        if (not backtracking):
            if (len(current_soln) == len(G.nodes())):
                return serialize(current_soln)
            if (len(current_soln) == 0 and bootstrap):
                (next_matrix, mat_id, rot_id) = bootstrap.pop()
                alternative_stack.append(bootstrap)
                y = [next_matrix, mat_id, rot_id]
                current_soln.append(y)
                pos += 1
                continue

            target_pos = placement_order[pos]
            dependencies = [d[0] for d in G.in_edges(target_pos)]
            dependency_pos = list(map(lambda x: find(placement_order, x), dependencies))
            must_intersect = set(filter(lambda z: z < pos, dependency_pos))
            must_intersect2 = [current_soln[xx][1] for xx in list(must_intersect)]
            alternatives = None
            in_use = set([yyy[1] for yyy in current_soln])
            if (len(must_intersect2) == 0):
                alternatives = set(tiles.keys()) - in_use
            else:
                alternatives = match_table[must_intersect2[0]] - in_use
                for xxx in must_intersect2[1:]:
                    alternatives = alternatives.intersection(match_table[xxx])

            possible = []
            for a in alternatives:
                for j in range(num_rotations):  # enumerate(rotation_labels):
                    matrix = permutation_table[a][j]
                    possible.append((matrix, a, j))

            if (len(possible) < 1):
                backtracking = True
                continue
            else:
                (next_matrix, mat_id, rot_id) = possible.pop()
                alternative_stack.append(possible)
                y = [next_matrix, mat_id, rot_id]
                current_soln.append(y)
                pos += 1
                continue


def q1(tiles, permutation_table, match_table, bootstrap=None):
    side_len = int(len(tiles.keys()) ** .5)
    G = nx.DiGraph()
    for i in range(side_len):
        for j in range(side_len):
            node = (i, j)
            G.add_node(node)
            if i >= 1:
                above = (i-1, j)
                if (above not in G.nodes()):
                    G.add_node(above)
                G.add_edge(node, above, weight='Y', color='red')
            if j >= 1:
                left = (i, j-1)
                if (left not in G.nodes()):
                    G.add_node(left)
                G.add_edge(node, left, weight='X', color='blue')

    placement_order = list(nx.dfs_preorder_nodes(G, source=(side_len - 1, side_len - 1)))
    a = iterative_traversal(G, placement_order, tiles, permutation_table,
                            match_table, bootstrap=bootstrap)
    point_dict = dict()
    point_list = []
    for idx, tilex in enumerate(a):
        d = placement_order[idx]
        coord_i = d[0]
        coord_j = d[1]
        point_dict[(coord_i, coord_j)] = tilex
        point_list.append([coord_i, coord_j, tilex['tile_id'], tilex['tile_rotation']])

    p1 = point_dict[(0, 0)]['tile_id']
    p2 = point_dict[(0, side_len - 1)]['tile_id']
    p3 = point_dict[(side_len - 1, 0)]['tile_id']
    p4 = point_dict[(side_len - 1, side_len - 1)]['tile_id']
    return (int(p1)*int(p2)*int(p3)*int(p4), point_list)


def q2(tiles, permutation_table, point_list):
    side_len = int(len(tiles.keys()) ** .5)
    point_dict = dict()
    for item in point_list:
        point_dict[(item[0], item[1])] = {'tile_id': item[2], 'tile_rotation': item[3]}

    rows = []
    for i in range(side_len):
        matrix_id = str(point_dict[(i, 0)]['tile_id'])
        rotation = point_dict[(i, 0)]['tile_rotation']
        row = permutation_table[matrix_id][rotation][1]
        for j in range(1, side_len):
            matrix_id = str(point_dict[(i, j)]['tile_id'])
            rotation = point_dict[(i, j)]['tile_rotation']
            mat = permutation_table[matrix_id][rotation][1]
            row = np.concatenate((row, mat), axis=1)
        rows.append(row)
    mat = rows[0]
    for i in range(1, len(rows)):
        r = rows[i]
        mat = np.concatenate((mat, r), axis=0)
    rows, cols = np.shape(mat)

    snake_re1 = r"(..................1.)"
    snake_re2 = r"(1....11....11....111)"
    snake_re3 = r"(.1..1..1..1..1..1...)"

    hashes_1 = sum([1 if c == '1' else 0 for c in snake_re1])
    hashes_2 = sum([1 if c == '1' else 0 for c in snake_re2])
    hashes_3 = sum([1 if c == '1' else 0 for c in snake_re3])

    pattern1 = re.compile(snake_re1)
    pattern2 = re.compile(snake_re2)
    pattern3 = re.compile(snake_re3)

    rotated = []
    for rotations in range(3):
        mx = np.rot90(mat, k=rotations)
        rotated.append(mx)
        for flip in range(1, 3):
            flp = np.flip(mx, flip-1)
            rotated.append(flp)

    hashmarks = 0
    for mx in rotated:
        s = ""
        (rows, cols) = np.shape(mx)
        count_hashes = 0
        for i in range(rows):
            for j in range(cols):
                s += (str(int(mx[i][j])))
                count_hashes += mx[i][j]
            s += "\n"
        hashmarks = count_hashes
        lines = s.split("\n")

        ones = dict()
        twos = dict()
        threes = dict()
        for idx, line in enumerate(lines):
            ones_matches = all_regex_matches(pattern1, line)
            twos_matches = all_regex_matches(pattern2, line)
            threes_matches = all_regex_matches(pattern3, line)
            for m in ones_matches:
                ones[(idx, m[0])] = m[1]
            for m in twos_matches:
                twos[(idx, m[0])] = m[1]
            for m in threes_matches:
                threes[(idx, m[0])] = m[1]

        snakes = 0
        total_hashes_in_matches = 0  # total number of matches in the pattern group
        for k, v in ones.items():
            cur_hashes_in_matches = v
            line_no, pos = k
            if (line_no + 1, pos) in twos:
                cur_hashes_in_matches += twos[(line_no + 1, pos)]
                if (line_no + 2, pos) in threes:
                    cur_hashes_in_matches += twos[(line_no + 1, pos)]
                    snakes += 1
                    total_hashes_in_matches += cur_hashes_in_matches
                    cur_hashes_in_matches = 0
        if (snakes > 0):
            snake_hashes = snakes * (hashes_1 + hashes_2 + hashes_3)
            nonsnake_hashes = hashmarks - snake_hashes
            return nonsnake_hashes


if __name__ == '__main__':
    s = [x.strip() for x in open("day20.txt").readlines()]
    tiles = parse(s, False)
    table = build_permutation_table(tiles)
    match_table = get_match_table(table)

    b = bootstrap(tiles, match_table=match_table)

    (q1_ans, point_list) = q1(tiles, table, match_table, bootstrap=b)
    print("q1:", q1_ans)
    q2_tiles = parse(s, True)
    rot_table = build_orienttion_table(q2_tiles)
    p = q2(q2_tiles, rot_table, point_list)
    print("q2:", p)
'''
import math
import re


def transposed(tile):
    return list(''.join(row) for row in zip(*tile))


def reversed_tile(tile):
    return [''.join(reversed(row)) for row in tile]


def rotations(tile):
    ans = [tile]
    for _ in range(3):
        ans.append(reversed_tile(transposed(ans[-1])))
    return ans


def group(tile):
    return rotations(tile) + rotations(transposed(tile))


tiles = {}
for tile in open('day20.txt').read().split('\n\n'):
    lines = tile.strip().split('\n')
    print(lines[0])
    tile_id = int(re.search(r'Tile (\d+):', lines[0]).group(1))
    rows = lines[1:]
    tiles[tile_id] = group(rows)

n = int(math.sqrt(len(tiles)))
arranged = [[0] * n for _ in range(n)]
stack = list(reversed(list((r, c) for c in range(n) for r in range(n))))


def solve():
    if not stack:
        print(arranged[0][0][0] * arranged[-1][0][0] * arranged[0][-1][0] *
              arranged[-1][-1][0])
        return True
    (r, c) = stack.pop()
    for tile_id in list(tiles):
        tile_group = tiles[tile_id]
        del tiles[tile_id]
        for tile in tile_group:
            if r > 0:
                if arranged[r - 1][c][1][-1] != tile[0]:
                    continue
            if c > 0:
                if list(row[-1] for row in arranged[r][c - 1][1]) != list(
                        row[0] for row in tile):
                    continue
            arranged[r][c] = (tile_id, tile)
            if solve():
                return True
        tiles[tile_id] = tile_group
    stack.append((r, c))


solve()


def remove_border(tile):
    return [row[1:-1] for row in tile[1:-1]]


board = [[remove_border(tile[1]) for tile in row] for row in arranged]
tile_n = len(board[0][0])


def get(r, c):
    return board[r // tile_n][c // tile_n][r % tile_n][c % tile_n]


board = [
    ''.join(get(r, c) for c in range(n * tile_n)) for r in range(n * tile_n)
]
for pattern in group(
    ['                  # ', '#    ##    ##    ###', ' #  #  #  #  #  #   ']):
    matches = 0
    for dr in range(len(board) - len(pattern) + 1):
        for dc in range(len(board[0]) - len(pattern[0]) + 1):
            matches += all(pattern[r][c] == ' ' or board[r + dr][c + dc] == '#'
                           for r in range(len(pattern))
                           for c in range(len(pattern[0])))
    if matches:
        print(''.join(board).count('#') -
              ''.join(pattern).count('#') * matches)
        break
