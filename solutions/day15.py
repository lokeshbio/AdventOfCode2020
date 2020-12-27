#!/usr/bin/env python
in_lst = [0,13,1,8,6,15]
'''
main_dict = {}

count = len(in_lst)
while count < 30000000:
	indices = [i for i, x in enumerate(in_lst) if x == in_lst[count-1]]
	if len(indices) == 1:
		in_lst.append(0)
	elif indices[len(indices)-1] - indices[len(indices)-2] == 1:
		in_lst.append(1)
	else:
		in_lst.append(indices[len(indices)-1] - indices[len(indices)-2])
	count = len(in_lst)
	print(count, end = '\r')
print(in_lst[count-1])

'''
def sequence(startingNumbers):
    spokenNumbers = {x: i for i, x in enumerate(startingNumbers)}
    yield from spokenNumbers.keys()

    nextNumber = 0
    turn = len(startingNumbers)
    while True:
        yield nextNumber

        lastOcc = spokenNumbers.get(nextNumber, turn)
        spokenNumbers[nextNumber] = turn

        nextNumber = turn - lastOcc
        turn += 1


def nth(iterable, n):
    for _ in range(n - 1):
        next(iterable)
    return next(iterable)


if __name__ == '__main__':
    print("Part one:", nth(sequence(in_lst), 2020))
    print("Part two:", nth(sequence(in_lst), 30000000))