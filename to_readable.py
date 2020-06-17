import sys

file = []
with open(sys.argv[1]) as f:
    for line in f:
        file.append(line)

to_remove = list("[](,'")
file = ''.join(file)
for r in to_remove:
    file = file.replace(r, '')
file = file.replace(')', '\n')

with open(sys.argv[1], 'w') as f:
    print(file, file=f)
