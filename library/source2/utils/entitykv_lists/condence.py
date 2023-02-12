from pathlib import Path

new_keys = set()

for file in Path('./').glob('*.txt'):
    with file.open('r') as f:
        for line in f.readlines():
            new_keys.add(line.strip('\n'))

with open('condenced.txt', 'w') as f:
    for key in new_keys:
        f.write(key + '\n')
