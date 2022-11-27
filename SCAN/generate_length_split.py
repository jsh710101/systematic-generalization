import argparse
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', default=26, type=int)
args = parser.parse_args()

split_type = f'length{args.cutoff}'
Path(split_type).mkdir(parents=True, exist_ok=True)

with open('tasks.txt', 'r') as file:
    lines = file.readlines()

lines_dict = {split: [] for split in ['train', 'test']}
for line in tqdm(lines):
    output_line = line.split('OUT:')[1]
    length = len(output_line.split())
    split = 'train' if length <= args.cutoff else 'test'
    lines_dict[split].append(line)

for split in ['train', 'test']:
    with open(f'{split_type}/tasks_{split}_{split_type}.txt', 'w') as file:
        file.writelines(lines_dict[split])
