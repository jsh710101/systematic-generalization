import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--split_type', default='simple', type=str)
args = parser.parse_args()

with open('dataset.txt', 'r') as file:
    lines = file.readlines()

FILENAMES = {'simple': 'random_split', 'length': 'query_complexity_split'}

filename = FILENAMES.get(args.split_type, args.split_type)
with open(f'{args.split_type}/{filename}.json') as file:
    indices = json.load(file)

for split in ['train', 'dev', 'test']:
    split_lines = [lines[i] for i in indices[f'{split}Idxs']]

    with open(f'{args.split_type}/{split}.txt', 'w') as file:
        file.writelines(split_lines)
