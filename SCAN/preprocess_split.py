import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--split_type', default='simple', type=str)
args = parser.parse_args()

for split in ['train', 'dev', 'test']:
    with open(f'{args.split_type}/tasks_{split}_{args.split_type}.txt', 'r') as file:
        lines = file.readlines()

    preprocessed_lines = []
    for line in tqdm(lines):
        input_line, output_line = line.split('OUT:')
        input_line = input_line.split('IN:')[1]
        preprocessed_lines.append(f'{input_line.strip()} <SEP> {output_line.strip()}\n')

    with open(f'{args.split_type}/{split}.txt', 'w') as file:
        file.writelines(preprocessed_lines)
