import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--split_type', default='simple', type=str)
args = parser.parse_args()

for split in ['train', 'dev', 'test']:
    with open(f'{args.split_type}/{split}.src', 'r') as file:
        input_lines = file.readlines()
    with open(f'{args.split_type}/{split}.tgt', 'r') as file:
        output_lines = file.readlines()

    preprocessed_lines = []
    for input_line, output_line in tqdm(zip(input_lines, output_lines)):
        preprocessed_lines.append(f'{input_line.strip()} <SEP> {output_line.strip()}\n')

    with open(f'{args.split_type}/{split}.txt', 'w') as file:
        file.writelines(preprocessed_lines)
