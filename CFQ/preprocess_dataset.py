import json
from tqdm import tqdm

with open('dataset.json', 'r') as file:
    lines = json.load(file)

preprocessed_lines = []
for line in tqdm(lines):
    input_line, output_line = line['questionPatternModEntities'], line['sparqlPatternModEntities'].replace('\n', ' ')
    preprocessed_lines.append(f'{input_line.strip()} <SEP> {output_line.strip()}\n')

with open(f'dataset.txt', 'w') as file:
    file.writelines(preprocessed_lines)
