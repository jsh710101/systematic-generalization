import argparse
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SCAN', type=str)
parser.add_argument('--split_type', default='simple', type=str)
args = parser.parse_args()

dirname = f'{args.dataset}/{args.split_type}'
data_files = {split: f'{dirname}/{split}.txt' for split in ['train', 'dev', 'test']}

tokenizer = PreTrainedTokenizerFast(
    pad_token='<PAD>', bos_token='<BOS>', eos_token='<EOS>',
    tokenizer_file=f'{args.dataset}/tokenizer.json')
dataset = load_dataset('text', data_files=data_files, keep_in_memory=True)


def encode(example):
    texts = example['text'].split('<SEP>')
    input_text, output_text = map(tokenizer, texts)

    return {
        'input_ids': input_text.input_ids,
        'labels': output_text.input_ids[1:],
    }


dataset = dataset.map(encode, remove_columns='text', keep_in_memory=True)
dataset.save_to_disk(f'{dirname}/data')
