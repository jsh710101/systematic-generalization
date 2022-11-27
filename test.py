import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, DataCollatorForSeq2Seq
from model import Transformer, T5, RoFormer, BertLSTM

DATASETS = ['SCAN', 'PCFG', 'COGS', 'CFQ']
MODELS = {'transformer': Transformer, 't5': T5, 'roformer': RoFormer, 'bertlstm': BertLSTM}

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dataset', default='SCAN', type=str, choices=DATASETS)
parser.add_argument('--split_type', default='simple', type=str)
parser.add_argument('--model', default='transformer', type=str, choices=MODELS.keys())
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--disable_tqdm', action='store_true')
args = parser.parse_args()

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
dirname = f'{args.dataset}/{args.split_type}'

tokenizer = PreTrainedTokenizerFast(
    pad_token='<PAD>', bos_token='<BOS>', eos_token='<EOS>',
    tokenizer_file=f'{args.dataset}/tokenizer.json')
dataset = load_from_disk(f'{dirname}/data', keep_in_memory=True)

model = MODELS[args.model].from_pretrained(f'{dirname}/model/{args.model}').to(device)
model.prepare()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
dataloader = {
    split: DataLoader(
        dataset[split], batch_size=args.batch_size,
        num_workers=4, collate_fn=data_collator, pin_memory=True) for split in ['train', 'dev', 'test']}

model.eval()
accuracy = {split: 0 for split in ['train', 'dev', 'test']}

with torch.no_grad():
    for split in ['train', 'dev', 'test']:
        for batch in tqdm(dataloader[split], disable=args.disable_tqdm):
            batch = batch.to(device)
            logits = model(**batch).logits

            predictions = logits.argmax(-1)
            predictions[batch.labels == -100] = -100

            num_correct = (predictions == batch.labels).all(-1).sum()
            accuracy[split] += num_correct.item() / len(dataset[split])

print(f'Accuracy: {accuracy}')


# def predict(input_text):
#     with torch.no_grad():
#         input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
#         output_ids = model.generate(input_ids=input_ids, use_cache=False)[0]
#         return tokenizer.decode(output_ids, skip_special_tokens=True)
# breakpoint()
