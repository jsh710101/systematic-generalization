import argparse
from tqdm import tqdm
from copy import deepcopy
from time import perf_counter
# from datetime import timedelta, timezone, datetime
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
# from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, BertConfig, EncoderDecoderConfig, T5Config, RoFormerConfig, DataCollatorForSeq2Seq
from model import Transformer, T5, RoFormer, BertLSTM
from utils import get_max_length, get_elapsed_time

DATASETS = ['SCAN', 'PCFG', 'COGS', 'CFQ']
MODELS = {'transformer': Transformer, 't5': T5, 'roformer': RoFormer, 'bertlstm': BertLSTM}

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dataset', default='SCAN', type=str, choices=DATASETS)
parser.add_argument('--split_type', default='simple', type=str)
parser.add_argument('--model', default='transformer', type=str, choices=MODELS.keys())
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--num_layers', default=6, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--intermediate_size', default=2048, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--max_norm', default=1.0, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--eval_interval', default=10, type=int)
parser.add_argument('--disable_tqdm', action='store_true')
args = parser.parse_args()

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
dirname = f'{args.dataset}/{args.split_type}'

tokenizer = PreTrainedTokenizerFast(
    pad_token='<PAD>', bos_token='<BOS>', eos_token='<EOS>',
    tokenizer_file=f'{args.dataset}/tokenizer.json')
dataset = load_from_disk(f'{dirname}/data', keep_in_memory=True)

max_input_length = get_max_length(dataset, 'input_ids')
max_output_length = get_max_length(dataset, 'labels')
max_position_embeddings = max(max_input_length, max_output_length) + 3

if args.model in ['transformer', 'bertlstm']:
    config = BertConfig(
        max_length=max_output_length + 3, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1, tie_word_embeddings=True, vocab_size=len(tokenizer),
        hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads, intermediate_size=args.intermediate_size,
        hidden_act='gelu_new', max_position_embeddings=max_position_embeddings, position_embedding_type='relative_key')
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config, deepcopy(config), **config.to_dict())

elif args.model == 't5':
    config = T5Config(
        max_length=max_output_length + 3, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1, tie_word_embeddings=True, vocab_size=len(tokenizer),
        d_model=args.hidden_size, d_kv=args.hidden_size // args.num_heads, d_ff=args.intermediate_size, num_layers=args.num_layers, num_heads=args.num_heads,
        feed_forward_proj='gelu_new', relative_attention_num_buckets=32, relative_attention_max_distance=128)

elif args.model == 'roformer':
    config = RoFormerConfig(
        max_length=max_output_length + 3, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1, tie_word_embeddings=True, vocab_size=len(tokenizer),
        hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads, intermediate_size=args.intermediate_size,
        hidden_act='gelu_new', max_position_embeddings=max_position_embeddings)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config, deepcopy(config), **config.to_dict())

model = MODELS[args.model](config).to(device)
model.prepare()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
dataloader = {
    split: DataLoader(
        dataset[split], batch_size=args.batch_size, shuffle=(split == 'train'),
        num_workers=4, collate_fn=data_collator, pin_memory=True) for split in ['train', 'dev', 'test']}

criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, args.num_epochs * len(dataloader['train']))

# KST = timezone(timedelta(hours=9))
# time = datetime.now(KST).strftime('%Y-%m-%d_%H:%M')
# writer = SummaryWriter(f'{dirname}/log/{args.model}/{time}')

for epoch in range(1, args.num_epochs + 1):
    start_time = perf_counter()

    model.train()
    train_loss = 0

    for batch in tqdm(dataloader['train'], disable=args.disable_tqdm):
        batch = batch.to(device)
        loss = model(**batch).loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * len(batch.labels) / len(dataset['train'])
    # writer.add_scalar('Loss/train', train_loss, epoch)

    accuracy = None
    if epoch % args.eval_interval == 0:
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
                # writer.add_scalar(f'Accuracy/{split}', accuracy[split], epoch)

    end_time = perf_counter()
    elapsed_time = get_elapsed_time(start_time, end_time)

    print(f'[Epoch {epoch:3}/{args.num_epochs}] Loss: {train_loss:6.4f} | Accuracy: {accuracy} | {elapsed_time}')

model.save_pretrained(f'{dirname}/model/{args.model}')
# writer.add_hparams(vars(args), accuracy, hparam_domain_discrete={'dataset': DATASETS, 'model': list(MODELS.keys())}, run_name='.')
# writer.close()
