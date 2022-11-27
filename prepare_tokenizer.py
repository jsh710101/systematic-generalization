import argparse
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SCAN', type=str)
args = parser.parse_args()

split_type = 'gen' if args.dataset == 'COGS' else 'simple'
files = [f'{args.dataset}/{split_type}/{split}.txt' for split in ['train', 'dev', 'test']]

tokenizer = Tokenizer(WordLevel())
tokenizer.pre_tokenizer = WhitespaceSplit()
tokenizer.post_processor = TemplateProcessing(
    single='<BOS> $A <EOS>',
    special_tokens=[('<BOS>', 1), ('<EOS>', 2)])

trainer = WordLevelTrainer(special_tokens=['<PAD>', '<BOS>', '<EOS>'])
tokenizer.train(files, trainer=trainer)
tokenizer.save(f'{args.dataset}/tokenizer.json')
