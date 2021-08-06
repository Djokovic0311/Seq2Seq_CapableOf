import re
import spacy
import random
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
from torchtext.datasets import Multi30k
import pandas as pd
from sklearn.model_selection import train_test_split
#
# def load_dataset1(batch_size):
#     spacy_de = spacy.load('de')
#     spacy_en = spacy.load('en')
#     url = re.compile('(<url>.*</url>)')
#
#     def tokenize_de(text):
#         return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]
#
#     def tokenize_en(text):
#         return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
#
#     DE = Field(tokenize=tokenize_de, include_lengths=True,
#                init_token='<sos>', eos_token='<eos>')
#     EN = Field(tokenize=tokenize_en, include_lengths=True,
#                init_token='<sos>', eos_token='<eos>')
#     train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
#     DE.build_vocab(train.src, min_freq=2)
#     EN.build_vocab(train.trg, max_size=10000)
#     train_iter, val_iter, test_iter = BucketIterator.splits(
#             (train, val, test), batch_size=batch_size, repeat=False)
#     return train_iter, val_iter, test_iter, DE, EN

def load_dataset(batch_size):

    def tokenize(text):
        result = []
        for item in text.split():
            result += item.split('_')
            # print(result)
        return result

    INPUT = Field(sequential=True, tokenize=tokenize, fix_length=200)
    OUTPUT = Field(sequential=True, tokenize=tokenize, fix_length=200)
    train, val, test = TabularDataset.splits(
        path='data/', train='train.csv',
        validation='val.csv', test='test.csv', format='csv', skip_header=True,
        fields=[('',None),('Input', INPUT),('Output', OUTPUT)])
    INPUT.build_vocab(train.Input)
    OUTPUT.build_vocab(train.Output)
    print(vars(train.examples[0]))
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False, sort_key=lambda x: x.Input)


    return train_iter, val_iter, test_iter,INPUT,OUTPUT

# load_dataset(1)