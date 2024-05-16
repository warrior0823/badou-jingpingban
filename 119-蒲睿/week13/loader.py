# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.load()
        
    def load(self):
        self.data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            segments = f.read().split('\n\n')
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split('\n'):
                    if line.strip() == "":
                        continue
                    word, label = line.split()
                    sentence.append(word)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_id = self.tokenizer.encode(self.sentences[-1], max_length=self.config["max_length"], 
                                                  pad_to_max_length=True)
                labels = self.padding(labels, 0)
                self.data.append([torch.LongTensor(input_id), torch.LongTensor(labels)])
        return
    
    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id
    
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]    
                
    def load_schema(self, schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dataloader = DataLoader(dataset=dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader