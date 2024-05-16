# -*- coding: utf-8 -*-

import json
import re
import os
from collections import defaultdict

import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


# 加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line['title'], line['content']])
    return corpus


def new_mask(len_title, len_content):
    len_v = len_title + len_content
    mask = torch.ones(len_v, len_v)
    mask[len_title:, len_title:] = 0
    for i in range(len_content):
        mask[len_title + i, len_title + i + 1:] = 0
    return mask


def pad_mask(mask, t_h, t_w):
    h, w = mask.shape
    result = torch.zeros(t_h, t_w, dtype=mask.dtype, device=mask.device)
    t_h = min(h, t_h)
    t_w = min(w, t_w)
    result[:t_h, :t_w] = mask[:t_h, :t_w]
    return result


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
        self.corpus = load_corpus(config.corpus)
        self.load()

    def build_sample(self, prompt, answer):
        max_length = self.config.max_length
        tokenizer = self.tokenizer
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        # 构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = new_mask(len(prompt_encode) + 2, len(answer_encode) + 1)
        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, max_length, max_length)
        return x, mask, y

    def build_dataset(self):
        dataset = []
        for i, (prompt, answer) in enumerate(self.corpus):
            dataset.append(self.build_sample(prompt, answer))
        return dataset

    def load(self):
        self.data = self.build_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_data(config, shuffle=True):
    dg = DataGenerator(config)
    dl = DataLoader(dg.data, batch_size=config.batch_size, shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    data = load_data(Config)
    dg = DataGenerator(Config)

    print(data, len(data), len(dg.data))
    for x, mask, y in dg:
        print(x, mask, y)
