# -*- coding: utf-8 -*-

import json
import torch
import jieba
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.index_to_label = {0: "文化", 1: "时尚", 2: "健康", 3: "教育", 4: "股票",
                               5: "娱乐", 6: "游戏", 7: "科技", 8: "房产", 9: "家居",
                               10: "财经", 11: "汽车", 12: "体育", 13: "军事", 14: "国际",
                               15: "彩票", 16: "社会", 17: "旅游"}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        if len(input_id) >= self.config["max_length"]:
            input_id = input_id[:self.config["max_length"]]
        else:
            input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1   #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("data/valid_tag_news.json", Config)
    print(dg[1])

