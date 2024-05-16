import numpy as np
import json
import re
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd

"""
数据加载
"""

class DataGenerator:
    def __init__(self,data_path, config):
        self.path =data_path
        self.config=config   # 参数需要先引入
        self.config["class_num"] = 2  # 2类，好评与差评
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])  # 加载词表
        self.config["vocab_size"] = len(self.vocab)
        self.load()  # 加载数据

    def load(self):
        self.data = []
        df = pd.read_csv(self.path, index_col=False)
        for i in range(len(df)):
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(df['review'][i], max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(df['review'][i])
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([df['label'][i]])
            self.data.append([input_id, label_index])
        return


    def encode_sentence(self, text):
        input_id = []
        for i in text:
            input_id.append(self.vocab.get(i,self.vocab["[UNK]"])) #如果词表没有用[UNK]代表
        return self.padding(input_id)

    def padding(self,input_id):
        #如果本身长度大于最大长度，仅保留最大长度，如果小于最大长度，则后面补0
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_data(data_path, config, shuffle=True):
    dg=DataGenerator(data_path, config)
    # shuffle = True,数据随机打乱
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config_hw import Config
    dg = DataGenerator("文本分类练习.csv", Config)
    # print(len(dg))
    # print(dg[0:20])
    # for index, batch_data in enumerate(dg):
    #     # input_ids, labels = batch_data
    #     print(batch_data)
    #     i=i+1
    # print(i)




