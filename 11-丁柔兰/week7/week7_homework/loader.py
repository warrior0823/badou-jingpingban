# -*- coding: utf-8 -*-
import csv
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []  # 初始化一个空列表，用于存储处理后的数据
        with open(self.path, 'r', encoding='utf-8',errors='ignore') as f:
            data = csv.reader(f)
            next(data)  # 跳过第一行的标题部分
            for line in data:
                label = int(line[0])
                review = line[1]
                if self.config["model_type"] == "bert":  # 检查配置是否指定使用 BERT 模型
                    # 如果是 BERT 模型，使用 tokenizer 对标题进行编码，转换为 input_id
                    # max_length 参数限制编码后的长度，pad_to_max_length 参数确保所有编码长度一致，不足的部分使用填充
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],
                                                     padding='max_length', truncation=True)
                else:
                    # 如果不是 BERT 模型，则使用自定义的 encode_sentence 方法对标题进行编码
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)  # 把编码后的 input_id 转换为 PyTorch 的 LongTensor 类型
                label_index = torch.LongTensor([label])  # 把标签索引也转换为 PyTorch 的 LongTensor 类型
                self.data.append([input_id, label_index])  # 将处理后的 input_id 和标签索引作为一个列表添加到 self.data 中
        return  # 方法结束，没有返回值，因为处理后的数据存储在 self.data 属性中

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
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
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    vocab_size = len(dg)# 获取词汇表大小
    return dl,vocab_size

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/文本分类练习.csv", Config)
    print(dg[1])

    # # 读取CSV文件
    # data = pd.read_csv("../data/文本分类练习.csv", encoding='utf8')
    # # 将DataFrame转换为JSON格式的字符串，并写入文件
    # with open('output.json', 'w', encoding='utf-8') as f:
    #     f.write(data.to_json(orient='records', force_ascii=False))
    # # 读取并加载JSON文件
    # with open('output.json', 'r', encoding='utf-8') as f:
    #     data_list = [json.loads(line) for line in f]
    # # 现在data_list变量包含了一个字典列表，每个字典对应文件中的一行JSON数据
    # print(data_list)