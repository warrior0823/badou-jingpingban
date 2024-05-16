# coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import logging

"""
基于pytorch的LSTM语言模型
"""

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageModel(nn.Module):
    def __init__(self, input_dim, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        # self.classify = nn.Linear(input_dim, len(vocab))
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

class DataGenerator:
    def __init__(self, path, tokenizer, max_length):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load_sample()

    # 加载样本
    def load_sample(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                question = line["question"]
                answer = line["answer"]
                question_encode = self.tokenizer.encode(question)
                answer_encode = self.tokenizer.encode(answer, add_special_tokens=False)
                x = question_encode + answer_encode + [self.tokenizer.sep_token_id]
                y = (len(question_encode) - 1) * [-1] + answer_encode + [self.tokenizer.sep_token_id] + [-1]
                ori_x_len = len(x)
                # padding
                x = x[:self.max_length] + [0] * (self.max_length - len(x))
                y = y[:self.max_length] + [0] * (self.max_length - len(y))
                mask = self.build_mask(torch.LongTensor(x), len(question), ori_x_len)
                mask = mask.squeeze()
                self.data.append([torch.LongTensor(x), torch.LongTensor(y), mask])

    def build_mask(self, x, question_len, ori_x_len):
        x = x.unsqueeze(0)
        ones = torch.ones(x.shape[0], x.shape[1], x.shape[1])
        for i in range(x.shape[1] - question_len):
            ones[:, :question_len + i, question_len + i:] = 0
        ones[:, ori_x_len:, :] = 0
        return ones

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(sample_path, tokenizer, batch_size, max_length, shuffle=True):
    dg = DataGenerator(sample_path, tokenizer, max_length)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl


# 建立模型
def build_model(pretrain_model_path, char_dim):
    model = LanguageModel(char_dim, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        # 生成了换行符，或生成文本超过30字则终止迭代
        while len(openings) <= 30:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)

    return tokenizer.decode(openings)


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(sample_path):
    epoch_num = 40  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    char_dim = 256  # 每个字的维度
    max_length = 30
    pretrain_model_path = "../week6/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    train_data = load_data(sample_path, tokenizer, batch_size, max_length)
    model = build_model(pretrain_model_path, char_dim)  # 建立模型
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        epoch += 1
        print("epoch %d begin" % epoch)
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optim.zero_grad()  # 梯度归零
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            x, y, mask = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(x, y, mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
        print("epoch average loss: %f" % np.mean(train_loss))
        print(generate_sentence("你好吗？", model, tokenizer))
        print(generate_sentence("你喜欢吃什么？", model, tokenizer))


# if __name__ == "__main__":
#     sample_path = "sample_data.json"
#     train(sample_path)
