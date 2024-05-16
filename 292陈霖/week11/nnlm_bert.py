#coding:utf8
import json

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
import math
import random
import os
import re
from torch.utils.data import DataLoader

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        BertPath = r"E:\学习资料_summary\八斗课程-精品班\第六周\bert-base-chinese"
        self.bert = BertModel.from_pretrained(BertPath, return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, len(vocab))
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None,y=None):
        if y is not None:
            x, _ = self.bert(x,attention_mask = mask)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"],line["content"]])
    return corpus

# sft的数据构造
def build_sample(vocab, prompt,answer, max_length):
    prompt_encode = vocab.encode(prompt,add_special_tokens=False)
    answer_encode = vocab.encode(answer, add_special_tokens=False)
    x = [vocab.cls_token_id] + prompt_encode + [vocab.sep_token_id] + answer_encode + [
        vocab.sep_token_id]
    y = len(prompt_encode) * [-1] + [-1] + answer_encode + [vocab.sep_token_id] + [-1]
    mask = create_mask(len(prompt_encode), len(answer_encode))
    x = x[:max_length] + [0] * (max_length - len(x))
    y = y[:max_length] + [0] * (max_length - len(y))
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y,mask

def create_mask(s1,s2):
    len_s1=s1+2
    len_s2=s2+1
    mask = torch.ones(len_s1+len_s2,len_s1+len_s2)
    for i in range(len_s1):
        mask[i,len_s1:]=0
    for i in range(len_s2):
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask

def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(vocab, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        x,y,mask = build_sample(vocab,prompt,answer,max_length)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#建立模型
def build_model(vocab):
    model = LanguageModel(vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    openings = vocab.encode(openings)
    with torch.no_grad():
        while len(openings) <= 30:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
        return vocab.decode(openings)

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



def train(corpus_path, save_weight=True):
    epoch_num = 100        #训练轮数
    batch_size = 64       #每次训练样本个数
    max_length = 50  # 样本文本长度
    # char_dim = 756        #每个字的维度
    # vocab_size = 21128
    BertPath = r"E:\学习资料_summary\八斗课程-精品班\第六周\bert-base-chinese"
    vocab = build_vocab(BertPath)       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    train_data = build_dataset(vocab, corpus, max_length, batch_size)
    model = build_model(vocab)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x,mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, vocab))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, vocab))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
