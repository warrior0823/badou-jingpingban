
# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
# 导入 bert 模型
from transformers import BertModel, BertTokenizer
import json

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        hidden_size = self.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None, attention_mask=None):
        x = self.bert.forward(x, attention_mask=attention_mask)[0]
        x = self.dropout(x)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)

        if self.training:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        corpus_max_length = 0
        for line in f:
            data = json.loads(line)
            corpus.append((data["title"], data["content"]))
            corpus_max_length = max(corpus_max_length, len(data["title"]) + len(data["content"]))
    return corpus, corpus_max_length


def build_dataset(batch_size, corpus_max_length, tokenizer, corpus):
    dataset_x, dataset_y, attention_mask_arr = [], [], []
    for _ in range(batch_size):
        title, content = random.choice(corpus)
        x, y, attention_mask = build_sample(tokenizer, title, content, corpus_max_length)

        dataset_x.append(x)
        dataset_y.append(y)
        attention_mask_arr.append(attention_mask)

    # 转换列表为张量
    dataset_x = torch.LongTensor(dataset_x)
    dataset_y = torch.LongTensor(dataset_y)

    # 将 attention_mask_arr 转换为张量并堆叠
    attention_mask_arr = torch.stack(
        [torch.tensor(arr, dtype=torch.long).clone().detach() for arr in attention_mask_arr])
    return dataset_x, dataset_y, attention_mask_arr


def build_sample(tokenizer, question, answer, corpus_max_length):
    question_encode = tokenizer.encode(question, add_special_tokens=False)
    answer_encode = tokenizer.encode(answer, add_special_tokens=False)

    question_encode = tokenizer.encode(question_encode, add_special_tokens=False) + [tokenizer.sep_token_id]
    answer_encode = tokenizer.encode(answer_encode, add_special_tokens=False) + [tokenizer.sep_token_id]

    x = [tokenizer.cls_token_id] + question_encode + answer_encode
    y = question_encode + [tokenizer.sep_token_id] + answer_encode

    padding_length = corpus_max_length - len(x)

    x += [tokenizer.pad_token_id] * padding_length
    y += [tokenizer.pad_token_id] * padding_length

    # 构建 attention mask
    max_len = len(x)
    attention_mask = torch.ones(max_len, max_len)

    # 设置后面的部分为0
    for i in range(len(question_encode), max_len):
        attention_mask[i, :] = 0
    return x, y, attention_mask


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过20字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings


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
    epoch_num = 40  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    pretrain_model_path =  "D:\ai\toolutils\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    # 获取tokenizer的词表大小
    vocab_size = len(tokenizer.vocab)
    corpus, corpus_max_length = load_corpus(corpus_path)  # 加载语料
    model = LanguageModel(vocab_size, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, attention_mask = build_dataset(batch_size, corpus_max_length, tokenizer, corpus)  # 构建一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, attention_mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北美洲发现肥皂人", model, tokenizer))
        print(generate_sentence("各路足坛名人陆续抵达", model, tokenizer))

    # 保存模型
    if save_weight:
        torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train("sample_data.json", True)