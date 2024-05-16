# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer

"""
基于pytorch的LSTM语言模型
"""

bert_path = r"C:\Users\28194\PycharmProjects\pythonProject2\week6\week6 语言模型和预训练\下午\bert-base-chinese"
bert = BertModel.from_pretrained(bert_path, return_dict=False)
tokenizer = BertTokenizer.from_pretrained(bert_path)


class LanguageModel(nn.Module):
    def __init__(self, input_dim):
        super(LanguageModel, self).__init__()
        self.bert = bert
        self.classify = nn.Linear(input_dim * 3, 21128)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x, _ = self.bert(x)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return y_pred


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_sample(window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    x = tokenizer.encode(window, max_length=100, padding='max_length', truncation=True)
    y = tokenizer.encode(target, max_length=100, padding='max_length', truncation=True)
    return x, y


def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 文本生成测试代码
def generate_sentence(openings, model):
    model.eval()
    with torch.no_grad():
        x = tokenizer.encode(openings, add_special_tokens=True, return_tensors='pt')
        # 将mask_token转换为一个张量，形状与x相同
        mask_tensor = torch.full_like(x, fill_value=tokenizer.mask_token_id, dtype=torch.long)
        # 寻找输入文本中[MASK]标记的位置
        mask_index = (x != mask_tensor).nonzero(as_tuple=True)[1]
        for _ in range(50):
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            predict_index = torch.argmax(y[mask_index]).item()
            predict_token = tokenizer.convert_ids_to_tokens([predict_index])[0]
            # 更新输入文本，替换下一个[MASK]标记
            x[0, mask_index[0]] = predict_index
            if predict_token == tokenizer.sep_token:
                break
        pre_text = tokenizer.decode(x.squeeze(), skip_special_tokens=True)
    return pre_text


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 13  # 样本文本长度
    corpus = load_corpus(corpus_path)  # 加载语料
    model = LanguageModel(char_dim)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
