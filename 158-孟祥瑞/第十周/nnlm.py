# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertTokenizer, BertModel

"""
基于pytorch的Bert模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.lstm = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # x = self.embedding(x)
        # x, _ = self.lstm(x)
        # y_pred = self.classify(x)
        # if y is not None:
        #     return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        # else:
        #     return torch.softmax(y_pred, dim=-1)
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


# 加载字表
# def build_vocab(vocab_path):
#     token_dict = {"<pad>": 0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             word = line.strip()
#             token_dict[word] = index + 1
#     return token_dict


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 建立模型
def build_model(char_dim, vocab_size, pretrain_model_path):
    model = LanguageModel(char_dim, vocab_size, pretrain_model_path)
    return model


# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    # 输入和输出错开一位
    target = corpus[start+1:end+1]
    # 将字转换成序号
    # x = [vocab.get(word, vocab["[UNK]"]) for word in window]
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
    # y = [vocab.get(word, vocab["[UNK]"]) for word in target]
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
    return x, y


# 创建数据集
def build_dataset(batch_size, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 文本生成测试
def generate_sentence(opening, model, tokenizer, window_size):
    # 要将编码后的文本转换回字符
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()    # 将模型切换到评估模式
    with torch.no_grad():       # 关闭梯度计算
        pred_char = ""
        # 生成了换行符，或者生成文本超过20个字，则终止迭代
        while pred_char != "\n" and len(opening) <= 30:
            opening += pred_char
            # 不管输入多少字，只取后面window_size个字
            # x = [vocab.get(char, vocab["<UNK>"]) for char in opening[-window_size:]]
            x = tokenizer.encode(opening, add_special_tokens=False)
            x = torch.LongTensor(x)
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            # pred_char = reverse_vocab[index]
            pred_char = ''.join(tokenizer.decode(index))
    return opening


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


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()    # 将模型切换到评估模式
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i-window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor(x)
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribution = model(x)[0][-1]
            target_prob = pred_prob_distribution[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20      # 训练轮数
    batch_size = 128     # 每次训练样本数
    train_sample = 10000     # 每轮训练总共训练的样本数
    char_dim = 768      # 每个字的维度
    window_size = 10        # 样本文本长度
    vocab_size = 21128      # bert字表大小
    learning_rate = 0.001   # 学习率
    pretrain_model_path = "../bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus = load_corpus(corpus_path)     # 加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)       # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)       # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()       # 切换到训练模式
        train_loss = []
        for batch in range(int(train_sample / batch_size)):
            # 构建一组训练样本
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # 梯度归零
            optim.zero_grad()
            # 计算loss
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            train_loss.append(loss.item())
        print("========\n第%d轮训练平均loss: %f" % (epoch + 1, np.mean(train_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return
    else:
        return


if __name__ == '__main__':
    train("corpus.txt", False)
    # mask = torch.tril(torch.ones((2, 2, 3)))
    # print(mask)
