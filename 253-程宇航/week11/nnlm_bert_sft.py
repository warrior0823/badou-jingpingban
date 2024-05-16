#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader

"""
基于pytorch的Bert语言模型,进行sft形式训练
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"D:\badou-jingpin\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)   # 表示目标值为-1的样本在计算损失时被忽略

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:   # 训练时构建mask矩阵，使得上下文无交互
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:   # 预测，不需要mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)




#加载语料   title作为问题，content作为答案
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus



#建立构造sft数据

def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        # 导入已构建好的mask，让prompt有交互， answer没有交互即前面看不到后面
        mask = build_mask(len(prompt_encode), len(answer_encode))
        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)



# 构建mask, 输入prompt和answer的长度
def build_mask(pro, ans):
    len_pro = pro + 2    # prompt长度加上cls和sep
    len_ans = ans + 1    # 加上sep
    # 创建mask全1张量
    mask = torch.ones(len_pro+len_ans, len_pro+len_ans)
    for i in range(len_pro):
        mask[i, len_pro:] = 0    # 遍历前半句， 每一行的后半句都为0
    for i in range(len_ans):
        mask[len_pro+i, len_pro+i+1:] = 0    # 当遍历到后半句时，第i个字符后面开始为0
    return mask


# 构建pad_mask, 获取输入张量和目标形状
def pad_mask(tensor, target_shape):
    # 定义输入张量即mask的长宽，定义目标形状即最大长度的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 做截断或者填充 对比一下前面mask形状小还是定义的最长长度形状小，取最小的
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_height)
    # 将上一个mask张量的对应部分填充到定义的result全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result




#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(768, 21128)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 50:
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


#计算文本ppl
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
            pred_prob_distribute = model(x)[0]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    tokenizer = BertTokenizer.from_pretrained(r"D:\badou-jingpin\bert-base-chinese", return_dict=False)
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    max_length = 50
    char_dim = 768
    vocab = 21228         #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for x, mask, y in train_data:       # 构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("观众热捧沪语版《新闻坊》：每周一期怎么够", model, tokenizer))
        print(generate_sentence("不仅胶囊，有些冰淇淋也用过工业明胶", model, tokenizer))
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
