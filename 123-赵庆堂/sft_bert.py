import torch
import numpy as np
import torch.nn as nn
import json
from transformers import BertTokenizer, BertModel
import random
import os
from torch.utils.data import Dataset, DataLoader

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r'D:\python\python_learn\python_learn\bert-base-chinese', return_dict=False)
        self.bert_tokenizer = BertTokenizer.from_pretrained(r'D:\python\python_learn\python_learn\bert-base-chinese')
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert_tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
    #     和构建数据时候的输出顺序要保持一致
    def forward(self, x, y = None, mask = None):
        if y is not None:
            x, _ = self.bert(x, attention_mask = mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim = -1)


def load_data(path):
    corpus = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            title = item.get("title")
            content = item.get("content")
            corpus.append([title, content])
        return corpus

def build_data(max_length, corpus, batch_size):
    bert_tokenizer = BertTokenizer.from_pretrained(r'D:\python\python_learn\python_learn\bert-base-chinese')
    data_set = []
    for i, item in enumerate(corpus):
        # 获取语料中的title和content
        title, content = item
        # 有cls_token和sep_token
        title_encode = bert_tokenizer.encode(title)
        # 无特殊token
        content_encode = bert_tokenizer.encode(content, add_special_tokens=False)
        # cls_token + title_encode + sep_token + content_encode + sep_token
        x = title_encode + content_encode + [bert_tokenizer.sep_token_id]
        # (cls_token + title_encode的个数) * -1(ignore_index) sep_token + content_encode + sep_encode + eos(ignore_index)
        y = (len(title_encode) - 1) * [-1] +  content_encode + [bert_tokenizer.sep_token_id] + [-1]
        # padding
        x = x[:max_length] + [bert_tokenizer.pad_token_id] * (max_length - len(x))
        y = y[:max_length] + [bert_tokenizer.pad_token_id] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        # 生成在不padding的情况下,x的mask
        mask = build_mask(len(title_encode) , len(content_encode))
        '''
        截取mask矩阵,方法有待优化,再研究一下
        '''
        mask = mask[:max_length, :max_length]
        data_set.append([x, y, mask])

    return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0)

def build_mask(title_length, content_length):
    title_encode_length = title_length
    content_encode_length = content_length + 1
    mask = torch.ones(title_encode_length + content_encode_length, title_encode_length + content_encode_length)
    for i in range(title_encode_length):
        mask[i, title_encode_length:] = 0
    for i in range(content_encode_length):
        mask[i + title_encode_length, i + title_encode_length + 1:] = 0
    return mask

def build_model():
    model = LanguageModel()
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
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



def main(corpus_path, save_weight=True):
    epoch_num = 50        #训练轮数
    batch_size = 32       #每次训练样本个数
    max_length = 50       #样本文本长度
    learning_rate = 0.0001  #学习率


    pretrain_model_path = r'D:\python\python_learn\python_learn\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_data(corpus_path)     #加载语料
    train_data = build_data(max_length, corpus, batch_size)  #建立数据集
    model = build_model()    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y, mask in train_data: #构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()

            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    main(r'D:\百度网盘\week10 文本生成\week10 文本生成问题\transformers-生成文章标题\sample_data.json', False)