import random
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import math
import numpy as np


#加载语料
def load_corpus(path):
    corpus=""
    s=''
    with open(path,encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus
corpus=load_corpus(r'E:\Pycharm_learn\pythonProject1\wk10\lstm语言模型生成文本\corpus.txt')
# print(corpus)

def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab
# vocab=build_vocab(r'E:\Pycharm_learn\pythonProject1\wk10\lstm语言模型生成文本\vocab.txt')
# reverse_vocab = dict((y, x) for x, y in vocab.items())
# print(reverse_vocab)

# for char in openings[-10:]:
#     print(char)
# x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-10:]]
# print(x)
#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus,tokenizer):
    start=random.randint(0,len(corpus)-1-window_size)
    end=start+window_size
    window=corpus[start:end]
    target=corpus[start+1:end+1]
    x=tokenizer.encode(window)
    y=tokenizer.encode(target)
    return x,y

# x,y=build_sample(10, corpus)
# print(x)
# print(y)

#建立数据集,sample_length 输入需要的样本数量。需要多少生成多少,vocab 词表,window_size 样本长度,corpus 语料字符串
def build_dataset(sample_length, window_size, corpus,tokenizer):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x,y=build_sample(window_size,corpus,tokenizer)
        if len(x)==len(y)== (window_size+2):
        # print(len(x),len(y))
            dataset_x.append(x)
            dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

# sample_x,sample_y=build_dataset(6, 10, corpus,tokenizer)
# print(sample_x)
# print(sample_y)

# 设置训练模型
class LanguageModel(nn.Module):
    def __init__(self,bert_path,class_num):
        super(LanguageModel,self).__init__()
        self.bert=BertModel.from_pretrained(bert_path,return_dict=False)
        hidden_size = self.bert.config.hidden_size
        self.classify=nn.Linear(hidden_size,class_num)
        # self.dropout = nn.Dropout(0.1)
        self.loss=nn.functional.cross_entropy

    def forward(self,x,y=None):
        x,x_=self.bert(x)
        y_p=self.classify(x)
        if y is not None:
            return self.loss(y_p.view(-1, y_p.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_p, dim=-1)


# 采样策略
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

#文本生成测试代码
def generate_sentence(openings,model,window_size,tokenizer):
    vocab = tokenizer.vocab
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()    # 训练
    with torch.no_grad(): # 训练时不考虑梯度
        pred_char=""
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def train(corpus_path, save_weight=True):
    epoch_num = 15        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    bert_path=r"E:\Pycharm_learn\pythonProject1\wk6\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    vocab = tokenizer.vocab
    class_num=len(vocab)
    # 加载语料
    corpus = load_corpus(corpus_path)
    model=LanguageModel(bert_path,class_num) #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus,tokenizer) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, window_size,tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, window_size,tokenizer))
    if not save_weight:
        return
    else:
        model_path = r"E:\Pycharm_learn\pythonProject1\wk10\lstm语言模型生成文本\model"
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    corpus_path=r'E:\Pycharm_learn\pythonProject1\wk10\lstm语言模型生成文本\corpus.txt'
    train("corpus.txt", False)