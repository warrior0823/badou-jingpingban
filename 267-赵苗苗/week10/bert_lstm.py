import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import random
import os

"""
基于pytorch的LSTM语言模型
"""
class LanguageModel(nn.Module):
    def __init__(self,hidden_size,vocab_size,pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert=BertModel.from_pretrained(pretrain_model_path,return_dict=False)
        self.classify=nn.Linear(hidden_size,vocab_size)  #全连接层
        self.loss=nn.functional.cross_entropy  #交叉熵损失函数
    
    #前向传播，输入真实标签，返回loss，无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 训练模式: 构建一个下三角的掩码矩阵，以防止上下文标记之间的交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)  # 使用 BERT 对输入进行编码，并应用注意力掩码
            y_pred = self.classify(x)   # 使用分类层预测输出概率，输出形状: (batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))  # 计算预测概率（重塑后）与目标之间的损失
        else:
            # 预测模式: 可以不使用掩码
            x, _ = self.bert(x)  # 使用 BERT 对输入进行编码
            y_pred = self.classify(x)   # 使用分类层预测输出概率，输出形状: (batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)  # 返回预测的 softmax 概率



#加载语料
def load_corpus(corpus_path):
    corpus=""
    with open(corpus_path,encoding="gbk") as f:
        for line in f:
            corpus+=line.strip()
    return corpus
#建立模型
def build_model(vocab,char_dim,pretrain_model_path):
    model=LanguageModel(768,21128,pretrain_model_path)
    return model
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []  # 用于存储输入样本的列表
    dataset_y = []  # 用于存储输出样本的列表
    
    # 循环生成指定数量的样本
    for i in range(sample_length):
        # 调用 build_sample 函数生成单个样本的输入和输出
        x, y = build_sample(tokenizer, window_size, corpus)
        # 将生成的输入样本添加到 dataset_x 列表中
        dataset_x.append(x)
        # 将生成的输出样本添加到 dataset_y 列表中
        dataset_y.append(y)
    # 将列表转换为 PyTorch 的 LongTensor 格式并返回
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 随机生成样本
# 从文本中随机选择一个窗口，并将前 N 个字符作为输入，最后一个字符作为输出
def build_sample(tokenizer, window_size, corpus):
    # 随机选择窗口的起始位置
    start = random.randint(0, len(corpus) - 1 - window_size)
    # 计算窗口的结束位置
    end = start + window_size
    # 获取窗口内的文本片段作为输入
    window = corpus[start:end]
    # 获取窗口内的目标文本片段作为输出（移动一个位置）
    target = corpus[start + 1:end + 1]
    # 使用 tokenizer 对输入和输出进行编码，并进行长度控制和填充
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    # 返回编码后的输入和输出
    return x, y

#生成句子
def generate_sentence(openings, model, tokenizer, window_size):
    # 将模型设置为评估模式，不进行梯度计算
    model.eval()
    # 禁止梯度计算
    with torch.no_grad():
        # 初始化预测字符为空字符串
        pred_char = ""
        # 生成的文本不包含换行符并且长度不超过30字时继续生成
        while pred_char != "\n" and len(openings) <= 30:
            # 将当前生成的部分文本添加到开头
            openings += pred_char
            # 对当前文本进行编码
            x = tokenizer.encode(openings, add_special_tokens=False)
            # 将编码后的文本转换为 PyTorch Tensor
            x = torch.LongTensor([x])
            # 如果可用 GPU，则将数据移至 GPU
            if torch.cuda.is_available():
                x = x.cuda()
            # 使用模型生成下一个字符的概率分布
            y = model(x)[0][-1]
            # 根据采样策略选择下一个字符的索引
            index = sampling_strategy(y)
            # 将字符索引转换为字符串形式
            pred_char = ''.join(tokenizer.decode(index))
    # 返回生成的文本
    return openings
#采样策略函数
def sampling_strategy(prob_distribution):
    #以90%的概率使用贪婪策略，以10%的概率使用随机采样策略
    if random.random()>0.1:
        strategy="greedy"
    else:
        strategy="sampling"
    #如果使用贪婪策略，则直接返回概率最高的索引
    if strategy=="greedy":
        return int(torch.argmax(prob_distribution))
    #如果使用随机采样策略，则从概率分布中随机选择一个索引
    elif strategy=="sampling":
        prob_distribution=prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))),p=prob_distribution)

def train(corpus_path,save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率

    #加载预训练模型
    pretrain_model_path=r"D:\AI\nlp\八斗课程-精品班\第六周\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    #加载语料
    corpus=load_corpus(corpus_path)
    #建立模型
    model=build_model(vocab_size,char_dim,pretrain_model_path)
    if torch.cuda.is_available():
        model=model.cuda()
    #定义优化器
    optim= torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("文本词表模型加载完毕，开始训练...")
    #开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch in range(int(train_sample/batch_size)):
            #构建一组训练样本
            x,y=build_dataset(batch_size,tokenizer,window_size,corpus)
            if torch.cuda.is_available():
                x,y=x.cuda(),y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=======\n第%d轮平均loss:%f"%(epoch+1,np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出",model,tokenizer,window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸",model,tokenizer,window_size))
    if not save_weight:
        return
    else:
        #保存模型
        base_name=os.path.basename(corpus_path).replace("txt","pth")
        model_path=os.path.join("model",base_name)
        torch.save(model.state_dict(),model_path)
        print("模型已保存至%s"%model_path)
        return



if __name__ == '__main__':
    train("corpus.txt",False)