import json
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from config import Config

"""
数据加载
"""
#用于加载和处理数据，
class  DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.schema = load_schema(config["schema_path"]) #加载模式信息
        self.vocab = load_vocab(config["vocab_path"]) #加载词汇表
        self.config["vocab_size"]=len(self.vocab)  #将词汇表大小赋值
        self.train_data_size=config["epoch_data_size"]  #设置采样数量
        self.data_type=None  #标识加载的的是训练集还是测试集
        self.load()
    #加载数据
    def load(self):
        self.data=[]
        self.knwb=defaultdict(list)
        with open(self.path,encoding="utf8") as f:
            for line in f:
                line=json.loads(line)
                #加载训练集
                if isinstance(line,dict):
                    self.data_type="train"
                    questions=line["questions"]
                    label=line["target"]
                    for question in questions:
                        input_id=self.encode_sentence(question)
                        input_id=torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                #加载测试集
                else:
                    self.data_type="test"
                    assert isinstance(line,list)
                    question,label=line
                    input_id=self.encode_sentence(question)
                    input_id=torch.LongTensor(input_id)
                    label_index=torch.LongTensor([self.schema[label]])
                    self.data.append([input_id,label_index])
        return
    #将文本编码成模型可以接受的输入格式
    def encode_sentence(self,text):
        input_id=[]
        if self.config["vocab_path"]=="words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word,self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))
        input_id=self.padding(input_id)
        return input_id
    #补齐或者截断输入的序列，使其可以在一个batch内运算
    def padding(self,input_id):
        input_id=input_id[:self.config["max_length"]]
        input_id+=[0]*(self.config["max_length"]-len(input_id))
        return input_id
    #返回数据集的长度，训练集返回预设的训练数据大小，测试集返回数据集的长度
    def __len__(self):
        if self.data_type=="train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type=="test",self.data_type
            return len(self.data)
    #获取指定索引的数据，训练集生成一个随机训练样本，测试集返回对应索引数据
    def __getitem__(self,index):
        if self.data_type=="train":
            return self.random_train_sample()
        else:
            return self.data[index]
    #随机生成3元组样本，2正1负
    def random_train_sample(self):
        standard_question_index=list(self.knwb.keys())
        #从数据中随机选择两个意图，分别选择其中的两个问题作为正样本，再从另一个意图中选择一个问题作为负样本
        p,n=random.sample(standard_question_index,2)
        if len(self.knwb[p])==1:
            s1=s2=self.knwb[p][0]
        else:
            s1,s2=random.sample(self.knwb[p],2)
        #随机一个负样本
        s3=random.choice(self.knwb[n])
        return [s1,s2,s3]
#加载字表或词表
def load_vocab(vocab_path):
    token_dict={}
    with open(vocab_path,encoding="utf8") as f:
        for index,line in enumerate(f):
            token=line.strip()
            token_dict[token]=index+1
    return token_dict
#加载模式信息
def load_schema(schema_path):
    with open(schema_path,encoding="utf8") as f:
        return json.loads(f.read())
#使用torch自带的DataLoader类封装数据
def load_data(data_path,config,shuffle=True):
    dg=DataGenerator(data_path,config)
    d1=DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return d1

if __name__=="__main__":
    dg=DataGenerator("../data/train.json",Config)
    print(dg[1])



