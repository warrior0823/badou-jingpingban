from collections import defaultdict
from config import config
import json
import torch
import jieba
import random
from torch.utils.data import DataLoader

"""
    从训练数据中读成对的数据，并打上标签。
    需要把字符串转成数值类型
    读取字表，进行转换
"""

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path,"r",encoding="utf-8") as f:
        #0留给padding位置，所以从1开始
        for index,line in enumerate(f):
            token = line.strip()
            token_dict[token] = index+1
    return token_dict

def load_schema(schema_path):
   with open(schema_path,"r",encoding="utf-8") as f: 
       return json.loads(f.read())

class DataGenerator:
    def __init__(self,data_path,config) -> None:
        self.path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        # 由于采取随机采样，需要设置一个采样数量
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        self.load()
    
    def load(self):
        self.data= []
        self.knwb = defaultdict(list)
        with open(self.path,"r",encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line,dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label  = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    self.data_type = "test"
                    assert isinstance(line,list)
                    question,label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id,label_index])
        return 
    
    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else :
            assert self.data_type == "test",self.data_type
            return len(self.data)
    
    def __getitem__(self,index):
        if self.data_type =="train":
            return self.random_train_sample()
        else:
            return self.data[index]
    
    def random_train_sample(self):
        # 采样组成a,p,n对
        standard_question_index = list(self.knwb.keys())
        p,n = random.sample(standard_question_index,2)
        # if len(self.knwb[p]) < 2:
        #     return self.random_train_sample()
        if len(self.knwb[p])==1:
            # 只有一个的也不能不选择
            a=p=self.knwb[p][0]
            n=  random.choice(self.knwb[n])
        else:
            a,p = random.sample(self.knwb[p],2)
            n=  random.choice(self.knwb[n])
        return [a,p,n]
    

    def encode_sentence(self,text):
        # 句子从词表中转成索引  注意vocab["unk"] 与padding不同
        input_id = []
        # 如果需要分词，词表的路径也要是不同的，否则就都是UNK了
        if self.config["need_cut"] == "1":
            for word in jieba.lcut(text):
                input_id.append(self.vocab.get(word,self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))

        return self.padding(input_id)
    

    def padding(self,input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
    

def load_data(data_path,config,shuffle=True):
    dg = DataGenerator(data_path,config)
    dl = DataLoader(dg,batch_size = config["batch_size"],shuffle=shuffle)
    return dl

if __name__ == "__main__":
    data_path = config["train_data_path"]
    dg = DataGenerator(data_path,config)
    print(dg[1])