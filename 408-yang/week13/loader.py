from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer

import json

def load_vocab(vocab_path):
    """
    加载词典,预留位置词位置 
    """
    vocab = {}
    with open(vocab_path,"r",encoding="utf-8") as f:
        for idx,item in enumerate(f):
            token = item.strip()
            vocab[token] = idx+1 # 0预留给padding位置
    return vocab
        


class DataGenerator:

    def __init__(self,data_path,config) -> None:
        self.config = config
        self.path = data_path

        self.schema = self.load_schema(config["schema_path"])
        # 加载的是自己的词表
        self.vocab = load_vocab(config["vocab_path"])
        if self.config["model_type"] =="bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
            self.vocab = self.tokenizer.get_vocab()
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.data = []
        self.load()

    def load_schema(self,schema_path):
        with open(schema_path,"r",encoding="utf-8") as f:
            return json.load(f)

    def load(self):
        """
           读取数据
        """
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
        for segment in segments:
            sentenece = []
            labels = []
            for line in segment.split("\n"):
                if line.strip() == "":
                    continue
                char, label = line.split()
                sentenece.append(char)
                labels.append(self.schema[label])
            self.sentences.append("".join(sentenece))
            if self.config["model_type"] =="bert":
                input_ids = self.encode_sentence_bert(sentenece)
                labels = [8]+labels[:self.config["max_length"]-2]+[8]
                labels = self.padding(labels,-1)
            else:
                input_ids = self.encode_sentence(sentenece)
                labels = self.padding(labels, -1)
            self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return
    
    def encode_sentence_bert(self,sentence):
        """
        当sentence为字符列表，会对单个元素分别编码
        """
        
        input_ids = self.tokenizer.encode(sentence, 
                                          max_length=self.config["max_length"],
                                          padding="max_length", truncation=True,
                                          # add_special_tokens=False  # 不添加特殊标记符
                                          )

        return input_ids

    def encode_sentence(self,text):
        input_id = []
        for char in text :
            input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id
    
    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
    




def load_data(data_path,config,shuffle=True):
    dg = DataGenerator(data_path,config)
    dl = DataLoader(dg,batch_size = config["batch_size"],shuffle=shuffle)
    return dl

if __name__ =="__main__":
    from config import Config

    dl = load_data(Config["train_data_path"], Config, shuffle=False)
    # dl = load_data(Config["valid_path"], Config, shuffle=False)
    for batch in dl:
        print(batch)
    #     print(batch[0], batch[Text_Matching_Presentation])
