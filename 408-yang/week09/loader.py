from config import config
import json
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer


"""
    使用bert的tokenizer进行向量化
    
"""

def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer

def load_schema(schema_path):
   with open(schema_path,"r",encoding="utf-8") as f: 
       return json.loads(f.read())

class DataGenerator:
    def __init__(self,data_path,config) -> None:
        self.path = data_path
        self.config = config
        self.max_length = config["max_length"]
        self.tokenizer = load_vocab(config["vocab_path"])
        self.schema = load_schema(config["schema_path"])
        self.sentences = []
        self.load()
    
    def load(self):
        self.data= []
        with open(self.path,"r",encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = [8] #cls_token
                for line in segment.split("\n"):
                    if line.strip() =="":
                        continue
                    char,label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels,-1)
                self.data.append([torch.LongTensor(input_ids),torch.LongTensor(labels)])
        return 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]

    def encode_sentence(self,text,padding=True):
        self.tokenizer.truncation = True
        self.tokenizer.padding = 'max_length'
        old_level = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        encode_input =  self.tokenizer.encode_plus(text,
                              truncation='longest_first',
                              max_length=self.max_length,
                              padding='max_length',
                              return_attention_mask=True,  # 返回 attention mask
                            #   return_tensors='pt'          # 返回 PyTorch 张量
                              )        # 返回 PyTorch 张量)
        transformers.logging.set_verbosity(old_level)
        return encode_input["input_ids"]

    

    def padding(self,input_id,pad_token = 0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id
    

def load_data(data_path,config,shuffle=True):
    dg = DataGenerator(data_path,config)
    dl = DataLoader(dg,batch_size = config["batch_size"],shuffle=shuffle)
    return dl

if __name__ == "__main__":
    data_path = config["train_data_path"]
    dg = DataGenerator(data_path,config)
    print(dg[1])