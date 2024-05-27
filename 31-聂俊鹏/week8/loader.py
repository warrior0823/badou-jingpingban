import json
import torch
import jieba
from config import objConfig
from torch.utils.data import Dataset, DataLoader
'''
    数据加载
'''

class DataGenerator:
    def __init__(self ,path, objConfig):
        self.mConfig = objConfig
        self.mPath = path
        self.vocab = load_vocab(self.mConfig["vocab_path"])
        self.mConfig["vocab_size"] = len(self.vocab)
        self.schme = load_schme(self.mConfig["schema_path"])
        self.mConfig["class_num"] = len(self.schme)
        self.load()

    def load(self):     #把train语料转成张量。
        self.data = []
        with open(self.mPath , encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line , dict):
                    questions = line["questions"]
                    label = line["target"]
                    label_index = torch.LongTensor([self.schme[label]])
                    for question in questions :
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.data.append([input_id , label_index])
                else:
                    assert isinstance(line, list)
                    question , label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schme[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self , text):
        input_id = []
        if self.mConfig["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab .get(char , self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self , input_id):
        input_id = input_id[:self.mConfig["max_len"]]                   #只有在文本长度大于max_len时才会切割
        input_id += [0] * (self.mConfig["max_len"] - len(input_id))     #如果文本长度小于max_len，则用0填充
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self , index):
        return self.data[index]

#加载schme 文件
def load_schme(path):
    with open(path , encoding="utf-8") as data:
        return json.loads(data.read())

#加载字表或词表
def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for  i, line in enumerate(f):
            vocab[line.strip()] = i + 1
    return vocab

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == '__main__':
     model = DataGenerator( objConfig["train_data_path"] , objConfig)
     model.forward()
