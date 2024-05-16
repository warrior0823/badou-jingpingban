import torch
from torch.utils.data import DataLoader,Dataset
import Config
from collections import defaultdict
from transformers import BertTokenizer
import random

class Data_loader():
    def __init__(self,CFG):
        self.path=CFG.data_path
        self.window_size=CFG.window_size
        self.tokenizer=BertTokenizer.from_pretrained(CFG.vocab_path)
        self.cropus=self.load_cropus()
        self.cfg=CFG
        self.small_cropus=load_small_dict()

    def build_sample(self,cropus):
        start=random.randint(0,len(cropus)-1-self.window_size)
        end=start+self.window_size

        sentences=cropus[start:end]
        #取后一个字
        pred_sentences=cropus[end:end+1]

        sentences_dict=self.tokenizer.encode_plus(sentences,padding='max_length',max_length=self.cfg.max_char,truncation=True)

        pred=self.small_cropus[pred_sentences]
        pred_dict=[pred]

        return sentences_dict,pred_dict

    def load_cropus(self):
        corpus = ""
        with open(self.path, encoding="gbk") as f:
            for line in f:
                corpus += line.strip()
        return corpus

    def build_dataset(self,sample_num):
        x_ids,x_type,x_mask=[],[],[]
        y_ids,y_type,y_mask=[],[],[]
        for i in range(sample_num):
            inputs_dict,pred=self.build_sample(self.cropus)
            x_ids.append(inputs_dict['input_ids'])
            x_type.append(inputs_dict['token_type_ids'])
            x_mask.append(inputs_dict['attention_mask'])

            y_ids.append(pred)


        return torch.LongTensor(x_ids),torch.LongTensor(x_type),torch.LongTensor(x_mask),torch.LongTensor(y_ids)
        #torch.LongTensor(y_type),torch.LongTensor(y_mask)


def load_small_dict():
    new_dict_path=r"E:\python\pre_train_model\bert_file\vocab.txt"
    nd=defaultdict(int)
    with open(new_dict_path,encoding='UTF-8') as f:
        for line in f:
            line=line.strip()
            nd[line]=len(nd)
    return nd

if __name__ == '__main__':
    nd=load_small_dict()
    print(len(nd))
