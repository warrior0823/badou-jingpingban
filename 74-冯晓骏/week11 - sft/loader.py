import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import random
import json
import logging
logger = logging.getLogger(__name__)
hl = logging.FileHandler('test.txt','w')
logger.setLevel(logging.DEBUG)
logger.addHandler(hl)
class DataGenerator():
    def __init__(self,config,path):
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.max_len = config['max_len']
        self.path = path
        self.data_len = config['batch_size'] * 100
        self.load_corpus()
        self.extend_attn_mask = generate_extend_attn_mask(self.max_len)

        self.operate_data()

    #改造loader，变为文本摘要的形式 content-->title
    def operate_data(self):
        self.data = []
        for i in range(self.data_len):
            # window_size = int(self.max_len * 0.9)
            index = random.randint(0,len(self.corpus)-1)
            src_sentence = self.corpus[index]['content']
            trg_sentence = self.corpus[index]['title']

            # print(index,'\t',src_sentence,'\t',trg_sentence)

            src_result = self.tokenizer.encode_plus(src_sentence,trg_sentence,max_length=self.max_len,padding='max_length',truncation=True,return_tensors='pt',
                                                    return_attention_mask=True,
                                                    add_special_tokens=True,
                                                    )
            # print(self.tokenizer.decode(src_result['input_ids'][0]))
            # print(src_result['attention_mask'].shape)
            extend_attn_mask = generate_extend_attn_mask_for_2_seq(len(src_sentence)+2,self.max_len-len(src_sentence)-2)
            # print(extend_attn_mask.shape)
            extend_attn_mask = extend_attn_mask.bool() & src_result['attention_mask'].bool()
            extend_attn_mask = extend_attn_mask.int()


            input_id = src_result['input_ids']
            trg_id = self.tokenizer.encode(trg_sentence,max_length=self.max_len,padding='max_length',truncation=True,return_tensors='pt',add_special_tokens=False)
            # print(trg_id)
            # print(self.tokenizer.decode(trg_id[0]))
            # input()
            #
            self.data.append([input_id.squeeze(),trg_id.squeeze(),extend_attn_mask])






    def load_corpus(self):
        print('loading corpus...')
        with open(self.path,'r',encoding='utf-8') as f:
            self.corpus = json.load(f)

        print('finish load corpus.')


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data_len
def generate_extend_attn_mask(sen_len):
    extend_attn_mask = (1-torch.triu(torch.ones(sen_len,sen_len),diagonal=1)).unsqueeze(0).bool()
    return extend_attn_mask
def generate_extend_attn_mask_for_2_seq(sen_len_1,sen_len_2):
    right_down = 1-torch.triu(torch.ones(sen_len_2,sen_len_2),diagonal=1)
    right_up = torch.zeros(sen_len_1,sen_len_2)
    left = torch.ones(sen_len_1+sen_len_2,sen_len_1)
    extend_attn_mask = torch.cat([left,torch.cat([right_up,right_down],dim=0)],dim=-1)
    return extend_attn_mask

def load_data(config,path,shuffle=False):
    dg = DataGenerator(config,path)
    data_loader = DataLoader(dg,batch_size=config['batch_size'],shuffle=shuffle)
    return data_loader

if __name__ == '__main__':
    from config import Config
    dg = DataGenerator(Config,'./sample_data.json')
