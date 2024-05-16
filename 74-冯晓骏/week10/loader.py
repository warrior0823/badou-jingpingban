import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import random


class DataGenerator():
    def __init__(self,config,path):
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.max_len = config['max_len']
        self.path = path
        self.data_len = config['batch_size'] * 100
        self.load_corpus()
        self.extend_attn_mask = generate_extend_attn_mask(self.max_len)

        self.operate_data()


    def operate_data(self):
        self.data = []
        for i in range(self.data_len):
            window_size = int(self.max_len * 0.9)
            start = random.randint(1,len(self.corpus))

            src_sentence = self.corpus[start:start+window_size]
            trg_sentence = self.corpus[start+1:start+window_size+1]

            src_result = self.tokenizer.encode_plus(src_sentence,max_length=self.max_len,padding='max_length',truncation=True,return_tensors='pt',
                                                    return_attention_mask=True,
                                                    return_token_type_ids=False,
                                                    add_special_tokens=False,
                                                    )

            input_id = src_result['input_ids']
            attn_mask = src_result['attention_mask']

            if attn_mask != None:
                attn_mask = (attn_mask.unsqueeze(0).bool() & self.extend_attn_mask).int()

            trg_id = self.tokenizer.encode(trg_sentence,max_length=self.max_len,padding='max_length',truncation=True,return_tensors='pt',add_special_tokens=False)

            # print('input_id',input_id)
            # print('attn:',attn_mask.shape)
            # print('trg_id:',trg_id)
            #
            # print('input decode:',self.tokenizer.decode(input_id.squeeze()))
            # print('trg_id decode:',self.tokenizer.decode(trg_id.squeeze()))
            # input()

            self.data.append([input_id.squeeze(),trg_id.squeeze(),attn_mask.squeeze()])

        # print('src sen:',self.tokenizer.decode(src_result['input_ids']))
        # print('trg sen:',self.tokenizer.decode(trg_result['input_ids']))





    def load_corpus(self):
        print('loading corpus...')
        self.corpus = ''
        with open(self.path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                self.corpus += line.strip()
        print('finish load corpus.')

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data_len
def generate_extend_attn_mask(sen_len):
    extend_attn_mask = (1-torch.triu(torch.ones(sen_len,sen_len),diagonal=1)).unsqueeze(0).bool()
    return extend_attn_mask

def load_data(config,path,shuffle=False):
    dg = DataGenerator(config,path)
    data_loader = DataLoader(dg,batch_size=config['batch_size'],shuffle=shuffle)
    return data_loader

if __name__ == '__main__':
    from config import Config
    dg = DataGenerator(Config,'./corpus1.txt')
