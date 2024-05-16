from re import X
from config import config
from loader import load_data

from model import SiameseTripNetWork
import os
import jieba

import torch

class Predictor:
    def __init__(self,config,model,knwb_data) -> None:
        self.config = config
        self.model = model
        self.train_data = knwb_data
        if torch.cuda.is_available():
            self.model = model.cuda()
        self.model.eval()
        self.knwb_to_vector()
    
    def predict(self,x):
        input_id = self.encode_sentence(x)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            test_question_vector = self.model(input_id).unsqueeze(0)
            res = torch.mm(test_question_vector,self.knwb_vectors.T)
            print(f"res shape:{res.shape}")
            hit_index = int(torch.argmax(res.squeeze()))
            hit_index = self.question_idx_to_standard_idx[hit_index]
        return self.idx_to_standard_question[hit_index]

    def knwb_to_vector(self):
        self.question_idx_to_standard_idx = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.idx_to_standard_question = dict(  (y,x) for x,y in self.schema.items())

        # 把知识库中的所有问题都转成对应的标准问的ID
        for standard_quesiton_idx,question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_idx_to_standard_idx[len(self.question_ids)] = standard_quesiton_idx
                self.question_ids.append(question_id)
        with torch.no_grad():
            # question_id shape len(qustion), seq_len
            question_matrix = torch.stack(self.question_ids,dim=0)
            if torch.cuda.is_available():
                question_matrix = question_matrix.cuda()

            self.knwb_vectors = self.model(question_matrix) # len(qustion), embedding_dim
            print(f"knwb_vectors shape :{self.knwb_vectors.shape}")
            self.knwb_vectors = torch.nn.functional.softmax(self.knwb_vectors,dim=-1)
            return 
        
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


if __name__ == "__main__":
    knwb_data = load_data(config["train_data_path"],config)
    model = SiameseTripNetWork(config)
    model.load_state_dict(torch.load( os.path.join(config["model_path"],"epoch_10.pth")))

    pd = Predictor(config,model,knwb_data)
    res = pd.predict("手机话费是多少一个月")
    print(f"res:{res}")