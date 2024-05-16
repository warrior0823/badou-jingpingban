""""
文本匹配问题，判断一个句子和那些问题比较相似。
训练数据： 相似问题和主问题训练
模型结构： 目标是训练出来用户query的向量化表示
预测数据： 用户query ，转为向量库中的向量进行匹配
输出结果：输出主问题
"""

import torch
import torch.nn as nn
from torch.optim import Adam,SGD

class SentenceEncoder(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size,hidden_size,padding_idx=0)
        # 只有bert不需要encoder,本身自带。其他模型都需要有embedding层
        # self.lstm = nn.LSTM(hidden_size,hidden_size,1,batch_first=True,bidirectional=True)
        self.layer = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.embedding(x)
        x = self.layer(x)
        print(f"x.transpose(1,2):{x.transpose(1,2).shape},x.shape[1]:{x.shape[1]}")
        # kernel_size 大小就是要x.句子长度，整体计算一个最大值
        x = nn.functional.max_pool1d(x.transpose(1,2),x.shape[1]).squeeze()
        return x 
    
        
class SiameseTripNetWork(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.sentence_encoder = SentenceEncoder(config)
    
    def forward(self,a,p=None,n=None,margin=None):
        if p is not None and n is not None:
            a = self.sentence_encoder(a)
            p = self.sentence_encoder(p)
            n = self.sentence_encoder(n)
            return self.cosine_triplet_loss(a,p,n,margin)
        else:
            return self.sentence_encoder(a)
    
        
    def cosine_distance(self,tensor1,tensor2):
        tensor1 = nn.functional.normalize(tensor1,dim=-1)
        tensor2 = nn.functional.normalize(tensor2,dim=-1)
        cosine = torch.sum(torch.mul(tensor1,tensor2),axis=-1)
        return 1-cosine
    
    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)]) #greater than
    
    
def choose_optimizer(config,model):
    optimezer = config["optimizer"]
    lr = config["learning_rate"]
    if optimezer == "adam":
        return Adam(model.parameters(),lr=lr)
    elif optimezer == 'sgd':
        return SGD(model.parameters(),lr = lr)


if __name__ == "__main__":
    from config import config       
    config["vocab_size"] = 10
    config["max_length"] = 4
    model = SiameseTripNetWork(config)
    s1 = torch.LongTensor([[1,2,3,0],[2,2,0,0]])
    print(f"s1 shape:{s1.shape}")
    s2 = torch.LongTensor([[1,2,3,4],[3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1,s2,l)
    print(y)

                




 
        

    