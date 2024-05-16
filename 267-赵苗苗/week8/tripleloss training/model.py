import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import Config
"""
建立网络模型结构
"""
#SentenceEncoder类用于编码输入的句子，生成句子的表示向量，用于计算余弦距离或者三元组损失
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]+1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size,padding_idx=0) #创建一个词嵌入层，用于将词的索引映射为向量表示
        self.layer=nn.Linear(hidden_size,hidden_size)  #线性层引入非线性变换，用于学习
        self.dropout=nn.Dropout(0.5)
    
    #对输入的文本数据进行处理并生成文本的表示向量
    def forward(self,x):
        sentence_length=torch.sum(x.gt(0),dim=-1) #计算输入张量x中每个句子的实际长度
        x=self.embedding(x)
        x=self.layer(x)
        x=nn.functional.max_pool1d(x.transpose(1,2),x.shape[1]).squeeze()
        return x

#用于学习文本之间的相似度或差异
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder=SentenceEncoder(config)
        self.loss=nn.CosineEmbeddingLoss()

    #计算余弦距离，1-cos(a,b)
    #cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1.余弦距离的取值范围为[0,2]
    def cosine_distance(self,tensor1,tensor2):
        #对输入的向量进行归一化处理
        tensor1=torch.nn.functional.normalize(tensor1,dim=-1)
        tensor2=torch.nn.functional.normalize(tensor2,dim=-1)
        cosine=torch.sum(torch.mul(tensor1,tensor2),axis=-1)
        return 1-cosine
    
    #计算三元组损失函数
    def cosine_triplet_loss(self,a,p,n,margin=None):
        ap=self.cosine_distance(a,p)
        an=self.cosine_distance(a,n)
        if margin is None:
            diff=ap-an+0.1
        else:
            diff=ap-an+margin.squeeze()
        return torch.mean(diff[diff.gt(0)])
    #sentence:(batch_size,max_length)
    def forward(self,sentence1,sentence2=None,sentence3=None):
        #同时传入3个句子，分别对句子编码，计算三元组损失
        if sentence2 is not None and sentence3 is not None:
            vector1=self.sentence_encoder(sentence1)
            vector2=self.sentence_encoder(sentence2)
            vector3=self.sentence_encoder(sentence3)
            return self.cosine_triplet_loss(vector1,vector2,vector3)
        else:
            #一个句子，仅进行编码
            return self.sentence_encoder(sentence1)
#用于根据配置参数选择优化器
def choose_optimizer(config,model):
    optimizer=config["optimizer"]
    learning_rate=config["learning_rate"]
    if optimizer=="adam":
        return Adam(model.parameters(),lr=learning_rate)
    elif optimizer=="sgd":
        return SGD(model.parameters(),lr=learning_rate)

if __name__=="__main__":
    Config["vocab_size"]=10
    Config["max_length"]=4
    model=SiameseNetwork(Config) #创建一个SiameseNetwork对象，传入修改后的参数
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)

        
