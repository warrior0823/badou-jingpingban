""""
序列标注任务
训练数据： 文本序列 及相同长度的文本标签
模型结构： bert+lstm 或者embedding+lstm 可以加或者不加crf
预测数据： 用户query
输出结果： 用户query ，打上标签，自己转成要解析的数据
"""

import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torchcrf import CRF

from transformers import BertModel

class TorchModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        # max_length = config["max_length"]
        class_num = config["class_num"]
        self.bert_encoder = BertModel.from_pretrained(config["pretrain_model_path"],return_dict = False)
        hidden_size = self.bert_encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size,class_num)
        self.crf_layer = CRF(class_num,batch_first=True)
        self.use_crf = config["use_crf"]
        # self.dropout = nn.Dropout(0.1)
        # loss采用交叉熵
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,x,target=None):
        # x:sequence_output  _:pooler_output
        x,_ = self.bert_encoder(x)
        predict = self.classify(x)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict,target,mask,reduction="mean")
            else:
                return self.loss(predict.view(-1,predict.shape[-1]),target.view(-1))        
        else :
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else :
                return predict
    
    
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
    model = TorchModel(config)
    s1 = torch.LongTensor([[1,2,3,0],[2,2,0,0]])
    print(f"s1 shape:{s1.shape}")
    s2 = torch.LongTensor([[1,2,3,4],[3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1,s2,l)
    print(y)

                




 
        

    