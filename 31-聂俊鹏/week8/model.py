
import torch
import torch.nn as nn
from torch.optim import Adam , SGD
from config import objConfig
'''
    建立网络模型结构
'''

class TorchModel(nn.Module):
    def __init__(self , mConfig):
        super().__init__()
        hidden_size = mConfig["hidden_size"]
        vocab_size = mConfig["vocab_size"] + 1
        max_len = mConfig["max_len"]
        class_num = mConfig["class_num"]
        self.embedding = nn.Embedding(vocab_size , hidden_size , padding_idx= 0)
        self.layer = nn.Linear(hidden_size , hidden_size )
        self.classify = nn.Linear(hidden_size , class_num)
        self.pool = nn.AvgPool1d(max_len)
        self.poolmax = nn.MaxPool1d(max_len)
        self.activation = torch.relu
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self ,x , target = None):
        x = self.embedding(x)
        # print("经过embedding" , x.shape)
        x = self.layer(x)
        # print("经过layer" , x.shape)
        x = self.pool(x.transpose(1,2)).squeeze()
        # print("经过pool" , x.shape)
        predict = self.classify(x)
        if target is not None:
            return self.loss(predict , target.squeeze())
        else:
            return predict

#优化器
def choose_optimizer(config , model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters() , lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters() , lr=learning_rate)


if __name__ == '__main__':
    model = TorchModel(objConfig)
