import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from transformers import BertModel

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self,config):
        super(TorchModel,self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1  # padding+1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]

        self.use_bert = False
        self.embedding=nn.Embedding(vocab_size,hidden_size,padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder =nn.LSTM(hidden_size,hidden_size,num_layers=num_layers)
            # 第1个参数与词表维度对应，第2个参数隐含层，第三个参数是层数
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    def forward(self,x,target=None):
        if self.use_bert:
            # bert 自带embedding
            x=self.encoder(x)
        else:
            x=self.embedding(x)
            x=self.encoder(x)
        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        predict = self.classify(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict




    # 卷积
class CNN(nn.Module):
    def __init__(self,config):
        super(CNN,self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)
    def forward(self, x):  #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class BertLSTM(nn.Module):
    def __init__(self,config):
        super(BertLSTM,self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config_hw import Config
    # Config["class_num"] = 3
    # Config["vocab_size"] = 20
    # Config["max_length"] = 5
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    # print(pooler_output)
    # print(x[2], type(x[2]), len(x[2]))

