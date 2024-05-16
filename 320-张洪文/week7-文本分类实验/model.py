import math

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from config import Config

"""
构建网络模型结构
"""
class TorchModel(nn.Module):
    def __init__(self, config):  # 传入config参数对象
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        self.config = config
        self.use_bert = False   # 是否使用bert
        self.embedding = nn.Embedding(config["vocab_size"]+1, hidden_size, padding_idx=0)
        # encoder 判断
        if config["model"] == "fast_text":
            self.encoder = lambda x: x
        elif config["model"] == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        elif config["model"] == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        elif config["model"] == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        elif config["model"] == "cnn":
            self.encoder = CNN(config)
        elif config["model"] == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif config["model"] == "rcnn":
            self.encoder = RCNN(config)
        elif config["model"] == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["bert_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size  # 768
        elif config["model"] == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif config["model"] == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif config["model"] == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, config["class_num"])
        self.activation = nn.Sigmoid()
        self.loss = nn.BCELoss()  # loss采用交叉熵损失，0的label会被忽略

    def forward(self, x, y=None):
        # 是否使用bert
        if self.use_bert:
            # output: [batch_size, max_len, hidden_size], [batch_size, hidden_size]
            x = self.encoder(x)
        else:
            # print(x, x.shape)
            x = self.embedding(x)  # output: [batch_size, max_len, embedding_dim]
            x = self.encoder(x)
        # 判断x是否为元组
        if isinstance(x, tuple):
            # 类RNN和bert的模型会同时返回隐单元向量，我们取第一个元素：序列结果
            x = x[0]

        # 池化降维
        if self.config["pooling_type"] == "max":
            self.pooling = nn.MaxPool1d(x.shape[1])  # x.shape[Text_Matching_Presentation] = max_len
        elif self.config["pooling_type"] == "avg":
            self.pooling = nn.AvgPool1d(x.shape[1])
        # output: [batch_size, sen_len, hidden_size] -> [batch_size, hidden_size]
        x = self.pooling(x.transpose(1, 2)).squeeze()

        y_classify = self.classify(x)  # 分类
        y_pred = self.activation(y_classify)  # 激活归一化
        if y is not None:
            return self.loss(y_pred.squeeze(), y)
        else:
            return y_pred


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        padding = math.ceil((config["kernel_size"] - 1) / 2)  # 向上取整
        self.cnn_layer = nn.Conv1d(hidden_size, hidden_size, config["kernel_size"], padding=padding)

    def forward(self, x):
        # x [batch_size, sequence_length, hidden_size]->[batch_size, hidden_size, sequence_length]->
        # output [batch_size, sequence_length, hidden_size]
        return self.cnn_layer(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn_layer = CNN(config)
        self.gate_layer = CNN(config)

    def forward(self, x):
        a = self.cnn_layer(x)
        b = self.gate_layer(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)  # 点乘

class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        self.rnn_layer = nn.RNN(config["hidden_size"], config["hidden_size"])
        self.cnn_layer = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn_layer(x)
        x = self.cnn_layer(x)
        return x

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        # 忽略不匹配的参数
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False, ignore_mismatched_sizes=True)
        self.lstm_rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]  # 最后一层的隐藏状态: [batch_size, sequence_length, hidden_size]
        x, _ = self.lstm_rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False, ignore_mismatched_sizes=True)
        config["hidden_size"] = self.bert.config.hidden_size  # 768
        self.cnn = CNN(config)

    def forward(self, x):
        # [0] 最后一层的隐藏状态: [batch_size, sequence_length, hidden_size]
        x = self.bert(x)[0]
        x = self.cnn(x)  # 过CNN
        return x

class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False, ignore_mismatched_sizes=True)
        self.bert.config.output_hidden_states = True  # 在前向传播时会返回一个包含所有隐藏层状态的列表

    def forward(self, x):
        # x batch_size, sen_len, hidden
        # 获取第三个输出（即所有隐藏层的输出）
        layer_states = self.bert(x)[2]
        # 将倒数第二层和最后一层的隐藏状态相加
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states

# 优化器选择
def choose_optimizer(model, config):
    if config["optimizer"] == 'adm':
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError('optimizer not supported')
    return optimizer

