# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
from transformers import AutoModelForTokenClassification
from config import Config
import json
from transformers import BertForSequenceClassification

TorchModel = AutoModelForTokenClassification.from_pretrained(Config["bert_path"],num_labels=9, return_dict=False)
# print(TorchModel)
# class TorchModel(nn.Module):
#     def __init__(self, config):
#         super(TorchModel, self).__init__()
#         class_num = config["class_num"]
#         self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
#         self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
#         self.crf_layer = CRF(class_num, batch_first=True)
#         self.use_crf = config["use_crf"]
#         self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失
# #
# #     #当输入真实标签，返回loss值；无真实标签，返回预测值
#     def forward(self, x,target=None):
#         x, _ = self.bert(x)
#         predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
# #
#         if target is not None:
#             if self.use_crf:
#                 mask = target.gt(-1)
#                 return - self.crf_layer(predict, target, mask, reduction="mean")
#             else:
#                 return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
#         else:
#             if self.use_crf:
#                 return self.crf_layer.decode(predict)
#             else:
#                 return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


# def load_schema(path):
#     with open(path, encoding="utf8") as f:
#         return json.load(f)

# schema2id=load_schema(r'E:\Pycharm_learn\pythonProject1\wk13\peft_ner\ner_data\schema.json')
#
# id2schema={}
# for key,val in schema2id.items():
#     id2schema[val]=key

# TorchModel = AutoModelForTokenClassification.from_pretrained(Config["bert_path"],num_labels=9,id2label=id2schema,label2id=schema2id)


# if __name__ == "__main__":
#     from config import Config
#     model = TorchModel(Config)