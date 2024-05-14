# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
from peft import get_peft_model, LoraConfig, TaskType, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
from transformers import BertForTokenClassification
from config import Config
TorchModel = BertForTokenClassification.from_pretrained(Config["bert_path"],return_dict=False,num_labels=Config["class_num"])



# """
# 建立网络模型结构
# """

# class TorchModel(nn.Module):
#     def __init__(self, config):
#         super(TorchModel, self).__init__()
#         hidden_size = config["hidden_size"]
#         vocab_size = config["vocab_size"] + 1
#         max_length = config["max_length"]
#         class_num = config["class_num"]
#         num_layers = config["num_layers"]
#         self.use_bert = config["use_bert"]
        
        
        
#         #大模型微调策略
#         tuning_tactics = config["tuning_tactics"]
#         if tuning_tactics == "lora_tuning":
#             peft_config = LoraConfig(
#                 task_type=TaskType.TOKEN_CLS,
#                 inference_mode=False,
#                 r=8,
#                 lora_alpha=32,
#                 lora_dropout=0.1,
#                 target_modules=["query", "key", "value"]
#             )
#         elif tuning_tactics == "p_tuning":
#             peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
#         elif tuning_tactics == "prompt_tuning":
#             peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
#         elif tuning_tactics == "prefix_tuning":
#             peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
        
#         # print(model.state_dict().keys())

#         if config["use_bert"] == True:
#             # BertModel.forward
#             # self.layer2 = BertModel.from_pretrained(config["bert_path"],return_dict=False)
#             # self.layer = get_peft_model(self.layer2, peft_config)
#             # self.classify = nn.Linear(self.layer.config.hidden_size, class_num)
#             from transformers import BertForTokenClassification
            
#             self.model = BertForTokenClassification.from_pretrained(config["bert_path"],return_dict=False,num_labels=class_num)
#             self.layer = get_peft_model(self.model, peft_config)
        
#         else:  
#             self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
#             self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
#             self.classify = nn.Linear(hidden_size * 2, class_num)
#             self.crf_layer = CRF(class_num, batch_first=True)
#         self.use_crf = config["use_crf"]
#         self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

#     #当输入真实标签，返回loss值；无真实标签，返回预测值
#     def forward(self, x, target=None):
#         # x = self.embedding(x)  #input shape:(batch_size, sen_len)
#         x= self.layer(x)
        
#         if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
#             x = x[0]
        
#         #x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
#         predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

#         if target is not None:
#             if self.use_crf:
#                 mask = target.gt(-1)
#                 return - self.crf_layer(predict, target, mask, reduction="mean")
#             else:
#                 #(number, class_num), (number)
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


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)