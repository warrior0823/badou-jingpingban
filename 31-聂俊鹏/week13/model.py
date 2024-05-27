# -*- coding: utf-8 -*-
from config import Config
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
# from torchcrf import CRF
from transformers import BertTokenizer, BertModel, AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.optim import Adam, SGD
"""
建立网络模型结构
"""
from peft import get_peft_model, LoraConfig, TaskType

bert = AutoModelForTokenClassification.from_pretrained(r"".join(Config["bert_path"]), return_dict=False, num_labels=9)
peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
pert_model = get_peft_model(bert, peft_config)

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.encoder = pert_model
        self.config = self.encoder.config
        self.classify_bert = nn.Linear(768, class_num)  #应该为bert隐藏层维数
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        # self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
#        x = self.embedding(x)  #input shape:(batch_size, sen_len)
#        x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
#        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
        x = self.encoder(x)  #input:(batch_size, sen_len, hidden_size)
        predict = x[0]  #output:(batch_szie,sen_len,numtags)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))   #这需要转为batch_size * sen_len, num_tags
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


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