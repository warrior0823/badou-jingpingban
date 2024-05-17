# -*- coding: utf-8 -*-

from torch.optim import Adam, SGD
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from config import Config

"""
建立网络模型结构
"""

TorchModel = AutoModelForSequenceClassification.from_pretrained(Config["bert_path"])


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)



