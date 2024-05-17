# -*- coding: utf-8 -*-
import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForTokenClassification
from torch.optim import Adam, SGD

TorchModel = AutoModelForTokenClassification.from_pretrained(Config["bert_path"], num_labels=9)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

