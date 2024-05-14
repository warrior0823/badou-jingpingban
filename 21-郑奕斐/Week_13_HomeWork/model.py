# -*- coding: utf-8 -*-
from torchcrf import CRF
from config import Config
from transformers import AutoModelForTokenClassification
from torch.optim import Adam, SGD
from icecream import ic

"""
建立网络模型结构
"""

diyTorchModel = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"])

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
