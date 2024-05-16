# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

class LLM_Module(nn.Module):
    def __init__(self, config):
        super(LLM_Module, self).__init__()
        self.config = config
        hidden_size = 768
        vocab_size = 21128
        
        self.embedding = BertModel.from_pretrained(self.config['pretrain_model_path'], return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(self, x, mask=None, y=None):
        if y is not None:
            x, _ = self.embedding(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.embedding(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)
            