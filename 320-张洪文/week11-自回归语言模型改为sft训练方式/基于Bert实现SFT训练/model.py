import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from transformers import BertModel

"""
模型构建
"""
class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config["vocab_size"]

        self.encoder = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, vocab_size)  # 得到每一个字符对应的概率
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)  # -1的标签不计算损失

    def forward(self, x, mask=None, y=None):
        if y is not None and mask is not None:
            x_encoder, _ = self.encoder(x, attention_mask=mask)  # batch_size, seq_len, hidden_size(768)
            y_pred = self.classify(x_encoder)  # batch_size, seq_len, vocab_size
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x_encoder, _ = self.encoder(x)
            y_pred = self.classify(x_encoder)  # batch_size, seq_len, vocab_size
            return torch.softmax(y_pred, dim=-1)

# 选择优化器
def choose_optimizer(model, config):
    if config["optimizer"] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'adam':
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError('Unsupported optimizer: {}'.format(config["optimizer"]))
    return optimizer
