import torch
from torch import nn
from transformers import BertTokenizer, BertModel

from config import config


class LLMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(config['bert_path'],return_dict=False)
        # 丢弃率
        self.dropout = nn.Dropout(config["dropout_rate"])
        # 分类层
        self.classify = nn.Linear(768, vocab_size)
        # 损失函数
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['learning_rate'])

        # 序列化器
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])

    def forward(self, input_ids, target=None, attention_mask=None):
        if target is not None:
            # 进行模型训练，需要传入mask
            # outputs shape : batch_size * seq_len * 768
            outputs, _ = self.bert(input_ids, attention_mask=attention_mask)
            outputs = self.dropout(outputs)
            outputs = self.classify(outputs)
            loss = self.loss(outputs.view(-1,outputs.shape[-1]), target.view(-1))
            return loss
        else:
            # 进行模型预测，不需要传入mask
            # outputs shape : batch_size * seq_len * 768
            outputs, _ = self.bert(input_ids)
            # outputs shape : batch_size * seq_len * vocab_size
            outputs = self.classify(outputs)

            # 返回最后一个预测值对应的词元
            return outputs[0][-1].argmax(dim=-1)