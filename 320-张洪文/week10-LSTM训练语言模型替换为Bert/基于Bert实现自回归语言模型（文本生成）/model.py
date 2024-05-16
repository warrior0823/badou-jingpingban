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
        embedding_size = config["embedding_size"]
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        self.config = config
        self.use_bert = False

        if config["model"] == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["bert_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif config["model"] == "gru":
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            self.encoder = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)
            hidden_size = hidden_size * 2

        self.classify = nn.Linear(hidden_size, vocab_size)  # 得到每一个字符对应的概率
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        if self.config["model"] == "bert":
            if y is not None:
                # mask应该是和x同维度的吗？
                mask = torch.tril(torch.ones((x.shape[0], x.shape[1])))
                mask = mask.to(x.device)
                x_encoder, _ = self.encoder(x, attention_mask=mask)  # batch_size, seq_len, hidden_size(768)
            else:
                x_encoder, _ = self.encoder(x)  # batch_size, seq_len, hidden_size(768)
        else:
            x_embedded = self.embedding(x)
            x_encoder, _ = self.encoder(x_embedded)
            
        pred = self.classify(x_encoder)  # batch_size, seq_len, vocab_size
        
        if y is not None:
            pred_view = pred.view(-1, pred.shape[-1])
            y_view = y.view(-1)
            return self.loss(pred_view, y_view)
        else:
            return torch.softmax(pred, dim=-1)

# 选择优化器
def choose_optimizer(model, config):
    if config["optimizer"] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'adam':
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError('Unsupported optimizer: {}'.format(config["optimizer"]))
    return optimizer
