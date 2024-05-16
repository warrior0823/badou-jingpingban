import torch.nn as nn
from torch.optim import SGD, Adam
from torchcrf import CRF
from transformers import BertModel, BertConfig

class NERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_size = config["embedding_size"]
        hidden_size = config["hidden_size"]
        self.config = config
        self.use_bert = False
        if config["model"] == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["bert_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
            # self.linear1 = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
            # self.linear2 = nn.Linear(hidden_size*2, hidden_size)
        elif config["model"] == "lstm":
            self.embedding = nn.Embedding(config["vocab_size"]+1, embedding_size, padding_idx=0)
            self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
            hidden_size = hidden_size * 2
        # self.activation = nn.ReLU()

        self.classify = nn.Linear(hidden_size, config["class_num"])
        self.crf = CRF(config["class_num"], batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)  # label的padding为-1

    def forward(self, x, y=None):
        if self.config["model"] == "bert":
            x_encoder, _ = self.encoder(x)  # batch_size, seq_len, hidden_size(768)\
            # x_encoder, _ = self.linear1(x_encoder)
            # x_encoder = self.linear2(x_encoder)
        else:
            x_embedded = self.embedding(x)
            x_encoder, _ = self.encoder(x_embedded)
        # x_encoder = self.activation(x_encoder)
        pred = self.classify(x_encoder)
        if y is not None:
            if self.config["use_crf"]:
                mask = y.gt(-1)  # 去掉填充的-1
                return - self.crf(pred, y, mask, reduction="mean")
            else:
                pred_view = pred.view(-1, pred.shape[-1])  # batch_size * sen_len, class_num
                y_view = y.view(-1)  # batch_size * sen_len,
                return self.loss(pred_view, y_view)
        else:
            if self.config["use_crf"]:
                return self.crf.decode(pred)  # 解码
            else:
                return pred

# 选择优化器
def choose_optimizer(model, config):
    if config["optimizer"] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'adam':
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError('Unsupported optimizer: {}'.format(config["optimizer"]))
    return optimizer
