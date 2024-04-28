#Import Library
import config as Config
import torch
import torch.nn as nn
from transformers import BertModel

#Define the model
class LanguageModel(nn.Module):
    def __init__(self, config, vocab):
        super(LanguageModel, self).__init__()
        #self.embedding = nn.Embedding(len(vocab), input_dim)
        #self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict = False)
        self.classify = nn.Linear(self.bert.config.hidden_size, 21128)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

        self.masked_attention = nn.MultiheadAttention(embed_dim = self.bert.config.hidden_size, num_heads = 8, dropout = 0.1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        #x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        #x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        x, _ = self.bert(x)

        #masked_attention
        x = x.permute(1,0,2)
        x, _ = self.masked_attention(x,x,x)
        x = x.permute(1,0,2)

        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim = -1)

def build_model(config,vocab):
    model = LanguageModel(config,vocab)
    return model

if __name__ == '__main__':
    from config import Config
    from loader import build_vocab
    vocab = build_vocab(Config["vocab_path"])
    model = build_model(Config, vocab)
    print(model)