import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
class LeftRightModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.bert:BertModel = BertModel.from_pretrained(config['bert_path'])
        bert_hidden_size = self.bert.config.hidden_size
        bert_vocab_size = self.bert.config.vocab_size
        print('bert_hidden_size:',bert_hidden_size)
        print('bert_vocab_size:',bert_vocab_size)
        # self.dropout = nn.Dropout(config['dropout'])
        # self.lstm_layer = nn.LSTM(input_size=bert_hidden_size,hidden_size=config['hidden_size'],num_layers=config['num_layer'],batch_first=True,bidirectional=config['bidirectional'])
        # self.classfy = nn.Linear(config['hidden_size']*2 if config['bidirectional'] else config['hidden_size'],bert_vocab_size)
        self.classfy = nn.Linear(bert_hidden_size,bert_vocab_size)
        self.loss = nn.functional.cross_entropy


    def forward(self,x,attn_mask=None,y=None):
        bert_output = self.bert(input_ids = x,attention_mask=attn_mask)
        # print('bert output:',bert_output.last_hidden_state.shape)
        # bert_output = self.dropout(bert_output.last_hidden_state)
        # lstm_output,_ = self.lstm_layer(bert_output)
        # pred = self.classfy(lstm_output)
        pred = self.classfy(bert_output.last_hidden_state)
        # print('pred:',pred.view(-1,pred.shape[-1]).shape)
        # print('target:',y.view(-1).shape)

        if y != None:
            loss = self.loss(pred.view(-1,pred.shape[-1]),y.view(-1))
            return loss
        else:
            return torch.softmax(pred,dim=-1)



def choose_optimizer(model,lr,config):
    opt = config['optimizer']
    if opt =='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt =='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer
