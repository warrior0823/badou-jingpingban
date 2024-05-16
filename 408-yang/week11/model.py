import torch 
import torch.nn as nn

from transformers import BertModel

class LanguageModel(nn.Module):
    def __init__(self,pretrain_model_path) -> None:
        super().__init__()
        self.bert_layer = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(self.bert_layer.config.hidden_size,self.bert_layer.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        nn.functional.cross_entropy

    def forward(self,x,mask=None,y=None):
        if y is not None:
            # print(f"x.shape:{x.shape},mask:{mask.shape},y shape:{y.shape}")
            x,_ = self.bert_layer(x,attention_mask=mask)
            # print(f"x.shape:{x.shape},_:{_.shape}")
            y_pred = self.classify(x)
            # print(f"y_pred.shape:{y_pred.shape}")
            return self.loss(y_pred.view(-1,y_pred.shape[-1]),y.view(-1))
        else :
            x,_ = self.bert_layer(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred,dim=-1)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
