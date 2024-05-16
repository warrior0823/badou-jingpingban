import Config
import torch.nn as nn
from transformers import BertModel
import torch


class Bert(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        cfg=Config.CFG()
        self.bert=BertModel.from_pretrained(cfg.pre_model_path)
        self.fc=nn.Linear(cfg.hidden_size,21128)
        self.dropout=nn.Dropout(0.1)
        self.classfy=nn.Linear(21128,21128)
        self.loss=nn.CrossEntropyLoss()
        if cfg.cuda:
            self.loss=self.loss.cuda()

    def forward(self,input_ids,input_type_ids,mask,pred_input=None):
        # with torch.no_grad():
        output=self.bert(input_ids,input_type_ids,mask)

            #[batch,hidden]
        output=output.pooler_output
        # identity=output
        output=self.fc(output)
      #  output=self.dropout(output)

        #output=self.classfy(output)

        if pred_input is not None:
            #[b,num_class],   y [b,sen_len]
            return self.loss(output,pred_input.view(-1))
        else:
            #batch_size,num_class
            return torch.softmax(output, dim=-1)
