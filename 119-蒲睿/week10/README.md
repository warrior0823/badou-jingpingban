# Week10 作业  

Model:
```python
class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # 原始代码
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        
        # 文本表征改用bert
        self.bert = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1) 
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        seq_out, pooler_out = self.bert(x)    # out:(batch_size, seq_len, hidden_size -> [768])
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(seq_out)   #output shape:(batch_size, vocab_size)
        if y is not None:
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=-1)
```
使用Bert做文本表征，使用seq_out作为seq2seq的输出  
