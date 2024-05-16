import tokenize
import torch
import torch.nn as nn
import random
import numpy as np

from transformers import BertModel,BertTokenizer
import transformers

def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer

def encode_sentence(tokenizer, text,max_length,padding=True):
    tokenizer.truncation = True
    tokenizer.padding = 'max_length'
    old_level = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    encode_input =  tokenizer.encode_plus(text,
                            truncation='longest_first',
                            max_length= max_length,
                            padding='max_length',
                            add_special_tokens=False,
                            return_attention_mask=True,  # 返回 attention mask
                        #   return_tensors='pt'          # 返回 PyTorch 张量
                            )     
    transformers.logging.set_verbosity(old_level)
    return encode_input["input_ids"],encode_input["attention_mask"]
    
def load_corpus(corpus_path):
    corpus = ""
    with open(corpus_path,"r",encoding="utf8") as f:
        for line in f:
            corpus += line.strip()
    return corpus

class LanguageModel(nn.Module):
    def __init__(self,pretrain_model_path) -> None:
        super().__init__()
        self.bert_layer = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(self.bert_layer.config.hidden_size,self.bert_layer.config.vocab_size)
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):
        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x,_ = self.bert_layer(x,attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1,y_pred.shape[-1]),y.view(-1))
        else :
            x,_ = self.bert_layer(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred,dim=-1)

def build_sample(corpus,window_size,tokenizer):
    # 随机选择window_size 大小的文字，
    start = random.randint(0,len(corpus)-window_size-1)
    end = start+window_size
    window = corpus[start:end]
    target =corpus[start+1:end+1]
    x,mask_x = encode_sentence(tokenizer,window,window_size)
    y,mask_y = encode_sentence(tokenizer,target,window_size)
    return x,y


def build_dataset(corpus,batch_size,window_size,tokenizer):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x,y = build_sample(corpus,window_size,tokenizer)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)


def build_model(pretrain_model_path):
    model = LanguageModel(pretrain_model_path)
    return model 

def sampling_strategy(prob_distribution):
    if random.random()>0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    # print(f"strategy:{strategy}")
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy=="sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))),p=prob_distribution)
    
def generate_sentence(text,model,tokenizer,window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(text) <=30:
            text += pred_char
            x = tokenizer.encode(text[-window_size:], add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)
            y = y[0][-1] #返回的是整个句子的[bs,seq_len,hidden_size] 
            index = sampling_strategy(y)
            pred_char = tokenizer.decode(index)
    return text


def main(corpus_path,vocab_path,model_path,pretrain_model_path):
    epochs = 20
    train_sample = 5000
    window_size = 20
    batch_size = 128

    corpus = load_corpus(corpus_path)

    lr = 1e-3
    tokenizer = load_vocab(vocab_path)

    model = build_model(pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(),lr = lr)

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample//batch_size)):
            optim.zero_grad()
            x,y = build_dataset(corpus,batch_size,window_size,tokenizer)
            if torch.cuda.is_available():
                x,y = x.cuda(),y.cuda()
            loss = model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"当期训练到第{epoch+1}轮，loss为：{np.mean(watch_loss)}")
        print(generate_sentence("让他在半年之前，就不能做出",model,tokenizer,window_size))


if __name__ == "__main__":
    corpus_path = './data/lstm/corpus.txt'
    model_path = './model/'
    pretrain_model_path="/root/pretrain/bert_base_chinese"
    vocab_path= "/root/pretrain/bert_base_chinese/vocab.txt"

    # tokenizer = load_vocab(vocab_path)
    # bert endocer 会加上开始结束token[101,102]
    # [101, 6375, 800, 1762, 1288, 2399, 722, 1184, 8024, 2218, 679, 5543, 976, 1139, 102, 0, 0, 0, 0, 0
    # input_ids,attention_mask = encode_sentence(tokenizer,"让他在半年之前，就不能做出",20)
    # print(input_ids)
    # print(attention_mask)
    # x = torch.randn(1,3,4)
    # y = torch.ones((x.shape[0], x.shape[1], x.shape[1])) #[1,3,3]
    # mask = torch.tril(y)
    # print(mask)
    main(corpus_path,vocab_path,model_path,pretrain_model_path)