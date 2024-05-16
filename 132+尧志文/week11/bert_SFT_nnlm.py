from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class LanguageModel(nn.Module):
    def __init__(self, input_dim):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained(r'E:\models\bert-base-chinese', return_dict=False) #加载bert
        self.classify = nn.Linear(input_dim, 21128)      #输出换成词表大小21128，input要传入bert的hidden_size 768
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, m=None):
        x, _ = self.layer(x, attention_mask=m)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=-1)
        else:
            return torch.softmax(y_pred, dim=-1)

#加载训练数据
def load_trian_data(path, batch_szie):
    tokenizer = BertTokenizer.from_pretrained(r'E:\models\bert-base-chinese')
    data = []
    train_data = []
    max_len_title = 0
    max_len_content = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            train_data.append(line)
            max_len_title = max(max_len_title, len(line["title"].strip()))
            max_len_content = max(max_len_content, len(line["content"].strip()))
    max_sen_len = max_len_title + max_len_content + 2
    for line in train_data:
        prompt = line["content"].strip() + "[SEP]" + line["title"].strip()
        prompt_code = tokenizer(prompt, add_special_tokens=False, return_tensors='pt', padding='max_length', truncation=True, max_length=max_sen_len)
        title_code = tokenizer(line["title"].strip(), add_special_tokens=False, padding='max_length', truncation=True, max_length=max_len_title)
        input_ids = prompt_code['input_ids']
        attention_mask = prompt_code['attention_mask']
        custom_attention_mask = torch.tril(torch.ones((input_ids.shape[1], input_ids.shape[1])))
        final_attention_mask = custom_attention_mask * attention_mask
        label = [-1] * len(line["content"].strip()) + title_code['input_ids'] + [tokenizer.sep_token_id]
        if len(label) < max_sen_len:
            label = label + [-1] * (max_sen_len - len(label))
        data.append([input_ids.squeeze(), torch.LongTensor([label]), final_attention_mask])
    dl = DataLoader(data, batch_size=batch_szie, shuffle=True, drop_last=True)
    return dl

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings += "[SEP]"
    with torch.no_grad():
        pred_char = ""
        abstract = ""
        title_num = 0
        while pred_char != "[SEP]" and title_num <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
#            pred_char = reverse_vocab[index]
            pred_char = tokenizer.decode([index], skip_special_tokens=True)
            abstract += pred_char
            title_num += 1
    return abstract

def train(path):
    tokenizer = BertTokenizer.from_pretrained(r'E:\models\bert-base-chinese')  # 加载bert
    cuda_flag = torch.cuda.is_available()
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    dg = load_trian_data(path, batch_size)
    print("数据加载完毕")
    model = LanguageModel(768)
    if cuda_flag:
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index, data in enumerate(dg):
            if cuda_flag:
                data = [d.cuda() for d in data]
            optim.zero_grad()
            input_ids, labels, mask = data
            loss = model(input_ids, labels, mask)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        sentence = "李阳表示为了少林寺能更好地与世界通用语言接轨，愿为少林慈幼院僧众进行英语培训。少林寺官网开设了“少林弟子之家”栏目，愿皈依佛门的人可通过在线申请——后台确认——反馈信息——方丈确认的方式，实现在线皈依。"
        print("原文:\n", sentence)
        print("摘要:\n", generate_sentence(sentence, model, tokenizer))

if __name__ == "__main__":
    train("sample_data.json")
