import torch
from transformers import BertTokenizer,BertModel
from torch.optim import Adam
import matplotlib.pyplot as plt
from Config import CFG
import numpy as np
import random
from model import Bert
from loader import Data_loader,load_small_dict


small_dict=load_small_dict()

new_dict = {v : k for k, v in small_dict.items()}


def train():
    cfg=CFG()
    bert_model=Bert(vocab_size=21130)
    optimizer=Adam(bert_model.parameters(),lr=cfg.lr)
    batch_size=cfg.batch_size
    dataloder=Data_loader(cfg)
    train_sample=5000

    if cfg.cuda:
        bert_model=bert_model.cuda()
    watch_loss = []
    for i in range(cfg.epoch):
        bert_model.train()

        for batch in range(int(train_sample / batch_size)):
            x_ids,x_type,x_mask, y = dataloder.build_dataset(batch_size)  # 构建一组训练样本
            if cfg.cuda:
                x_ids,x_type,x_mask, y = x_ids.cuda(),x_type.cuda(),x_mask.cuda(),y.cuda()

            optimizer.zero_grad()  # 梯度归零
            loss = bert_model(x_ids,x_type,x_mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            watch_loss.append(loss.item())

            print("Epoch: %d, Batch: %d, loss: %.5f" % (i, batch, loss.item()))
        evaluate_model("让他在半年之前，就不能做出",bert_model,cfg)
    torch.save(bert_model,'bert_model.pth')
    plt.plot(watch_loss)
    plt.show()



def evaluate_model(opening,model,CFG):

    tokenizer=BertTokenizer.from_pretrained(CFG.vocab_path)

    pred_char=" "
    model.eval()
    with torch.no_grad():
        while pred_char != "\n" and len(opening) <= 30:
            opening += pred_char
            x_dict=tokenizer.encode_plus(opening,max_length=CFG.max_char,pad_to_max_length=True,truncation=True)
            x_ids,token_type,attention_mask=x_dict['input_ids'],x_dict['token_type_ids'],x_dict['attention_mask']
            x_ids,token_type,attention_mask=torch.LongTensor([x_ids]),torch.LongTensor([token_type]),torch.LongTensor([attention_mask])
            if CFG.cuda:
                x_ids,token_type,attention_mask=x_ids.cuda(),token_type.cuda(),attention_mask.cuda()

            y_pred=model(x_ids,token_type,attention_mask)

            y_pred=y_pred[0]

            y_pred=sampling_strategy(y_pred)



            pred_char=new_dict[y_pred]
            print(pred_char)

    return opening

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

if __name__ == '__main__':
    train()
    cfg=CFG()
    model=torch.load('bert_model.pth')

    open="让他在半年之前，就不能做出"
    pad=evaluate_model(open,model,cfg)
    print(pad)
