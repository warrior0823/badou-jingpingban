# import Library
import torch
import os
import numpy as np

from transformers import BertTokenizer
from config import Config
from loader import build_dataset, build_vocab, load_corpus
from model import build_model
from text_generate import generate_sentence

# define main function
def main(Config, save_weight=True):
    train_sample = Config['train_sample']
    batch_size = Config['batch_size']
    window_size = Config['window_size']
    vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(Config['corpus_path'])     #加载语料
    model = build_model(Config, vocab)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(Config['epoch'] ):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus, tokenizer) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(Config['corpus_path']).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == '__main__':
    from config import Config
    tokenizer = BertTokenizer.from_pretrained(r"/Users/henryzheng/Desktop/NLP/八斗学院/bert-base-chinese")
    main(Config)