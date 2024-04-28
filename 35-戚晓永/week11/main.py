import datetime

import numpy as np
import torch
from transformers import BertTokenizer
from config import config
from model import LLMModel
from datetime import datetime
import evaluator
import loader


def main():
    tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
    model = LLMModel(tokenizer.vocab_size)
    train_data = loader.build_dataset(tokenizer, max_length=config['max_seq_len'])
    for i in range(config['num_epoch']):
        model.train()
        watch_loss = []
        for x, y, mask in train_data:
            model.optimizer.zero_grad()
            loss = model(x, y, mask)
            loss.backward()
            model.optimizer.step()
            watch_loss.append(loss.item())
        print(f'{datetime.now()},epoch {i} loss {np.mean(watch_loss)}')
        pred = evaluator.eval(model, tokenizer, "北京明年拟推工作日半价观看电影")
        print(pred)
        pred = evaluator.eval(model, tokenizer, "邓亚萍：互联网要有社会担当")
        print(pred)
        torch.save(model.state_dict(), f'{config["model_path"]}/model_{i}.pt')


def test():
    tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
    model = LLMModel(tokenizer.vocab_size)
    model.load_state_dict(torch.load(f'{config["model_path"]}/model_67.pt'))
    pred = evaluator.eval(model, tokenizer, "今天")
    print(f"今天   :{pred.replace(' ', '')}\n")
    pred = evaluator.eval(model, tokenizer, "我叫")
    print(f"我叫   :{pred.replace(' ', '')}\n")
    pred = evaluator.eval(model, tokenizer, "你是")
    print(f"你是   :{pred.replace(' ', '')}\n")
    pred = evaluator.eval(model, tokenizer, "以后")
    print(f"以后   :{pred.replace(' ', '')}\n")
    # data = loader.load_corpus()
    # for title, content in data:
    #     print(f'title:{title}')
    #     print(f'content:{content}')
    #     pred = evaluator.eval(model, tokenizer, title)
    #     print(f"pred   :{pred.replace(' ', '')}\n")


if __name__ == '__main__':
    # main()
    test()
    # tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
    # l  = tokenizer.encode([["a"],["b"]])
    # print(l)
