import re
from collections import defaultdict
import torch
from model import LanguageModel
from loader import load_vocab
from config import Config
import random
import numpy as np

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

class Prediction:
    def __init__(self, config, weights_path):
        self.config = config
        self.vocab = load_vocab(config["bert_vocab_path"])
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.config["vocab_size"] = len(self.vocab)
        self.window_size = self.config["window_size"]
        self.model = LanguageModel(config)  # 加载模型
        self.model.load_state_dict(torch.load(weights_path))  # 加载权重

    # 对传入的文本进行ids转换
    def encode_sentence(self, sentence):
        input_ids = []
        for char in sentence:
            input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_ids

    def predict(self, sentence, pred_length):
        self.model.eval()
        with torch.no_grad():
            pred_sentence = sentence
            while len(pred_sentence) < pred_length:
                x = self.encode_sentence(pred_sentence)
                x = torch.LongTensor([x])
                pred_results = self.model(x)[0][-1]  # 得到最后一个字符的预测值
                pred_index = sampling_strategy(pred_results)
                pred_sentence += self.reverse_vocab[pred_index]
        return pred_sentence


if __name__ == '__main__':
    weight_path = "./models/NER_gru_15.pth"
    # weight_path = "./models/NER_lstm_30.pth"
    pd = Prediction(Config, weight_path)
    text1 = "李慕"
    text2 = "就在李慕迷茫前路"
    text3 = "酒楼饭菜的味道虽然比不上后世的各种美食"

    print(pd.predict(text1, 50))
    print(pd.predict(text2, 30))
    print(pd.predict(text3, 30))


