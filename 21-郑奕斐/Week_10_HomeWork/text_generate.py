import torch
import random
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r"/Users/henryzheng/Desktop/NLP/八斗学院/bert-base-chinese")
def generate_sentence(openings, model,reverse_vocab, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过20字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings[- window_size:])
            x = torch.LongTensor([x ])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

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
    from config import Config
    from loader import build_vocab, reverse_vocab
    from model import build_model

    tokenizer = BertTokenizer.from_pretrained(r"/Users/henryzheng/Desktop/NLP/八斗学院/bert-base-chinese")
    vocab = build_vocab("vocab.txt")
    reverse_vocab = reverse_vocab(vocab)
    model = build_model(Config, vocab)
    for i in range(200):
        x = generate_sentence("让他在半年之前，就不能做出", model, reverse_vocab, 5)
        print(x)