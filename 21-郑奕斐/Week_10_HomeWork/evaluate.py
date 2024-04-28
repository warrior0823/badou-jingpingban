#Loading Library
import torch
import math
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r"/Users/henryzheng/Desktop/NLP/八斗学院/bert-base-chinese")

def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))

if __name__ == '__main__':
    from config import Config
    from loader import build_vocab
    from model import build_model

    vocab = build_vocab("vocab.txt")
    model = build_model(Config, vocab)
    sentence = "让他在半年之前，就不能做出"
    print(calc_perplexity(sentence, model,vocab, Config['window_size']))