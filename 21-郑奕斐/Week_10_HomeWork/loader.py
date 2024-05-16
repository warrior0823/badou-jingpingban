# Import Library
from config import Config
import random
import torch
from transformers import BertTokenizer

# Loading vocab
def build_vocab(vocab_path):
    vocab = BertTokenizer.from_pretrained(vocab_path)
    vocab = vocab.get_vocab()
    return vocab

def reverse_vocab(vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return reverse_vocab

# Loading corpus
def load_corpus(corpus_path):
    corpus = ""
    with open(corpus_path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def build_sample(window_size, corpus, tokenizer):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1 : end + 1]  #输入输出错开一位
    x = tokenizer.encode(window, padding = 'max_length', max_length = Config['max_length'], truncation = True)   #将字转换成序号
    y = tokenizer.encode(target, padding = 'max_length', max_length = Config['max_length'], truncation = True)
    return x,y

def build_dataset(sample_length, window_size, corpus, tokenizer):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus, tokenizer)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

if __name__ == '__main__':
    from config import Config
    tokenizer = BertTokenizer.from_pretrained(r"/Users/henryzheng/Desktop/NLP/八斗学院/bert-base-chinese")
    vocab = build_vocab("vocab.txt")
    reverse_vocab = reverse_vocab(vocab)
    corpus = load_corpus("corpus.txt")
    print(len(vocab),'vocab')