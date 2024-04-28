import torch
from torch.utils.data import Dataset, DataLoader
import random
"""
数据加载
"""

class DataGenerator(Dataset):
    def __init__(self, path, config, sample_number):
        self.config = config
        self.path = path
        self.window_size = config["window_size"]
        self.sample_number = sample_number  # 样本数量
        # 根据模型加载对应的字符集
        if self.config["model"] == "bert":
            self.vocab = load_vocab(config["bert_vocab_path"])
            # self.corpus, _ = self.load_corpus_vocab()   # 加载语料库和语料库专用字符集
        else:
            self.vocab = load_vocab(config["vocab_path"])
            # self.corpus, _ = self.load_corpus_vocab()   # 加载语料库和语料库专用字符集

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.config["vocab_size"] = len(self.vocab)  # 更新配置参数

        self.data = []
        self.load()  # 加载训练数据

    # 这样好像不行，计算loss的时候更新bert的权重会出错
    def load_corpus_vocab(self):
        corpus = ""
        corpus_vocab = dict()
        with open(self.path, 'r', encoding='gbk') as f:
            for line in f:
                corpus += line.strip()
        print("语料库长度: ", len(corpus))
        # 只使用语料库中的字符作为字符集
        corpus_vocab["[PAD]"] = 0
        corpus_vocab["[UNK]"] = 1
        for char in corpus:
            if char in self.vocab.keys():
                index = self.vocab[char]
                corpus_vocab[char] = index
        return corpus, corpus_vocab

    def load(self):
        corpus = ""
        with open(self.path, 'r', encoding='gbk') as f:
            for line in f:
                corpus += line.strip()
        print("语料库长度: ", len(corpus))

        # 生成训练样本
        for _ in range(self.sample_number):
            # 如果取的句子长度小于等于窗口大小，则重新选择句子
            start_index = random.randint(0, len(corpus) - self.window_size-1)  # -1是为下一个预测的字留出位置
            end_index = start_index + self.window_size
            window_string = corpus[start_index: end_index]
            target_string = corpus[start_index+1: end_index+1]  # 预测的是一个字
            # ids转换
            x = self.encode_sentence(window_string)
            y = self.encode_sentence(target_string)
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)
            self.data.append([x, y])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 常见字符集加载
def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            word = line.replace("\n", "")
            vocab[word] = index
    return vocab

def load_data(path, config, sample_number, shuffle=True):
    dg = DataGenerator(path, config, sample_number)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    # dl = load_data(Config["corpus_path"], Config, shuffle=False)
    # dl = load_data(Config["valid_path"], Config, shuffle=False)
    # for batch in dl:
    #     print(batch[0], batch[Text_Matching_Presentation])
