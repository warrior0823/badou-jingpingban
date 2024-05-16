import json
import random
import jieba
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""
# 字符集加载
def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            word = line.strip()
            vocab[word] = index + 1  # 0是padding的位置
    return vocab

# 分类加载
def load_schema(path):
    with open(path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

class DataGenerator(Dataset):
    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.samples_number = config["samples_number"]  # 采样样本数
        self.vocab = load_vocab(config["vocab_path"])
        self.schema = load_schema(config["schema_path"])
        config["vocab_size"] = len(self.vocab)
        self.mode = None  # 用于标识训练模式还是测试模式
        self.data = []    # 保存测试集数据
        self.knwb = defaultdict(list)  # 保存训练集总数据   k {{1：2}， {2：3}}
        self.index_questions = {}  # 每个子问题对应的索引
        self.load()       # 加载数据

    def load(self):
        i = 0
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                # 字典类型为训练数据，知识库
                if isinstance(line, dict):
                    self.mode = "train"
                    questions, label = line['questions'], line['target']
                    for question in questions:
                        self.index_questions[i] = question
                        x = self.encode_sentence(question)  # 对每个问题做编码
                        x = torch.LongTensor(x)
                        # k: 类别id  v: 问题的编码集合
                        self.knwb[self.schema[label]].append(x)
                        i += 1
                # 列表类型为测试数据
                elif isinstance(line, list):
                    self.mode = "test"
                    question, label = line
                    x = self.encode_sentence(question)
                    x = torch.LongTensor(x)
                    y = torch.LongTensor([self.schema[label]])
                    self.data.append([x, y])
        return

    def encode_sentence(self, text):
        encode_text = []
        if "chars" in self.config["vocab_path"]:  # 以字作为分词
            for char in text:
                encode_text.append(self.vocab.get(char, self.vocab["[UNK]"]))
        else:  # 否则以词作为分词
            for word in jieba.cut(text):
                encode_text.append(self.vocab.get(word, self.vocab["[UNK]"]))
        encode_text = self.padding(encode_text)
        return encode_text

    # 截断补全字符
    def padding(self, x):
        if len(x) < self.config["max_len"]:
            x.extend([0] * (self.config["max_len"] - len(x)))
        else:
            x = x[:self.config["max_len"]]
        return x

    # 生成训练样本: a p n
    def random_train_sample(self):
        standard_question_id = list(self.knwb.keys())  # 获取所有标准问的id
        a_p, n = random.sample(standard_question_id, 2)  # 选择2个不同的标准问id
        while len(self.knwb[a_p]) < 2:
            a_p = random.choice(standard_question_id)
        a, p = random.choices(self.knwb[a_p], k=2, )  # a p 为同样本问题
        n = random.choice(self.knwb[n])
        return [a, p, n]

    def __len__(self):
        if self.mode == "train":
            return self.samples_number
        elif self.mode == "test":
            return len(self.data)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.random_train_sample()
        elif self.mode == "test":
            return self.data[index]


def load_data(path, config, shuffle=True):
    dg = DataGenerator(path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    # dl = load_data(Config["train_path"], Config, shuffle=False)
    # dl = load_data(Config["valid_path"], Config, shuffle=False)
    # for batch in dl:
        # print(batch)
        # print(batch[0], batch[Text_Matching_Presentation])
