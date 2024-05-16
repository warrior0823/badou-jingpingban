import json
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""
class DataGenerator(Dataset):
    def __init__(self, path, config):
        self.config = config
        self.path = path
        self.vocab = load_vocab(config["vocab_path"])
        self.schema = load_schema(config["schema_path"])
        config["vocab_size"] = len(self.vocab)  # 更新配置参数
        config["class_num"] = len(self.schema)
        if self.config["model"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.data = []
        self.sentences = []  # 保存每一个句子
        self.load()  # 加载数据

    def load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            segments = f.read().split('\n\n')  # 按双换行符做句子分割
        for segment in segments:  # 遍历每个句子
            sentence = []
            labels = []
            for line in segment.split('\n'):  # 遍历每一行
                if line.strip() == "":
                    continue
                char, label = line.split()
                sentence.append(char)
                labels.append(self.schema[label])
            self.sentences.append("".join(sentence))
            if self.config["model"] == "bert":
                input_ids = self.encode_sentence_bert(sentence)
                # 所以对应的labels也需要添加特殊标记的label
                labels = [8] + labels + [8]
            else:
                input_ids = self.encode_sentence(sentence)
            labels = self.padding(labels, -1)  # 对labels进行补全截断
            self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence_bert(self, sentence):
        # 使用tokenizer对句子进行编码时，会在句子开头与结尾处添加开始标记[CLS] 和 结束标记[SEP]
        # 还需要注意的是tokenizer分词器做分词对数字可能会出现分错的情况，连续的数字会分在一起，导致和label对不上，从而不是单字符
        input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_len"],
                                          padding="max_length", truncation=True)
        return input_ids

    def encode_sentence(self, text, padding=True):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # 截断补全
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 截断补全字符
    def padding(self, input_id, pad_token=0):
        input_id = input_id[: self.config["max_len"]]
        input_id += [pad_token] * (self.config["max_len"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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
        return json.load(f)

def load_data(path, config, shuffle=True):
    dg = DataGenerator(path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_path"], Config, shuffle=False)
    # dl = load_data(Config["valid_path"], Config, shuffle=False)
    for batch in dl:
        print(batch)
    #     print(batch[0], batch[Text_Matching_Presentation])
