import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
import json
"""
数据加载
"""

class DataGenerator(Dataset):
    def __init__(self, path, config):
        self.config = config
        self.path = path
        self.max_len = config["max_len"]
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_vocab_path"])
        self.config["vocab_size"] = self.tokenizer.vocab_size

        self.data = []
        self.load()  # 加载训练数据

    def load(self):
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                text = json.loads(line)
                title, content = text["title"], text["content"]
                prompt_seq = self.tokenizer.encode(title, add_special_tokens=False)
                answer_seq = self.tokenizer.encode(content, add_special_tokens=False)

                # 输入 [cls] + prompt + [sep] + answer + [sep]
                x = [self.tokenizer.cls_token_id] + prompt_seq + [self.tokenizer.sep_token_id] + answer_seq + [self.tokenizer.sep_token_id]
                # 输出 [-1]*len(prompt) + [-1] + answer + [sep] + [-1]
                y = len(prompt_seq)*[-1] + [-1] + answer_seq + [self.tokenizer.sep_token_id] + [-1]
                # 构造掩码
                mask = self.get_mask(len(prompt_seq), len(answer_seq))
                # padding,先截断再补全
                x = x[:self.max_len] + [0]*(self.max_len - len(x))
                y = y[:self.max_len] + [0]*(self.max_len - len(y))
                x, y = torch.LongTensor(x), torch.LongTensor(y)
                # mask padding
                pad_mask = self.get_pad_mask(mask, (self.max_len, self.max_len))
                self.data.append([x, pad_mask, y])

    def get_mask(self, s1, s2):
        len_s1 = s1 + 2  # cls、sep
        len_s2 = s2 + 1  # sep
        # 创建掩码张量
        mask = torch.ones(len_s1+len_s2, len_s1+len_s2)
        mask[:len_s1, len_s1:] = 0  # s1的当前token不能看到s2的任何token
        # 遍历s2的每个token
        for i in range(len_s2):
            # s2的当前token不能看到后面的s2 token
            mask[len_s1 + i, len_s1+i+1:] = 0
        return mask

    def get_pad_mask(self, mask, target_shape):
        h, w = mask.shape  # 获取输入mask的形状
        t_h, t_w = target_shape  # 目标形状
        # 创建一个全零张量，形状为目标形状
        t_mask = torch.zeros(target_shape, dtype=mask.dtype, device=mask.device)
        # 计算需要填充或截断的区域
        h_start, w_start = 0, 0  # 开始索引
        h_end, w_end = min(h, t_h), min(w, t_w)  # 避免索引超出
        # 将输入mask中对应的目标形状区域的值复制到目标mask中
        t_mask[h_start:h_end, w_start:w_end] = mask[:h_end-h_start, :w_end-w_start]
        return t_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(path, config, shuffle=True):
    dg = DataGenerator(path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle, num_workers=0)
    return dl


if __name__ == "__main__":
    from config import Config

    # dl = load_data(Config["valid_path"], Config, shuffle=False)
    # for batch in dl:
    #     print(batch)
