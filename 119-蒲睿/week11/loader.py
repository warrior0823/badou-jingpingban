import json
import torch
import numpy as np
import transformers
from config import Config
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

transformers.logging.set_verbosity_error()
def load_data(data_path):
    data_set = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            data_set.append([line['tag'], line['title']])
    return data_set

def build_dataset(tokenizer, data, max_length, batch_size):
    dataset =[]
    for i, (prompt, answer) in enumerate(data):
        p_e = tokenizer.encode(prompt, add_special_tokens=False)
        # p_e = [3152, 1265]
        a_e = tokenizer.encode(answer, add_special_tokens=False)
        # a_e = [100, 2208, 2399, 679, 1377, 3619, 100, 8024, 4276, 3326, 771, 679, 1377, 3619]
        x = [tokenizer.cls_token_id] + p_e + [tokenizer.sep_token_id] + a_e + [tokenizer.sep_token_id]
        # x:[101, 3152, 1265, 102, 100, 2208, 2399, 679, 1377, 3619, 100, 8024, 4276, 3326, 771, 679, 1377, 3619, 102]
        y = len(p_e) * [-1] + [-1] + a_e + [tokenizer.sep_token_id] + [-1]
        # y:[-1, -1, -1, 100, 2208, 2399, 679, 1377, 3619, 100, 8024, 4276, 3326, 771, 679, 1377, 3619, 102, -1]
        
        mask = create_mask(len(p_e), len(a_e))
        
        x = x[:max_length] + [0]*(max_length - len(x))
        y = y[:max_length] + [0]*(max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def create_mask(s1, s2):
    len_s1 = s1 + 2
    len_s2 = s2 + 1
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    for i in range(len_s1):
        mask[i, len_s1:] = 0
    for i in range(len_s2):
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask

def pad_mask(mask, shape):
    h, w = mask.shape
    target_h, target_w = shape
    res = torch.zeros(shape, dtype=mask.dtype, device=mask.device)
    # 填充或截断
    h_start = 0
    w_start = 0
    h_end = min(h, target_h)
    w_end = min(w, target_w)
    res[h_start:h_end, w_start:w_end] = mask[:h_end - h_start, :w_end - w_start]
    return res
# data_set = load_data(config['train_data_path'])
# print(data_set[1])