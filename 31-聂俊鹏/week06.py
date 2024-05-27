import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"D:\ai\toolutils\bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数


embedding_parameters = vocab * embedding_size + max_sequence_length * embedding_size + n * embedding_size + embedding_size + embedding_size

self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

pool_fc_parameters = embedding_size * embedding_size + embedding_size

all_paramerters = embedding_parameters + self_attention_parameters + self_attention_out_parameters + \
    feed_forward_parameters + pool_fc_parameters
print("模型实际参数个数为%d" % sum(p.numel() for p in model.parameters()))
print("计算参数个数为%d" % all_paramerters)