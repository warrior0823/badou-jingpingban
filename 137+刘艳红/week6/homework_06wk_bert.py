# 第6周作业：计算bert的可训练参数总量，写出计算过程
import torch
from transformers import BertModel
import numpy as np
import math

bert=BertModel.from_pretrained(r"E:\Pycharm_learn\pythonProject1\wk6\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
# print(bert)
print(state_dict.keys())
n=2       # 输入句子最大个数
vocab_size=21128    # 词表大小
max_sequence_length=512 # 最大句子长度
embedding_size=768
hide_size=3072

# embeddings的参数5个
vocab_embeddings=vocab_size*embedding_size
position_embeddings=max_sequence_length*embedding_size
segmment_embeddings=n*embedding_size

embedding_parameters=vocab_embeddings+position_embeddings+segmment_embeddings+embedding_size+embedding_size
print(embedding_parameters) 

# self_attention过程的参数, 6个
# 其中embedding_size * embedding_size是权重参数，embedding_size是bias， *3是K Q V三个
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

# self_attention_out参数 4个
# 其中 embedding_size * embedding_size + embedding_size是self输出的线性层参数，
# embedding_size + embedding_size是layer_norm层参数
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

# Feed Forward参数 ，6个
# 其中embedding_size * hide_size + hide_size第一个线性层，
# embedding_size * hide_size + embedding_size第二个线性层，
# embedding_size + embedding_size是layer_norm层
feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size
print(feed_forward_parameters)

# pool_fc层参数，2个
pool_fc_parameters = embedding_size * embedding_size + embedding_size
print(pool_fc_parameters)

# 模型总参数 = embedding层参数 + self_attention参数 + self_attention_out参数 + Feed_Forward参数 + pool_fc层参数
all_paramerters = embedding_parameters + self_attention_parameters + self_attention_out_parameters + \
    feed_forward_parameters + pool_fc_parameters
print("总参数",all_paramerters)
m=0
for p in bert.parameters():
    print("===============================")
    print(p.numel())
    # x,y=p.shape
    print(p.shape)
    m=m+1
    # print(x*y)
print(m)