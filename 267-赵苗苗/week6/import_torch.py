"""
比较通过直接调用transformers库中的BertModel加载的预训练BERT模型的参数数量，
与手动计算BERT模型参数数量的差异
"""
import  torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
# 加载预训练的BERT模型
model = BertModel.from_pretrained(r"D:\AI\nlp\八斗课程-精品班\第六周\bert-base-chinese", return_dict=False)

#手动DIY计算参数个数
def main():
    n=2                      #输入最大句子个数      
    vocab=21128              #词表数目
    max_sequence_length=512  #最大句子长度
    embedding_size=768       #embedding维度
    hide_size=3072           #隐藏层维数

    #计算词嵌入、位置嵌入、句子类型嵌入的参数数量
    #vocab * embedding_size是词表embedding参数，max_sequence_length * embedding_size是位置参数，
    #n * embedding_size是句子参数，embedding_size + embedding_size是layer_norm层参数
    embedding_parameters=vocab * embedding_size + max_sequence_length * embedding_size + n * embedding_size + embedding_size + embedding_size
    print("embedding层参数数量：",embedding_parameters)
    #计算自注意力机制中的参数数量，包括权重和偏置,*3是K Q V三个
    self_attention_parameters=(embedding_size*embedding_size + embedding_size) * 3
    print("self_attention层参数数量：",self_attention_parameters)
    #计算自注意力输出后的线性层和LayerNorm层的参数数量,其中embedding_size是self输出的线性层参数
    self_attention_out_parameters=embedding_size*embedding_size+embedding_size+embedding_size+embedding_size
    print("self_attention输出层参数数量：",self_attention_out_parameters)
    #计算前馈网络中的参数数量(两个线性层+一个layer_norm层)
    feed_forward_parameters=embedding_size*hide_size+hide_size+embedding_size*hide_size+embedding_size+embedding_size+embedding_size
    print("forward层参数数量：",feed_forward_parameters)
    #计算池化后的全连接层参数数量(权重+偏置)
    pool_fc_parameters=embedding_size*embedding_size+embedding_size
    print("池化层参数数量：",pool_fc_parameters)
    #模型总参数
    all_paramerters=embedding_parameters+self_attention_parameters+self_attention_out_parameters+feed_forward_parameters+pool_fc_parameters
    print("手动DIY计算参数个数为%d" % all_paramerters)

if __name__ == '__main__':
    main()
    print("模型实际参数个数为%d"% sum(p.numel() for p in model.parameters()))
    