from transformers import BertModel

bert = BertModel.from_pretrained(r"D:\LearnNLP\bert-base-chinese", return_dict=False)
vocab_size = 21128  # 词表大小
embedding_hidden_size = 768  # embedding维度
max_sequence_num = 2  # 最大句子数量
max_sequence_len = 512  # 最大句子长度
feed_hidden_size = 3072  # Feed Forward 会放大到3072

# layer Normalization
w_layer = embedding_hidden_size
b_layer = embedding_hidden_size
layer_parameters = w_layer + b_layer

# embedding_parameters
token_embedding = vocab_size * embedding_hidden_size
segment_embedding = max_sequence_num * embedding_hidden_size
position_embedding = max_sequence_len * embedding_hidden_size
word_embedding = token_embedding + segment_embedding + position_embedding + layer_parameters

# self attention  w的形状 embedding_hidden_size * embedding_hidden_size  b的形状 embedding_hidden_size
w_q = embedding_hidden_size * embedding_hidden_size + embedding_hidden_size
w_k = embedding_hidden_size * embedding_hidden_size + embedding_hidden_size
w_v = embedding_hidden_size * embedding_hidden_size + embedding_hidden_size
self_attention_param = w_q + w_k + w_v

# self attention Liner
w_liner = embedding_hidden_size * embedding_hidden_size
b_liner = embedding_hidden_size
liner = w_liner + b_liner

# self attention output
# 过 layerNorm
self_attention_output = liner + layer_parameters

# feed forward
# 放大到 embedding_hidden_size * feed_hidden_size
# 再缩小回 feed_hidden_size * embedding_hidden_size
# 再过 layerNorm
liner1 = embedding_hidden_size * feed_hidden_size + feed_hidden_size
liner2 = feed_hidden_size * embedding_hidden_size + embedding_hidden_size
feed_forward_output = liner1 + liner2 + layer_parameters

# pooler (cls token的层)
pooler_out_param = embedding_hidden_size * embedding_hidden_size + embedding_hidden_size

# 总参数数量
all_parameters = word_embedding + self_attention_param + self_attention_output + feed_forward_output + pooler_out_param

print("模型实际参数个数为%d" % sum(p.numel() for p in bert.parameters()))
print("diy计算参数个数为%d" % all_parameters)

