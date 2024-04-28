from transformers import BertModel


model = BertModel.from_pretrained(r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese", return_dict=False)
state_dict = model.state_dict()
# for k, v in state_dict.items():
#     print(k, v.size())

vocab_size = 21128    # 词表大小
position_size = 512   # 位置编码: 最大句子长度
token_size = 2        # CLS，SEP 编码
hidden_size = 768     # 隐藏层大小
feed_forward_hidden_size = 3072  # feed forward层隐藏层大小

# embedding层
# (vocab_size+position_size+token_size)*hidden_size: 词表大小+位置编码+token_type_id的embedding
# hidden_size + hidden_size: layer_norm层的weights和bias
embedding_parameters = (vocab_size + position_size + token_size)*hidden_size + hidden_size+hidden_size
print("embedding_parameters:", embedding_parameters)

# self-attention层: qkv的weights加上bias
self_attention_parameters = 3 * (hidden_size * hidden_size + hidden_size)
print("self_attention_qkv层:", self_attention_parameters)

# self_attention_out层：output层的w、b 加上 layer norm层的权重
self_attention_out_parameters = (hidden_size * hidden_size + hidden_size) + (hidden_size + hidden_size)
print("self_attention_out层:", self_attention_parameters)

# feed forward层: 2个线性层，1个layer norm层
feed_forward_parameters = feed_forward_hidden_size * hidden_size + feed_forward_hidden_size
feed_forward_parameters += hidden_size * feed_forward_hidden_size + hidden_size
feed_forward_parameters += hidden_size + hidden_size
print("feed_forward_parameters层:", feed_forward_parameters)

# pool_fc层:
pool_fc_parameters = hidden_size * hidden_size + hidden_size
print("pool_fc层:", pool_fc_parameters)

# 1层的总参数
total_parameters = (embedding_parameters + self_attention_parameters + self_attention_out_parameters
                    + feed_forward_parameters + pool_fc_parameters)
print("1层bert的参数量:", total_parameters)
total_parameters *= 12
print("12层bert的参数量:", total_parameters)

print("模型实际参数量:    %d" % sum(p.numel() for p in model.parameters()))

