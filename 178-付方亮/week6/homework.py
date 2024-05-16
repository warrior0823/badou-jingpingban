from transformers import BertModel

# 加载预训练的 BERT 模型
bert = BertModel.from_pretrained(r"..\..\..\bert-base-chinese", return_dict=False)

# 定义模型的配置参数
vocab_size = 21128  # 词汇表大小
num_layers = 1  # BERT 层数
hidden_size = 768  # 隐藏层大小
max_position_embeddings = 512  # 最大位置嵌入
type_vocab_size = 2  # 类型词汇表大小，表示一次输入token包含的段话数量
intermediate_size = 3072  # Transformer 层中间层的大小

# 计算各层参数数量
# Embedding 层参数
embed_params = (vocab_size * hidden_size) + (max_position_embeddings * hidden_size) + (type_vocab_size * hidden_size)
# 第一层的 LayerNorm 层参数
layer_norm1_params = hidden_size

# Transformer 层参数
q_k_v_W_b = (hidden_size * hidden_size * 3)  # Query, Key, Value 的权重
attention_output_params = (hidden_size * hidden_size) + hidden_size
layer_norm2_params = hidden_size
feed_forward_params = (hidden_size * intermediate_size * 3) + hidden_size
transformer_layer_params = (q_k_v_W_b + attention_output_params + layer_norm2_params + feed_forward_params) * num_layers

# Pooler 层参数
pooler_params = hidden_size * hidden_size

# 计算总参数数量
total_params = embed_params + layer_norm1_params + transformer_layer_params + pooler_params
print("手动计算的 BERT 模型总参数数量为：%d", total_params)

# 获取模型的实际参数数量
num_parameters = sum(p.numel() for p in bert.parameters() if p.requires_grad)
print("模型的实际总参数数量为：", num_parameters)
