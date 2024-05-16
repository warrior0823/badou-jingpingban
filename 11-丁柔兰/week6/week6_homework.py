'''
描述一个基于BERT模型的编码过程，包括词嵌入(embedding)、
                            位置嵌入(position_embedding)、
                            段落类型嵌入(token_type_embedding)，还有对应的
                            transformer的encoder层的参数量计算
'''
import torch
from torch import nn

# ======================================================BERT的输入嵌入过程====================================================================
# 假设输入的句子是 "深度学习"
input_sentence = "深度学习"
# BERT的tokenizer将句子分词并转换为索引
# 假设得到的索引是 [2450, 15486, 102, 2110]
input_ids = [2450, 15486, 102, 2110]
# -------------------------------------------------------嵌入矩阵的前置操作：分割线头-------------------------------------------------------
vocab_size = 30522  # BERT Base使用的词汇表大小
hidden_size = 768  # BERT Base使用的隐藏层大小
max_position_embeddings = 512  # BERT Base中的最大序列长度
type_vocab_size = 2  # BERT中的Token类型数量（一般为2：句子A和句子B）
# -------------------------------------------------------嵌入矩阵的前置操作：分割线尾-------------------------------------------------------
# 假设有以下嵌入矩阵
word_embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=hidden_size)  # 词嵌入矩阵, shape: [vocab_size, hidden_size]
position_embedding = nn.Embedding(num_embeddings=max_position_embeddings,
                                  embedding_dim=hidden_size)  # 位置嵌入矩阵, shape: [max_position_embeddings, hidden_size]
token_type_embedding = nn.Embedding(num_embeddings=type_vocab_size,
                                    embedding_dim=hidden_size)  # 类型嵌入矩阵, shape: [type_vocab_size, hidden_size]

# -------------------------------------------------------执行嵌入操作的前置操作：分割线头-------------------------------------------------------
# 定义位置索引
position_ids = torch.arange(len(input_ids))
# 定义token类型索引, 假设为单句子输入
token_type_ids = torch.zeros(len(input_ids), dtype=torch.long)
# 转换为PyTorch的Tensor
input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
position_ids_tensor = torch.tensor(position_ids, dtype=torch.long)
token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
# -------------------------------------------------------执行嵌入操作的前置操作：分割线尾-------------------------------------------------------
# 执行嵌入操作
embedded_input = word_embedding(input_ids_tensor) + position_embedding(position_ids_tensor) + token_type_embedding(
    token_type_ids_tensor)

# -------------------------------------------------------执行LayerNormalization和线性层变换的前置操作：分割线头-------------------------------------------------------
# 定义层归一化
layer_norm = nn.LayerNorm(hidden_size)
# 定义线性层
linear_layer = nn.Linear(hidden_size, hidden_size)
# -------------------------------------------------------执行LayerNormalization和线性层变换的前置操作：分割线尾-------------------------------------------------------
# 执行Layer Normalization和线性层变换
normalized_input = layer_norm(embedded_input)  # Layer normalization
transformed_input = linear_layer(normalized_input)  # 线性层变换


# ===================================================transformer的encoder层=======================================================================
# -------------------------------------------------------transformer_encoder的前置操作：分割线头-------------------------------------------------------
# linear、matmul、softmax、gelu
def linear(input, weight, bias=None):
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


def matmul(matrix1, matrix2):
    return torch.matmul(matrix1, matrix2)


def softmax(input):
    return nn.functional.softmax(input, dim=-1)


def gelu(input):
    return nn.functional.gelu(input)


# q_w q_b ...
q_w = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
q_b = nn.Parameter(torch.Tensor(hidden_size))
k_w = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
k_b = nn.Parameter(torch.Tensor(hidden_size))
v_w = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
v_b = nn.Parameter(torch.Tensor(hidden_size))
attention_output_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
attention_output_bias = nn.Parameter(torch.Tensor(hidden_size))
# 初始化权重和偏置
nn.init.xavier_uniform_(q_w)
nn.init.constant_(q_b, 0)
nn.init.xavier_uniform_(k_w)
nn.init.constant_(k_b, 0)
nn.init.xavier_uniform_(v_w)
nn.init.constant_(v_b, 0)
nn.init.xavier_uniform_(attention_output_weight)
nn.init.constant_(attention_output_bias, 0)

# w1 w2 b1 b2
# 假设我们有一个FFN的内部维度
intermediate_size = 3072  # 通常是hidden_size的4倍
# 初始化FFN的第一层权重和偏置
w1 = nn.Parameter(torch.Tensor(intermediate_size, hidden_size))
b1 = nn.Parameter(torch.Tensor(intermediate_size))
nn.init.xavier_uniform_(w1)
nn.init.constant_(b1, 0)
# 初始化FFN的第二层权重和偏置
w2 = nn.Parameter(torch.Tensor(hidden_size, intermediate_size))
b2 = nn.Parameter(torch.Tensor(hidden_size))
nn.init.xavier_uniform_(w2)
nn.init.constant_(b2, 0)


# -------------------------------------------------------transformer_encoder的前置操作：分割线尾-------------------------------------------------------

def transformer_encoder(x):
    # 1. Attention
    q = linear(x, q_w, q_b)  # Query
    k = linear(x, k_w, k_b)  # Key
    v = linear(x, v_w, v_b)  # Value

    # 计算self-attention
    qk = matmul(q, k.transpose())  # QK^T
    qk = softmax(qk)  # Softmax
    qkv = matmul(qk, v)  # Attention Value

    # 线性层和残差连接
    attention_output = linear(qkv, attention_output_weight, attention_output_bias)
    x = x + attention_output
    x = layer_norm(x)  # Layer Normalization

    # 2. Feedforward
    intermediate_output = gelu(matmul(x, w1) + b1)
    output = matmul(intermediate_output, w2) + b2

    # 残差连接和Layer Normalization
    x = x + output
    x = layer_norm(x)

    return x


# -------------------------------------------------------执行pooler操作的前置操作：分割线头-------------------------------------------------------
# tanh，pooler_w，pooler_b
def tanh(input):
    return torch.tanh(input)


# 假设pooler层的权重和偏置的初始化
pooler_w = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
pooler_b = nn.Parameter(torch.Tensor(hidden_size))
nn.init.xavier_uniform_(pooler_w)
nn.init.constant_(pooler_b, 0)


# -------------------------------------------------------执行pooler操作的前置操作：分割线尾-------------------------------------------------------
# 执行pooler操作
def pooler(x):
    pooled_output = tanh(matmul(x[0], pooler_w) + pooler_b)
    return pooled_output


# 假设我们只有一层encoder
encoder_output = transformer_encoder(transformed_input)
pooler_output = pooler(encoder_output)

# ===================================================参数量计算逻辑=======================================================================
# 词嵌入矩阵word_embedding = vocab_size * hidden_size,
# 位置嵌入矩阵position_embedding = max_position_embeddings * hidden_size,
# 段落（句子）类型嵌入矩阵token_type_embedding = type_vocab_size * hidden_size
# 嵌入参数
embedding_params = (vocab_size * hidden_size) + (max_position_embeddings * hidden_size) + (type_vocab_size * hidden_size)
# Layer Normalization参数
layer_norm_params = 2 * hidden_size  # 一个Layer Normalization层的参数

# Transformer参数
attention_params = 4 * (hidden_size * hidden_size + hidden_size)
feed_forward_params = 2 * (4 * hidden_size * hidden_size + 4 * hidden_size)

# Encoder层参数量
# 注意这里只有一个Layer Normalization层被计算两次（自注意力和前馈网络之后各有一个）
encoder_params = attention_params + feed_forward_params + 2 * layer_norm_params

# Pooler层参数量
pooler_params = hidden_size * hidden_size + hidden_size

# 总参数量
total_params = embedding_params + encoder_params + pooler_params

