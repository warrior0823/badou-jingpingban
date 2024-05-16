import torch
import torch.nn as nn
from torch.optim import SGD, Adam

"""
建立网络模型
"""

"""
计算余弦距离： 1 - cos(a, b)
当两个向量完全相同时（即夹角为0度），余弦相似度为1，余弦距离为0。
当两个向量正交时（即夹角为90度），余弦相似度为0，余弦距离为1。
"""
def cosine_distance(tensor1, tensor2):
    # 归一化
    tensor1 = nn.functional.normalize(tensor1, dim=-1)
    tensor2 = nn.functional.normalize(tensor2, dim=-1)
    cosine = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
    return 1 - cosine

# 表示文本匹配的另外一种计算方式: Triplet Loss
def cosine_triplet_loss(a, p, n, margin=0.1):
    """
    :param a: 原点
    :param p: positive 与a同一类别的样本
    :param n: negative 与a不同类别的样本
    :param margin: 间隔常数，默认0.1
    :return: 违规三元组的平均损失差异，即锚点与正样本之间的距离加上间隔值大于锚点与负样本之间的距离
    """
    # 计算a与p的余弦距离
    pos_dist = cosine_distance(a, p)
    # 计算a与n的余弦距离
    neg_dist = cosine_distance(a, n)
    # 计算损失, 并限定最小值为0
    loss1 = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    loss2 = loss1[loss1.gt(0)]
    loss = torch.mean(loss2)
    return loss

# 文本向量化
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        vocab_size = config["vocab_size"] + 1
        embedding_size = config["embedding_size"]
        hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.linear = nn.Linear(embedding_size, hidden_size)
        # self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.pooling = nn.MaxPool1d(config["max_len"])

    def forward(self, x):
        x_embedding = self.embedding(x)  # batch_size, max_len, embedding_size
        x_linear = self.linear(x_embedding)  # batch_size, max_len, hidden_size
        # x_lstm = self.lstm(x_embedding)
        x_pool = self.pooling(x_linear.transpose(1, 2))
        x_encoder = x_pool.squeeze()  # # batch_size, hidden_size
        return x_encoder

# 表示型文本匹配模型
class PresentationModel(nn.Module):
    def __init__(self, config):
        super(PresentationModel, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = cosine_triplet_loss  # 使用余弦相似度作为损失函数

    def forward(self, x, x_p=None, x_n=None, margin=0.1):
        # 同时传入3个句子,则计算 triplet loss
        if x_p is not None and x_n is not None:
            x_a_encoder = self.sentence_encoder(x)
            x_p_encoder = self.sentence_encoder(x_p)
            x_n_encoder = self.sentence_encoder(x_n)
            return self.loss(x_a_encoder, x_p_encoder, x_n_encoder, margin)
        # 只传入一个句子时，认为在进行Encoder
        else:
            return self.sentence_encoder(x)


# 选择优化器
def choose_optimizer(model, config):
    if config["optimizer"] == 'adam':
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError('Unsupported optimizer: {}'.format(config["optimizer"]))
    return optimizer


if __name__ == '__main__':
    from config import Config
    # Config["vocab_size"] = 10
    # Config["max_len"] = 4
    # model = PresentationModel(Config)
    # s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    # s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    # t = torch.LongTensor([[1], [0]])
    # y = model(s1, s2)
    # print(y)
    # y = model(s1, s2, t)
    # print(y)
