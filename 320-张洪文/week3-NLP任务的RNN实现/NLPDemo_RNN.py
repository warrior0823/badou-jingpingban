import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch网络编写: 
    实现一个简单的NLP任务，用于判断文本中是否有某些特定字符的出现
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, hidden_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding 层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.rnn = nn.RNN(vector_dim, hidden_dim, batch_first=True, num_layers=2)
        self.classify = nn.Linear(hidden_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # 激活函数 sigmoid 归一化
        self.loss = nn.functional.mse_loss  # 均方差loss

    def forward(self, x, y=None):
        # 词向量
        x = self.embedding(x)  # (batch_size,sen_len) -> (batch_size,sen_len,vector_dim)
        # 池化层
        x = x.transpose(1, 2)  # 池化前准备(batch_size,sen_len,vector_dim)->(batch_size,vector_dim,sen_len)
        x = self.pool(x)  # (batch_size,vector_dim,sen_len) -> (batch_size,vector_dim,1)
        x = x.squeeze()  # 去除多余维度(batch_size,vector_dim,1)->(batch_size,vector_dim)
        # RNN
        x, _ = self.rnn(x)  # rnn (batch_size,vector_dim)->(batch_size,hidden_dim)
        # 线性层
        x = self.classify(x)  # 线性层 (batch_size,vector_dim) -> (batch_size,1)
        y_pred = self.activation(x)  # 激活函数，shape不变
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 字符集设定
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 26个基本字母
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index  # 字符对应序号
    vocab["unk"] = len(vocab)
    return vocab

# 根据字符集随机生成样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取 sentence_length 个字符，可能会重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 设定正负样本规则
    if set("abc") & set(x):  # & 表示2个集合做交集运算
        y = 1
    else:
        y = 0
    # 将字符转为序号，为了做embedding. 如果word不在vocab中，则返回"unk"对应的索引值
    x = [vocab.get(word, vocab["unk"]) for word in x]
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

# 测试代码：测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()  # 模型设置为测试模式
    x, y = build_dataset(500, vocab, sample_length)  # 200个测试样本
    print("本次测试集中正样本：%d, 负样本：%d" % (sum(y), 500 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():  # 不计算梯度
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):
            # 与真实标签做对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 10  # 迭代次数
    batch_size = 20  # 每次训练样本数量
    train_sample = 500  # 训练样本
    char_dim = 64  # 每个字的维度
    hidden_dim = 128  # 隐藏层维度
    sentence_length = 6  # 样本字符长度
    learning_rate = 0.001  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(char_dim, hidden_dim, sentence_length, vocab)
    # 设定优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []  # 日志留存
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 训练模式
        watch_loss = []  # 保存loss
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 按批次构造训练样本
            optim.zero_grad()  # 梯度归0
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("===========\n第%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮结果
        log.append([acc, np.mean(watch_loss)])
    # 绘图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # loss曲线
    plt.legend()  # 图例
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as f:
        f.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    return


# 模型预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 64  # 每个字维度
    hidden_dim = 128  # 隐藏层维度
    sentence_length = 6  # 样本文字长度
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)  # 加载字符表
    model = TorchModel(char_dim, hidden_dim, sentence_length, vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载权重

    x = []
    for input_string in input_strings:
        # 输入序列化
        x.append([vocab.get(char, vocab["unk"]) for char in input_string])
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        # 打印结果
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))


if __name__ == '__main__':
    main()
    test_strings = ["fbvfee", "wbsdfg", "fqwdwg", "nakwww", "odsjhq", "sadjss"]
    predict("model.pth", "vocab.json", test_strings)
