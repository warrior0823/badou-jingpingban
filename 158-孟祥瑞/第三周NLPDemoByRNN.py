# coding: utf-8

import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.use("TkAgg")


class TorchModel(nn.Module):
    def __init__(self, vocab, sentence_length, vector_length):
        super(TorchModel, self).__init__()
        # Embedding层将字符串转换成数字
        self.embedding = nn.Embedding(len(vocab), vector_length)
        self.rnn = nn.RNN(vector_length, vector_length, batch_first=True)
        self.classify = nn.Linear(vector_length, sentence_length + 1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijk"
    vocab = {"pad": 0}
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab["unk"] = len(chars) + 1
    print("vocab 是:", vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    without_a = list(vocab.keys())
    without_a.remove("a")
    x = random.sample(without_a, sentence_length)
    x.append("a")
    random.shuffle(x)
    # print(f"构造的样本为: {x}")
    if set("a") & set(x):
        y = x.index("a")
    else:
        y = sentence_length
    # 将字转换成序号，为了做embedding
    # print(x)
    x = [vocab.get(word, vocab["unk"]) for word in x]
    print(f"样本返回的x: {x}, y: {y}")
    return x, y


def build_dataset(vocab, batch_size, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 测试模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(vocab, 200, sentence_length)
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数: %d, 正确率: %f" % (correct, correct/(correct + wrong)))
    return correct/(correct + wrong)


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    sentence_length = 6
    char_dim = 20
    learning_rate = 0.005
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(vocab, sentence_length, char_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            # 建立数据集
            x, y = build_dataset(vocab, batch_size, sentence_length)
            # 梯度归零
            optim.zero_grad()
            # 计算loss
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch +1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding='utf8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, test_data):
    # 每个字的维度
    char_dim = 20
    # 样本文本长度
    sentence_length = 6
    # 加载字符表
    vocab = json.load(open(vocab_path, "r", encoding='utf8'))
    # 建立模型
    model = TorchModel(vocab, sentence_length, char_dim)
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in test_data:
        # 将输入序列化
        x.append([vocab[char] for char in input_string])
    # 测试模式
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        # 模型预测
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(test_data):
        print("输入: %s, 预测类别: %s, 概率: %s" % (input_string, torch.argmax(result[i]), result[i]))


if __name__ == '__main__':
    # build_vocab()
    # main()
    test_strings = ["kijabc", "bcdeaf", "gkijad", "defacb"]
    predict("model.pth", "vocab.json", test_strings)
