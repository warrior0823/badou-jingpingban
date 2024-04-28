import torch
import os
from loader import load_data
os.environ["CUDA_LAUNCH_BLOCKING"] = "Text_Matching_Presentation"

"""
模型效果测试
"""
class Evaluator:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.valid_data = load_data(config["valid_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果
        self.train_data = load_data(config["train_path"], config)  # 训练集就是知识库

    # 知识库问题向量化，为匹配做准备
    # 且每轮训练的参数都不一样，所以需要重新计算
    def knwb_to_vector(self):
        self.question_labels = {}  # k:知识库中每一个问题的索引 v:问题在知识库中的标准问label
        question_list = []  # 保存每一个问题的字符ids
        i = 0
        for standard_question_index, questions in self.train_data.dataset.knwb.items():
            for question in questions:
                # 记录每一个子问题到标准问题label的映射
                self.question_labels[i] = standard_question_index
                question_list.append(question)
                i += 1
        # 问题encoder
        with torch.no_grad():  # 不计算梯度
            # shape=(n, max_len)  n 为问题总数
            question_matrix = torch.stack(question_list, dim=0)
            question_matrix = question_matrix.to(self.config["device"])
            # shape = (n, hidden_size)
            self_knwb_vectors = self.model(question_matrix)
            # 向量归一化：v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self_knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        # self.logger.info("\t开始测试第 %d 轮模型效果: " % epoch)
        # 重置上一轮结果
        self.stats_dict["correct"] = 0
        self.stats_dict["wrong"] = 0
        self.model.eval()  # 训练模式
        self.knwb_to_vector()  # 知识库问题向量初始化
        for x, y in self.valid_data:
            x, y = x.to(self.config["device"]), y.to(self.config["device"])
            with torch.no_grad():  # 不计算梯度
                x_vector = self.model(x)  # 问题向量化
            # 计算预测结果
            self.write_stats(x_vector, y)
        acc = self.show_stats()
        return acc

    def write_stats(self, x_vector, y):
        # 计算预测结果
        for x, label in zip(x_vector, y):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # x shape=(hidden_size,) knwb_vector shape=(n, hidden_size)
            res = torch.mm(x.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # 命中的问题索引
            hit_index = self.question_labels[hit_index]  # 转化为标准问编号
            if hit_index == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        # 避免除0错误
        if total == 0:
            total += 1
        acc = correct / total
        self.logger.info("Evaluating Total:%d, Correct:%d, Wrong:%d, Accuracy:%.4f" % (total, correct, wrong, acc))
        return acc
