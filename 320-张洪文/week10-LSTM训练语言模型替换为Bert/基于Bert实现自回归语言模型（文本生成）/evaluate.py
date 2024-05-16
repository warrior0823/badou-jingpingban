import re
import numpy as np
import torch
import os
from loader import load_data
from collections import defaultdict

"""
模型效果测试
"""
class Evaluator:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.valid_data = load_data(config["valid_path"], config, config["test_sample_number"], shuffle=False)
        self.reverse_vocab = self.valid_data.dataset.reverse_vocab

    def eval(self, epoch):
        # self.logger.info("\t开始测试第 %d 轮模型效果: " % epoch)
        # 每轮重置测试结果
        self.stats_dict = {"correct": 0, "error": 0}
        self.model.eval()  # 训练模式
        for index, batch_data in enumerate(self.valid_data):
            x, y = [i.to(self.config["device"]) for i in batch_data]  # 转移到GPU
            with torch.no_grad():  # 不计算梯度
                # 使用模型当前参数进行预测
                pred_y = self.model(x)
            # 计算预测结果
            self.write_stats(y, pred_y)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        # 选择最大的概率值的索引，即字符集中对应字符的索引
        pred_results = torch.argmax(pred_results, dim=-1).cpu().tolist()
        labels = labels.cpu().tolist()
        # 遍历每一个样本
        for pred_result, label in zip(pred_results, labels):
            text_true = ""
            text_pred = ""
            # 遍历每一个字符
            for c1, c2 in zip(pred_result, label):
                text_pred += self.reverse_vocab[c1]
                text_true += self.reverse_vocab[c2]
            if text_pred == text_true:
                # print(text_pred, text_true)
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["error"] += 1

    def show_stats(self):
        correct = self.stats_dict["correct"]
        error = self.stats_dict["error"]
        total = correct + error
        correct_rate = correct / total
        # 一行打印正确数、错误数、正确率
        self.logger.info(f"正确数：{correct}，错误数：{error}，正确率：{correct_rate * 100:.4f}%")
        return correct_rate
