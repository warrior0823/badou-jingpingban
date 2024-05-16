import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "Text_Matching_Presentation"

"""
模型效果测试
"""
class Evaluator:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def evaluate(self, valid_data):
        # self.logger.info("Evaluating at epoch %d" % epoch)
        # 重置上一轮结果
        self.stats_dict["correct"] = 0
        self.stats_dict["wrong"] = 0

        x, y = valid_data
        x, y = x.to(self.config["device"]), y.to(self.config["device"])
        self.model.eval()  # 训练模式
        with torch.no_grad():
            # 使用模型当前参数进行预测
            y_pred = self.model(x)
            # 计算预测结果
            assert len(y) == len(y_pred)  # 断言
            for y_p, y_t in zip(y_pred, y):
                y_p = int(y_p >= 0.5)  # >= 0.5 为1，反之为0
                # 与真实标签做对比
                if y_p == int(y_t):
                    self.stats_dict["correct"] += 1  # 负样本判断正确
                else:
                    self.stats_dict["wrong"] += 1  # 负样本判断正确
        acc = self.show_stats()
        return acc

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        # 避免除0错误
        if total == 0:
            total += 1
        acc = correct / total
        # self.logger.info("\tEvaluating Total: %d, Correct: %d, Wrong: %d, Accuracy: %.4f" % (total, correct, wrong, acc))
        return acc
