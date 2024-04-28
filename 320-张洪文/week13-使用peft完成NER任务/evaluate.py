import re
import numpy as np
import torch
import os
from loader import load_data
from collections import defaultdict
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
        self.sentences = self.valid_data.dataset.sentences

    def eval(self, epoch):
        # self.logger.info("\t开始测试第 %d 轮模型效果: " % epoch)
        # 每轮重置测试结果
        self.stats_dict = {  # 用于存储测试结果, 也可从分类文件中获取实体名
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int),
        }
        self.model.eval()  # 训练模式
        for index, batch_data in enumerate(self.valid_data):
            # 得到一个原始句子列表，句子索引和label索引对应，按批次取
            sentences = self.sentences[index*self.config["batch_size"]: (index+1)*self.config["batch_size"]]
            x, y = [i.to(self.config["device"]) for i in batch_data]  # 转移到GPU
            with torch.no_grad():  # 不计算梯度
                # 使用模型当前参数进行预测
                pred = self.model(x)
            # 计算预测结果
            self.write_stats(y, pred, sentences)
        macro_f1, micro_f1 = self.show_stats()
        return macro_f1, micro_f1

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        # 没有使用crf时则直接选择最大的概率值的索引，即类别label
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()  # 转为cpu上的list
            true_label = true_label.cpu().detach().tolist()

            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)

            for key in self.stats_dict.keys():
                self.stats_dict[key]["正确识别数"] += len([e for e in pred_entities[key] if e in true_entities[key]])
                self.stats_dict[key]["识别实体数"] += len(pred_entities[key])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])

    def show_stats(self):
        # 正确率 = 识别出的正确实体数 / 识别出的实体数
        # 召回率 = 识别出的正确实体数 / 样本的实体数
        F1_scores = []
        correct_number_total = 0
        recognition_number_total = 0
        sample_number_total = 0
        for key in self.stats_dict.keys():
            correct_number = self.stats_dict[key]["正确识别数"]
            recognition_number = self.stats_dict[key]["识别实体数"]
            sample_number = self.stats_dict[key]["样本实体数"]
            correct_number_total += correct_number
            recognition_number_total += recognition_number
            sample_number_total += sample_number
            # 避免除0错误
            precision = correct_number / (recognition_number+1e-5)
            recall = correct_number / (sample_number+1e-5)
            # 避免除0错误
            F1 = 2 * precision * recall / (precision + recall+1e-5)
            self.logger.info("准确率: %.4f, 召回率: %.4f, F1: %.4f, 类别: %s" % (precision, recall, F1, key))
            F1_scores.append(F1)
        # 计算Macro-F1
        macro_f1 = np.mean(F1_scores)
        # 计算Micro-F1
        micro_precision = correct_number_total / (recognition_number_total+1e-5)
        micro_recall = correct_number_total / (sample_number_total+1e-5)
        micro_f1 = 2*micro_precision*micro_recall / (micro_precision+micro_recall+1e-5)

        self.logger.info("样本实体总数: %d, 识别实体总数: %d, 识别正确实体总数: %d" %
                         (sample_number_total, recognition_number_total, correct_number_total))
        self.logger.info("Macro-F1: %f, Micro-F1: %f" % (macro_f1, micro_f1))
        return macro_f1, micro_f1

    def decode(self, sentence, labels):
        """
          {
            "B-LOCATION": 0,
            "B-ORGANIZATION": Transformers模型-生成文章标题,
            "B-PERSON": 2,
            "B-TIME": 3,
            "I-LOCATION": 4,
            "I-ORGANIZATION": 5,
            "I-PERSON": 6,
            "I-TIME": 7,
            "O": 8
          }
          """
        labels = "".join([str(x) for x in labels[:len(sentence)]])   # 根据实际的句子获取对应的标签长度
        results = defaultdict(list)
        # 通过正则匹配标注的实体
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
