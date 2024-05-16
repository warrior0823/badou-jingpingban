"""
    模型评估类
    加载评估数据，带有标签，是单独一个句子，
    需要已知训练集所有的encoder之后的表示
    计算cosine loss
    评估正确率、错误率
"""
from collections import defaultdict
from loader import load_data
import torch
import torch.nn as nn
import re
import numpy as np

ENTITY_LIST = ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]

class Evaluator:

    def __init__(self,config,model,logger) -> None:
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"],config,shuffle=False)
        self.stats_dict = {"correct":0,"wrong":0}


    def eval(self,epoch):
        self.logger.info(f"开始进行第{epoch}轮模型效果评估")
        # 对不同类型的实体需要分开进行计算
        self.stats_dict = {"LOCATION":defaultdict(int),
                           "TIME":defaultdict(int),
                           "PERSON":defaultdict(int),
                           "ORGANIZATION":defaultdict(int)
                           }
        self.model.eval()
        # eval是模型级别的设置,不进行dropout操作，batch norm 在评估模式下会使用全局统计量等。
        # 模型不会更新权重，但是输入的梯度依然会被计算，是为了确保能过进行正确的反向传播，例如在验证过程中使用梯度下降来计算模型的性能指标
        for idx,batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[idx*self.config["batch_size"]:(idx+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id,labels = batch_data
            # print(f"labels:{len(labels)}")
            # torch.nograd() 是在计算级别的设置，不进行梯度计算也不会进行反向传播，会节省大量的资源
            with torch.no_grad():
                # 只输入原始数据，不输入标签，返回每个序列的标签数据
                predict_results = self.model(input_id)
                # print(f"predict_result:{len(predict_results)}")
            self.write_stats(predict_results,labels,sentences)
        self.show_stats()
        return

    
    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")


    def write_stats(self,pred_results,labels,sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return
    
    def decode(self,sentence,labels):
        '''
            {
            "B-LOCATION": 0,
            "B-ORGANIZATION": 1,
            "B-PERSON": 2,
            "B-TIME": 3,
            "I-LOCATION": 4,
            "I-ORGANIZATION": 5,
            "I-PERSON": 6,
            "I-TIME": 7,
            "O": 8
            }
        '''
        sentence = "$"+sentence
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)",labels):
            s,e = location.span()
            # print(f"location: {sentence[s:e]}")
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)",labels):
            s,e = location.span()
            # print(f"orgnization: {sentence[s:e]}")
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
            # print(f"PERSON: {sentence[s:e]}")
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
            # print(f"TIME: {sentence[s:e]}")
        return results

    


# if __name__ == "__main__":
#     from config import config       
  
#     eval = Evaluator(config,model,logger)
    