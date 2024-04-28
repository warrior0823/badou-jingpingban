import random
import numpy as np
import torch

"""
模型效果测试
"""
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

class Evaluator:
    def __init__(self, model, config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

    def eval(self,  openings):
        openings_ids = self.tokenizer.encode(openings)
        self.model.eval()  # 训练模式
        with torch.no_grad():  # 不计算梯度
            while len(openings_ids) < self.config["max_len"]:
                x = torch.LongTensor([openings_ids])
                x = x.to(self.config["device"])

                pred_y = self.model(x)
                y = pred_y[0][-1]
                index = sampling_strategy(y)
                openings_ids.append(index)
        return self.tokenizer.decode(openings_ids)

