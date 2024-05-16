from loader import  load_vocab
import torch
import random
import numpy as np

class Translator:

    def __init__(self,config,model,logger) -> None:
        self.config = config
        self.model = model
        self.logger = logger
        self.tokenizer = load_vocab(config["pretrain_model_path"])


    def generateSentence(self,epoch,text):
        self.logger.info(f"开始进行第{epoch}轮模型效果评估")
        self.model.eval()
        # eval是模型级别的设置,不进行dropout操作，batch norm 在评估模式下会使用全局统计量等。
        # 模型不会更新权重，但是输入的梯度依然会被计算，是为了确保能过进行正确的反向传播，例如在验证过程中使用梯度下降来计算模型的性能指标
        input_ids = self.tokenizer.encode(text)
        with torch.no_grad():
            while len(input_ids) <= self.config["max_length"]:
                x = torch.LongTensor([input_ids])
                if torch.cuda.is_available():
                    x = x.cuda()
                y = self.model(x)
                y = y[0][-1] #返回的是整个句子的[bs,seq_len,hidden_size] 
                index = self.sampling_strategy(y)
                input_ids.append(index)
        return self.tokenizer.decode(input_ids)


    def sampling_strategy(self,prob_distribution):
        if random.random()>0.1:
            strategy = "greedy"
        else:
            strategy = "sampling"
        # print(f"strategy:{strategy}")
        if strategy == "greedy":
            return int(torch.argmax(prob_distribution))
        elif strategy=="sampling":
            prob_distribution = prob_distribution.cpu().numpy()
            return np.random.choice(list(range(len(prob_distribution))),p=prob_distribution)

