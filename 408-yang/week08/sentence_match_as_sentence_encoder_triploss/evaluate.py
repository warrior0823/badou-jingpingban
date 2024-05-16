"""
    模型评估类
    加载评估数据，带有标签，是单独一个句子，
    需要已知训练集所有的encoder之后的表示
    计算cosine loss
    评估正确率、错误率
"""
from loader import load_data
import torch
import torch.nn as nn

class Evaluator:

    def __init__(self,config,model,logger) -> None:
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"],config,shuffle=False)
        # 效果测试需要训练集当做知识库
        self.train_data = load_data(config["train_data_path"],config)
        self.stats_dict = {"correct":0,"wrong":0}

    # 这部分数据已经提前存储在向量数据库中了。
    # 因为训练过程中每一次迭代的模型参数都不同，所以每一次向量都要重新计算
    def knwb_to_vector(self):
        self.question_idx_to_standard_question_idx = {}
        self.question_ids = []
        for standard_question_idx,question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_idx_to_standard_question_idx[len(self.question_ids)] = standard_question_idx
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrix = torch.stack(self.question_ids,dim=0)
            if torch.cuda.is_available():
                question_matrix = question_matrix.cuda()
            self.knwb_vectors = self.model(question_matrix)
            self.knwb_vectors = nn.functional.normalize(self.knwb_vectors)
        return 


    def eval(self,epoch):
        # 对句子进行编码，模型预测出向量，找向量数据库中最阶级的那个向量对应的问题
        self.logger.info(f"开始进行第{epoch}轮模型效果评估")
        self.stats_dict = {"correct":0,"wrong":0}
        self.model.eval()
        # eval是模型级别的设置,不进行dropout操作，batch norm 在评估模式下会使用全局统计量等。
        # 模型不会更新权重，但是输入的梯度依然会被计算，是为了确保能过进行正确的反向传播，例如在验证过程中使用梯度下降来计算模型的性能指标
        self.knwb_to_vector()
        for idx,batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id,labels = batch_data
            # torch.nograd() 是在计算级别的设置，不进行梯度计算也不会进行反向传播，会节省大量的资源
            with torch.no_grad():
                # 只输入原始数据，不输入标签，返回输入的向量化表示
                eval_question_vectors = self.model(input_id)
            self.write_stats(eval_question_vectors,labels)
        self.show_stats()

    
    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info(f"预测集合总数为：{correct+wrong}")
        self.logger.info(f"预测集合正确数目为：{correct}，错误数目为：{wrong},正确率为：{correct/(correct+wrong)}")
        self.logger.info(f"-----------------")


    def write_stats(self,eval_question_vectors,labels):
        assert len(eval_question_vectors) == len(labels)
        for eval_question_vector,label in zip(eval_question_vectors,labels):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            res = torch.mm(eval_question_vector.unsqueeze(0),self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze())) #命中问题编号
            hit_index = self.question_idx_to_standard_question_idx[hit_index]
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else :
                self.stats_dict["wrong"] += 1

        return 

    


# if __name__ == "__main__":
#     from config import config       
  
#     eval = Evaluator(config,model,logger)
    