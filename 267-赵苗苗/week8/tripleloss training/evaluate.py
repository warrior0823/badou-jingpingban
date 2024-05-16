import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self,config,model,logger):
        self.config=config
        self.model=model
        self.logger=logger
        self.valid_data=load_data(config["valid_data_path"],config,shuffle=False)#加载验证数据集存储起来
        self.train_data=load_data(config["train_data_path"],config)#加载训练集存储起来
        self.stats_dict={"correct":0,"wrong":0}#初始化一个字典，用于存储测试结果，包括正确和错误的数量
    #将知识库中的问题向量化，为匹配做准备
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index={}
        self.question_ids=[]
        for standard_question_index,question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)]=standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad(): #上下文管理器，表示接下来的计算不会进行梯度跟踪
            question_matrixs=torch.stack(self.question_ids,dim=0)  #将问题编号列表转换为张量，dim=0 表示沿着行的方向进行堆叠
            if torch.cuda.is_available():
                question_matrixs=question_matrixs.cuda()
            self.knwb_vectors=self.model(question_matrixs)  #使用模型对问题张量进行计算，得到知识库的向量表示
            self.knwb_vectors=torch.nn.functional.normalize(self.knwb_vectors,dim=-1) #将所有向量做归一化处理
        return
    
    #记录测试的统计信息
    def write_stats(self,test_question_vectors,labels):
        assert len(labels)==len(test_question_vectors)
        for test_question_vector,label in zip(test_question_vectors,labels):
            #计算输入问题向量test_question_vector和知识库中所有问题向量self.knwb_vectors的相似度
            #test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res=torch.mm(test_question_vector.unsqueeze(0),self.knwb_vectors.T)
            hit_index=int(torch.argmax(res.squeeze())) #得到命中的问题的索引
            hit_index=self.question_index_to_standard_question_index[hit_index] #命中的问题的索引映射到标准问题编号
            if int(hit_index)==int(label):
                self.stats_dict["correct"]+=1
            else:
                self.stats_dict["wrong"]+=1
        return
    #展示测试结果的统计信息
    def show_stats(self):
        correct=self.stats_dict["correct"]
        wrong=self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" %(correct+wrong))
        self.logger.info("预测正确条目:%d, 预测错误条目:%d"%(correct,wrong))
        self.logger.info("预测正确率:%f" %(correct/(correct+wrong)))
        self.logger.info("---------------")
        return
    #进行模型效果的测试
    def eval(self,epoch):
        self.logger.info("开始测试第%d轮模型效果："% epoch)
        self.stats_dict={"correct":0,"wrong":0} #清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index,batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data=[d.cuda() for d in batch_data]
            input_id,labels=batch_data #输入变化这里需要修改
            with torch.no_grad():
                test_question_vectors=self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(test_question_vectors,labels)
        self.show_stats()
        return