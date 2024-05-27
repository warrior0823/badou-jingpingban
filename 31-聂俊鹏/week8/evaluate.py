
import torch
from loader import load_data

'''
    模型测试效果
'''

#    self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
#         self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果
class Evaluate:
    def __init__(self , config , model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"] , config , shuffle=False)
        self.stats_dict = {"correct" : 0 , "wrong" : 0}

    def eval(self , epoch):
        self.logger.info("开始测试， 第%d轮模型效果："%epoch)
        self.stats_dict = {"correct" : 0 , "wrong" : 0}         # 重置测试结果
        self.model.eval()                   # 设置模型为评估模式
        for index , batch_date in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_date]
            input_id , labels = batch_date      #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_result = self.model(input_id)       #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_result)
        self.show_stats()
        return


    def write_stats(self ,  labels,pred_result ):
        assert len(labels) == len(pred_result)
        for label , pred_label in zip (labels , pred_result):
            pred_label = torch.argmax(pred_label)
            if int(label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
