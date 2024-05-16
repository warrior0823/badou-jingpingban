from config import config
import os 

import logging
import random
import numpy as np
import torch
import json

from loader import load_data
from translator import Translator
from model import LanguageModel,choose_optimizer

logging.basicConfig(level= logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型训练主程序
"""

seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载模型
    logger.info(json.dumps(config,ensure_ascii=False,indent=2))
    print(config["pretrain_model_path"])
    model = LanguageModel(config["pretrain_model_path"])
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移到gpu")
        model = model.cuda()

    # 配置优化器
    optim = choose_optimizer(config,model)

    # 生成训练数据
    train_data= load_data(config["train_data_path"],config,logger)
    evaluator = Translator(config,model,logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"开始第{epoch}轮训练")
        train_loss = []
        for idx,batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq,mask,y = batch_data
            loss = model(input_seq,mask,y)
            train_loss.append(loss.item())
            loss.backward()
            optim.step()
            optim.zero_grad()
        logger.info(f"epoch average loss {np.mean(train_loss)}")
        res = evaluator.generateSentence(epoch,"北京明年拟推工作日半价观看电影")
        print(res)
    model_path = os.path.join(config["model_path"],"transformer_epoch_%d.pth"%epoch)
    torch.save(model.state_dict(),model_path)
    return 
    



if __name__ == "__main__":
    main(config)